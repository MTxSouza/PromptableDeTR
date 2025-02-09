"""
Main module used to declare the text encoder block to process the input text data. It is 
a fully implementation of MobileBERT model, which is a lightweight version of BERT model 
that is designed to be more efficient in terms of computational resources and memory usage.

Paper: https://arxiv.org/abs/2004.02984
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from logger import Logger

# Logger.
logger = Logger(name="model")


# Structures.
@dataclass
class MobileBertOutput:
    """
    Structure used to store the output of the MobileBERT model.

    Attributes:
        last_hidden_state (torch.Tensor): The last hidden state tensor. (shape: (batch_size, seq_length, hidden_size))
        pooled_output (torch.Tensor): The pooled output tensor. (shape: (batch_size, hidden_size))
    """
    last_hidden_state: torch.FloatTensor
    pooled_output: torch.FloatTensor


# Classes.
class NoNorm(nn.Module):


    # Special methods.
    def __init__(self, feat_size = 512):
        """
        Initializes the NoNorm class that is used to replace the LayerNorm layer in the 
        MobileBERT model.

        Args:
            feat_size (int): The size of the features. (Default: 512)
        """
        super().__init__()

        # Parameters.
        self.bias = nn.Parameter(data=torch.zeros(feat_size))
        self.weight = nn.Parameter(data=torch.ones(feat_size))


    # Methods.
    def forward(self, input_tensor):
        """
        Applies the NoNorm layer to the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `NoNorm` layer to the input tensor.")
        logger.debug(msg="Input shape: %s" % (input_tensor.shape,))

        return input_tensor * self.weight + self.bias


class MobileBertEmbedding(nn.Module):


    # Special methods.
    def __init__(
            self, 
            vocab_size = 30522, 
            emb_dim = 128, 
            hidden_size = 512, 
            max_positional_emb = 512, 
            type_vocab_size = 2, 
            padding_idx = 0, 
            dropout_rate = 0.0
        ):
        """
        Initializes the MobileBertEmbedding class that converts the input tokens into 
        vector representations. It is a fully implementation of the MobileBERT embedding 
        layer with a few modifications.

        Args:
            vocab_size (int): The size of the vocabulary. (Default: 30522)
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
            max_positional_emb (int): The maximum number of positional embeddings. (Default: 512)
            type_vocab_size (int): The size of the type vocabulary. (Default: 2)
            padding_idx (int): The index used to pad the input tokens. (Default: 0)
            dropout_rate (float): The dropout rate. (Default: 0.1)
        """
        super().__init__()

        # Layers.
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(num_embeddings=max_positional_emb, embedding_dim=hidden_size)
        self.token_type_embeddings = nn.Embedding(num_embeddings=type_vocab_size, embedding_dim=hidden_size)

        emb_trans = 3 * emb_dim if emb_dim != hidden_size else emb_dim
        self.embedding_transformation = nn.Linear(in_features=emb_trans, out_features=hidden_size, bias=True)
        self.LayerNorm = NoNorm(feat_size=hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)

        # Attributes.
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.max_positional_emb = max_positional_emb
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = dropout_rate

        # Buffers.
        self.register_buffer(
            "position_ids", 
            torch.arange(max_positional_emb, dtype=torch.long).expand(1, -1), 
            persistent=False
        )


    # Methods.
    def forward(self, input_ids, token_type_ids, position_ids = None):
        """
        Applies the MobileBERT embedding layer to the input tokens.

        Args:
            input_ids (torch.Tensor): The input tokens. (shape: (batch_size, seq_length))
            token_type_ids (torch.Tensor): The token type ids. (shape: (batch_size, seq_length))
            position_ids (torch.Tensor): The positional ids. (shape: (1, seq_length))

        Returns:
            torch.Tensor: The embeddings of the input tokens. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertEmbedding` layer to the input tokens.")
        logger.debug(msg="Input tokens shape: %s" % (input_ids.shape,))
        logger.debug(msg="Token type ids shape: %s" % (token_type_ids.shape,))
        logger.debug(msg="Positional ids shape: %s" % (position_ids.shape if position_ids is not None else None))

        # Define positional indices.
        if position_ids is None:
            logger.warning(msg="No positions indices, using the default positional indices.")
            seq_length = input_ids.size(1)
            position_ids = self.position_ids[:, :seq_length]

        # Word embeddings.
        word_embeddings = self.word_embeddings(input_ids)
        logger.debug(msg="Word embeddings shape: %s" % (word_embeddings.shape,))
        position_embeddings = self.position_embeddings(position_ids)
        logger.debug(msg="Position embeddings shape: %s" % (position_embeddings.shape,))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        logger.debug(msg="Token type embeddings shape: %s" % (token_type_embeddings.shape,))

        # Apply transformation to the word embeddings.
        # Implementation from Hugging Face:
        # https://github.com/huggingface/transformers/blob/241c04d36867259cdf11dbb4e9d9a60f9cb65ebc/src/transformers/models/mobilebert/modeling_mobilebert.py#L226
        word_embeddings = torch.cat(
            tensors=[
                nn.functional.pad(input=word_embeddings[:, 1:], pad=[0, 0, 0, 1, 0, 0], value=0.0),
                word_embeddings,
                nn.functional.pad(input=word_embeddings[:, :-1], pad=[0, 0, 1, 0, 0, 0], value=0.0),
            ],
            dim=2,
        )
        logger.debug(msg="Concatenated word embeddings shape: %s" % (word_embeddings.shape,))
        word_embeddings = self.embedding_transformation(word_embeddings)
        logger.debug(msg="Transformed word embeddings shape: %s" % (word_embeddings.shape,))

        # Sum the embeddings.
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        logger.debug(msg="LayerNorm embeddings shape: %s" % (embeddings.shape,))
        embeddings = self.dropout(embeddings)

        return embeddings


class BottleneckLayer(nn.Module):


    # Special methods.
    def __init__(self, hidden_size = 512, intra_bottleneck_dim = 128):
        """
        Initializes the BottleneckLayer class that is used to reduce the dimensionality of 
        the hidden states in the MobileBERT model.

        Args:
            hidden_size (int): The size of the hidden states. (Default: 512)
            intra_bottleneck_dim (int): The size of the bottleneck layer. (Default: 128)
        """
        super().__init__()

        # Layers.
        self.dense = nn.Linear(in_features=hidden_size, out_features=intra_bottleneck_dim)
        self.LayerNorm = NoNorm(feat_size=intra_bottleneck_dim)


    # Methods.
    def forward(self, input_tensor):
        """
        Applies the bottleneck layer to the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, intra_bottleneck_dim))
        """
        logger.debug(msg="Calling `BottleneckLayer` layer to the input tensor.")
        logger.debug(msg="Input tensor shape: %s" % (input_tensor.shape,))

        hidden_states = self.dense(input_tensor)
        logger.debug(msg="Dense layer shape: %s" % (hidden_states.shape,))
        hidden_states = self.LayerNorm(hidden_states)
        logger.debug(msg="LayerNorm layer shape: %s" % (hidden_states.shape,))

        return hidden_states


class Bottleneck(nn.Module):


    # Special methods.
    def __init__(self, hidden_size = 512, intra_bottleneck_dim = 128):
        """
        Initializes the Bottleneck class that is used to apply the bottleneck layer to the input tensor.

        Args:
            hidden_size (int): The size of the hidden states. (Default: 512)
            intra_bottleneck_dim (int): The size of the bottleneck layer. (Default: 128)
        """
        super().__init__()

        # Layers.
        self.input = BottleneckLayer(hidden_size=hidden_size, intra_bottleneck_dim=intra_bottleneck_dim)
        self.attention = BottleneckLayer(hidden_size=hidden_size, intra_bottleneck_dim=intra_bottleneck_dim)
    

    # Methods.
    def forward(self, input_tensor):
        """
        Applies the bottleneck layer to the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, intra_bottleneck_dim))
        """
        logger.debug(msg="Calling `Bottleneck` layer to the input tensor.")
        logger.debug(msg="Input tensor shape: %s" % (input_tensor.shape,))

        bottlenecked_hidden_states = self.input(input_tensor)
        shared_attention = self.attention(input_tensor)

        logger.debug(msg="Returning two `shared_attention` tensors, the `input_tensor` and the `bottlenecked_hidden_states` tensor.")
        return shared_attention, shared_attention, input_tensor, bottlenecked_hidden_states


class MobileBertSelfAttention(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, hidden_size = 512, num_heads = 1, dropout_rate = 0.1):
        """
        Initializes the MobileBertSelfAttention class that is used to apply the self-attention 
        mechanism to the input tokens.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
            num_heads (int): The number of attention heads. (Default: 1)
            dropout_rate (float): The dropout rate. (Default: 0.1)
        """
        super().__init__()

        # Attributes.
        intermediate_emb_dim = emb_dim // num_heads
        self.num_attention_heads = num_heads
        self.attention_head_size = intermediate_emb_dim

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Layers.
        self.query = nn.Linear(in_features=emb_dim, out_features=intermediate_emb_dim)
        self.key = nn.Linear(in_features=emb_dim, out_features=intermediate_emb_dim)
        self.value = nn.Linear(in_features=hidden_size, out_features=intermediate_emb_dim)

        self.dropout = nn.Dropout(p=dropout_rate)


    # Methods.
    def transpose_for_scores(self, x):
        """
        Performs the transpose operation to the input tensor. Reference from 
        Hugging Face implementation: https://github.com/huggingface/transformers/blob/241c04d36867259cdf11dbb4e9d9a60f9cb65ebc/src/transformers/models/mobilebert/modeling_mobilebert.py#L261

        Args:
            x (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, num_attention_heads, seq_length, attention_head_size))
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, query_input, key_input, value_input, attention_mask):
        """
        Applies the self-attention mechanism to the input tensors.

        Args:
            query_input (torch.Tensor): The query tensor. (shape: (batch_size, seq_length, emb_dim))
            key_input (torch.Tensor): The key tensor. (shape: (batch_size, seq_length, emb_dim))
            value_input (torch.Tensor): The value tensor. (shape: (batch_size, seq_length, hidden_size))
            attention_mask (torch.Tensor): The attention mask tensor. (shape: (batch_size, 1, 1, seq_length))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertSelfAttention` layer to the input tensors.")
        logger.debug(msg="Query tensor shape: %s" % (query_input.shape,))
        logger.debug(msg="Key tensor shape: %s" % (key_input.shape,))
        logger.debug(msg="Value tensor shape: %s" % (value_input.shape,))

        # Compute query, key and value tensors.
        query = self.query(query_input)
        logger.debug(msg="Query tensor shape: %s" % (query.shape,))
        key = self.key(key_input)
        logger.debug(msg="Key tensor shape: %s" % (key.shape,))
        value = self.value(value_input)
        logger.debug(msg="Value tensor shape: %s" % (value.shape,))

        # Transpose the tensors.
        query_layer = self.transpose_for_scores(query)
        logger.debug(msg="Query layer shape: %s" % (query_layer.shape,))
        key_layer = self.transpose_for_scores(key)
        logger.debug(msg="Key layer shape: %s" % (key_layer.shape,))
        value_layer = self.transpose_for_scores(value)
        logger.debug(msg="Value layer shape: %s" % (value_layer.shape,))

        # Compute attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        logger.debug(msg="Attention scores shape: %s" % (attention_scores.shape,))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_scores = attention_scores + attention_mask
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        logger.debug(msg="Attention scores shape: %s" % (attention_scores.shape,))

        attention_probs = self.dropout(attention_scores)

        # Compute final embeddings representations.
        context_layer = attention_probs @ value_layer
        logger.debug(msg="Context layer shape: %s" % (context_layer.shape,))
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        logger.debug(msg="Context layer shape: %s" % (context_layer.shape,))
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        logger.debug(msg="New context layer shape: %s" % (context_layer.shape,))

        return context_layer


class MobileBertSelfOutput(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, num_heads = 1):
        """
        Initializes the MobileBertSelfOutput class that is used to apply the self-output 
        layer to the input tensor.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            num_heads (int): The number of attention heads. (Default: 1)
        """
        super().__init__()

        # Attributes.
        intermediate_emb_dim = emb_dim // num_heads

        # Layers.
        self.dense = nn.Linear(in_features=intermediate_emb_dim, out_features=intermediate_emb_dim)
        self.LayerNorm = NoNorm(feat_size=intermediate_emb_dim)
    

    # Methods.
    def forward(self, hidden_states, residual_tensor):
        """
        Applies the self-output layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor. (shape: (batch_size, seq_length, hidden_size))
            residual_tensor (torch.Tensor): The input tensor used to perform a residual connection. (shape: (batch_size, seq_length, emb_dim))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertSelfOutput` layer to the input tensors.")
        logger.debug(msg="Hidden states shape: %s" % (hidden_states.shape,))
        logger.debug(msg="Residual tensor shape: %s" % (residual_tensor.shape,))

        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = hidden_states + residual_tensor
        hidden_states = self.LayerNorm(hidden_states)
        logger.debug(msg="Output tensor shape: %s" % (hidden_states.shape,))

        return hidden_states


class MobileBertAttention(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, hidden_size = 512, num_heads = 1):
        """
        Initializes the MobileBertAttention class that is used to apply the attention mechanism to 
        the input tensors.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
            num_heads (int): The number of attention heads. (Default: 1)
        """
        super().__init__()

        # Layers.
        self.self = MobileBertSelfAttention(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads)
        self.output = MobileBertSelfOutput(emb_dim=emb_dim, num_heads=num_heads)
    

    # Methods.
    def forward(self, query_input, key_input, value_input, input_tensor, attention_mask):
        """
        Applies the attention layer to the input tensors.

        Args:
            query_input (torch.Tensor): The query tensor. (shape: (batch_size, seq_length, emb_dim))
            key_input (torch.Tensor): The key tensor. (shape: (batch_size, seq_length, emb_dim))
            value_input (torch.Tensor): The value tensor. (shape: (batch_size, seq_length, hidden_size))
            input_tensor (torch.Tensor): The input tensor used to perform a skip connection. (shape: (batch_size, seq_length, hidden_size))
            attention_mask (torch.Tensor): The attention mask tensor. (shape: (batch_size, 1, 1, seq_length))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertAttention` layer to the input tensors.")
        logger.debug(msg="Query tensor shape: %s" % (query_input.shape,))
        logger.debug(msg="Key tensor shape: %s" % (key_input.shape,))
        logger.debug(msg="Value tensor shape: %s" % (value_input.shape,))
        logger.debug(msg="Input tensor shape: %s" % (input_tensor.shape,))
        logger.debug(msg="Attention mask shape: %s" % (attention_mask.shape,))

        # Applies self attention.
        self_output = self.self(query_input, key_input, value_input, attention_mask)

        attention_output = self.output(self_output, input_tensor)

        return attention_output


class MobileBertIntermediate(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, intermediate_size = 512):
        """
        Initializes the MobileBertIntermediate class that is used to apply an intermediate 
        layer to the input tensor.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            intermediate_size (int): The size of the intermediate layer. (Default: 512)
        """
        super().__init__()

        # Layers.
        self.dense = nn.Linear(in_features=emb_dim, out_features=intermediate_size)
        self.intermediate_act_fn = nn.ReLU()


    # Methods.
    def forward(self, hidden_states):
        """
        Applies the intermediate layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, intermediate_size))
        """
        logger.debug(msg="Calling `MobileBertIntermediate` layer to the input tensor.")
        logger.debug(msg="Input tensor shape: %s" % (hidden_states.shape,))

        hidden_states = self.dense(hidden_states)
        logger.debug(msg="Dense layer shape: %s" % (hidden_states.shape,))
        hidden_states = self.intermediate_act_fn(hidden_states)
        logger.debug(msg="Intermediate activation function shape: %s" % (hidden_states.shape,))

        return hidden_states


class FFNOutput(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, intermediate_size = 512):
        """
        Initializes the FFNOutput class that is used to apply the feed-forward network (FFN) output 
        layer to the input tensor.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            intermediate_size (int): The size of the intermediate layer. (Default: 512)
        """
        super().__init__()
        self.dense = nn.Linear(in_features=intermediate_size, out_features=emb_dim)
        self.LayerNorm = NoNorm(feat_size=emb_dim)


    # Methods.
    def forward(self, hidden_states, residual_tensor):
        """
        Applies the feed-forward network (FFN) output layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor. (shape: (batch_size, seq_length, intermediate_size))
            residual_tensor (torch.Tensor): The input tensor used to perform a skip connection. (shape: (batch_size, seq_length, emb_dim))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, emb_dim))
        """
        logger.debug(msg="Calling `FFNOutput` layer to the input tensors.")
        logger.debug(msg="Hidden states shape: %s" % (hidden_states.shape,))
        logger.debug(msg="Residual tensor shape: %s" % (residual_tensor.shape,))

        hidden_states = self.dense(hidden_states)
        logger.debug(msg="Dense layer shape: %s" % (hidden_states.shape,))
        hidden_states = hidden_states + residual_tensor
        hidden_states = self.LayerNorm(hidden_states)
        logger.debug(msg="Output tensor shape: %s" % (hidden_states.shape,))

        return hidden_states


class FFNLayer(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, intermediate_size = 512):
        """
        Initializes the FFNLayer class that is used to apply the feed-forward network (FFN) layer to 
        the input tensor.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            intermediate_size (int): The size of the intermediate layer. (Default: 512)
        """
        super().__init__()

        self.intermediate = MobileBertIntermediate(emb_dim=emb_dim, intermediate_size=intermediate_size)
        self.output = FFNOutput(emb_dim=emb_dim, intermediate_size=intermediate_size)


    # Methods.
    def forward(self, hidden_states):
        """
        Applies the feed-forward network (FFN) layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, emb_dim))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, emb_dim))
        """
        logger.debug(msg="Calling `FFNLayer` layer to the input tensor.")
        logger.debug(msg="Input tensor shape: %s" % (hidden_states.shape,))

        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)

        return layer_output


class OutputBottleneck(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, hidden_size = 512, dropout_rate = 0.0):
        """
        Initializes the OutputBottleneck class that is used to reduce the dimensionality of the 
        hidden states in the MobileBERT model.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
            dropout_rate (float): The dropout rate. (Default: 0.1)
        """
        super().__init__()

        # Layers.
        self.dense = nn.Linear(in_features=emb_dim, out_features=hidden_size)
        self.LayerNorm = NoNorm(feat_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)


    # Methods.
    def forward(self, hidden_states, residual_tensor):
        """
        Applies the output bottleneck layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, emb_dim))
            residual_tensor (torch.Tensor): The residual tensor used to perform a skip connection. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `OutputBottleneck` layer to the input tensors.")
        logger.debug(msg="Hidden states shape: %s" % (hidden_states.shape,))
        logger.debug(msg="Residual tensor shape: %s" % (residual_tensor.shape,))

        hidden_states = self.dense(hidden_states)
        logger.debug(msg="Dense layer shape: %s" % (hidden_states.shape,))
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual_tensor
        hidden_states = self.LayerNorm(hidden_states)
        logger.debug(msg="Output tensor shape: %s" % (hidden_states.shape,))

        return hidden_states


class MobileBertOutputBlock(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, hidden_size = 512):
        """
        Initializes the MobileBertOutputBlock class that is used to apply the output layer to the 
        input tensor.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
        """
        super().__init__()

        # Layers.
        self.dense = nn.Linear(hidden_size, emb_dim)
        self.LayerNorm = NoNorm(emb_dim)
        self.bottleneck = OutputBottleneck(emb_dim, hidden_size)


    # Methods.
    def forward(self, hidden_states, residual_tensor_1, residual_tensor_2):
        """
        Applies the output layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))
            residual_tensor_1 (torch.Tensor): The residual tensor used to perform a skip connection with the linear projection. (shape: (batch_size, seq_length, emb_dim))
            residual_tensor_2 (torch.Tensor): The residual tensor used to perform a skip connection with the bottleneck layer. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, emb_dim))
        """
        logger.debug(msg="Calling `MobileBertOutputBlock` layer to the input tensors.")
        logger.debug(msg="Hidden states shape: %s" % (hidden_states.shape,))
        logger.debug(msg="Residual tensor 1 shape: %s" % (residual_tensor_1.shape,))
        logger.debug(msg="Residual tensor 2 shape: %s" % (residual_tensor_2.shape,))

        hidden_states = self.dense(hidden_states)
        logger.debug(msg="Dense layer shape: %s" % (hidden_states.shape,))
        hidden_states = hidden_states + residual_tensor_1
        hidden_states = self.LayerNorm(hidden_states)
        logger.debug(msg="LayerNorm layer shape: %s" % (hidden_states.shape,))
        hidden_states = self.bottleneck(hidden_states, residual_tensor_2)

        return hidden_states


class MobileBertLayer(nn.Module):


    # Special methods.
    def __init__(
        self, 
        emb_dim = 128, 
        hidden_size = 512, 
        intermediate_size = 512, 
        intra_bottleneck_dim = 128, 
        num_heads = 1
        ):
        """
        Initializes the MobileBertLayer class that is used to process the input tokens in the 
        MobileBERT model.

        Args:
            emb_dim (int): The dimension of the embeddings. (Default: 128)
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
            intermediate_size (int): The size of the intermediate layer. (Default: 512)
            intra_bottleneck_dim (int): The size of the bottleneck layer. (Default: 128)
            num_heads (int): The number of attention heads. (Default: 1)
        """
        super().__init__()

        # Layers.
        self.attention = MobileBertAttention(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads)
        self.intermediate = MobileBertIntermediate(emb_dim=emb_dim, intermediate_size=intermediate_size)
        self.output = MobileBertOutputBlock(emb_dim=emb_dim, hidden_size=hidden_size)
        self.bottleneck = Bottleneck(hidden_size=hidden_size, intra_bottleneck_dim=intra_bottleneck_dim)
        self.ffn = nn.ModuleList([FFNLayer(emb_dim=emb_dim, intermediate_size=intermediate_size) for _ in range(3)])


    # Methods.
    def forward(self, input_tensor, input_mask):
        """
        Applies the MobileBERT layer to the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))
            input_mask (torch.Tensor): The attention mask tensor. (shape: (batch_size, 1, 1, seq_length))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertLayer` layer to the input tensor.")
        logger.debug(msg="Input tensor shape: %s" % (input_tensor.shape,))
        logger.debug(msg="Attention mask shape: %s" % (input_mask.shape,))

        # Reduce the dimensionality of the hidden states.
        bnk_out_1, bnk_out_2, bnk_out_3, bnk_out_4 = self.bottleneck(input_tensor)
        logger.debug(msg="Bottleneck output 1 shape: %s" % (bnk_out_1.shape,))
        logger.debug(msg="Bottleneck output 2 shape: %s" % (bnk_out_2.shape,))
        logger.debug(msg="Bottleneck output 3 shape: %s" % (bnk_out_3.shape,))
        logger.debug(msg="Bottleneck output 4 shape: %s" % (bnk_out_4.shape,))

        # Apply the attention mechanism.
        att_out = self.attention(bnk_out_1, bnk_out_2, bnk_out_3, bnk_out_4, input_mask)
        logger.debug(msg="Attention output shape: %s" % (att_out.shape,))

        # Apply the feed-forward network (FFN) layer.
        ffn_out = att_out
        for ff in self.ffn:
            ffn_out = ff(ffn_out)
            logger.debug(msg="FFN output shape: %s" % (ffn_out.shape,))


        # Expand the dimensionality of the hidden states.
        itr_out = self.intermediate(ffn_out)
        logger.debug(msg="Intermediate output shape: %s" % (itr_out.shape,))

        # Apply the output layer.
        out_out = self.output(itr_out, ffn_out, input_tensor)
        logger.debug(msg="Output output shape: %s" % (out_out.shape,))

        return out_out


class MobileBertEncoder(nn.Module):


    # Special methods.
    def __init__(
        self, 
        emb_dim = 128, 
        hidden_size = 512, 
        intermediate_size = 512, 
        intra_bottleneck_dim = 128, 
        num_heads = 1, 
        num_layers = 24
        ):
        super().__init__()

        # Layers.
        self.layer = nn.ModuleList([
            MobileBertLayer(
                emb_dim=emb_dim, 
                hidden_size=hidden_size, 
                intermediate_size=intermediate_size, 
                intra_bottleneck_dim=intra_bottleneck_dim, 
                num_heads=num_heads
            ) 
            for _ in range(num_layers)
        ])


    # Methods.
    def forward(self, hidden_states, attention_mask):
        """
        Applies the MobileBERT encoder to the input tensor.

        Args:
            hidden_states (torch.Tensor): The input tensor. (shape: (batch_size, seq_length, hidden_size))
            attention_mask (torch.Tensor): The attention mask tensor. (shape: (batch_size, 1, 1, seq_length))

        Returns:
            torch.Tensor: The output tensor. (shape: (batch_size, seq_length, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertEncoder` layer to the input tensor.")
        logger.debug(msg="Input tensor shape: %s" % (hidden_states.shape,))
        logger.debug(msg="Attention mask shape: %s" % (attention_mask.shape,))

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            logger.debug(msg="Hidden states shape: %s" % (hidden_states.shape,))

        return hidden_states


class MobileBertPooler(nn.Module):


    # Special methods.
    def __init__(self, hidden_size = 512):
        """
        Initializes the MobileBertPooler class that is used to pool the hidden states of the 
        MobileBERT model.

        Args:
            hidden_size (int): The size of the hidden layer in the embedding layer. (Default: 512)
        """
        super().__init__()

        # Layers.
        self.dense = nn.Linear(hidden_size, hidden_size)


    # Methods.
    def forward(self, hidden_states):
        """
        Applies the pooling layer to the hidden states.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor. (shape: (batch_size, seq_length, hidden_size))

        Returns:
            torch.Tensor: The pooled output tensor. (shape: (batch_size, hidden_size))
        """
        logger.debug(msg="Calling `MobileBertPooler` layer to the input tensor.")
        logger.debug(msg="Hidden states shape: %s" % (hidden_states.shape,))

        # Get the first token tensor.
        first_token_tensor = hidden_states[:, 0]
        logger.debug(msg="First token tensor shape: %s" % (first_token_tensor.shape,))

        pooled_output = self.dense(first_token_tensor)
        logger.debug(msg="Dense layer shape: %s" % (pooled_output.shape,))
        pooled_output = torch.tanh(pooled_output)

        return pooled_output


class MobileBert(nn.Module):


    # Special methods.
    def __init__(
        self, 
        vocab_size = 30522, 
        emb_dim = 128, 
        hidden_size = 512, 
        max_positional_emb = 512, 
        type_vocab_size = 2, 
        padding_idx = 0, 
        emb_dropout_rate = 0.0, 
        intermediate_size = 512, 
        intra_bottleneck_dim = 128, 
        num_heads = 1, 
        num_layers = 24
        ):
        super().__init__()

        # Layers.
        self.embeddings = MobileBertEmbedding(
            vocab_size=vocab_size, 
            emb_dim=emb_dim, 
            hidden_size=hidden_size, 
            max_positional_emb=max_positional_emb, 
            type_vocab_size=type_vocab_size, 
            padding_idx=padding_idx, 
            dropout_rate=emb_dropout_rate
        )
        self.encoder = MobileBertEncoder(
            emb_dim=emb_dim, 
            hidden_size=hidden_size, 
            intermediate_size=intermediate_size, 
            intra_bottleneck_dim=intra_bottleneck_dim, 
            num_heads=num_heads, 
            num_layers=num_layers
        )
        self.pooler = MobileBertPooler(hidden_size=hidden_size)


    # Methods.
    def get_extended_attention_mask(self, attention_mask):
        """
        Extends the attention mask tensor to match the shape of the attention scores tensor.

        Args:
            attention_mask (torch.Tensor): The attention mask tensor. (shape: (batch_size, seq_length))

        Returns:
            torch.Tensor: The extended attention mask tensor. (shape: (batch_size, 1, 1, seq_length))
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask


    def forward(self, input_ids, token_type_ids = None, attention_mask = None):
        """
        Applies the MobileBERT model to the input tokens.

        Args:
            input_ids (torch.Tensor): The input tokens. (shape: (batch_size, seq_length))
            token_type_ids (torch.Tensor): The token type ids. (shape: (batch_size, seq_length))
            attention_mask (torch.Tensor): The attention mask tensor. (shape: (batch_size, 1, 1, seq_length))

        Returns:
            MobileBertOutput: The output of the MobileBERT model.
        """
        # Get the input shape.
        input_shape = input_ids.size()

        # Define the attention mask.
        if attention_mask is None:
            logger.warning(msg="No attention mask tensor, using the default attention mask.")
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=input_ids.device)
            attention_mask = self.get_extended_attention_mask(attention_mask)
        logger.debug(msg="Attention mask shape: %s" % (attention_mask.shape,))

        # Define the token type ids.
        if token_type_ids is None:
            logger.warning(msg="No token type ids tensor, using the default token type ids.")
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        logger.debug(msg="Token type ids shape: %s" % (token_type_ids.shape,))

        # Compute the embeddings.
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Compute the encoder output.
        encoder_output = self.encoder(embedding_output, attention_mask)

        # Compute the pooled output.
        pooled_output = self.pooler(encoder_output)

        return MobileBertOutput(last_hidden_state=encoder_output, pooled_output=pooled_output)


if __name__ == "__main__":

    # Get arguments.
    import argparse
    import os

    from torchsummary import summary


    def check_weight_file(path):

        # Check if the file exists.
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError("The file '%s' does not exist." % path)

        # Check if it is a file.
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError("The path '%s' is not a file." % path)

        # Check if the file is a PyTorch model file.
        if not path.endswith(".pth"):
            raise argparse.ArgumentTypeError("The file '%s' is not a valid PyTorch model file." % path)

        return path

    parser = argparse.ArgumentParser(prog="MobileBERT model", description=__doc__)
    parser.add_argument("--weight-path", "-w", type=check_weight_file, required=True, help="The path to the pre-trained weights.")

    args = parser.parse_args()

    # Initialize the MobileBERT model and load the pre-trained weights.
    encoder = MobileBert()

    # Load the pre-trained weights.
    encoder.load_state_dict(state_dict=torch.load(f=args.weight_path, weights_only=True))
    summary(model=encoder, input_data=torch.randint(low=0, high=30522, size=(1, 490)))
