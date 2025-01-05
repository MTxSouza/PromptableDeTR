"""
Main module used to declare the text encoder block to process the input text data. It is 
a fully implementation of MobileBERT model, which is a lightweight version of BERT model 
that is designed to be more efficient in terms of computational resources and memory usage.

Paper: https://arxiv.org/abs/2004.02984
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


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

        # Define positional indices.
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = self.position_ids[:, :seq_length]

        # Word embeddings.
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

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
        word_embeddings = self.embedding_transformation(word_embeddings)

        # Sum the embeddings.
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
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
        hidden_states = self.dense(input_tensor)
        hidden_states = self.LayerNorm(hidden_states)

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
        bottlenecked_hidden_states = self.input(input_tensor)
        shared_attention = self.attention(input_tensor)

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
        # Compute query, key and value tensors.
        query = self.query(query_input)
        key = self.key(key_input)
        value = self.value(value_input)

        # Transpose the tensors.
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Compute attention scores.
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_scores = attention_scores + attention_mask
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_scores)

        # Compute final embeddings representations.
        context_layer = attention_probs @ value_layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

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
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + residual_tensor
        hidden_states = self.LayerNorm(hidden_states)

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
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

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
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + residual_tensor
        hidden_states = self.LayerNorm(hidden_states)

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
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual_tensor
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class MobileBertOutput(nn.Module):


    # Special methods.
    def __init__(self, emb_dim = 128, hidden_size = 512):
        """
        Initializes the MobileBertOutput class that is used to apply the output layer to the 
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
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + residual_tensor_1
        hidden_states = self.LayerNorm(hidden_states)
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
        self.output = MobileBertOutput(emb_dim=emb_dim, hidden_size=hidden_size)
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

        # Reduce the dimensionality of the hidden states.
        bnk_out_1, bnk_out_2, bnk_out_3, bnk_out_4 = self.bottleneck(input_tensor)

        # Apply the attention mechanism.
        att_out = self.attention(bnk_out_1, bnk_out_2, bnk_out_3, bnk_out_4, input_mask)

        # Apply the feed-forward network (FFN) layer.
        ffn_out = att_out
        for ff in self.ffn:
            ffn_out = ff(ffn_out)

        # Expand the dimensionality of the hidden states.
        itr_out = self.intermediate(ffn_out)

        # Apply the output layer.
        out_out = self.output(itr_out, ffn_out, input_tensor)

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
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

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
        # Get the first token tensor.
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)
        pooled_output = torch.tanh(pooled_output)

        return pooled_output


class MobileBertModel(nn.Module):


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
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=input_ids.device)

        # Define the token type ids.
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        # Compute the embeddings.
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Compute the encoder output.
        encoder_output = self.encoder(embedding_output, attention_mask)

        # Compute the pooled output.
        pooled_output = self.pooler(encoder_output)

        return MobileBertOutput(last_hidden_state=encoder_output, pooled_output=pooled_output)
