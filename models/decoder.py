"""
This module contains the Decoder class of the model. It is based entirely on ViT 
but it uses on cross-attention layers the text embeddings as Query and the image 
embeddings as Key and Value.
"""
import torch
import torch.nn as nn


# Classes.
class Attention(nn.Module):


    # Special methods.
    def __init__(self, base_embedding_dim, embedding_dim):
        """
        Initializes the Attention class that applies the Scale 
        Dot-Product Attention mechanism.

        Args:
            base_embedding_dim (int): The dimension of the base embeddings. It is refering \
                to the input and output embedding dimensions.
            embedding_dim (int): The dimension of the embeddings that will be computed \
                by the attention mechanism.
        """
        super(Attention, self).__init__()

        # Attributes.
        self.__n_dim = embedding_dim
        self.__score = None
        self.__attention = None

        # Layers.
        self.__query = nn.Linear(in_features=base_embedding_dim, out_features=embedding_dim)
        self.__key = nn.Linear(in_features=base_embedding_dim, out_features=embedding_dim)
        self.__value = nn.Linear(in_features=base_embedding_dim, out_features=embedding_dim)


    # Properties.
    @property
    def score(self):
        """
        torch.Tensor: The attention scores.
        """
        return self.__score


    @property
    def attention(self):
        """
        torch.Tensor: The attention matrix.
        """
        return self.__attention


    # Methods.
    def forward(self, query, key, value):
        """
        Applies the Scale Dot-Product Attention mechanism.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.

        Returns:
            torch.Tensor: The attention output.
        """
        # Compute the query, key and value tensors.
        q = self.__query(query)
        k = self.__key(key)
        v = self.__value(value)

        # Compute the attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.__n_dim ** 0.5)
        self.__score = scores

        # Compute the attention matrix.
        attention = torch.nn.functional.softmax(scores, dim=-1)
        self.__attention = attention

        # Compute the attention output.
        return torch.matmul(attention, v)


class MultiHeadAttention(nn.Module):


    # Special methods.
    def __init__(self, embedding_dim, num_heads):
        """
        Initializes the HeadAttention class that applies the 
        Multi-Head Attention mechanism.

        Args:
            embedding_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()

        # Check if the embedding dimension is divisible by the number of heads.
        if not embedding_dim % num_heads == 0:
            raise ValueError("The embedding dimension must be divisible by the number of heads.")

        # Compute embdding dimension per head.
        head_embedding_dim = embedding_dim // num_heads

        # Layers.
        self.__attention = nn.ModuleList([
            Attention(base_embedding_dim=embedding_dim, embedding_dim=head_embedding_dim) 
            for _ in range(num_heads)
        ])
        self.__linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)


    # Methods.
    def forward(self, query, key, value):
        """
        Applies the Multi-Head Attention mechanism.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.

        Returns:
            torch.Tensor: The attention output.
        """
        # Compute the attention heads.
        attention_heads = self.__attention(query, key, value)

        # Concatenate the attention heads.
        attention_heads = torch.cat(attention_heads, dim=-1)

        # Apply the linear layer.
        return self.__linear(attention_heads)


class MLP(nn.Module):


    # Special methods.
    def __init__(self, embedding_dim, hidden_dim):
        """
        Initializes the MLP class that applies a Multi-Layer 
        Perceptron to the input tensor.

        Args:
            embedding_dim (int): The dimension of the input tensor.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super(MLP, self).__init__()

        # Layers.
        self.__linear_1 = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.__linear_2 = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)


    # Methods.
    def forward(self, x):
        """
        Applies the Multi-Layer Perceptron to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Apply the first linear layer.
        x = self.__linear_1(x)
        x = torch.nn.functional.gelu(x)

        # Apply the second linear layer.
        return self.__linear_2(x)


class DecoderBlock(nn.Module):


    # Special methods.
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        """
        Initializes the DecoderBlock class that applies the 
        Multi-Head Attention mechanism and the MLP to the input 
        tensor.

        Args:
            embedding_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super(DecoderBlock, self).__init__()

        # Layers.
        self.__attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.__mlp = MLP(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.__norm_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.__norm_2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.__norm_3 = nn.LayerNorm(normalized_shape=embedding_dim)


    # Methods.
    def forward(self, image_embedding, text_embedding):
        """
        Applies the DecoderBlock to the input tensor.

        Args:
            image_embedding (torch.Tensor): The image embedding tensor.
            text_embedding (torch.Tensor): The text embedding tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute Self-Attention with image embeddings.
        image_attention = self.__attention(image_embedding, image_embedding, image_embedding)

        # Apply skip connection and normalization.
        image_embedding = image_embedding + image_attention
        image_embedding = self.__norm_1(image_embedding)

        # Compute Cross-Attention with text embeddings 
        # to compute object embeddings.
        object_attention = self.__attention(text_embedding, image_embedding, image_embedding)

        # Apply skip connection and normalization.
        object_embedding = image_embedding + object_attention
        object_embedding = self.__norm_2(object_embedding)

        # Apply the MLP.
        object_embedding = self.__mlp(object_embedding)

        # Apply skip connection and normalization.
        object_embedding = object_embedding + object_embedding
        object_embedding = self.__norm_3(object_embedding)

        return object_embedding


class Decoder(nn.Module):


    # Special methods.
    def __init__(self, num_blocks, embedding_dim, num_heads, hidden_dim):
        """
        Initializes the Decoder class that applies the DecoderBlock 
        multiple times.

        Args:
            num_blocks (int): The number of DecoderBlocks.
            embedding_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super(Decoder, self).__init__()

        # Layers.
        self.__blocks = nn.ModuleList([
            DecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim) 
            for _ in range(num_blocks)
        ])


    # Methods.
    def forward(self, image_embedding, text_embedding):
        """
        Applies the Decoder to the input tensor.

        Args:
            image_embedding (torch.Tensor): The image embedding tensor.
            text_embedding (torch.Tensor): The text embedding tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Apply the DecoderBlocks.
        for block in self.__blocks:
            image_embedding = block(image_embedding, text_embedding)

        return image_embedding
