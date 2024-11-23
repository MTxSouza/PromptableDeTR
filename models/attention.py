"""
Main module that contains the attention classes to compose both the text and image encoders.
"""
import torch
import torch.nn as nn


# Classes.
class Attention(nn.Module):


    # Special methods.
    def __init__(self, in_embedding_dim, out_embedding_dim, attention_size = None):
        """
        Initializes the Attention class that computes the attention weights with the normal 
        dot-product mechanism. Optionally, it can also compute the attention weights with a 
        deformable mechanism that uses an offset tensor from image embeddings used to deform the 
        attention weights on image based on both text and image embeddings.

        Args:
            in_embedding_dim (int): The input embedding dimension.
            out_embedding_dim (int): The output embedding dimension.
            attention_size (int): The size of the attention tensor. (Default: None)
        """
        super().__init__()

        # Layers.
        self.__query = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)
        self.__key = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)
        self.__value = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)

        self.__offset = None
        if attention_size is not None:
            self.__offset = nn.Linear(in_features=in_embedding_dim, out_features=attention_size, bias=False)

        # Attributes.
        self.__attention = None
        self.__offset_attention = None


    # Properties.
    @property
    def attention(self):
        """
        Get the attention matrix computed by the forward pass.

        Returns:
            torch.Tensor: The attention matrix.
        """
        return self.__attention


    @property
    def offset_attention(self):
        """
        Get the offset attention matrix computed by the forward pass.

        Returns:
            torch.Tensor: The offset attention matrix.
        """
        return self.__offset_attention


    # Methods.
    def forward(self, query, key, value, offset = None):
        """
        Forward pass of the attention mechanism. Make sure to enable the offset layer defining 
        the `attention_size` parameter in the constructor, otherwise it will raise an error if 
        you pass the `offset` parameter during the forward pass.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            offset (torch.Tensor): The offset tensor. (Default: None)

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the attention weights.
        attention_weights = self.__get_attention_weights(query=query, key=key)

        # Try to compute the offset tensor and aggregate to 
        # the attention weights if it is defined.
        if offset is not None:
            assert self.__offset is not None, "The offset layer is not defined."
            o = self.__offset(offset)

            # Aggregate the offset.
            assert attention_weights.size() == o.size(), "The attention weights and offset tensor must have the same size."
            attention_weights = attention_weights + o

            # Save the offset attention matrix.
            self.__offset_attention = attention_weights.clone()

        # Compute the output tensor.
        v = self.__value(value)
        output = torch.matmul(attention_weights, v)

        return output


    # Private methods.
    def __get_attention_weights(self, query, key):
        """
        Computes the attention weights given the query and key tensors.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.

        Returns:
            torch.Tensor: The attention weights.
        """
        # Get the query and key tensors.
        q = self.__query(query)
        k = self.__key(key)

        # Compute the attention weights.
        attention_weights = q @ k.transpose(-2, -1)
        attention_weights = attention_weights / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

        # Save the attention matrix.
        self.__attention = attention_weights.clone()

        return attention_weights


class MultiHeadAttention(nn.Module):


    # Special methods.
    def __init__(self, num_heads, in_embedding_dim, attention_size = None):
        """
        Initializes the MultiHeadAttention class that computes the attention weights with the 
        normal dot-product mechanism. Optionally, it can also compute the attention weights with a 
        deformable mechanism that uses an offset tensor from image embeddings used to deform the 
        attention weights on image based on both text and image embeddings.

        Args:
            num_heads (int): The number of heads.
            in_embedding_dim (int): The input embedding dimension.
            attention_size (int): The size of the attention tensor. (Default: None)
        """
        super().__init__()

        # Validate the number of heads.
        assert in_embedding_dim % num_heads == 0, "The number of heads must be divisible by the input embedding dimension."
        head_embedding_dim = in_embedding_dim // num_heads

        # Layers.
        self.__attention_heads = nn.ModuleList([
            Attention(
                in_embedding_dim=in_embedding_dim, 
                out_embedding_dim=head_embedding_dim, 
                attention_size=attention_size
            ) for _ in range(num_heads)
        ])

        self.__linear = nn.Linear(in_features=in_embedding_dim, out_features=in_embedding_dim)


    # Properties.
    @property
    def attention(self):
        """
        Get the attention matrix computed by the forward pass.

        Returns:
            Dict[str, torch.Tensor]: The attention matrix for each head.
        """
        return {"head_%d" % i: head.attention for i, head in enumerate(self.__attention_heads)}


    @property
    def offset_attention(self):
        """
        Get the offset attention matrix computed by the forward pass.

        Returns:
            Dict[str, torch.Tensor]: The offset attention matrix for each head.
        """
        return {"head_%d" % i: head.offset_attention for i, head in enumerate(self.__attention_heads)}


    # Methods.
    def forward(self, query, key, value, offset = None):
        """
        Forward pass of the multi-head attention mechanism. Make sure to enable the offset layer 
        defining the `attention_size` parameter in the constructor, otherwise it will raise an error 
        if you pass the `offset` parameter during the forward pass.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            offset (torch.Tensor): The offset tensor. (Default: None)

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the attention heads.
        attention_heads = [head(query, key, value, offset) for head in self.__attention_heads]

        # Concatenate the attention heads.
        concatenated_attention_heads = torch.cat(attention_heads, dim=-1)

        # Compute the output tensor.
        output = self.__linear(concatenated_attention_heads)

        return output
