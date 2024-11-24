"""
Main module that contains the attention classes to compose both the text and image encoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Classes.
class Attention(nn.Module):


    # Special methods.
    def __init__(self, in_embedding_dim, out_embedding_dim):
        """
        Initializes the Attention class that computes the attention weights with the normal 
        dot-product mechanism.

        Args:
            in_embedding_dim (int): The input embedding dimension.
            out_embedding_dim (int): The output embedding dimension.
        """
        super().__init__()

        # Layers.
        self.__query = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)
        self.__key = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)
        self.__value = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)

        # Attributes.
        self.__attention = None


    # Properties.
    @property
    def attention(self):
        """
        Get the attention matrix computed by the forward pass.

        Returns:
            torch.Tensor: The attention matrix.
        """
        return self.__attention


    # Methods.
    def forward(self, query, key, value):
        """
        Forward pass of the attention mechanism.

        Args:
            query (torch.Tensor): The query tensor, usually the text embeddings.
            key (torch.Tensor): The key tensor, usually the image embeddings.
            value (torch.Tensor): The value tensor, usually the image embeddings.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the attention weights.
        attention_weights = self.__get_attention_weights(query=query, key=key)

        # Compute the output tensor.
        output = self.__get_output_embedding(attention_weights=attention_weights, value=value)

        return output


    # Private methods.
    def __get_attention_weights(self, query, key):
        """
        Computes the attention weights given the query and key tensors.

        Args:
            query (torch.Tensor): The query tensor, usually the text embeddings.
            key (torch.Tensor): The key tensor, usually the image embeddings.

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


    def __get_output_embedding(self, attention_weights, value):
        """
        Computes the output embedding given the attention weights and value tensor.

        Args:
            attention_weights (torch.Tensor): The attention weights.
            value (torch.Tensor): The value tensor, usually the image embeddings.

        Returns:
            torch.Tensor: The output embedding.
        """
        # Compute the output tensor.
        v = self.__value(value)
        output = torch.matmul(attention_weights, v)

        return output


class DeformableAttention(nn.Module):


    # Special methods.
    def __init__(self, in_embedding_dim, out_embedding_dim, num_points, patch_size):
        """
        Initializes the DeformableAttention class that computes the attention weights with a 
        deformable mechanism that uses an offset tensor from image embeddings used to deform the 
        attention weights on image based on both text and image embeddings.

        Args:
            in_embedding_dim (int): The input embedding dimension.
            out_embedding_dim (int): The output embedding dimension.
            num_points (int): The number of points to deform the attention weights.
            patch_size (int): The size of the patch used to compute the reference points.
        """
        super().__init__()

        # Layers.
        self.__offset = nn.Linear(in_features=in_embedding_dim, out_features=num_points * 2, bias=False)
        self.__att_weight = nn.Linear(in_features=in_embedding_dim, out_features=num_points, bias=False)
        self.__value = nn.Linear(in_features=in_embedding_dim, out_features=out_embedding_dim, bias=False)

        # Attributes.
        self.__offset_attention = None
        self.__patch_size = patch_size
        self.__num_points = num_points

        # Buffers.
        self.register_buffer(name="reference_points", tensor=self.__get_reference_points(height=patch_size, width=patch_size))


    # Properties.
    @property
    def offset_attention(self):
        """
        Get the offset attention matrix computed by the forward pass.

        Returns:
            torch.Tensor: The offset attention matrix.
        """
        return self.__offset_attention


    # Methods.
    def forward(self, query, value):
        """
        Forward pass of the deformable attention mechanism.

        Args:
            query (torch.Tensor): The query tensor, usually the text embeddings.
            value (torch.Tensor): The value tensor, usually the image embeddings.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the attention weights.
        offsets, attention_weights = self.__get_offsets_and_attention_weights(query=query)

        # Apply the offsets to the reference points.
        sampled_values = self.__apply_offsets_to_reference_points(
            reference_points=self.reference_points,
            offsets=offsets,
            value=value
        )

        # Compute the output tensor.
        output = (sampled_values * attention_weights[:, None, :, :]).sum(dim=-1)
        output = output.permute(0, 2, 1)

        return output


    # Private methods.
    def __get_offsets_and_attention_weights(self, query):
        """
        Computes the offsets and attention weights given the query tensor.

        Args:
            query (torch.Tensor): The query tensor, usually the text embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The offsets and attention weights.
        """
        # Get tensor shapes.
        B, N, _ = query.size()

        # Compute the attention weights.
        attention_weights = self.__att_weight(query)
        attention_weights = attention_weights.view(B, N, self.__num_points)
        attention_weights = F.softmax(input=attention_weights, dim=-1)
        attention_weights = attention_weights.view(B, N, self.__num_points)

        # Compute the offsets.
        offsets = self.__offset(query)
        offsets = offsets.view(B, N, self.__num_points, 2)

        return offsets, attention_weights


    def __get_reference_points(self, height, width):
        """
        Computes the reference points given the height and width of the image.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            torch.Tensor: The reference points.
        """
        # Create the reference points.
        ref_x = torch.linspace(start=0, end=1, steps=width)
        ref_y = torch.linspace(start=0, end=1, steps=height)
        ref_x, ref_y = torch.meshgrid(ref_x, ref_y)

        # Reshape the reference points.
        ref_x = ref_x.reshape(-1)[None]
        ref_y = ref_y.reshape(-1)[None]

        # Stack the reference points.
        reference_points = torch.stack(tensors=[ref_x, ref_y], dim=-1)

        return reference_points


    def __apply_offsets_to_reference_points(self, reference_points, offsets, value):
        """
        Applies the offsets to the reference points.

        Args:
            reference_points (torch.Tensor): The reference points.
            offsets (torch.Tensor): The offsets.
            value (torch.Tensor): The value tensor, usually the image embeddings.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Get tensor shapes.
        B, N, _, _ = offsets.size()

        # Compute the sampling locations.
        sampling_locations = reference_points[:, :, None, :] + offsets
        sampling_locations = (
            sampling_locations / 
            torch.tensor([self.__patch_size - 1, self.__patch_size - 1], device=value.device) - 
            0.5
        ) * 2

        # Compute the sampled values.
        v = self.__value(value)
        v = v.view(B, -1, self.__patch_size, self.__patch_size)

        sampled_values = F.grid_sample(
            input=v,
            grid=sampling_locations,
            mode="bilinear",
            align_corners=False
        )

        return sampled_values


class MultiHeadAttention(nn.Module):


    # Special methods.
    def __init__(self, num_heads, in_embedding_dim):
        """
        Initializes the MultiHeadAttention class that computes the attention weights with the 
        normal dot-product mechanism.

        Args:
            num_heads (int): The number of heads.
            in_embedding_dim (int): The input embedding dimension.
        """
        super().__init__()

        # Validate the number of heads.
        assert in_embedding_dim % num_heads == 0, "The number of heads must be divisible by the input embedding dimension."
        head_embedding_dim = in_embedding_dim // num_heads

        # Layers.
        self.__attention_heads = nn.ModuleList([
            Attention(
                in_embedding_dim=in_embedding_dim, 
                out_embedding_dim=head_embedding_dim
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


    # Methods.
    def forward(self, query, key, value):
        """
        Forward pass of the multi-head attention mechanism. Make sure to enable the offset layer 
        defining the `attention_size` parameter in the constructor, otherwise it will raise an error 
        if you pass the `offset` parameter during the forward pass.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the attention heads.
        attention_heads = [head(query, key, value) for head in self.__attention_heads]

        # Concatenate the attention heads.
        concatenated_attention_heads = torch.cat(attention_heads, dim=-1)

        # Compute the output tensor.
        output = self.__linear(concatenated_attention_heads)

        return output


class DeformableMultiHeadAttention(nn.Module):


    # Special methods.
    def __init__(self, num_heads, in_embedding_dim, num_points, patch_size):
        """
        Initializes the DeformableMultiHeadAttention class that computes the attention weights with 
        a deformable mechanism that uses an offset tensor from image embeddings used to deform the 
        attention weights on image based on both text and image embeddings.

        Args:
            num_heads (int): The number of heads.
            in_embedding_dim (int): The input embedding dimension.
            num_points (int): The number of points to deform the attention weights.
            patch_size (int): The size of the patch used to compute the reference points.
        """
        super().__init__()

        # Validate the number of heads.
        assert in_embedding_dim % num_heads == 0, "The number of heads must be divisible by the input embedding dimension."
        head_embedding_dim = in_embedding_dim // num_heads

        # Layers.
        self.__attention_heads = nn.ModuleList([
            DeformableAttention(
                in_embedding_dim=in_embedding_dim, 
                out_embedding_dim=head_embedding_dim, 
                num_points=num_points, 
                patch_size=patch_size
            ) for _ in range(num_heads)
        ])

        self.__linear = nn.Linear(in_features=in_embedding_dim, out_features=in_embedding_dim)


    # Properties.
    @property
    def offset_attention(self):
        """
        Get the offset attention matrix computed by the forward pass.

        Returns:
            Dict[str, torch.Tensor]: The offset attention matrix for each head.
        """
        return {"head_%d" % i: head.offset_attention for i, head in enumerate(self.__attention_heads)}


    # Methods.
    def forward(self, query, value):
        """
        Forward pass of the deformable multi-head attention mechanism.

        Args:
            query (torch.Tensor): The query tensor, usually the text embeddings.
            value (torch.Tensor): The value tensor, usually the image embeddings.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the attention heads.
        attention_heads = [head(query, value) for head in self.__attention_heads]

        # Concatenate the attention heads.
        concatenated_attention_heads = torch.cat(attention_heads, dim=-1)

        # Compute the output tensor.
        output = self.__linear(concatenated_attention_heads)

        return output
