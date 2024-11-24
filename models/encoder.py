"""
This module contains the classes that define the encoder layers of the 
Transformer model for both text and image inputs.
"""
import torch
import torch.nn as nn

from models.attention import DeformableMultiHeadAttention, MultiHeadAttention


# Classes.
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
        super().__init__()

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


class TextEncoderLayer(nn.Module):


    # Special methods.
    def __init__(self, emb_dim, num_heads, hidden_dim):
        """
        Initializes the TextEncoderLayer class that applies the base attention 
        mechanism to the input tensor.

        Args:
            emb_dim (int): The dimension of the input tensor.
            num_heads (int): The number of heads in the attention mechanism.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()

        # Layers.
        self.__attention = MultiHeadAttention(in_embedding_dim=emb_dim, num_heads=num_heads)
        self.__mlp = MLP(embedding_dim=emb_dim, hidden_dim=hidden_dim)
        self.__layer_norm_1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.__layer_norm_2 = nn.LayerNorm(normalized_shape=emb_dim)


    # Properties.
    @property
    def attention(self):
        """
        torch.nn.Module: The attention mechanism.
        """
        return self.__attention.attention


    # Methods.
    def forward(self, text_emb):
        """
        Applies the base attention mechanism to the input tensor.

        Args:
            text_emb (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Apply the attention mechanism.
        text_emb_att = self.__attention(text_emb, text_emb, text_emb)

        # Apply the layer normalization.
        text_emb_norm = self.__layer_norm_1(text_emb_att + text_emb)

        # Apply the MLP.
        text_emb_mlp = self.__mlp(text_emb_norm)

        # Apply the layer normalization.
        text_emb = self.__layer_norm_2(text_emb_mlp + text_emb_norm)

        return text_emb


class ImageEncoderLayer(nn.Module):


    # Special methods.
    def __init__(self, emb_dim, num_heads, hidden_dim, num_points, patch_size):
        """
        Initializes the ImageEncoderLayer class that applies the deformable 
        attention mechanism to the input tensor.

        Args:
            emb_dim (int): The dimension of the input tensor.
            num_heads (int): The number of heads in the attention mechanism.
            hidden_dim (int): The dimension of the hidden layer.
            num_points (int): The number of points in the image.
            patch_size (int): The size of the patch.
        """
        super().__init__()

        # Layers.
        self.__attention = DeformableMultiHeadAttention(
            in_embedding_dim=emb_dim, 
            num_heads=num_heads, 
            num_points=num_points, 
            patch_size=patch_size
        )
        self.__mlp = MLP(embedding_dim=emb_dim, hidden_dim=hidden_dim)
        self.__layer_norm_1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.__layer_norm_2 = nn.LayerNorm(normalized_shape=emb_dim)


    # Methods.
    def forward(self, text_emb, image_emb):
        """
        Applies the deformable attention mechanism to the input tensor.

        Args:
            text_emb (torch.Tensor): The input text tensor.
            image_emb (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Apply the attention mechanism.
        img_emb_att = self.__attention(text_emb, image_emb)

        # Apply the layer normalization.
        img_emb_norm = self.__layer_norm_1(img_emb_att + image_emb)

        # Apply the MLP.
        img_emb_mlp = self.__mlp(img_emb_norm)

        # Apply the layer normalization.
        img_emb = self.__layer_norm_2(img_emb_mlp + img_emb_norm)

        return img_emb


class TextEncoder(nn.Module):


    # Special methods.
    def __init__(self, num_layers, emb_dim, num_heads, hidden_dim):
        """
        Initializes the TextEncoder class that applies the base attention 
        mechanism to the input tensor.

        Args:
            num_layers (int): The number of layers in the encoder.
            emb_dim (int): The dimension of the input tensor.
            num_heads (int): The number of heads in the attention mechanism.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()

        # Layers.
        self.__layers = nn.ModuleList([
            TextEncoderLayer(emb_dim=emb_dim, num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])


    # Properties.
    @property
    def attention(self):
        """
        Returns the attention mechanism of all layers.

        Returns:
            Dict[str, dict]: The attention mechanisms of all layers.
        """
        return {"layer_%d" % i: layer.attention for i, layer in enumerate(self.__layers)}


    # Methods.
    def forward(self, text_emb):
        """
        Applies the base attention mechanism to the input tensor.

        Args:
            text_emb (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.__layers:
            text_emb = layer(text_emb, text_emb, text_emb)

        return text_emb


class ImageEncoder(nn.Module):


    # Special methods.
    def __init__(self, num_layers, emb_dim, num_heads, hidden_dim, num_points, patch_size):
        """
        Initializes the ImageEncoder class that applies the deformable 
        attention mechanism to the input tensor.

        Args:
            num_layers (int): The number of layers in the encoder.
            emb_dim (int): The dimension of the input tensor.
            num_heads (int): The number of heads in the attention mechanism.
            hidden_dim (int): The dimension of the hidden layer.
            num_points (int): The number of points in the image.
            patch_size (int): The size of the patch.
        """
        super().__init__()

        # Layers.
        self.__layers = nn.ModuleList([
            ImageEncoderLayer(
                emb_dim=emb_dim, 
                num_heads=num_heads, 
                hidden_dim=hidden_dim, 
                num_points=num_points, 
                patch_size=patch_size
            )
            for _ in range(num_layers)
        ])


    # Methods.
    def forward(self, text_emb, image_emb):
        """
        Applies the deformable attention mechanism to the input tensor.

        Args:
            text_emb (torch.Tensor): The input text tensor.
            image_emb (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.__layers:
            image_emb = layer(text_emb, image_emb)

        return image_emb
