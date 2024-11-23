"""
This module contains the Embedding classes for both text and image embeddings.
"""
from abc import ABC

import torch
import torch.nn as nn


# Classes.
class _BaseEmbedding(ABC):


    # Special methods.
    def __init__(self, num_embedding, embedding_dim, padding_idx):
        """
        Initializes the _BaseEmbedding class. This is an abstract class has 
        only the positional encoding layer and should not be instantiated.

        Args:
            num_embedding (int): The number of embeddings.
            embedding_dim (int): The dimension of the embeddings.
            padding_idx (int): The index of the padding token in the vocabulary.
        """
        super().__init__()

        # Layers.
        self.__pe = nn.Embedding(
            num_embeddings=num_embedding, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )

        # Attributes.
        self.__num_embedding = num_embedding
        self.__embedding_dim = embedding_dim
        self.__padding_idx = padding_idx

        self.__indices_pe = torch.randint(low=0, high=num_embedding, size=(1, num_embedding))


    # Properties.
    @property
    def num_embedding(self):
        """
        Returns the number of embeddings.

        Returns:
            int: The number of embeddings.
        """
        return self.__num_embedding


    @property
    def embedding_dim(self):
        """
        Returns the dimension of the embeddings.

        Returns:
            int: The dimension of the embeddings.
        """
        return self.__embedding_dim


    @property
    def padding_idx(self):
        """
        Returns the index of the padding token in the vocabulary.

        Returns:
            int: The index of the padding token in the vocabulary.
        """
        return self.__padding_idx


    @property
    def positional_encoding(self):
        """
        Returns the positional encoding layer.

        Returns:
            nn.Embedding: The positional encoding layer.
        """
        return self.__pe


    @property
    def indices_pe(self):
        """
        Returns the indices used to generate the positional encoding.

        Returns:
            torch.Tensor: The indices used to generate the positional encoding.
        """
        return self.__indices_pe


class TextEmbedding(_BaseEmbedding, nn.Module):


    # Special methods.
    def __init__(self, context_size, embedding_dim, padding_idx):
        """
        Initializes the TextEmbedding class that computes the embeddings of the text tokens. It 
        uses a predefined positional encoding layer to add positional information to the embeddings.

        Args:
            context_size (int): The size of the context.
            embedding_dim (int): The dimension of the embeddings.
            padding_idx (int): The index of the padding token in the vocabulary.
        """
        # Initialize the parent class.
        super().__init__(
            num_embedding=context_size, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )

        # Layers.
        self.__emb = nn.Embedding(
            num_embeddings=context_size, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )

        # Buffers.
        self.register_buffer(name="positional_tokens", tensor=self.indices_pe)


    # Methods.
    def forward(self, x):
        """
        Performs a forward pass through the text embedding layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Get embeddings.
        txt_emb = self.__emb(x)

        # Apply positional encoding.
        pe = self.positional_encoding(self.positional_tokens)
        txt_emb_pe = txt_emb + pe

        return txt_emb_pe


class ImageEmbedding(_BaseEmbedding, nn.Module):


    # Special methods.
    def __init__(self, image_size, num_patches, padding_idx):
        """
        Initializes the ImageEmbedding class.

        Args:
            image_size (int): The size of the image.
            num_patches (int): The number of patches in the image.
            padding_idx (int): The index of the padding token in the vocabulary.
        """
        # Check if the image size is divisible by the patch size.
        if image_size % num_patches != 0:
            raise ValueError("The image size must be divisible by the number of patches.")

        # Compute patch size.
        patch_size = image_size // num_patches

        # Computes embedding dimension.
        self.__embedding_dim = patch_size ** 2 * 3

        # Compute the number of embeddings.
        num_embedding = num_patches ** 2

        # Initialize the parent class.
        super().__init__(
            num_embedding=num_embedding, 
            embedding_dim=self.__embedding_dim, 
            padding_idx=padding_idx
        )

        # Layers.
        self.__patch_image = nn.Unfold(
            kernel_size=patch_size, 
            stride=patch_size
        )

        # Buffers.
        self.register_buffer(name="positional_tokens", tensor=self.indices_pe)


    # Methods.
    def forward(self, x):
        """
        Performs a forward pass through the image embedding layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Get batch size.
        B = x.size(0)

        # Extract patches.
        patches = self.__patch_image(x)

        # Reshape patches and get embeddings.
        img_emb = patches.view(B, -1, self.embedding_dim)

        # Apply positional encoding.
        pe = self.positional_encoding(self.positional_tokens)
        img_emb_pe = img_emb + pe

        return img_emb_pe
