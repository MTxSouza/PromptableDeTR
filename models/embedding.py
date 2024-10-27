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
        super(_BaseEmbedding, self).__init__()

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
        super(ImageEmbedding, self).__init__(
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
        self.register_buffer(
            name="positional_tokens", 
            tensor=torch.randint(low=0, high=self.num_embedding, size=(1, self.num_embedding))
        )


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
