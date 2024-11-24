"""
Main module that contains the PromptVision model class.
"""
import torch
import torch.nn as nn

from models.embedding import ImageEmbedding, TextEmbedding
from models.encoder import ImageEncoder, TextEncoder


# Classes.
class PromptVisionTrainer(nn.Module):


    # Special methods.
    def __init__(
            self, 
            num_text_encoder_layers, 
            num_image_encoder_layers, 
            text_encoder_hidden_dim, 
            image_encoder_hidden_dim, 
            num_heads, 
            embedding_dim, 
            context_length, 
            image_size, 
            num_patches, 
            num_points, 
            padding_idx = 0
        ):
        super().__init__()

        # Embedding layers.
        self.__text_embedding = TextEmbedding(
            context_size=context_length, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )
        self.__image_embedding = ImageEmbedding(
            image_size=image_size,
            num_patches=num_patches, 
            padding_idx=padding_idx
        )

        # Encoder layers.
        self.__text_encoder = TextEncoder(
            num_layers=num_text_encoder_layers, 
            emb_dim=embedding_dim, 
            num_heads=num_heads, 
            hidden_dim=text_encoder_hidden_dim
        )
        self.__image_encoder = ImageEncoder(
            num_layers=num_image_encoder_layers, 
            emb_dim=embedding_dim, 
            num_heads=num_heads, 
            hidden_dim=image_encoder_hidden_dim, 
            num_points=num_points, 
            patch_size=num_patches
        )


    # Methods.
    def forward(self, text, image):
        """
        Performs a forward pass through the model.

        Args:
            text (torch.Tensor): The input text tensor.
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Embedding layers.
        text_emb = self.__text_embedding(text)
        image_emb = self.__image_embedding(image)

        # Encoder layers.
        text_enc = self.__text_encoder(text_emb)
        image_enc = self.__image_encoder(text_enc, image_emb)

        return image_enc
