"""
Main module that contains the PromptVision model class.
"""
import torch.nn as nn

from models.decoder import Decoder as ImageDecoder
from models.encoder import TextEncoder


# Classes.
class PromptVisionTrainer(nn.Module):


    # Special methods.
    def __init__(
            self, 
            text_encoder_name, 
            num_image_decoder_blocks, 
            num_image_decoder_heads, 
            image_decoder_hidden_dim
        ):
        """
        Initializes the PromptVisionTrainer class.

        Args:
            text_encoder_name (str): The name of the text encoder model.
            num_image_decoder_blocks (int): The number of blocks in the image decoder.
            num_image_decoder_heads (int): The number of heads in the image decoder.
            image_decoder_hidden_dim (int): The hidden dimension of the image decoder.
        """
        super().__init__()

        # Layers.
        self.text_encoder = TextEncoder(model_name=text_encoder_name)
        self.image_decoder = ImageDecoder(
            num_blocks=num_image_decoder_blocks,
            num_heads=num_image_decoder_heads,
            hidden_dim=image_decoder_hidden_dim,
            embedding_dim=768 # The BERT model has an embedding dimension of 768.
        )


    # Methods.
    def forward(self, image_tensor, tokenized_text_tensor):
        """
        Forward pass of the model used specifically for training.

        Args:
            image_tensor (torch.Tensor): The input image tensor.
            tokenized_text_tensor (torch.Tensor): The tokenized text tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Encode the text.
        text_embedding = self.text_encoder(tokenized_text_tensor)

        # Decode the image.
        output = self.image_decoder(image_tensor, text_embedding)

        return output
