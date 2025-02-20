"""
This module contains the Aligner model class used to train the Joiner block only, it 
aims to train the model first to model the relationship between the text and the image 
before training the whole model for detection.
"""
import torch.nn as nn

from logger import Logger
from models import Encoder
from models.joiner import Joiner

# Logger.
logger = Logger(name="model")


# Classes.
class Aligner(Encoder):


    # Special methods.
    def __init__(
            self, 
            vocab_size = 30522, 
            emb_dim = 128, 
            proj_dim = 512, 
            emb_dropout_rate = 0.1
        ):
        """
        Initializes the Aligner class used to align images and text informations.

        Args:
            vocab_size (int): The size of the vocabulary. (Default: 30522)
            emb_dim (int): The embedding dimension. (Default: 128)
            proj_dim (int): The projection dimension. (Default: 512)
            emb_dropout_rate (float): The embedding dropout rate. (Default: 0.1)
        """
        super().__init__(vocab_size=vocab_size, emb_dim=emb_dim, proj_dim=proj_dim, emb_dropout_rate=emb_dropout_rate)

        # Aligner.
        self.aligner = nn.Sequential(
            nn.LayerNorm(normalized_shape=proj_dim),
            nn.Linear(in_features=proj_dim, out_features=vocab_size)
        )
    

    # Methods.
    def forward(self, image, prompt):
        """
        Forward pass of the aligner.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoded image and text tensors and alignment tensor.
        """
        logger.info(msg="Calling `Aligner` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Encode images and text.
        image_emb, text_emb = super().forward(image=image, prompt=prompt)

        # Align image and text embeddings.
        logger.debug(msg="- Calling the `nn.Sequential` block to the tensor %s." % (image_emb.shape,))
        alignment = self.aligner(image_emb)
        logger.debug(msg="- Result of the `nn.Sequential` block: %s." % (alignment.shape,))

        logger.info(msg="Returning the final output of the `Aligner` model with one tensor.")
        logger.debug(msg="- Alignment shape: %s" % (alignment.shape,))
        return alignment
