"""
This module contains the Aligner model class used to train the Joiner block only, it 
aims to train the model first to model the relationship between the text and the image 
before training the whole model for detection.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
from models.base import BasePromptableDeTR

# Logger.
logger = Logger(name="model")


# Classes.
class Aligner(BasePromptableDeTR):


    # Special methods.
    def __init__(
            self, 
            image_tokens, 
            vocab_size = 30522, 
            emb_dim = 128, 
            proj_dim = 512, 
            emb_dropout_rate = 0.1
        ):
        """
        Initializes the Aligner class used to align images and text informations.

        Args:
            image_tokens (List[int]): The image tokens for each level.
            vocab_size (int): The size of the vocabulary. (Default: 30522)
            emb_dim (int): The embedding dimension. (Default: 128)
            proj_dim (int): The projection dimension. (Default: 512)
            emb_dropout_rate (float): The embedding dropout rate. (Default: 0.1)
        """
        super().__init__(image_tokens=image_tokens, vocab_size=vocab_size, emb_dim=emb_dim, proj_dim=proj_dim, emb_dropout_rate=emb_dropout_rate)

        # Aligner.
        self.aligner = nn.Sequential(
            nn.LayerNorm(normalized_shape=proj_dim),
            nn.Linear(in_features=proj_dim, out_features=vocab_size)
        )
    

    # Methods.
    def forward(self, image, prompt, prompt_mask = None):
        """
        Forward pass of the aligner.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.
            prompt_mask (torch.Tensor): Prompt mask tensor. (Default: None)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoded image and text tensors and alignment tensor.
        """
        logger.info(msg="Calling `Aligner` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Encode images and text.
        logger.debug(msg="- Calling `BasePromptableDeTR` block to the tensors %s and %s." % (image.shape, prompt.shape))
        joiner_emb = super().forward(image=image, prompt=prompt, prompt_mask=prompt_mask)
        logger.debug(msg="- Result of the `BasePromptableDeTR` block: %s." % (joiner_emb.shape,))

        # Align image and text embeddings.
        logger.debug(msg="- Calling the `nn.Sequential` block to the tensor %s." % (joiner_emb.shape,))
        alignment = self.aligner(joiner_emb)
        logger.debug(msg="- Result of the `nn.Sequential` block: %s." % (alignment.shape,))

        logger.info(msg="Returning the final output of the `Aligner` model with one tensor.")
        logger.debug(msg="- Alignment shape: %s" % (alignment.shape,))
        return alignment


    def save_joiner_weights(self, dir_path, ckpt_step = None):
        """
        Save the joiner weights.

        Args:
            dir_path (str): The path to the directory where the weights will be saved.
            ckpt_step (int): The checkpoint step. (Default: None)
        """
        logger.info(msg="Saving the joiner weights.")

        # Define the checkpoint path.
        ckpt_name = "joiner.pth"
        if ckpt_step is not None:
            ckpt_name = "joiner-ckpt-%d.pth" % ckpt_step
        ckpt_fp = os.path.join(dir_path, ckpt_name)

        torch.save(obj=self.joiner.state_dict(), f=ckpt_fp)
        logger.info(msg="Joiner weights saved successfully.")


    def compute_aligner_loss(self, y_pred, y_true):
        """
        Compute the loss needed to train the aligner model.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The true tensor.

        Returns:
            torch.Tensor: The loss value.
        """
        logger.info(msg="Computing the aligner loss.")

        # Compute the loss.
        B, N, _ = y_pred.shape
        f_y_pred = y_pred.view(B * N, -1)
        f_y_true = y_true.view(B * N)

        loss = F.cross_entropy(
            input=f_y_pred, 
            target=f_y_true, 
            reduction="mean"
        )
        logger.info(msg="Returning the loss value.")
        return loss
