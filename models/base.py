"""
This module contains the core of the main model. It defines the entire architecture except the 
last block that can be used either for the Aligner or the Detector.
"""
import torch
import torch.nn as nn

from logger import Logger
from models.image_encoder import MobileNetV3
from models.joiner import Joiner
from models.text_encoder import MobileBert

# Logger.
logger = Logger(name="model")


# Classes.
class Encoder(nn.Module):


    # Special methods.
    def __init__(
            self, 
            vocab_size = 30522, 
            emb_dim = 128, 
            proj_dim = 512, 
            emb_dropout_rate = 0.1
        ):
        """
        Initializes the Encoder class used to encode images and text.

        Args:
            vocab_size (int): The size of the vocabulary. (Default: 30522)
            emb_dim (int): The embedding dimension. (Default: 128)
            proj_dim (int): The projection dimension. (Default: 512)
            emb_dropout_rate (float): The embedding dropout rate. (Default: 0.1)
        """
        super().__init__()

        # Encoders.
        self.image_encoder = MobileNetV3(emb_dim=proj_dim)
        self.text_encoder = MobileBert(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_size=proj_dim,
            emb_dropout_rate=emb_dropout_rate,
            intermediate_size=proj_dim,
            intra_bottleneck_dim=emb_dim
        )
    

    # Methods.
    def forward(self, image, prompt, prompt_mask = None):
        """
        Forward pass of the encoder.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.
            prompt_mask (torch.Tensor): Prompt mask tensor. (Default: None)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded image and text tensors.
        """
        logger.info(msg="Calling `Encoder` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))
        logger.debug(msg="- Prompt mask shape: %s" % (prompt_mask.shape if prompt_mask is not None else None,))

        # Encode images and text.
        logger.debug(msg="- Calling `MobileNetV3` block to the tensor %s." % (image.shape,))
        image_emb = self.image_encoder(image)

        logger.debug(msg="- Calling `MobileBert` block to the tensor %s." % (prompt.shape,))
        logger.debug(msg="- Using the prompt mask: %s." % (prompt_mask is not None))
        text_emb = self.text_encoder(prompt, attention_mask=prompt_mask)

        logger.info(msg="Returning the final output of the `Encoder` model with five tensors.")
        logger.debug(msg="- High resolution image shape: %s" % (image_emb.high_resolution_feat.shape,))
        logger.debug(msg="- Medium resolution image shape: %s" % (image_emb.mid_resolution_feat.shape,))
        logger.debug(msg="- Low resolution image shape: %s" % (image_emb.low_resolution_feat.shape,))
        logger.debug(msg="- Last hidden state shape: %s" % (text_emb.last_hidden_state.shape,))
        logger.debug(msg="- Pooled output shape: %s" % (text_emb.pooled_output.shape,))
        return image_emb, text_emb


class BasePromptableDeTR(Encoder):


    # Special methods.
    def __init__(
            self, 
            image_tokens, 
            vocab_size = 30522, 
            emb_dim = 128, 
            proj_dim = 512, 
            num_heads = 8, 
            ff_dim = 2048, 
            emb_dropout_rate = 0.1, 
            num_joiner_layers = 3
        ):
        super().__init__(vocab_size=vocab_size, emb_dim=emb_dim, proj_dim=proj_dim, emb_dropout_rate=emb_dropout_rate)

        # Joiner.
        self.joiner = Joiner(
            image_tokens=image_tokens, 
            emb_dim=proj_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            num_joins=num_joiner_layers
        )


    # Methods.
    def load_base_weights(self, image_encoder_weights = None, text_encoder_weights = None, joiner_weights = None):
        """
        Load the weights of the image and text encoders.

        Args:
            image_encoder_weights (str): Path to the image encoder weights. (Default: None)
            text_encoder_weights (str): Path to the text encoder weights. (Default: None)
            joiner_weights (str): Path to the joiner weights. (Default: None)
        """
        logger.info(msg="Loading the weights of the image and text encoders.")
        logger.debug(msg="- Image encoder weights: %s" % image_encoder_weights)
        logger.debug(msg="- Text encoder weights: %s" % text_encoder_weights)
        logger.debug(msg="- Joiner weights: %s" % joiner_weights)

        # Load weights.
        if image_encoder_weights is not None:
            logger.debug(msg="- Loading the image encoder weights.")
            self.image_encoder.load_state_dict(torch.load(f=image_encoder_weights, weights_only=True))

        if text_encoder_weights is not None:
            logger.debug(msg="- Loading the text encoder weights.")
            self.text_encoder.load_state_dict(torch.load(f=text_encoder_weights, weights_only=True))
        
        if joiner_weights is not None:
            logger.debug(msg="- Loading the joiner weights.")
            self.joiner.load_state_dict(torch.load(f=joiner_weights, weights_only=True))


    def forward(self, image, prompt, prompt_mask = None):
        """
        Forward pass of the PromptableDeTR model.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.
            prompt_mask (torch.Tensor): Prompt mask tensor. (Default: None)

        Returns:
            PromptableDeTROutput: Bounding box and presence predictions.
        """
        logger.info(msg="Calling `BasePromptableDeTR` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Encode images and text.
        image_emb, text_emb = super().forward(image=image, prompt=prompt, prompt_mask=prompt_mask)

        # Join image and text embeddings.
        logger.debug(msg="- Calling `Joiner` block to the image and text tensors.")
        joint_emb = self.joiner(image_emb, text_emb)
        logger.debug(msg="- Result of the `Joiner` block: %s." % (joint_emb.shape,))

        logger.info(msg="Returning the final output of the `BasePromptableDeTR` model with one tensor.")
        logger.debug(msg="- Joint embeddings shape: %s" % (joint_emb.shape,))
        return joint_emb
