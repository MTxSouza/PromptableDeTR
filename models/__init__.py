"""
Main module that contains the PromptableDeTR model class.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from logger import Logger
from models.image_encoder import MobileNetV3
from models.joiner import Joiner
from models.text_encoder import MobileBert

# Logger.
logger = Logger(name="model")


# Structure.
@dataclass
class PromptableDeTROutput:
    """
    Dataclass to hold the output of the PromptableDeTR model.

    Attributes:
        bbox (torch.Tensor): Bounding box predictions.
        presence (torch.Tensor): Presence predictions.
    """
    bbox: torch.Tensor
    presence: torch.Tensor


# Class.
class Detector(nn.Module):


    # Special methods.
    def __init__(self, proj_dim = 512):
        """
        Initializes the Detector class used to predict bounding boxes and presence 
        of objects in the image.

        Args:
            proj_dim (int): The projection dimension of the image and text embeddings. (Default: 512)
        """
        super().__init__()

        # Layers.
        self.detector = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=proj_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=proj_dim * 4, out_features=proj_dim * 2),
            nn.ReLU()
        )
        self.bbox_predictor = nn.Linear(in_features=proj_dim * 2, out_features=4)
        self.presence_predictor = nn.Linear(in_features=proj_dim * 2, out_features=1)


    # Methods.
    def forward(self, joint_emb):
        """
        Forward pass of the detector.

        Args:
            joint_emb (torch.Tensor): Joint image and text embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Bounding box and presence predictions.
        """
        logger.info(msg="Calling  `Detector` forward method.")
        logger.debug(msg="- Input shape: %s" % (joint_emb.shape,))

        # Detect objects.
        logger.debug(msg="- Calling the `nn.Sequential` block to the tensor %s." % (joint_emb.shape,))
        detection_emb = self.detector(joint_emb)
        logger.debug(msg="- Result of the `nn.Sequential` block: %s." % (detection_emb.shape,))

        # Predict bounding box and presence.
        logger.debug(msg="- Calling the `nn.Linear` block to the tensor %s." % (detection_emb.shape,))
        bbox = self.bbox_predictor(detection_emb)
        logger.debug(msg="- Result of the `nn.Linear` block: %s." % (bbox.shape,))
        
        logger.debug(msg="- Calling the `nn.Linear` block to the tensor %s." % (detection_emb.shape,))
        presence = self.presence_predictor(detection_emb)
        logger.debug(msg="- Result of the `nn.Linear` block: %s." % (presence.shape,))

        logger.info(msg="Final output of the `Detector` block: %s and %s." % (bbox.shape, presence.shape))
        return bbox, presence


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
    def forward(self, image, prompt):
        """
        Forward pass of the encoder.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded image and text tensors.
        """
        logger.info(msg="Calling `Encoder` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Encode images and text.
        logger.debug(msg="- Calling `MobileNetV3` block to the tensor %s." % (image.shape,))
        image_emb = self.image_encoder(image)

        logger.debug(msg="- Calling `MobileBert` block to the tensor %s." % (prompt.shape,))
        text_emb = self.text_encoder(prompt)

        logger.info(msg="Returning the final output of the `Encoder` model with two tensors.")
        logger.debug(msg="- Image embedding shape: %s" % (image_emb.shape,))
        logger.debug(msg="- Text embedding shape: %s" % (text_emb.shape,))
        return image_emb, text_emb


class Aligner(Encoder):


    # Special methods.
    def __init__(
            self, 
            vocab_size = 30522, 
            emb_dim = 128, 
            proj_dim = 512, 
            emb_dropout_rate = 0.1
        ):
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

        # Detection head.
        self.detector = Detector(proj_dim=proj_dim)


    # Methods.
    def load_base_weights(self, image_encoder_weights = None, text_encoder_weights = None):
        """
        Load the weights of the image and text encoders.

        Args:
            image_encoder_weights (str): Path to the image encoder weights. (Default: None)
            text_encoder_weights (str): Path to the text encoder weights. (Default: None)
        """
        logger.info(msg="Loading the weights of the image and text encoders.")
        logger.debug(msg="- Image encoder weights: %s" % image_encoder_weights)
        logger.debug(msg="- Text encoder weights: %s" % text_encoder_weights)

        # Load weights.
        if image_encoder_weights is not None:
            logger.debug(msg="- Loading the image encoder weights.")
            self.image_encoder.load_state_dict(torch.load(f=image_encoder_weights, weights_only=True))

        if text_encoder_weights is not None:
            logger.debug(msg="- Loading the text encoder weights.")
            self.text_encoder.load_state_dict(torch.load(f=text_encoder_weights, weights_only=True))


    def forward(self, image, prompt):
        """
        Forward pass of the PromptableDeTR model.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.

        Returns:
            PromptableDeTROutput: Bounding box and presence predictions.
        """
        logger.info(msg="Calling `BasePromptableDeTR` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Encode images and text.
        image_emb, text_emb = super().forward(image=image, prompt=prompt)

        # Join image and text embeddings.
        logger.debug(msg="- Calling `Joiner` block to the image and text tensors.")
        joint_emb = self.joiner(image_emb, text_emb)
        logger.debug(msg="- Result of the `Joiner` block: %s." % (joint_emb.shape,))

        # Detect objects.
        logger.debug(msg="- Calling `Detector` block to the tensor %s." % (joint_emb.shape,))
        bbox, presence = self.detector(joint_emb)
        logger.debug(msg="- Result of the `Detector` block: %s and %s." % (bbox.shape, presence.shape))

        logger.info(msg="Returning the final output of the `BasePromptableDeTR` model with two tensors.")
        logger.debug(msg="- Bounding box shape: %s" % (bbox.shape,))
        logger.debug(msg="- Presence shape: %s" % (presence.shape,))
        return PromptableDeTROutput(bbox=bbox, presence=presence)
