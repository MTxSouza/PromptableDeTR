"""
This module contains the Detector model class used to predict bounding boxes and presence of objects 
in the image.
"""
import torch.nn as nn

from logger import Logger
from models.base import BasePromptableDeTR

# Logger.
logger = Logger(name="model")


# Classes.
class PromptableDeTR(BasePromptableDeTR):


    # Special methods.
    def __init__(self, proj_dim = 512, **kwargs):
        """
        Initializes the Detector class used to predict bounding boxes and presence 
        of objects in the image.

        Args:
            proj_dim (int): The projection dimension of the image and text embeddings. (Default: 512)
        """
        super().__init__(**kwargs)

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
    def forward(self, image, prompt):
        """
        Forward pass of the detector.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Bounding box and presence predictions.
        """
        logger.info(msg="Calling `Detector` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Compute joint embeddings.
        logger.debug(msg="- Calling `BasePromptableDeTR` forward method.")
        joint = super().forward(image=image, prompt=prompt)
        logger.debug(msg="- Result of the `BasePromptableDeTR` forward method: %s." % (joint.shape,))

        # Predict bounding boxes and presence.
        logger.debug(msg="- Calling `nn.Sequential` block to the tensor %s." % (joint.shape,))
        out = self.detector(joint)
        logger.debug(msg="- Result of the `nn.Sequential` block: %s." % (out.shape,))

        logger.debug(msg="- Calling `nn.Linear` block to the tensor %s." % (out.shape,))
        bbox = self.bbox_predictor(out)
        presence = self.presence_predictor(out)
        logger.debug(msg="- Result of the `nn.Linear` block: %s and %s." % (bbox.shape, presence.shape))

        return bbox, presence
