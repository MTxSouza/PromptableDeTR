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
        pass
