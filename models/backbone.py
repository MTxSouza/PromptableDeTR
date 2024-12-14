"""
Main module that stores the image encoder block of the model. It is a MobileNetV3 based model without 
the classification head.

The model is divided into three parts: the base backbone, the first feature 
map and the second feature map. The first feature map is used to detect small objects, while the second 
feature map is used to detect medium objects. 

The feature maps are then passed through a 1x1 convolution 
layer to increase the number of channels. The model is pre-trained on the ImageNet dataset.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import (MobileNet_V3_Large_Weights,
                                            mobilenet_v3_large)


# Structures.
@dataclass
class ImageEncoderOutput:
    """
    Dataclass that stores the output of the image encoder.
    """
    small_obj: torch.Tensor
    medium_obj: torch.Tensor


# Classes.
class MobileNetv3Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pre-trained model.
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone[0] = nn.Sequential(*list(backbone[0].children())[:-1])

        # Get feature map layers.
        n_children = len(list(backbone[0].children()))

        base_backbone = list(backbone[0].children())[:n_children - 10]
        self.base_backbone = nn.Sequential(*base_backbone)

        feature_0 = list(backbone[0].children())[n_children - 10:n_children - 4]
        self.feature_0 = nn.Sequential(*feature_0)

        feature_1 = list(backbone[0].children())[n_children - 4:]
        self.feature_1 = nn.Sequential(*feature_1)

        # Define final layers.
        self.small_feat = nn.Conv2d(in_channels=112, out_channels=256, kernel_size=1, stride=1)
        self.medium_feat = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1)


    def forward(self, x):
        """
        Forward pass of the model. It returns the feature maps of the small and medium objects.

        Args:
            x (torch.Tensor): Image tensor with shape (B, C, H, W).

        Returns:
            ImageEncoderOutput: Dataclass that stores the feature maps of the small and medium objects.
        """
        base = self.base_backbone(x)

        # Multi-Scale feature maps.
        feature_map_0 = self.feature_0(base)
        feature_map_1 = self.feature_1(feature_map_0)

        # Compute new channels.
        feature_small_obj = self.small_feat(feature_map_0)
        feature_medium_obj = self.medium_feat(feature_map_1)

        return ImageEncoderOutput(small_obj=feature_small_obj, medium_obj=feature_medium_obj)
