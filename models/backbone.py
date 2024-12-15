"""
Main module that stores the image encoder block of the model and the text encoder. It is a MobileNetV3 based model 
without the classification head and a MobileBERT model.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import (MobileNet_V3_Large_Weights,
                                            mobilenet_v3_large)
from transformers import AutoTokenizer, MobileBertModel


# Structures.
@dataclass
class ImageEncoderOutput:
    """
    Dataclass that stores the output of the image encoder.
    """
    high_resolutution_feat: torch.Tensor
    medium_resolutution_feat: torch.Tensor


# Classes.
class MobileNetv3Backbone(nn.Module):
    def __init__(self):
        """
        The model is divided into three parts: the base backbone, the first feature map and the second 
        feature map. The first feature map is used to detect small objects, while the second feature 
        map is used to detect medium objects. 

        The feature maps are then passed through a 1x1 convolution 
        layer to increase the number of channels. The model is pre-trained on the ImageNet dataset.
        """
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
        self.high_resolutution_feat = nn.Conv2d(in_channels=112, out_channels=256, kernel_size=1, stride=1)
        self.medium_resolutution_feat = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1)


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
        high_resolutution_feat = self.high_resolutution_feat(feature_map_0)
        medium_resolutution_feat = self.medium_resolutution_feat(feature_map_1)

        return ImageEncoderOutput(
            high_resolutution_feat=high_resolutution_feat, 
            medium_resolutution_feat=medium_resolutution_feat
        )


class MobileBertEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pre-trained model.
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="google/mobilebert-uncased")
        self.encoder = MobileBertModel.from_pretrained(pretrained_model_name_or_path="google/mobilebert-uncased")


    def forward(self, text):
        """
        Forward pass of the model. It returns the last hidden state of the model.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: Last hidden state of the model.
        """

        # Tokenize input text.
        input_ids = self.tokenizer(text, return_tensors="pt")
        input_ids = input_ids.get("input_ids")

        # Forward pass.
        outputs = self.encoder(input_ids)

        return outputs.last_hidden_state
