"""
Main module used to declare the image encoder block to process the input image data. It is 
a fully implementation of MobileNetV3 model with a small modification at end, with a new 
InvertedResidual used to process a new feature map in a smaller resolution.

Paper: https://arxiv.org/pdf/1905.02244
"""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


# Structures.
@dataclass
class MobileNetV3Output:
    """
    Structure used to store the output of the MobileNetV3 model. It has three feature maps 
    with different resolutions.

    Attributes:
        high_resolution_feat (torch.FloatTensor): High resolution feature map.
        mid_resolution_feat (torch.FloatTensor): Medium resolution feature map.
        low_resolution_feat (torch.FloatTensor): Low resolution feature map.
    """
    high_resolution_feat: torch.FloatTensor
    mid_resolution_feat: torch.FloatTensor
    low_resolution_feat: torch.FloatTensor


# Functions.
def make_ntuple(value, n):
    """
    Convert a number into a tuple representation with `n` repetitions. If `value` 
    was an iterable, it will just convert it into a tuple.

    Args:
        value (int|Iterable): Number to be converted into a tuple.
        n (int): Number of repetitions to be make.

    Returns:
        Tuple: Tuple object with `n` elements of `value` number.
    """
    if isinstance(value, Iterable):
        return tuple(value)
    return tuple([value] * n)


def make_divisible(v, divisor, min_value = None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Classes.
class Conv2dNormActivation(nn.Sequential):


    # Special methods.
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            dilation=1, 
            groups=1, 
            conv_layer=nn.Conv2d, 
            norm_layer=nn.BatchNorm2d, 
            act_func=nn.ReLU, 
            padding=None, 
            bias=None
        ):
        """
        Main class used to define the convolutional block in MobileNetV3 model. It is a 
        sequential block with convolutional layer, normalization layer and activation 
        function.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int|Tuple): Size of the convolutional kernel. (Default: 3)
            stride (int): Stride of the convolutional layer. (Default: 1)
            dilation (int): Dilation of the convolutional layer. (Default: 1)
            groups (int): Number of groups to be used in the convolutional layer. (Default: 1)
            conv_layer (nn.Module): Convolutional layer to be used. (Default: nn.Conv2d)
            norm_layer (nn.Module): Normalization layer to be used. (Default: nn.BatchNorm2d)
            act_func (nn.Module): Activation function to be used. (Default: nn.ReLU)
            padding (int|Tuple): Padding to be used in the convolutional layer. (Default: None)
            bias (bool): If `True`, the convolutional layer will have bias. (Default: None)
        """

        # Check the padding value.
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = make_ntuple(kernel_size, conv_dim)
                dilation = make_ntuple(dilation, conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(conv_dim))

        # Check the bias value.
        if bias is None:
            bias = norm_layer is None

        # Build the layers.
        layers = [conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels, eps=0.001, momentum=0.01))
        if act_func is not None:
            layers.append(act_func())

        super().__init__(*layers)


class SqueezeExcitation(nn.Module):


    # Special methods.
    def __init__(self, in_channels, squeeze_channels, act_func, scale_act):
        """
        Main class used to define the Squeeze-and-Excitation block in MobileNetV3 model.

        Args:
            in_channels (int): Number of input channels.
            squeeze_channels (int): Number of channels to be used in the squeeze layer.
            act_func (nn.Module): Activation function to be used in the block.
            scale_act (nn.Module): Activation function to be used in the scale layer.
        """
        super().__init__()

        # Layers.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = act_func()
        self.scale_activation = scale_act()


    # Methods.
    def forward(self, x):
        """
        Forward method of the Squeeze-and-Excitation block.

        Args:
            x (torch.Tensor): Input tensor to be processed. (shape: (batch_size, in_channels, height, width))

        Returns:
            torch.Tensor: Output tensor processed by the block. (shape: (batch_size, in_channels, height, width))
        """
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)

        return scale * x


class InvertedResidual(nn.Module):


    # Special methods.
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            dilation=1, 
            conv_layer=nn.Conv2d, 
            norm_layer=nn.BatchNorm2d, 
            act_func=nn.ReLU, 
            expanded_channels=None, 
            padding=None, 
            bias=None, 
            pool=False
        ):
        """
        Main class used to define the InvertedResidual block in MobileNetV3 model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. (Default: 3)
            stride (int): Stride of the convolutional layer. (Default: 1)
            dilation (int): Dilation of the convolutional layer. (Default: 1)
            conv_layer (nn.Module): Convolutional layer to be used. (Default: nn.Conv2d)
            norm_layer (nn.Module): Normalization layer to be used. (Default: nn.BatchNorm2d)
            act_func (nn.Module): Activation function to be used. (Default: nn.ReLU)
            expanded_channels (int): Number of channels to be used in the expanded layer. (Default: None)
            padding (int): Padding to be used in the convolutional layer. (Default: None)
            bias (bool): If `True`, the convolutional layer will have bias. (Default: None)
            pool (bool): If `True`, a Squeeze-and-Excitation block will be used before the last Convolutional block. (Default: False)
        """
        super().__init__()

        # Check the padding value.
        self.use_res_connect = stride == 1 and in_channels == out_channels

        # Build the layers.
        layers = []
        if not expanded_channels is None and expanded_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels, expanded_channels, kernel_size=1, norm_layer=norm_layer, act_func=act_func
                )
            )
            in_channels = expanded_channels

        stride = 1 if dilation > 1 else stride
        layers.append(
            Conv2dNormActivation(
                in_channels, expanded_channels, kernel_size, stride, dilation, expanded_channels, conv_layer, norm_layer, act_func, padding, bias
            )
        )

        # Check if the pool is enabled.
        if pool:
            squeeze_channels = make_divisible(expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(expanded_channels, squeeze_channels, nn.ReLU, nn.Hardsigmoid))

        layers.append(
            Conv2dNormActivation(
                expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer, act_func=None
            )
        )

        self.block = nn.Sequential(*layers)


    # Methods.
    def forward(self, x):
        """
        Forward method of the InvertedResidual block.

        Args:
            x (torch.Tensor): Input tensor to be processed. (shape: (batch_size, in_channels, height, width))

        Returns:
            torch.Tensor: Output tensor processed by the block. (shape: (batch_size, out_channels, height, width))
        """
        output = self.block(x)
        if self.use_res_connect:
            output += x

        return output


class Custom_MobileNetV3(nn.Module):


    # Special methods.
    def __init__(self):
        """
        Main class used to define the base MobileNetV3 model. It is a the base backbone of 
        MobileNetV3 model without the pooling and fully connected layers. It is used to 
        extract the features of the input image data.

        At end, it has two new InvertedResidual blocks to process the feature maps in a 
        smaller resolution.
        """
        super().__init__()

        # Backbone.
        self.backbone = nn.Sequential(
            Conv2dNormActivation(3, 16, 3, 2, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 1, False),
            InvertedResidual(16, 16, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, 16, 1, False),
            InvertedResidual(16, 24, 3, 2, 1, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, 64, 1, False),
            InvertedResidual(24, 24, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, 72, 1, False),
            InvertedResidual(24, 40, 5, 2, 1, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, 72, 2, False, True),
            InvertedResidual(40, 40, 5, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, 120, 2, False, True),
            InvertedResidual(40, 40, 5, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, 120, 2, False, True)
        )

        # Features maps.
        self.features_1 = nn.Sequential(
            InvertedResidual(40, 80, 3, 2, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 240, 1, False),
            InvertedResidual(80, 80, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 200, 1, False),
            InvertedResidual(80, 80, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 184, 1, False),
            InvertedResidual(80, 80, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 184, 1, False),
            InvertedResidual(80, 112, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 480, 1, False, True),
            InvertedResidual(112, 112, 3, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 672, 1, False, True)
        )
        self.features_2 = nn.Sequential(
            InvertedResidual(112, 160, 5, 2, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 672, 2, False, True),
            InvertedResidual(160, 160, 5, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 960, 2, False, True),
            InvertedResidual(160, 160, 5, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 960, 2, False, True)
        )

        # Extra features.
        self.features_3 = nn.Sequential(
            InvertedResidual(160, 320, 5, 2, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 960, 2, False, True), 
            InvertedResidual(320, 320, 5, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 1280, 2, False, True), 
            InvertedResidual(320, 320, 5, 1, 1, nn.Conv2d, nn.BatchNorm2d, nn.Hardswish, 1280, 2, False, True)
        )


    def forward(self, x):
        """
        Forward method of the MobileNetV3 model.

        Args:
            x (torch.Tensor): Input tensor to be processed. (shape: (batch_size, 3, height, width))

        Returns:
            MobileNetV3Output: Output of the MobileNetV3 model with three feature maps with different resolutions.
        """
        # Forward pass on the backbone.
        backbone_out = self.backbone(x)

        # Extra multi-resolution features.
        high_res = self.features_1(backbone_out)
        mid_res = self.features_2(high_res)
        low_res = self.features_3(mid_res)

        return MobileNetV3Output(high_resolution_feat=high_res, mid_resolution_feat=mid_res, low_resolution_feat=low_res)
