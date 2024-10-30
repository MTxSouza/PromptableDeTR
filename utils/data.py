"""
This module contains useful functions to manipulate any data type and objects 
in the PromptVision project.
"""
import warnings

import torch


# Functions.
def box_cxcywh_to_xyxy(boxes, width, height):
    """
    Converts bounding boxes from (center_x, center_y, width, height) to 
    (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (torch.Tensor): The bounding boxes with shape (B, N, 4).
        width (int): The width of the image.
        height (int): The height of the image.
    
    Returns:
        torch.Tensor: The bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    # Check if the class column is present.
    class_col = None
    bbox_index = 0
    if boxes.size(-1) == 5:
        warnings.warn("The class column is present but it will be ignored during the conversion.")
        class_col = boxes[:, :, 0]
        bbox_index = 1

    # Compute bounding box coordinates.
    half_width = (boxes[:, :, bbox_index + 2] * width) / 2
    half_height = (boxes[:, :, bbox_index + 3] * height) / 2
    x1 = boxes[:, :, bbox_index] * width - half_width
    y1 = boxes[:, :, bbox_index + 1] * height - half_height
    x2 = boxes[:, :, bbox_index] * width + half_width
    y2 = boxes[:, :, bbox_index + 1] * height + half_height

    # Clamp the bounding box coordinates.
    x1 = torch.clamp(x1, min=0)
    y1 = torch.clamp(y1, min=0)
    x2 = torch.clamp(x2, max=width)
    y2 = torch.clamp(y2, max=height)

    # Concatenate the bounding box coordinates.
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    if class_col is not None:
        xyxy = torch.cat([class_col.unsqueeze(-1), xyxy], dim=-1)

    return xyxy


def box_xyxy_to_cxcywh(boxes, width, height):
    """
    Converts bounding boxes from (x_min, y_min, x_max, y_max) to 
    (center_x, center_y, width, height) format.

    Args:
        boxes (torch.Tensor): The bounding boxes with shape (B, N, 4).
        width (int): The width of the image.
        height (int): The height of the image.
    
    Returns:
        torch.Tensor: The bounding boxes in (center_x, center_y, width, height) format.
    """
    # Check if the class column is present.
    class_col = None
    bbox_index = 0
    if boxes.size(-1) == 5:
        warnings.warn("The class column is present but it will be ignored during the conversion.")
        class_col = boxes[:, :, 0]
        bbox_index = 1

    # Compute bounding box coordinates.
    cx = (boxes[:, :, bbox_index] + boxes[:, :, bbox_index + 2]) / 2
    cy = (boxes[:, :, bbox_index + 1] + boxes[:, :, bbox_index + 3]) / 2
    w = (boxes[:, :, bbox_index + 2] - boxes[:, :, bbox_index]).float()
    h = (boxes[:, :, bbox_index + 3] - boxes[:, :, bbox_index + 1]).float()

    # Normalize the bounding box coordinates.
    cx /= width
    cy /= height
    w /= width
    h /= height

    # Concatenate the bounding box coordinates.
    cxcywh = torch.stack([cx, cy, w, h], dim=-1)
    if class_col is not None:
        cxcywh = torch.cat([class_col.unsqueeze(-1), cxcywh], dim=-1)

    return cxcywh
