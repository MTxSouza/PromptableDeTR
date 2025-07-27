"""
This module contains useful functions to manipulate any data type and objects 
in the PromptVision project.
"""
import torch


# Functions.
def xywh_to_xyxy(boxes, width=640, height=640):
    """
    Converts bounding boxes from (x_min, y_min, width, height) to 
    (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (torch.Tensor): The bounding boxes with shape (N, 4).
        width (int): The width of the image. (Default: 640)
        height (int): The height of the image. (Default: 640)
    
    Returns:
        torch.Tensor: The bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    # Check boxes type.
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)

    # Convert 0-1 coordinates to pixel coordinates.
    new_boxes = boxes.clone()
    new_boxes[:, 0::2] *= width
    new_boxes[:, 1::2] *= height

    new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]
    new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]

    # Clamp the bounding box coordinates.
    new_boxes[:, 0::2] = torch.clamp(new_boxes[:, 0::2], min=0, max=width)
    new_boxes[:, 1::2] = torch.clamp(new_boxes[:, 1::2], min=0, max=height)

    new_boxes = new_boxes.numpy()

    return new_boxes
