"""
This module contains useful functions to manipulate any data type and objects 
in the PromptVision project.
"""
import torch
from torchvision.ops.boxes import box_area


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


def xywh2xyxy(boxes):
    """
    This function converts the coordinates from (x, y, w, h) to (x1, y1, x2, y2).

    Args:
        boxes (torch.Tensor): The boxes with shape (N, 4).

    Returns:
        torch.Tensor: The converted boxes with shape (N, 4).
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    new_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return new_boxes


def generalized_iou(boxes1, boxes2):
    """
    Compute the Generalized Intersection over Union (GIoU) between two sets of boxes.

    Args:
        boxes1 (torch.Tensor): The first set of boxes with shape (N, 4).
        boxes2 (torch.Tensor): The second set of boxes with shape (M, 4).

    Returns:
        torch.Tensor: The GIoU between the two sets of boxes.
    """
    # Convert coordinates to (x1, y1, x2, y2).
    boxes1 = xywh_to_xyxy(boxes=boxes1)
    boxes2 = xywh_to_xyxy(boxes=boxes2)

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # Compute area.
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Compute intersection.
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Compute union.
    union = area1[:, None] + area2 - inter

    # Compute IoU.
    iou = inter / union

    # Compute GIoU.
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area

    return giou
