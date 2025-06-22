"""
This module contains the algorithm used to align the bounding boxes with the respective label for 
detection task.
"""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area

from logger import Logger

# Logger.
logger = Logger(name="model")


# Functions.
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
    logger.debug(msg="- Computing the IoU between the two sets of boxes.")

    # Convert coordinates to (x1, y1, x2, y2).
    boxes1 = xywh2xyxy(boxes=boxes1)
    boxes2 = xywh2xyxy(boxes=boxes2)

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


# Classes.
class HuggarianMatcher(nn.Module):


    # Special methods.
    def __init__(self, presence_loss_weight = 1.0, l1_loss_weight = 1.0, giou_loss_weight = 1.0):
        """
        Class constructor for the HungarianMatcher class.

        Args:
            presence_loss_weight (float): The weight for the presence loss. (Default: 1.0)
            l1_loss_weight (float): The weight for the L1 loss. (Default: 1.0)
            giou_loss_weight (float): The weight for the GIoU loss. (Default: 1.0)
        """
        super().__init__()

        # Attribute.
        self.presence_loss_weight = presence_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.giou_loss_weight = giou_loss_weight


    # Methods.
    @torch.no_grad()
    def forward(self, predict_scores, predict_boxes, scores, boxes):
        """
        Find the best matching between the predicted boxes and the ground truth 
        boxes using the Hungarian algorithm.

        Args:
            predict_scores (torch.Tensor): The predicted scores from the model with shape (B, N, 2).
            predict_boxes (torch.Tensor): The predicted boxes from the model with shape (B, N, 4).
            scores (torch.Tensor): The ground truth scores with shape (B, M).
            boxes (torch.Tensor): The ground truth boxes with shape (B, M, 4).

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: The best matching between the predicted and ground truth boxes.
        """
        # Get the batch size and number of predicted objects.
        B, N = predict_scores.size()[:2]
        indices = []

        for batch in range(B):

            # Get the predicted scores and boxes for the current batch.
            batch_predict_scores = predict_scores[batch].softmax(dim=1)
            batch_predict_boxes = predict_boxes[batch]
            batch_scores = scores[batch]
            batch_boxes = boxes[batch]

            obj_idx = batch_scores == 1
            batch_scores = batch_scores[obj_idx]
            batch_boxes = batch_boxes[obj_idx]

            # Compute presence loss.
            presence_loss = -batch_predict_scores[:, batch_scores]

            # Compute L1 loss.
            l1_loss = torch.cdist(batch_predict_boxes, batch_boxes, p=1)

            # Compute GIoU loss.
            giou_loss = 1 - generalized_iou(batch_predict_boxes, batch_boxes)

            # Compute matrix loss.
            mtx_loss = self.presence_loss_weight * presence_loss + self.l1_loss_weight * l1_loss + self.giou_loss_weight * giou_loss
            row_idx, col_idx = linear_sum_assignment(mtx_loss.cpu())
            indices.append((row_idx, col_idx))

        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices
