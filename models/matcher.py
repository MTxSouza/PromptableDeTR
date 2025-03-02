"""
This module contains the algorithm used to align the bounding boxes with the respective label for 
detection task.
"""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from logger import Logger

# Logger.
logger = Logger(name="model")


# Functions.
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

    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
    boxes1 = cxcywh_to_xyxy(boxes1)
    boxes2 = cxcywh_to_xyxy(boxes2)

    # Compute area.
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Compute intersection.
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Compute union.
    union = area1[:, None] + area2[None, :] - inter

    # Compute IoU.
    iou = inter / union

    # Compute GIoU.
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / (area + 1e-6)

    return giou


def cxcywh_to_xyxy(boxes):
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        boxes (torch.Tensor): The boxes to be converted with shape (N, 4).

    Returns:
        torch.Tensor: The converted boxes with shape (N, 4).
    """
    logger.debug(msg="- Converting boxes from (cx, cy, w, h) to (x1, y1, x2, y2).")
    cx, cy, w, h = boxes.unbind(-1)
    new_boxes = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_boxes, dim=-1)


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

        # Flatten the predicted scores and boxes.
        flt_predict_scores = predict_scores.view(-1, 2)
        flt_predict_boxes = predict_boxes.view(-1, 4)

        # Join all targets.
        all_scores = scores.view(-1).long()
        all_boxes = boxes.view(-1, 4)

        # Compute presence loss.
        presence_loss = -flt_predict_scores[torch.arange(B * N)[:, None], all_scores[None, :]]

        # Compute L1 loss.
        l1_loss = torch.cdist(flt_predict_boxes, all_boxes, p=1)

        # Compute GIoU loss.
        giou_loss = -generalized_iou(flt_predict_boxes, all_boxes)

        # Compute matrix loss.
        mtx_loss = self.presence_loss_weight * presence_loss + self.l1_loss_weight * l1_loss + self.giou_loss_weight * giou_loss
        mtx_loss = mtx_loss.view(B, N, -1).cpu()

        # Compute the best matching.
        n_targets = [scores.shape[1] for _ in range(B)]
        indices = [linear_sum_assignment(mtx[i]) for i, mtx in enumerate(iterable=mtx_loss.split(split_size=n_targets, dim=-1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices
