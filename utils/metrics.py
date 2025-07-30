"""
This module contains all evaluation metrics used to validate the PromptableDeTR model.
"""
import torch
from torchvision.ops.boxes import box_iou


# Functions.
def iou_accuracy(labels, logits):
    """
    Computes the accuracy of the model based on the Intersection over Union (IoU) 
    between the ground truth and the predicted bounding boxes.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 4).
        logits (numpy.ndarray): The logits from the model with shape (N, 4).

    Returns:
        float: The accuracy of the model.
    """
    # Check if the labels and logits are empty.
    if not labels.shape[0] and not logits.shape[0]:
        return 1.0
    elif labels.shape[0] and not logits.shape[0]:
        return 0.0
    elif not labels.shape[0] and logits.shape[0]:
        return 0.0

    # Compute the accuracy.
    iou_score = box_iou(
        torch.from_numpy(labels),
        torch.from_numpy(logits)
    )
    iou_score = iou_score[iou_score > 0.0]
    iou_score = iou_score.float().mean().item()

    # Check if the IoU score is NaN.
    if torch.isnan(iou_score):
        iou_score = 0.0

    return iou_score
