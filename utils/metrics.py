"""
This module contains all evaluation metrics used to validate the PromptableDeTR model.
"""
import torch
from torchvision.ops.boxes import box_iou

from models.matcher import HuggarianMatcher


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
    iou_score = iou_score.float().mean()

    # Check if the IoU score is NaN.
    if torch.isnan(iou_score):
        iou_score = torch.tensor(0.0)

    return iou_score.item()


def average_precision_open_vocab(labels, logits, iou_threshold=0.5):
    """
    Computes the average precision for open vocabulary object detection. The metric
    is based on the paper https://arxiv.org/pdf/2102.01066.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 4).
        logits (numpy.ndarray): The logits from the model with shape (N, 4).
        iou_threshold (float): The IoU threshold for positive samples. (Default: 0.5)

    Returns:
        float: The average precision score.
    """
    # Check types.
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    # Check number of dimensions.
    if labels.ndim == 2:
        labels = labels[None, :, :]
    if logits.ndim == 2:
        logits = logits[None, :, :]

    labels = labels.reshape((-1, 4))
    logits = logits.reshape((-1, 4))

    # Compute the IoU between the predicted and true boxes.
    iou_matrix = box_iou(
        torch.from_numpy(logits),
        torch.from_numpy(labels)
    )

    # Compute precision and recall based on the IoU threashold.
    n_pred, n_true = iou_matrix.shape
    matched = set()
    true_positive = torch.zeros(n_pred)
    false_positive = torch.zeros(n_pred)

    for idx_iou in range(n_pred):
        iou_vec = iou_matrix[idx_iou]
        max_iou, idx = iou_vec.max(0)
        idx = idx.item()
        if max_iou >= iou_threshold and idx not in matched:
            matched.add(idx)
            true_positive[idx_iou] = 1
        else:
            false_positive[idx_iou] = 1
    
    true_positive = torch.cumsum(true_positive, dim=0)
    false_positive = torch.cumsum(false_positive, dim=0)
    recall = true_positive / n_true
    precision = true_positive / (true_positive + false_positive)

    recall = torch.cat(tensors=[torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat(tensors=[torch.tensor([1.0]), precision, torch.tensor([0.0])])

    for i in range(precision.size(0) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:]).item()

    return ap
