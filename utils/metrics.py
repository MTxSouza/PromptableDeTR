"""
This module contains all evaluation metrics used to validate the PromptableDeTR model.
"""
import numpy as np
import torch


# Functions.
def dist_accuracy(labels, logits, conf, threshold=0.5, return_logits=False):
    """
    Computes the accuracy of the model based on the L1 distance
    between the ground truth and the predicted bounding points.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 2).
        logits (numpy.ndarray): The logits from the model with shape (N, 2).
        conf (numpy.ndarray): The confidence scores with shape (N, 2).
        threshold (float): The presence threshold for positive samples. (Default: 0.5)
        return_logits (bool): Whether to return the filtered logits. (Default: False)

    Returns:
        tensor: The distance accuracy score.
    """
    # Check types.
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Filter out the points based on confidence.
    logits = logits[conf[:, 1] >= threshold]

    # Check if the labels and logits are empty.
    if not labels.size(0) and not logits.size(0):
        if return_logits:
            return torch.tensor(1.0), logits
        return torch.tensor(1.0)
    elif (labels.size(0) and not logits.size(0)) or (not labels.size(0) and logits.size(0)):
        if return_logits:
            return torch.tensor(0.0), logits
        return torch.tensor(0.0)

    # Compute the accuracy.
    dist_score = 1 - torch.cdist(
        logits,
        labels
    ).mean()

    # Check if the distance score is NaN.
    if torch.isnan(dist_score):
        dist_score = torch.tensor(0.0)

    if return_logits:
        return dist_score, logits
    return dist_score


def average_precision_open_vocab(labels, logits, threshold=0.5):
    """
    Computes the average precision for open vocabulary object detection.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 1).
        logits (numpy.ndarray): The logits from the model with shape (N, 2).
        threshold (float): Threshold for positive samples. (Default: 0.5)

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

    labels = labels.reshape((-1, 1))
    logits = logits.reshape((-1, 2))

    # Filter out the invalid points.
    logits = (logits[:, 1] >= threshold).astype(int)

    # Compute precision and recall.
    true_positive = torch.tensor((labels == 1) & (logits == 1), dtype=torch.float32).sum()
    false_positive = torch.tensor((labels == 0) & (logits == 1), dtype=torch.float32).sum()
    false_negative = torch.tensor((labels == 1) & (logits == 0), dtype=torch.float32).sum()

    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)
    
    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive + false_negative)

    ap = precision * recall  # Simplified for binary classification.
    return ap.item()
