"""
This module contains all evaluation metrics used to validate the PromptableDeTR model.
"""
import torch


# Functions.
def dist_accuracy(labels, logits, threshold=0.25):
    """
    Computes the accuracy of the model based on the L1 distance
    between the ground truth and the predicted bounding points.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 2).
        logits (numpy.ndarray): The logits from the model with shape (N, 2).
        threshold (float): The distance threshold for positive samples. (Default: 0.25)

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
    dist_score = torch.cdist(
        torch.from_numpy(logits),
        torch.from_numpy(labels),
        p=2
    )
    dist_score = dist_score[threshold >= dist_score]
    dist_score = dist_score.float().mean()

    # Check if the distance score is NaN.
    if torch.isnan(dist_score):
        dist_score = torch.tensor(0.0)

    return dist_score.item()


def average_precision_open_vocab(labels, logits, dist_threshold=0.25):
    """
    Computes the average precision for open vocabulary object detection. The metric
    is based on the paper https://arxiv.org/pdf/2102.01066.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 4).
        logits (numpy.ndarray): The logits from the model with shape (N, 4).
        dist_threshold (float): Distance threshold for positive samples. (Default: 0.25)

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

    labels = labels.reshape((-1, 2))
    logits = logits.reshape((-1, 2))

    # Compute the IoU between the predicted and true boxes.
    dist_matrix = torch.cdist(
        torch.from_numpy(logits),
        torch.from_numpy(labels),
        p=2
    )

    # Compute precision and recall based on the IoU threashold.
    n_pred, n_true = dist_matrix.shape
    matched = set()
    true_positive = torch.zeros(n_pred)
    false_positive = torch.zeros(n_pred)

    for idx_iou in range(n_pred):
        iou_vec = dist_matrix[idx_iou]
        min_dist, idx = iou_vec.min(0)
        idx = idx.item()
        if dist_threshold >= min_dist and idx not in matched:
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
