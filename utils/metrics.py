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


def average_precision_open_vocab(labels, logits, threshold=0.5):
    """
    Computes the average precision for open vocabulary object detection. The metric
    is based on the paper https://arxiv.org/pdf/2102.01066.

    Args:
        labels (numpy.ndarray): The true labels with shape (N, 1).
        logits (numpy.ndarray): The logits from the model with shape (N, 2).
        threshold (float): Threshold for positive samples. (Default: 0.25)

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
    logits = (logits[:, 1] >= threshold).astype(float)

    # Compute precision and recall.
    n_true = (labels == 1).sum().item()
    true_positive = torch.tensor((labels == 1) & (logits == 1), dtype=torch.float32).sum()
    false_positive = torch.tensor((labels == 0) & (logits == 1), dtype=torch.float32).sum()

    recall = (true_positive / n_true).unsqueeze(dim=0)
    precision = (true_positive / (true_positive + false_positive)).unsqueeze(dim=0)

    recall = torch.cat(tensors=[torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat(tensors=[torch.tensor([1.0]), precision, torch.tensor([0.0])])

    for i in range(precision.size(0) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:]).item()

    return ap
