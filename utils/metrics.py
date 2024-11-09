"""
This module contains all cost functions and evaluation metrics used to train 
and validate the PromptVision model.
"""
import torch
import torch.nn.functional as F


# Functions.
def bipartite_matching_loss(y, yhat):
    """
    Computes the bipartite matching loss between the ground truth and the 
    predicted labels. It uses the Hungarian algorithm to find the optimal 
    matching between the two sets of object presence scores.

    Equation:
    > Loss = - 1 * {y[i] != 0} * yhat[i] + 1 * {y[i] != 0}

    Where y[i] is the ground truth object presence score and yhat[i] is the 
    predicted object presence score.

    Args:
        y (torch.Tensor): The ground truth labels with shape (B, N, 5).
        yhat (torch.Tensor): The predicted labels with shape (B, N, 5).

    Returns:
        torch.Tensor: The bipartite matching loss.
    """
    # Apply mask to consider only the object presence scores.
    object_presence = (y[:, :, 0] != 0).float()

    # Compute the object presence loss.
    object_presence_score = F.binary_cross_entropy(
        input=yhat[:, :, 0], 
        target=object_presence,
        reduction="sum"
    )

    # Compute the bipartite matching loss.
    return object_presence_score


def iou_loss(y, yhat, width, height):
    """
    Computes the Intersection over Union (IoU) loss between the ground truth 
    and the predicted bounding boxes.

    Equation:
    > Loss = 1 - IoU(y, yhat)

    Where y and yhat are the ground truth and predicted bounding boxes 
    respectively.

    Args:
        y (torch.Tensor): The ground truth bounding boxes with shape (B, N, 5).
        yhat (torch.Tensor): The predicted bounding boxes with shape (B, N, 5).
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        torch.Tensor: The IoU loss.
    """
    # Compute bounding box coordinates.
    y_x1 = y[:, :, 1] * width - (y[:, :, 3] * width) / 2
    y_y1 = y[:, :, 2] * height - (y[:, :, 4] * height) / 2
    y_x2 = y[:, :, 1] * width + (y[:, :, 3] * width) / 2
    y_y2 = y[:, :, 2] * height + (y[:, :, 4] * height) / 2

    yhat_x1 = yhat[:, :, 1] * width - (yhat[:, :, 3] * width) / 2
    yhat_y1 = yhat[:, :, 2] * height - (yhat[:, :, 4] * height) / 2
    yhat_x2 = yhat[:, :, 1] * width + (yhat[:, :, 3] * width) / 2
    yhat_y2 = yhat[:, :, 2] * height + (yhat[:, :, 4] * height) / 2

    # Compute the intersection coordinates.
    inter_x1 = torch.max(y_x1, yhat_x1)
    inter_y1 = torch.max(y_y1, yhat_y1)
    inter_x2 = torch.min(y_x2, yhat_x2)
    inter_y2 = torch.min(y_y2, yhat_y2)

    # Compute the intersection area.
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute the union area.
    y_area = (y_x2 - y_x1) * (y_y2 - y_y1)
    yhat_area = (yhat_x2 - yhat_x1) * (yhat_y2 - yhat_y1)
    union_area = y_area + yhat_area - inter_area

    # Compute the IoU.
    iou = inter_area / union_area

    # Compute the IoU loss.
    return 1 - iou
