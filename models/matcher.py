"""
This module contains the algorithm used to align the bounding boxes with the respective label for 
detection task.
"""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


# Classes.
class HuggarianMatcher(nn.Module):


    # Special methods.
    def __init__(self, presence_loss_weight = 1.0, l1_loss_weight = 1.0):
        """
        Class constructor for the HungarianMatcher class.

        Args:
            presence_loss_weight (float): The weight for the presence loss. (Default: 1.0)
            l1_loss_weight (float): The weight for the L1 loss. (Default: 1.0)
        """
        super().__init__()

        # Attribute.
        self.presence_loss_weight = presence_loss_weight
        self.l1_loss_weight = l1_loss_weight


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
            List[Tuple[numpy.ndarray, numpy.ndarray]]: The best matching between the predicted and ground truth boxes.
        """
        # Get the batch size and number of predicted objects.
        B, N = predict_scores.size()[:2]

        # Flatten the predicted scores and boxes.
        flt_predict_scores = predict_scores.view(-1, 2)
        flt_predict_boxes = predict_boxes.view(-1, 4)

        # Join all targets.
        all_scores = scores.view(-1)
        all_boxes = boxes.view(-1, 4)

        # Compute presence loss.
        presence_loss = -flt_predict_scores[torch.arange(N)[:, None], all_scores[None, :]]

        # Compute L1 loss.
        l1_loss = torch.cdist(flt_predict_boxes, all_boxes, p=1)

        # Compute matrix loss.
        mtx_loss = self.presence_loss_weight * presence_loss + self.l1_loss_weight * l1_loss
        mtx_loss = mtx_loss.view(B, N, -1).cpu()

        # Compute the best matching.
        n_targets = [scores.shape[1] for _ in range(B)]
        indices = [linear_sum_assignment(mtx[i]) for i, mtx in enumerate(iterable=mtx_loss.split(split_size=n_targets, dim=0))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return indices
