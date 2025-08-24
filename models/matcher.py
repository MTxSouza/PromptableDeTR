"""
This module contains the algorithm used to align the bounding boxes with the respective label for 
detection task.
"""
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from logger import Logger
from utils.data import generalized_iou

# Logger.
logger = Logger(name="model")


# Functions.



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
    def forward(self, predict_scores, predict_points, scores, points):
        """
        Find the best matching between the predicted points and the ground truth 
        points using the Hungarian algorithm.

        Args:
            predict_scores (torch.Tensor): The predicted scores from the model with shape (B, N, 2).
            predict_points (torch.Tensor): The predicted points from the model with shape (B, N, 2).
            scores (torch.Tensor): The ground truth scores with shape (B, M).
            points (torch.Tensor): The ground truth points with shape (B, M, 2).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The batch, source, and target indices.
        """
        # Get the batch size and number of predicted objects.
        B, N = predict_scores.size()[:2]
        indices = []

        for batch in range(B):

            # Get the predicted scores and points for the current batch.
            batch_predict_scores = predict_scores[batch].softmax(dim=1)
            batch_predict_points = predict_points[batch]
            batch_scores = scores[batch]
            batch_points = points[batch]

            obj_idx = batch_scores == 1
            batch_scores = batch_scores[obj_idx]
            batch_points = batch_points[obj_idx]

            # Compute presence loss.
            presence_loss = -batch_predict_scores[:, batch_scores]

            # Compute L1 loss.
            l1_loss = torch.cdist(batch_predict_points, batch_points, p=1)

            # Compute matrix loss.
            mtx_loss = self.presence_loss_weight * presence_loss + self.l1_loss_weight * l1_loss
            row_idx, col_idx = linear_sum_assignment(mtx_loss.cpu())
            indices.append((row_idx, col_idx))

        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # Get the batch index, source index and target index.
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, src_idx, tgt_idx
