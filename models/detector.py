"""
This module contains the Detector model class used to predict bounding boxes and presence of objects 
in the image.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
from models.base import BasePromptableDeTR
from models.matcher import HuggarianMatcher, generalized_iou

# Logger.
logger = Logger(name="model")


# Classes.
class PromptableDeTR(BasePromptableDeTR):


    # Special methods.
    def __init__(self, proj_dim = 512, **kwargs):
        """
        Initializes the Detector class used to predict bounding boxes and presence 
        of objects in the image.

        Args:
            proj_dim (int): The projection dimension of the image and text embeddings. (Default: 512)
        """
        super().__init__(**kwargs)

        # Layers.
        self.detector = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=proj_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=proj_dim * 4, out_features=proj_dim * 2),
            nn.ReLU()
        )
        self.bbox_predictor = nn.Linear(in_features=proj_dim * 2, out_features=4)
        self.presence_predictor = nn.Linear(in_features=proj_dim * 2, out_features=2)

        # Matcher.
        self.__presence_weight = None
        self.__l1_weight = None
        self.__giou_weight = None
        self.matcher = None


    # Methods.
    def define_matcher(self, presence_loss_weight = 1.0, l1_loss_weight = 1.0, giou_loss_weight = 1.0):
        """
        Define the matcher used to align the bounding boxes with the respective label. It is only 
        defined during the training phase.

        Args:
            presence_loss_weight (float): The weight for the presence loss. (Default: 1.0)
            l1_loss_weight (float): The weight for the L1 loss. (Default: 1.0)
            giou_loss_weight (float): The weight for the GIoU loss. (Default: 1.0)
        """
        logger.info(msg="Defining the matcher for the detector.")
        self.__presence_weight = presence_loss_weight
        self.__l1_weight = l1_loss_weight
        self.__giou_weight = giou_loss_weight
        self.matcher = HuggarianMatcher(
            presence_loss_weight=presence_loss_weight,
            l1_loss_weight=l1_loss_weight,
            giou_loss_weight=giou_loss_weight
        )


    def forward(self, image, prompt):
        """
        Forward pass of the detector.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.

        Returns:
            torch.Tensor: The predicted bounding boxes and presence.
        """
        logger.info(msg="Calling `Detector` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Compute joint embeddings.
        logger.debug(msg="- Calling `BasePromptableDeTR` forward method.")
        joint = super().forward(image=image, prompt=prompt)
        logger.debug(msg="- Result of the `BasePromptableDeTR` forward method: %s." % (joint.shape,))

        # Predict bounding boxes and presence.
        logger.debug(msg="- Calling `nn.Sequential` block to the tensor %s." % (joint.shape,))
        out = self.detector(joint)
        logger.debug(msg="- Result of the `nn.Sequential` block: %s." % (out.shape,))

        logger.debug(msg="- Calling `nn.Linear` block to the tensor %s." % (out.shape,))
        bbox = self.bbox_predictor(out)
        presence = self.presence_predictor(out)
        logger.debug(msg="- Result of the `nn.Linear` block: %s and %s." % (bbox.shape, presence.shape))

        # Concatenate the predictions.
        logger.debug(msg="- Concatenating the predictions.")
        outputs = torch.cat(tensors=(bbox, presence), dim=-1)
        logger.debug(msg="- Result of the concatenation: %s." % (outputs.shape,))

        return outputs


    def save_model(self, dir_path, loss, samples, ckpt_step = None, is_best = False):
        """
        Save the model weights.

        Args:
            dir_path (str): The path to the directory where the weights will be saved.
            loss (float): The loss of the model at the checkpoint.
            samples (List[Tuple[str, str]]): The validation results at the checkpoint.
            ckpt_step (int): The checkpoint step. (Default: None)
            is_best (bool): Flag to indicate if the checkpoint is the best. (Default: False)
        """
        logger.info(msg="Saving the model weights.")
        
        # Define the checkpoint path.
        name = "promptable-detr"

        # Check if the model is the best.
        if is_best:
            name += "-best"
        
        if ckpt_step is not None:
            name += "-ckpt-%d" % ckpt_step
        
        os.makedirs(name=dir_path, exist_ok=True)
        ckpt_fp = os.path.join(dir_path, name + ".pth")
        log_fp = os.path.join(dir_path, name + ".log")

        # Save the weights.
        torch.save(obj=self.state_dict(), f=ckpt_fp)

        # Save the log.
        with open(file=log_fp, mode="w") as f:
            f.write("Detector results:\n")
            f.write("Loss: %s\n\n" % loss)
            f.write("Samples:\n")


    def compute_loss(self, logits, labels):
        """
        Compute the loss needed to train the detector model.

        Args:
            logits torch.Tensor): The predicted tensor.
            labels (torch.Tensor): The true tensor.

        Returns:
            torch.Tensor: The loss value.
        """
        logger.info(msg="Computing the detector loss.")

        # Sort the logits and labels.
        pred_presence = logits[:, :, 4:]
        pred_boxes = logits[:, :, :4]
        true_presence = labels[:, :, 4].view(-1, 1)
        true_boxes = labels[:, :, :4]
        indices = self.matcher(predict_scores=pred_presence, predict_boxes=pred_boxes, scores=true_presence, boxes=true_boxes)

        sorted_pred_presence = torch.gather(pred_presence, dim=1, index=indices)
        sorted_pred_boxes = torch.gather(pred_boxes, dim=1, index=indices)
        sorted_true_presence = torch.gather(true_presence, dim=1, index=indices)
        sorted_true_boxes = torch.gather(true_boxes, dim=1, index=indices)

        # Compute presence loss.
        flt_sorted_pred_presence = sorted_pred_presence.view(-1, 2)
        flt_sorted_true_presence = sorted_true_presence.view(-1)
        presence_loss = F.cross_entropy(input=flt_sorted_pred_presence, target=flt_sorted_true_presence, reduction="mean")

        # Compute bounding box loss.
        flt_sorted_pred_boxes = sorted_pred_boxes.view(-1, 4)
        flt_sorted_true_boxes = sorted_true_boxes.view(-1, 4)
        bbox_loss = F.l1_loss(input=flt_sorted_pred_boxes, target=flt_sorted_true_boxes, reduction="mean")
        giou_loss = -torch.diag(generalized_iou(sorted_pred_boxes, sorted_true_boxes))
        giou_loss = giou_loss.mean()

        # Compute the total loss.
        loss = self.__l1_weight * bbox_loss + self.__presence_weight * presence_loss + self.__giou_weight * giou_loss 
        logger.info(msg="Returning the loss value.")

        return loss
