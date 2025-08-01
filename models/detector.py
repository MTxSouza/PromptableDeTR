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
    def __init__(
            self, 
            image_tokens, 
            vocab_size = 30522, 
            emb_dim = 128, 
            proj_dim = 512, 
            num_heads = 8, 
            ff_dim = 2048, 
            emb_dropout_rate = 0.1, 
            num_joiner_layers = 3
        ):
        """
        Initializes the Detector class used to predict bounding boxes and presence 
        of objects in the image.

        Args:
            proj_dim (int): The projection dimension of the image and text embeddings. (Default: 512)
        """
        super().__init__(
            image_tokens=image_tokens, 
            vocab_size=vocab_size, 
            emb_dim=emb_dim, 
            proj_dim=proj_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            emb_dropout_rate=emb_dropout_rate, 
            num_joiner_layers=num_joiner_layers
        )

        # Layers.
        self.detector = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=proj_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=proj_dim * 4, out_features=proj_dim * 2),
            nn.ReLU()
        )
        self.bbox_predictor = nn.Linear(in_features=proj_dim * 2, out_features=4)
        self.presence_predictor = nn.Linear(in_features=proj_dim * 2, out_features=2)

        # Initialize weights.
        self.__initialize_weights()

        # Matcher.
        self.__presence_weight = None
        self.__l1_weight = None
        self.__giou_weight = None
        self.matcher = None


    # Private methods.
    def __initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.detector.apply(init_weights)
        self.bbox_predictor.apply(init_weights)
        self.presence_predictor.apply(init_weights)


    def __get_indices(self, matcher_indices):
        """
        Organize the indices to be used in the loss computation.

        Args:
            matcher_indices (List[Tuple[torch.Tensor, torch.Tensor]]): The indices from the matcher.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The batch, source, and target indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matcher_indices)])
        src_idx = torch.cat([src for (src, _) in matcher_indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in matcher_indices])
        return batch_idx, src_idx, tgt_idx


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


    def forward(self, image, prompt, prompt_mask = None):
        """
        Forward pass of the detector.

        Args:
            image (torch.Tensor): Image tensor.
            prompt (torch.Tensor): Prompt tensor.
            prompt_mask (torch.Tensor): Prompt mask tensor. (Default: None)

        Returns:
            torch.Tensor: The predicted bounding boxes and presence.
        """
        logger.info(msg="Calling `Detector` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Compute joint embeddings.
        logger.debug(msg="- Calling `BasePromptableDeTR` forward method.")
        joint = super().forward(image=image, prompt=prompt, prompt_mask=prompt_mask)
        logger.debug(msg="- Result of the `BasePromptableDeTR` forward method: %s." % (joint.shape,))

        # Predict bounding boxes and presence.
        logger.debug(msg="- Calling `nn.Sequential` block to the tensor %s." % (joint.shape,))
        out = self.detector(joint)
        logger.debug(msg="- Result of the `nn.Sequential` block: %s." % (out.shape,))

        logger.debug(msg="- Calling `nn.Linear` block to the tensor %s." % (out.shape,))
        bbox = self.bbox_predictor(out)
        bbox = F.sigmoid(input=bbox)
        presence = self.presence_predictor(out)
        logger.debug(msg="- Result of the `nn.Linear` block: %s and %s." % (bbox.shape, presence.shape))

        # Concatenate the predictions.
        logger.debug(msg="- Concatenating the predictions.")
        outputs = torch.cat(tensors=(bbox, presence), dim=-1)
        logger.debug(msg="- Result of the concatenation: %s." % (outputs.shape,))

        return outputs


    def save_model(self, dir_path, ckpt_step = None, is_best = False):
        """
        Save the model weights.

        Args:
            dir_path (str): The path to the directory where the weights will be saved.
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


    def load_base_model(self, base_model_weights):
        """
        Load the weights of the base model only.

        Args:
            base_model_weights (str): Path to the base model weights.
        """
        logger.info(msg="Loading the base model weights.")
        logger.debug(msg="- Base model weights: %s" % base_model_weights)

        # Load weights.
        super().load_full_weights(base_model_weights=base_model_weights)


    def compute_loss_and_accuracy(self, logits, labels):
        """
        Compute the loss needed to train the detector model and
        it also computes the accuracy of the model.

        Args:
            logits torch.Tensor): The predicted tensor.
            labels (torch.Tensor): The true tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The losses of model.
        """
        logger.info(msg="Computing the detector loss.")

        assert self.matcher is not None, "Matcher is not defined."

        # Sort the logits and labels.
        pred_presence = logits[:, :, 4:]
        pred_boxes = logits[:, :, :4]
        true_presence = labels[:, :, 4].long()
        true_boxes = labels[:, :, :4]
        logger.debug(msg="- Predicted presence shape: %s." % (pred_presence.shape,))
        logger.debug(msg="- Predicted boxes shape: %s." % (pred_boxes.shape,))
        logger.debug(msg="- True presence shape: %s." % (true_presence.shape,))
        logger.debug(msg="- True boxes shape: %s." % (true_boxes.shape,))
        indices = self.matcher(predict_scores=pred_presence, predict_boxes=pred_boxes, scores=true_presence, boxes=true_boxes)
        batch_idx, src_idx, tgt_idx = self.__get_indices(matcher_indices=indices)
        logger.debug(msg="- Batch index shape: %s." % (batch_idx.shape,))
        logger.debug(msg="- Source index shape: %s." % (src_idx.shape,))
        logger.debug(msg="- Target index shape: %s." % (tgt_idx.shape,))

        sorted_pred_presence = pred_presence[(batch_idx, src_idx)]
        sorted_pred_boxes = pred_boxes[(batch_idx, src_idx)]
        sorted_true_presence = true_presence[(batch_idx, tgt_idx)]
        sorted_true_boxes = true_boxes[(batch_idx, tgt_idx)]
        logger.debug(msg="- Sorted predicted presence shape: %s." % (sorted_pred_presence.shape,))
        logger.debug(msg="- Sorted predicted boxes shape: %s." % (sorted_pred_boxes.shape,))
        logger.debug(msg="- Sorted true presence shape: %s." % (sorted_true_presence.shape,))
        logger.debug(msg="- Sorted true boxes shape: %s." % (sorted_true_boxes.shape,))

        # Define new scores labels.
        new_scores = torch.full(size=pred_presence.shape[:2], fill_value=0, device=sorted_pred_presence.device).long()
        logger.debug(msg="- New scores shape: %s." % (new_scores.shape,))
        new_scores[(batch_idx, tgt_idx)] = sorted_true_presence

        # Compute number of boxes.
        obj_idx = new_scores == 1
        num_boxes = obj_idx.sum()
        logger.debug(msg="- Number of boxes: %s." % num_boxes)

        # Compute presence loss.
        presence_weight = None
        if self.__presence_weight != 1.0:
            presence_weight = torch.tensor([1.0, self.__presence_weight], device=pred_presence.device)
        B, N, C = pred_presence.shape
        presence_loss = F.cross_entropy(input=pred_presence.view(B * N, C), target=new_scores.view(-1), weight=presence_weight, reduction="mean")
        logger.debug(msg="- Presence loss: %s." % presence_loss)

        # Compute bounding box loss.
        bbox_loss = F.l1_loss(input=sorted_pred_boxes, target=sorted_true_boxes, reduction="none")
        bbox_loss = bbox_loss.sum() / num_boxes
        logger.debug(msg="- Bounding box loss: %s." % bbox_loss)

        diag_acc = torch.diag(input=generalized_iou(sorted_pred_boxes, sorted_true_boxes))
        giou_loss = 1 - diag_acc
        giou_loss = giou_loss.sum() / num_boxes
        logger.debug(msg="- GIoU loss: %s." % giou_loss)

        # Compute the total loss.
        final_l1_loss = self.__l1_weight * bbox_loss
        final_giou_loss = self.__giou_weight * giou_loss
        final_presence_loss = self.__presence_weight * presence_loss
        loss = final_l1_loss + final_presence_loss + final_giou_loss
        logger.debug(msg="- Total loss: %s." % loss)
        logger.info(msg="Returning the loss value.")

        return loss, final_l1_loss, final_giou_loss, final_presence_loss
