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
from models.matcher import HuggarianMatcher
from utils.metrics import average_precision_open_vocab

# Logger.
logger = Logger(name="model")


# Classes.
class PromptableDeTR(BasePromptableDeTR):


    # Special methods.
    def __init__(
            self, 
            image_tokens = [400, 100], 
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
            image_tokens (List[int]): The number of tokens in the image. (Default: [400, 100])
            vocab_size (int): The size of the vocabulary. (Default: 30522)
            emb_dim (int): The embedding dimension of the image and text embeddings. (Default: 128)
            proj_dim (int): The projection dimension of the image and text embeddings. (Default: 512)
            num_heads (int): The number of attention heads. (Default: 8)
            ff_dim (int): The dimension of the feed-forward network. (Default: 2048)
            emb_dropout_rate (float): The dropout rate for the embeddings. (Default: 0.1)
            num_joiner_layers (int): The number of joiner layers in the model. (Default: 3)
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
        self.point_predictor = nn.Linear(in_features=proj_dim * 2, out_features=2)
        self.presence_predictor = nn.Linear(in_features=proj_dim * 2, out_features=2)

        # Initialize weights.
        self.__initialize_weights()


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
        self.point_predictor.apply(init_weights)
        self.presence_predictor.apply(init_weights)


    # Methods.
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
        point = self.point_predictor(out)
        point = F.sigmoid(input=point)
        presence = self.presence_predictor(out)
        logger.debug(msg="- Result of the `nn.Linear` block: %s and %s." % (point.shape, presence.shape))

        # Concatenate the predictions.
        logger.debug(msg="- Concatenating the predictions.")
        outputs = torch.cat(tensors=(point, presence), dim=-1)
        logger.debug(msg="- Result of the concatenation: %s." % (outputs.shape,))

        return outputs


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


class PromptableDeTRTrainer(PromptableDeTR):
    """
    A subclass of PromptableDeTR that is used for training the model.
    It inherits from PromptableDeTR and adds the functionality to compute
    the loss and accuracy of the model.
    """

    # Special methods.
    def __init__(self, presence_loss_weight=1.0, l1_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Matcher.
        self.__presence_weight = presence_loss_weight
        self.__l1_weight = l1_loss_weight
        self.matcher = HuggarianMatcher(
            presence_loss_weight=self.__presence_weight,
            l1_loss_weight=self.__l1_weight
        )


    # Methods.
    def compute_loss_and_accuracy(self, logits, labels):
        """
        Compute the loss needed to train the detector model and
        it also computes the accuracy of the model.

        Args:
            logits torch.Tensor): The predicted tensor.
            labels (torch.Tensor): The true tensor.

        Returns:
            dict: A dictionary containing the loss and accuracy values.
        """
        logger.info(msg="Computing the detector loss.")

        assert self.matcher is not None, "Matcher is not defined."

        # Sort the logits and labels.
        pred_presence = logits[:, :, 2:]
        pred_points = logits[:, :, :2]
        true_presence = labels[:, :, 2].long()
        true_points = labels[:, :, :2]
        logger.debug(msg="- Predicted presence shape: %s." % (pred_presence.shape,))
        logger.debug(msg="- Predicted points shape: %s." % (pred_points.shape,))
        logger.debug(msg="- True presence shape: %s." % (true_presence.shape,))
        logger.debug(msg="- True points shape: %s." % (true_points.shape,))
        batch_idx, src_idx, tgt_idx = self.matcher(predict_scores=pred_presence, predict_points=pred_points, scores=true_presence, points=true_points)
        logger.debug(msg="- Batch index shape: %s." % (batch_idx.shape,))
        logger.debug(msg="- Source index shape: %s." % (src_idx.shape,))
        logger.debug(msg="- Target index shape: %s." % (tgt_idx.shape,))

        sorted_pred_presence = pred_presence[(batch_idx, src_idx)]
        sorted_pred_points = pred_points[(batch_idx, src_idx)]
        sorted_true_presence = true_presence[(batch_idx, tgt_idx)]
        sorted_true_points = true_points[(batch_idx, tgt_idx)]
        logger.debug(msg="- Sorted predicted presence shape: %s." % (sorted_pred_presence.shape,))
        logger.debug(msg="- Sorted predicted points shape: %s." % (sorted_pred_points.shape,))
        logger.debug(msg="- Sorted true presence shape: %s." % (sorted_true_presence.shape,))
        logger.debug(msg="- Sorted true points shape: %s." % (sorted_true_points.shape,))

        # Compute average precision.
        ap_50 = torch.tensor(average_precision_open_vocab(labels=sorted_true_presence, logits=sorted_pred_presence, threshold=0.50))
        ap_75 = torch.tensor(average_precision_open_vocab(labels=sorted_true_presence, logits=sorted_pred_presence, threshold=0.75))
        ap_90 = torch.tensor(average_precision_open_vocab(labels=sorted_true_presence, logits=sorted_pred_presence, threshold=0.90))
        logger.debug(msg="- Average precision @0.50: %s." % ap_50)
        logger.debug(msg="- Average precision @0.75: %s." % ap_75)
        logger.debug(msg="- Average precision @0.90: %s." % ap_90)

        # Compute number of points.
        obj_idx = sorted_true_presence == 1
        num_points = obj_idx.sum()
        logger.debug(msg="- Number of points: %s." % num_points)

        # Compute presence loss with focal loss.
        presence_weight = torch.tensor([1.0, self.__presence_weight], device=pred_presence.device)
        predictions = sorted_pred_presence.view(-1, 2)
        targets = sorted_true_presence.view(-1)
        presence_loss = F.cross_entropy(input=predictions, target=targets, weight=presence_weight, reduction="none")
        presence_loss = presence_loss[obj_idx].sum() / num_points

        # alpha = torch.tensor(0.25).to(device=targets.device)
        # gamma = torch.tensor(2.0).to(device=predictions.device)

        # pred_pos = F.logsigmoid(predictions[:, 1])
        # pred_neg = F.logsigmoid(predictions[:, 0])

        # pos_term = -pred_pos.exp().pow(gamma) * targets * pred_pos
        # neg_term = -pred_neg.exp().pow(gamma) * (1 - targets) * pred_neg

        # pos_term = alpha * pos_term
        # neg_term = (1 - alpha) * neg_term

        # if presence_weight is not None:
        #     pos_term = pos_term * presence_weight

        # focal_loss = (pos_term + neg_term).view(-1)
        # presence_loss = focal_loss.mean()
        logger.debug(msg="- Presence loss: %s." % presence_loss)

        # Compute L1 loss.
        l1_loss = F.l1_loss(input=sorted_pred_points, target=sorted_true_points, reduction="none")
        l1_loss = l1_loss.sum(dim=-1)
        l1_loss = l1_loss[obj_idx].sum() / num_points
        logger.debug(msg="- L1 loss: %s." % l1_loss)

        # Compute the total loss.
        final_l1_loss = self.__l1_weight * l1_loss
        final_presence_loss = self.__presence_weight * presence_loss
        loss = final_l1_loss + final_presence_loss
        logger.debug(msg="- Total loss: %s." % loss)
        logger.info(msg="Returning the loss value.")

        metrics = {
            "loss": loss,
            "l1_loss": final_l1_loss,
            "presence_loss": final_presence_loss,
            "ap_50": ap_50,
            "ap_75": ap_75,
            "ap_90": ap_90
        }

        return metrics


    def save_checkpoint(self, model, optimizer, scheduler, dir_path, step):
        """
        Save the model and optimizer state.

        Args:
            model (PromptableDeTRTrainer): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to save.
            dir_path (str): The path to the directory where the checkpoint will be saved.
            step (int): The current training step.
        """
        logger.info(msg="Saving the model and optimizer state.")
        
        # Define the checkpoint path.
        ckpt_fp = os.path.join(dir_path, "step-%d.ckpt" % step)

        # Save the model and optimizer state.
        torch.save(obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step
        }, f=ckpt_fp)
