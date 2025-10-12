"""
This module contains the Detector model class used to predict bounding boxes and presence of objects 
in the image.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from logger import Logger
from models.base import BasePromptableDeTR
from models.matcher import HungarianMatcher
from utils.data import generalized_iou
from utils.metrics import f1_accuracy_open_vocab, iou_accuracy

# Logger.
logger = Logger(name="model")


# Classes.
class PromptableDeTR(BasePromptableDeTR):

    # Special methods.
    def __init__(
            self, 
            image_size = 320, 
            num_queries = 10,
            vocab_size = 30522, 
            emb_dim = 512, 
            num_heads = 8, 
            ff_dim = 1024, 
            emb_dropout_rate = 0.1, 
            num_joiner_layers = 3
        ):
        """
        Initializes the Detector class used to predict bounding boxes and presence 
        of objects in the image.

        Args:
            image_size (int): The size of the input image. (Default: 320)
            vocab_size (int): The size of the vocabulary. (Default: 30522)
            emb_dim (int): The projection dimension of the image and text embeddings. (Default: 512)
            num_heads (int): The number of attention heads. (Default: 8)
            ff_dim (int): The dimension of the feed-forward network. (Default: 1024)
            emb_dropout_rate (float): The dropout rate for the embeddings. (Default: 0.1)
            num_joiner_layers (int): The number of joiner layers in the model. (Default: 3)
        """
        super().__init__(
            image_size=image_size, 
            num_queries=num_queries,
            vocab_size=vocab_size, 
            emb_dim=emb_dim, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            emb_dropout_rate=emb_dropout_rate, 
            num_joiner_layers=num_joiner_layers
        )

        # Layers.
        self.bbox_predictor = nn.Linear(in_features=emb_dim, out_features=4)
        self.presence_predictor = nn.Linear(in_features=emb_dim, out_features=2)

        # Initialize weights.
        self.__initialize_weights()

        self.params = {
            "image_size": image_size,
            "num_queries": num_queries,
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "emb_dropout_rate": emb_dropout_rate,
            "num_joiner_layers": num_joiner_layers
        }

    # Private methods.
    def __initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.bbox_predictor.apply(init_weights)
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
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The predicted bounding boxes and presence and the text and image embeddings.
        """
        logger.info(msg="Calling `Detector` forward method.")
        logger.debug(msg="- Image shape: %s" % (image.shape,))
        logger.debug(msg="- Prompt shape: %s" % (prompt.shape,))

        # Compute joint embeddings.
        logger.debug(msg="- Calling `BasePromptableDeTR` forward method.")
        joint, text_emb, img_emb = super().forward(image=image, prompt=prompt, prompt_mask=prompt_mask)
        logger.debug(msg="- Result of the `BasePromptableDeTR` forward method: %s." % (joint.shape,))

        logger.debug(msg="- Calling `nn.Linear` blocks to the tensor %s." % (joint.shape,))
        bbox = self.bbox_predictor(joint)
        bbox = F.sigmoid(input=bbox)
        presence = self.presence_predictor(joint)
        logger.debug(msg="- Result of the `nn.Linear` block: %s and %s." % (bbox.shape, presence.shape))

        # Concatenate the predictions.
        logger.debug(msg="- Concatenating the predictions.")
        outputs = torch.cat(tensors=(bbox, presence), dim=-1)
        logger.debug(msg="- Result of the concatenation: %s." % (outputs.shape,))

        return outputs, joint, text_emb, img_emb

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

    def load_weights(self, model_weights):
        """
        Load the weights of the entire model.

        Args:
            model_weights (str): Path to the model weights.
        """
        logger.info(msg="Loading the model weights.")
        logger.debug(msg="- Model weights: %s" % model_weights)

        # Load weights.
        state_dict = torch.load(f=model_weights, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict=state_dict, strict=True)

class PromptableDeTRTrainer(PromptableDeTR):
    """
    A subclass of PromptableDeTR that is used for training the model.
    It inherits from PromptableDeTR and adds the functionality to compute
    the loss and accuracy of the model.
    """

    # Special methods.
    def __init__(
            self,
            use_focal_loss=False,
            presence_loss_weight=1.0,
            giou_loss_weight=1.0,
            l1_loss_weight=1.0,
            contrastive_loss_weight=1.0,
            alpha=0.25,
            hm_presence_weight=5.0,
            hm_giou_weight=2.0,
            hm_l1_weight=3.0,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        # Loss weights.
        self.__use_focal_loss = use_focal_loss
        self.__presence_weight = presence_loss_weight
        self.__giou_weight = giou_loss_weight
        self.__l1_weight = l1_loss_weight
        self.__contrastive_weight = contrastive_loss_weight
        self.__alpha = alpha

        # Matcher.
        self.matcher = HungarianMatcher(
            presence_loss_weight=hm_presence_weight,
            l1_loss_weight=hm_l1_weight,
            giou_loss_weight=hm_giou_weight
        )

    # Methods.
    def compute_loss_and_accuracy(self, logits, labels, fusion_emb, txt_emb, img_emb, txt_mask = None):
        """
        Compute the loss needed to train the detector model and
        it also computes the accuracy of the model.

        Args:
            logits torch.Tensor): The predicted tensor.
            labels (torch.Tensor): The true tensor.
            fusion_emb (torch.Tensor): The joint embeddings.
            txt_emb (torch.Tensor): The text embeddings.
            img_emb (torch.Tensor): The image embeddings.
            txt_mask (torch.Tensor): The text mask tensor. (Default: None)

        Returns:
            dict: A dictionary containing the loss and accuracy values.
        """
        logger.info(msg="Computing the detector loss.")

        assert self.matcher is not None, "Matcher is not defined."

        # Check if the model outputs has nan values.
        if torch.isnan(logits).any():
            raise ValueError("The model outputs contains NaN values.")

        # Sort the logits and labels.
        pred_presence = logits[:, :, 4:]
        pred_boxes = logits[:, :, :4]
        true_presence = labels[:, :, 4].long()
        true_boxes = labels[:, :, :4]
        logger.debug(msg="- Predicted presence shape: %s." % (pred_presence.shape,))
        logger.debug(msg="- Predicted boxes shape: %s." % (pred_boxes.shape,))
        logger.debug(msg="- True presence shape: %s." % (true_presence.shape,))
        logger.debug(msg="- True boxes shape: %s." % (true_boxes.shape,))
        batch_idx, src_idx, tgt_idx = self.matcher(predict_scores=pred_presence, predict_boxes=pred_boxes, scores=true_presence, boxes=true_boxes)
        logger.debug(msg="- Batch index shape: %s." % (batch_idx.shape,))
        logger.debug(msg="- Source index shape: %s." % (src_idx.shape,))
        logger.debug(msg="- Target index shape: %s." % (tgt_idx.shape,))

        # Prepare new targets.
        new_true_presence = torch.zeros_like(true_presence).long().to(true_presence.device)
        new_true_presence[batch_idx, src_idx] = 1
        new_true_boxes = torch.zeros_like(pred_boxes).to(true_boxes.device)
        new_true_boxes[batch_idx, src_idx] = true_boxes[batch_idx, tgt_idx]

        # Compute F1 accuracy.
        f1_50 = torch.tensor(f1_accuracy_open_vocab(labels=new_true_presence.view(-1, 1), logits=pred_presence.view(-1, 2), threshold=0.50))
        f1_75 = torch.tensor(f1_accuracy_open_vocab(labels=new_true_presence.view(-1, 1), logits=pred_presence.view(-1, 2), threshold=0.75))
        f1_90 = torch.tensor(f1_accuracy_open_vocab(labels=new_true_presence.view(-1, 1), logits=pred_presence.view(-1, 2), threshold=0.90))
        logger.debug(msg="- F1 Score @0.50: %s." % f1_50)
        logger.debug(msg="- F1 Score @0.75: %s." % f1_75)
        logger.debug(msg="- F1 Score @0.90: %s." % f1_90)

        # Compute number of boxes.
        obj_idx = new_true_presence == 1
        num_boxes = obj_idx.sum()
        logger.debug(msg="- Number of boxes: %s." % num_boxes)

        # Compute Global and Local Contrastive Loss.
        valid_obj = obj_idx.any(dim=1) # (B,)
        if valid_obj.any():
            txt_emb = txt_emb[valid_obj] # (num_valid_obj, Tt, D)
            img_emb = img_emb[valid_obj] # (num_valid_obj, Ti, D)
            fusion_emb = fusion_emb[valid_obj] # (num_valid_obj, n_detections, D)
            tk_emb = txt_emb.clone()

            if txt_mask is not None:
                txt_mask = txt_mask[valid_obj] # (num_valid_obj, Tt)
                denom = txt_mask.sum(dim=1, keepdim=True).clamp(min=1e-6).to(txt_emb.device)
                txt_emb = (txt_emb * (txt_mask).unsqueeze(dim=-1)).sum(dim=1) / denom
                tk_emb = tk_emb * (txt_mask).unsqueeze(dim=-1)
            else:
                txt_emb = txt_emb.mean(dim=1)
            txt_emb = F.normalize(input=txt_emb, dim=-1)
            img_emb = F.normalize(input=img_emb.mean(dim=1), dim=-1)

            global_cos_sim = txt_emb @ img_emb.t()
            targets = torch.arange(global_cos_sim.size(0), device=global_cos_sim.device)
            global_txt_loss = F.cross_entropy(global_cos_sim, targets)
            global_img_loss = F.cross_entropy(global_cos_sim.t(), targets)
            global_contrastive_loss = (global_txt_loss + global_img_loss) / 2
            global_contrastive_loss = self.__contrastive_weight * global_contrastive_loss

            tk_emb = F.normalize(input=tk_emb, dim=-1)
            fusion_emb = F.normalize(input=fusion_emb, dim=-1)
            local_cos_sim = tk_emb @ fusion_emb.transpose(-2, -1)
            local_txt_loss = 0
            local_obj_loss = 0
            for batch in range(local_cos_sim.size(0)):
                cos_sim = local_cos_sim[batch]
                txt_targets = torch.arange(cos_sim.size(0), device=cos_sim.device) % cos_sim.size(1)
                obj_targets = torch.arange(cos_sim.size(1), device=cos_sim.device) % cos_sim.size(0)
                local_txt_loss += F.cross_entropy(cos_sim, txt_targets)
                local_obj_loss += F.cross_entropy(cos_sim.t(), obj_targets)
            local_txt_loss /= local_cos_sim.size(0)
            local_obj_loss /= local_cos_sim.size(0)
            local_contrastive_loss = (local_txt_loss + local_obj_loss) / 2
            local_contrastive_loss = self.__contrastive_weight * local_contrastive_loss
        
        else:
            global_contrastive_loss = torch.tensor(0.0, device=pred_boxes.device)
            local_contrastive_loss = torch.tensor(0.0, device=pred_boxes.device)

        logger.debug(msg="- Global Contrastive loss: %s." % global_contrastive_loss)
        logger.debug(msg="- Local Contrastive loss: %s." % local_contrastive_loss)

        # Compute presence loss with focal loss.
        predictions = pred_presence.view(-1, 2)
        targets = new_true_presence.view(-1)
        if not self.__use_focal_loss:
            presence_weight = torch.tensor([1.0, self.__presence_weight], device=pred_presence.device)
            presence_loss = F.cross_entropy(input=predictions, target=targets, weight=presence_weight, reduction="mean")
        else:
            presence_loss = sigmoid_focal_loss(
                inputs=predictions,
                targets=F.one_hot(targets, num_classes=2).float(),
                alpha=self.__alpha,
                gamma=self.__presence_weight,
                reduction="mean"
            )
        logger.debug(msg="- Presence loss: %s." % presence_loss)

        # Compute bounding box loss.
        bbox_loss = F.l1_loss(input=pred_boxes, target=new_true_boxes, reduction="none")
        bbox_loss = bbox_loss.sum(dim=-1)
        if num_boxes == 0:
            bbox_loss = torch.tensor(0.0, device=pred_boxes.device)
        else:
            bbox_loss = bbox_loss[obj_idx].sum() / num_boxes
        bbox_loss = self.__l1_weight * bbox_loss
        logger.debug(msg="- Bounding box loss: %s." % bbox_loss)

        # Compute GIoU loss.
        pred_boxes = pred_boxes.view(-1, 4)
        new_true_boxes = new_true_boxes.view(-1, 4)
        diag_accuracy = torch.diag(generalized_iou(boxes1=pred_boxes, boxes2=new_true_boxes))
        diag_accuracy = (1 - diag_accuracy)
        if num_boxes == 0:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
        else:
            diag_accuracy = diag_accuracy[obj_idx.view(-1)]
            giou_loss = diag_accuracy.sum() / num_boxes
        giou_loss = self.__giou_weight * giou_loss
        logger.debug(msg="- GIoU loss: %s." % giou_loss)

        # Compute the total loss.
        loss = bbox_loss + presence_loss + giou_loss + global_contrastive_loss + local_contrastive_loss
        logger.debug(msg="- Total loss: %s." % loss)
        logger.info(msg="Returning the loss value.")

        # Compute IoU accuracy.
        giou_50 = iou_accuracy(labels=new_true_boxes, logits=pred_boxes, threshold=0.50)
        giou_75 = iou_accuracy(labels=new_true_boxes, logits=pred_boxes, threshold=0.75)
        giou_90 = iou_accuracy(labels=new_true_boxes, logits=pred_boxes, threshold=0.90)

        metrics = {
            "loss": loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss,
            "presence_loss": presence_loss,
            "global_contrastive_loss": global_contrastive_loss,
            "local_contrastive_loss": local_contrastive_loss,
            "f1_50": f1_50,
            "f1_75": f1_75,
            "f1_90": f1_90,
            "giou_50": giou_50,
            "giou_75": giou_75,
            "giou_90": giou_90
        }

        return metrics

    def save_checkpoint(self, model, optimizers, dir_path, step, is_best = False):
        """
        Save the model and optimizer state.

        Args:
            model (PromptableDeTRTrainer): The model to save.
            optimizers (dict): Dictionary containing the optimizers to save.
            dir_path (str): The path to the directory where the checkpoint will be saved.
            step (int): The current training step.
        """
        logger.info(msg="Saving the model and optimizer state.")
        
        # Define the checkpoint path.
        if is_best:
            ckpt_fp = os.path.join(dir_path, "best_model.ckpt")
        else:
            ckpt_fp = os.path.join(dir_path, "model.ckpt")

        # Get state dict of the optimizers and schedulers.
        opt_state_dict = {}
        for name, opt_data in optimizers.items():
            if opt_data["scheduler"] is not None:
                opt_state_dict[name] = {
                    "opt": opt_data["opt"].state_dict(),
                    "scheduler": opt_data["scheduler"].state_dict()
                }
            else:
                opt_state_dict[name] = {
                    "opt": opt_data["opt"].state_dict(),
                    "scheduler": None
                }

        # Save the model and optimizer state.
        torch.save(obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt_state_dict,
            "step": step,
            "configs": self.params
        }, f=ckpt_fp)
