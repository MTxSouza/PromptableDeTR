"""
This module contains the main class used to train all models. It defines the training loop and the 
evaluation loop.
"""
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from data.loader import PromptableDeTRDataLoader
from utils.logger import Tensorboard


# Functions.
def convert_ckpt_into_pt(ckpt_file_path: str, output_dir: str):
    """
    Convert a checkpoint file path into a .pt file that
    can be used to instantiate the entire model using the
    `PromptableDeTRModel` class.

    Args:
        ckpt_file_path (str): Path to the ckpt file.
        output_dir (str): Directory to save the .pt file.
    """
    assert os.path.exists(ckpt_file_path), "The checkpoints does not exist."
    assert os.path.isdir(output_dir), "Directory path is invalid."

    # Load checkpoint.
    ckpt = torch.load(f=ckpt_file_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "configs" in ckpt

    # Define new state dict.
    model_pt = {"weights": ckpt.pop("model_state_dict"), "configs": ckpt.pop("configs")}
    pt_file_path = os.path.join(output_dir, "promptabledetr.pt")
    torch.save(obj=model_pt, f=pt_file_path)

# Classes.
class Trainer:

    # Static methods.
    @staticmethod
    def project_lr_curve(curve_limit, warmup_steps, frozen_steps, lr_factor):

        def lr_curve(step):
            if step < warmup_steps:
                return lr_factor + (1.0 - lr_factor) * step / warmup_steps
            elif step <= frozen_steps:
                return 1.0
            elif step >= curve_limit:
                return lr_factor
            decay_progress = (step - frozen_steps) / (curve_limit - frozen_steps)
            decay_factor = (1 - decay_progress) ** 2
            return lr_factor + (1.0 - lr_factor) * decay_factor

        return lr_curve

    # Special methods.
    def __init__(
            self, 
            trainer_name, 
            model, 
            optimizer, 
            train_dataset, 
            valid_dataset, 
            max_queries,
            max_lr,
            min_lr,
            warmup_steps,
            frozen_steps,
            log_interval,
            eval_interval,
            max_iter,
            curve_limit, 
            overfit_threshold,
            overfit_patience,
            exp_dir,
            log_grads,
            device
        ):
        """
        Initializes the Trainer class used to train models.

        Args:
            trainer_name (str): The name of the trainer.
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            train_dataset (PromptableDeTRDataLoader): The training dataset.
            valid_dataset (PromptableDeTRDataLoader): The validation dataset.
            max_queries (int): The maximum number of object queries.
            max_lr (float): Maximum learning rate for the Joiner optimizer.
            min_lr (float): Minimum learning rate for the Joiner optimizer.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            frozen_steps (int): The number of steps to freeze the learning rate.
            log_interval (int): Period to log the training status.
            eval_interval (int): Period to evaluate the model.
            max_iter (int): The maximum number of iterations.
            curve_limit (int): The maximum number of iterations for the LR curve.
            overfit_threshold (float): The threshold to consider overfitting.
            overfit_patience (int): The number of iterations to wait before considering overfitting.
            exp_dir (str): The directory to save the experiment.
            log_grads (bool): Whether to log the gradients.
            device (torch.device): The device to use for training.
        """
        # Attributes.
        self.trainer_name = trainer_name
        self.model = model
        self.optimizer = optimizer
        self.optimizers = None
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.max_queries = max_queries
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_factor = min_lr / max_lr
        self.warmup_steps = warmup_steps
        self.frozen_steps = frozen_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.max_iter = max_iter
        self.curve_limit = curve_limit
        self.overfit_threshold = overfit_threshold
        self.overfit_patience = overfit_patience
        self.exp_dir = exp_dir
        self.log_grads = log_grads
        self.tokenizer = None
    
        self.device = device

        # Training attributes.
        self.__total_samples = 8
        self.__current_iter = 1
        self.__best_acc = 0.0
        self.__is_overfitting = False
        self.__overfit_counter = 0
        self.__losses = []
        self.__bbox_losses = []
        self.__l1_losses = []
        self.__presence_losses = []
        self.__contrastive_losses = []
        self.__giou_50_accuracies = []
        self.__giou_75_accuracies = []
        self.__giou_90_accuracies = []
        self.__f1_50_accuracies = []
        self.__f1_75_accuracies = []
        self.__f1_90_accuracies = []

        self.__tensorboard = Tensorboard(log_dir=exp_dir)

    # Private methods.
    def __compile_model(self):
        """
        It initializes the model and the optimizer and compiles the model 
        for more efficient training.
        """
        # Project optimizer and LR curve.
        lr_curve = self.project_lr_curve(
            curve_limit=self.curve_limit,
            warmup_steps=self.warmup_steps,
            frozen_steps=self.frozen_steps,
            lr_factor=self.lr_factor
        )

        # Define the optimizers.
        self.optimizers = {
            "encoder": {
                "add_scheduler": False,
                "opt": self.optimizer(params=\
                                    list(self.model.image_encoder.parameters()) + \
                                    list(self.model.text_encoder.parameters()),
                                    lr=self.min_lr
                                    )
            },
            "head": {
                "add_scheduler": True,
                "opt": self.optimizer(params=\
                                    list(self.model.joiner.parameters()) + \
                                    list(self.model.bbox_predictor.parameters()) + \
                                    list(self.model.presence_predictor.parameters()),
                                    lr=self.min_lr
                                    )
            }
        }
        # Add LR Scheduler.
        for _, opt_data in self.optimizers.items():
            if opt_data["add_scheduler"]:
                opt_data["scheduler"] = optim.lr_scheduler.LambdaLR(optimizer=opt_data["opt"], lr_lambda=lr_curve)
            else:
                opt_data["scheduler"] = None

        # Move the model to the device.
        self.model.to(device=self.device)

    def __optimize_model(self, loss):
        """
        Run the backpropagation algorithm and updates the
        model parameters.

        Args:
            loss (torch.Tensor): Final loss of the model.
        """
        # Reset gradients.
        for opt_data in self.optimizers.values():
            opt_data["opt"].zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.2) # HARDCODED.

        # Update model.
        for name, opt_data in self.optimizers.items():
            opt_data["opt"].step()
            if opt_data["scheduler"] is not None:
                opt_data["scheduler"].step()

                # Store current LR
                self.__tensorboard.add_current_lr(lr=opt_data["scheduler"].get_last_lr()[0], name=name, step=self.__current_iter)
            else:
                self.__tensorboard.add_current_lr(lr=opt_data["opt"].param_groups[0]["lr"], name=name, step=self.__current_iter)

    def __compute_current_training_metrics(self):
        """
        It computes the current training metrics.

        Returns:
            dict: A dictionary containing the current training metrics.
        """
        # Check the length of the metrics.
        if len(self.__losses) > self.log_interval:
            self.__losses = self.__losses[-self.log_interval:]
            self.__bbox_losses = self.__bbox_losses[-self.log_interval:]
            self.__l1_losses = self.__l1_losses[-self.log_interval:]
            self.__presence_losses = self.__presence_losses[-self.log_interval:]
            self.__contrastive_losses = self.__contrastive_losses[-self.log_interval:]
            self.__giou_50_accuracies = self.__giou_50_accuracies[-self.log_interval:]
            self.__giou_75_accuracies = self.__giou_75_accuracies[-self.log_interval:]
            self.__giou_90_accuracies = self.__giou_90_accuracies[-self.log_interval:]
            self.__f1_50_accuracies = self.__f1_50_accuracies[-self.log_interval:]
            self.__f1_75_accuracies = self.__f1_75_accuracies[-self.log_interval:]
            self.__f1_90_accuracies = self.__f1_90_accuracies[-self.log_interval:]

        # Compute mean loss.
        mean_loss = sum(self.__losses) / len(self.__losses)
        mean_bbox_loss = sum(self.__bbox_losses) / len(self.__bbox_losses)
        mean_l1_loss = sum(self.__l1_losses) / len(self.__l1_losses)
        mean_presence_loss = sum(self.__presence_losses) / len(self.__presence_losses)
        mean_contrastive_loss = sum(self.__contrastive_losses) / len(self.__contrastive_losses)

        # Compute mean accuracy.
        mean_giou_50_acc = sum(self.__giou_50_accuracies) / len(self.__giou_50_accuracies)
        mean_giou_75_acc = sum(self.__giou_75_accuracies) / len(self.__giou_75_accuracies)
        mean_giou_90_acc = sum(self.__giou_90_accuracies) / len(self.__giou_90_accuracies)
        mean_f1_50_acc = sum(self.__f1_50_accuracies) / len(self.__f1_50_accuracies)
        mean_f1_75_acc = sum(self.__f1_75_accuracies) / len(self.__f1_75_accuracies)
        mean_f1_90_acc = sum(self.__f1_90_accuracies) / len(self.__f1_90_accuracies)

        metrics = {
            "mean_loss": mean_loss,
            "mean_bbox_loss": mean_bbox_loss,
            "mean_l1_loss": mean_l1_loss,
            "mean_presence_loss": mean_presence_loss,
            "mean_contrastive_loss": mean_contrastive_loss,
            "mean_giou_50_acc": mean_giou_50_acc,
            "mean_giou_75_acc": mean_giou_75_acc,
            "mean_giou_90_acc": mean_giou_90_acc,
            "mean_f1_50_acc": mean_f1_50_acc,
            "mean_f1_75_acc": mean_f1_75_acc,
            "mean_f1_90_acc": mean_f1_90_acc
        }

        return metrics

    def __run_forward(self, model, batch, is_training = True):
        """
        It runs the forward pass of the model.

        Args:
            model (nn.Module): The model to run the forward pass.
            batch (List[Sample]): The batch of data.
            is_training (bool): Whether the model is training or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The images, labels, logits and the joiner, text and image embeddings.
        """
        # Convert the batch into tensors.
        images, captions, mask, extra_data = PromptableDeTRDataLoader.convert_batch_into_tensor(batch=batch, max_queries=self.max_queries)
        images = images.to(device=self.device)
        captions = captions.to(device=self.device)
        mask = mask.to(device=self.device)

        def run_forward(model, images, captions, extra_data):
            boxes = extra_data["boxes"].to(device=self.device)
            logits, joiner_emb, txt_emb, img_emb = model(images, captions, mask) # Input: Image, caption and the mask to occlude padded tokens.
            return logits, boxes, joiner_emb, txt_emb, img_emb # Output: Pred boxes and presences and the true boxes and presences.

        # Run the forward pass.
        if not is_training:
            model.eval()
            with torch.no_grad():
                logits, labels, joiner_emb, txt_emb, img_emb = run_forward(model, images, captions, extra_data)
        else:
            model.train()
            logits, labels, joiner_emb, txt_emb, img_emb = run_forward(model, images, captions, extra_data)

        return images, captions, labels, logits, joiner_emb, txt_emb, img_emb

    def __get_sample(self, images, captions, y, logits):
        """
        It gets a sample from the logits and the true captions to be visualized further.

        Args:
            images (torch.Tensor): The input images from the model.
            captions (torch.Tensor): The true captions.
            y (torch.Tensor): The true captions.
            logits (torch.Tensor): The logits from the model.

        Returns:
            Tuple[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]: The image, input caption, true objects, and predicted objects.
        """
        # Retrieve samples.
        image = images.permute(1, 2, 0).detach().cpu().numpy()
        caption = captions.detach().cpu().numpy()
        logits_objs = logits.cpu()
        y_objs = y.detach().cpu().numpy()

        # Decode the caption.
        caption = self.valid_dataset.get_tokenizer().decode(caption.tolist())
        caption = "" if not caption else caption[0]

        # Filter the objects.
        logits_presence = logits_objs[:, 4:].softmax(dim=1).detach().cpu().numpy()
        logits_boxes = logits_objs[:, :4].detach().cpu().numpy()

        y_objs = y_objs[y_objs[:, 4] == 1][:, :4]

        return image, caption, y_objs, logits_presence, logits_boxes

    def __save_model(self, giou_50, giou_75, giou_90, l1_50, l1_75, l1_90):
        """
        It saves the model if the validation loss is better than the previous one.

        Args:
            valid_loss (float): The validation loss.
        """
        # Save the model weights.
        metrics = self.__compute_current_training_metrics()

        curr_train_l1_acc = (metrics["mean_giou_50_acc"] + metrics["mean_giou_75_acc"] + metrics["mean_giou_90_acc"]) / 3
        curr_train_giou_acc = (metrics["mean_f1_50_acc"] + metrics["mean_f1_75_acc"] + metrics["mean_f1_90_acc"]) / 3
        curr_train_acc = (curr_train_l1_acc + curr_train_giou_acc) / 2

        curr_valid_l1_acc = (giou_50 + giou_75 + giou_90) / 3
        curr_valid_giou_acc = (l1_50 + l1_75 + l1_90) / 3
        curr_valid_acc = (curr_valid_l1_acc + curr_valid_giou_acc) / 2

        if curr_valid_acc > self.__best_acc:
            self.__best_acc = curr_valid_acc
            self.__overfit_counter = 0 # Reset the overfit counter.

        # Check if it is overfitting.
        elif curr_valid_acc - curr_train_acc > self.overfit_threshold:
            self.__overfit_counter += 1
            if self.__overfit_counter >= self.overfit_patience:
                print("Overfitting detected. Stopping training.")
                self.__is_overfitting = True

        self.model.save_checkpoint(
            model=self.model,
            optimizers=self.optimizers,
            dir_path=self.exp_dir,
            step=self.__current_iter
        )
        if not self.__overfit_counter:
            self.model.save_checkpoint(
                model=self.model,
                optimizers=self.optimizers,
                dir_path=self.exp_dir,
                step=self.__current_iter,
                is_best=True
            )

    def __filter_samples_by_conf(self, sample, threshold):
        """
        It filters the samples by the confidence threshold.

        Args:
            sample (Tuple[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]): The sample containing the image, caption, true objects, and predicted objects.
            threshold (float): Presence threshold to filter the predicted objects.

        Returns:
            Tuple[np.ndarray, str, np.ndarray, np.ndarray]: The filtered samples.
        """
        img, cap, y, confs, boxes = sample
        new_logits = None
        for conf, bbox in zip(confs, boxes):
            if len(conf.shape) == 1:
                conf = conf[None, :]
                bbox = bbox[None, :]
            filt_boxes = bbox[conf[:, 1] >= threshold]
            if new_logits is None:
                new_logits = filt_boxes
            else:
                new_logits = np.concatenate((new_logits, filt_boxes), axis=0)
        sample = (img, cap, y, new_logits)
        return sample

    def __get_batch_metrics(self, batch):
        """
        Extract some stats about the current batch.

        Args:
            batch (List[Sample]): The batch of data.

        Returns:
            dict: A dictionary containing the batch metrics.
        """
        num_samples = len(batch)
        max_caption_length = max(sample.caption_tokens.size(0) for sample in batch)
        num_objects = sum(len(sample.boxes) for sample in batch)
        num_of_empty_captions = sum(1 for sample in batch if len(sample.caption) == 0)
        num_of_no_objects = sum(1 for sample in batch if len(sample.boxes) == 0)

        return {
            "num_samples": num_samples,
            "num_objects": num_objects,
            "num_of_empty_captions": num_of_empty_captions,
            "num_of_no_objects": num_of_no_objects,
            "max_caption_length": max_caption_length
        }

    def __main_loop(self):
        """
        It iterates over the training and validation datasets and trains the model.
        """
        print("Starting the training loop...")
        print("=" * 100)
        while self.__current_iter < self.max_iter and not self.__is_overfitting:

            # Loop over the training data loader.
            training_batch = next(self.train_dataset)
            batch_metrics = self.__get_batch_metrics(batch=training_batch)

            # Run the forward pass.
            images, captions, y, logits, joiner_emb, txt_emb, img_emb = self.__run_forward(model=self.model, batch=training_batch, is_training=True)

            # Compute the loss.
            metrics = self.model.compute_loss_and_accuracy(logits=logits, labels=y, fusion_emb=joiner_emb, txt_emb=txt_emb, img_emb=img_emb)
            loss = metrics["loss"]
            final_l1_loss = metrics["bbox_loss"].cpu().detach().numpy().item()
            final_bbox_loss = metrics["giou_loss"].cpu().detach().numpy().item()
            final_presence_loss = metrics["presence_loss"].cpu().detach().numpy().item()
            final_contrastive_loss = metrics["contrastive_loss"].cpu().detach().numpy().item()
            f1_50 = metrics["f1_50"].cpu().detach().numpy().item()
            f1_75 = metrics["f1_75"].cpu().detach().numpy().item()
            f1_90 = metrics["f1_90"].cpu().detach().numpy().item()
            giou_50 = metrics["giou_50"].cpu().detach().numpy().item()
            giou_75 = metrics["giou_75"].cpu().detach().numpy().item()
            giou_90 = metrics["giou_90"].cpu().detach().numpy().item()

            # Backward pass.
            self.__optimize_model(loss=loss)

            # Store accuracy.
            self.__giou_50_accuracies.append(giou_50)
            self.__giou_75_accuracies.append(giou_75)
            self.__giou_90_accuracies.append(giou_90)
            self.__f1_50_accuracies.append(f1_50)
            self.__f1_75_accuracies.append(f1_75)
            self.__f1_90_accuracies.append(f1_90)

            # Store the losses.
            self.__losses.append(loss.cpu().detach().numpy().item())
            self.__bbox_losses.append(final_bbox_loss)
            self.__l1_losses.append(final_l1_loss)
            self.__presence_losses.append(final_presence_loss)
            self.__contrastive_losses.append(final_contrastive_loss)

            # Check if it is time to log the loss.
            print("Iteration [%d/%d] : Num. Samples: %d - Max. Caption length: %d - Num. Objects: %d - Num. Empty Captions: %d - Num. No Objects: %d" % (
                self.__current_iter,
                self.max_iter, batch_metrics["num_samples"],
                batch_metrics["max_caption_length"],
                batch_metrics["num_objects"],
                batch_metrics["num_of_empty_captions"],
                batch_metrics["num_of_no_objects"]
            ))
            if self.__current_iter % self.log_interval == 0:
                metrics = self.__compute_current_training_metrics()
                current_loss = metrics["mean_loss"]
                current_bbox_loss = metrics["mean_bbox_loss"]
                current_l1_loss = metrics["mean_l1_loss"]
                current_presence_loss = metrics["mean_presence_loss"]
                current_contrastive_loss = metrics["mean_contrastive_loss"]
                current_giou_50_acc = metrics["mean_giou_50_acc"]
                current_giou_75_acc = metrics["mean_giou_75_acc"]
                current_giou_90_acc = metrics["mean_giou_90_acc"]
                current_f1_50_acc = metrics["mean_f1_50_acc"]
                current_f1_75_acc = metrics["mean_f1_75_acc"]
                current_f1_90_acc = metrics["mean_f1_90_acc"]

                # Log the training loss.
                self.__tensorboard.add_train_losses(
                    loss=current_loss, 
                    l1_loss=current_l1_loss,
                    bbox_loss=current_bbox_loss, 
                    presence_loss=current_presence_loss, 
                    contrastive_loss=current_contrastive_loss, 
                    step=self.__current_iter
                )

                # Log the training accuracy.
                self.__tensorboard.add_train_giou_accuracy(acc=current_giou_50_acc, step=self.__current_iter, th=0.50)
                self.__tensorboard.add_train_giou_accuracy(acc=current_giou_75_acc, step=self.__current_iter, th=0.75)
                self.__tensorboard.add_train_giou_accuracy(acc=current_giou_90_acc, step=self.__current_iter, th=0.90)
                self.__tensorboard.add_train_f1_accuracy(acc=current_f1_50_acc, step=self.__current_iter, th=0.50)
                self.__tensorboard.add_train_f1_accuracy(acc=current_f1_75_acc, step=self.__current_iter, th=0.75)
                self.__tensorboard.add_train_f1_accuracy(acc=current_f1_90_acc, step=self.__current_iter, th=0.90)

                # Log gradients.
                if self.log_grads:
                    self.__tensorboard.add_grad(model=self.model, step=self.__current_iter)

                print("Loss: %.4f - GIoU Loss: %.4f - L1 Loss: %.4f - Presence Loss: %.4f - Contrastive Loss: %.4f" % (current_loss, current_bbox_loss, current_l1_loss, current_presence_loss, current_contrastive_loss))
                print("GIoU@0.50: %.4f - GIoU@0.75: %.4f - GIoU@0.90: %.4f | F1@0.50: %.4f - F1@0.75: %.4f - F1@0.90: %.4f" % (current_giou_50_acc, current_giou_75_acc, current_giou_90_acc, current_f1_50_acc, current_f1_75_acc, current_f1_90_acc))
                print("-" * 100)

            # Check if it is time to validate the model.
            if self.__current_iter % self.eval_interval == 0:
                print("=" * 100)
                print("Validating the model...")

                # Loop over the validation data loader.
                total_loss = 0.0
                total_bbox_loss = 0.0
                total_l1_loss = 0.0
                total_presence_loss = 0.0
                total_contrastive_loss = 0.0
                total_giou_50_acc = 0.0
                total_giou_75_acc = 0.0
                total_giou_90_acc = 0.0
                total_f1_50_acc = 0.0
                total_f1_75_acc = 0.0
                total_f1_90_acc = 0.0
                samples = []
                init_time = time.time()
                for _ in tqdm(range(len(self.valid_dataset)), unit="batches (randomly sampled)"):
                    validation_batch = next(self.valid_dataset)

                    # Run the forward pass.
                    images, captions, y, logits, joiner_emb, txt_emb, img_emb = self.__run_forward(model=self.model, batch=validation_batch, is_training=False)

                    # Compute the loss.
                    metrics = self.model.compute_loss_and_accuracy(logits=logits, labels=y, fusion_emb=joiner_emb, txt_emb=txt_emb, img_emb=img_emb)
                    total_loss += metrics["loss"].cpu().detach().numpy().item()
                    total_l1_loss += metrics["bbox_loss"].cpu().detach().numpy().item()
                    total_bbox_loss += metrics["giou_loss"].cpu().detach().numpy().item()
                    total_presence_loss += metrics["presence_loss"].cpu().detach().numpy().item()
                    total_contrastive_loss += metrics["contrastive_loss"].cpu().detach().numpy().item()
                    total_f1_50_acc += metrics["f1_50"].cpu().detach().numpy().item()
                    total_f1_75_acc += metrics["f1_75"].cpu().detach().numpy().item()
                    total_f1_90_acc += metrics["f1_90"].cpu().detach().numpy().item()
                    total_giou_50_acc += metrics["giou_50"].cpu().detach().numpy().item()
                    total_giou_75_acc += metrics["giou_75"].cpu().detach().numpy().item()
                    total_giou_90_acc += metrics["giou_90"].cpu().detach().numpy().item()

                    # Get a random sample.
                    small_samples = [self.__get_sample(images=images[idx_batch], captions=captions[idx_batch], y=y[idx_batch], logits=logits[idx_batch]) for idx_batch in range(images.size(0))]
                    for sample in small_samples:
                        if self.__total_samples > len(samples):
                            samples.append(sample)
                        else:
                            i = random.randint(0, self.__total_samples - 1)
                            j = random.randint(0, self.__total_samples - 1)
                            if i > j:
                                samples[i] = sample
                    del small_samples

                # Compute final accuracy.
                samples_giou_50_acc = [self.__filter_samples_by_conf(sample=sample, threshold=0.50) for sample in samples]
                samples_giou_75_acc = [self.__filter_samples_by_conf(sample=sample, threshold=0.75) for sample in samples]
                samples_giou_90_acc = [self.__filter_samples_by_conf(sample=sample, threshold=0.90) for sample in samples]
                del samples

                total_loss /= len(self.valid_dataset)
                total_l1_loss /= len(self.valid_dataset)
                total_bbox_loss /= len(self.valid_dataset)
                total_presence_loss /= len(self.valid_dataset)
                total_contrastive_loss /= len(self.valid_dataset)
                total_giou_50_acc /= len(self.valid_dataset)
                total_giou_75_acc /= len(self.valid_dataset)
                total_giou_90_acc /= len(self.valid_dataset)
                total_f1_50_acc /= len(self.valid_dataset)
                total_f1_75_acc /= len(self.valid_dataset)
                total_f1_90_acc /= len(self.valid_dataset)
                end_time = (time.time() - init_time) / 60.0

                # Save the model weights.
                self.__save_model(giou_50=total_giou_50_acc, giou_75=total_giou_75_acc, giou_90=total_giou_90_acc, l1_50=total_f1_50_acc, l1_75=total_f1_75_acc, l1_90=total_f1_90_acc)

                # Log the valid losses.
                self.__tensorboard.add_valid_losses(
                    loss=total_loss, 
                    l1_loss=total_l1_loss,
                    bbox_loss=total_bbox_loss, 
                    presence_loss=total_presence_loss, 
                    contrastive_loss=total_contrastive_loss, 
                    step=self.__current_iter
                )

                # Log the valid accuracy.
                self.__tensorboard.add_valid_giou_accuracy(acc=total_giou_50_acc, step=self.__current_iter, th=0.50)
                self.__tensorboard.add_valid_giou_accuracy(acc=total_giou_75_acc, step=self.__current_iter, th=0.75)
                self.__tensorboard.add_valid_giou_accuracy(acc=total_giou_90_acc, step=self.__current_iter, th=0.90)
                self.__tensorboard.add_valid_f1_accuracy(acc=total_f1_50_acc, step=self.__current_iter, th=0.50)
                self.__tensorboard.add_valid_f1_accuracy(acc=total_f1_75_acc, step=self.__current_iter, th=0.75)
                self.__tensorboard.add_valid_f1_accuracy(acc=total_f1_90_acc, step=self.__current_iter, th=0.90)

                # Display the samples on Tensorboard.
                self.__tensorboard.add_image(samples=samples_giou_50_acc, step=self.__current_iter, giou_th=0.50)
                self.__tensorboard.add_image(samples=samples_giou_75_acc, step=self.__current_iter, giou_th=0.75)
                self.__tensorboard.add_image(samples=samples_giou_90_acc, step=self.__current_iter, giou_th=0.90)

                print("Validation time: %.2f minutes" % end_time)
                print("Overfit counter: %d" % self.__overfit_counter)
                print("Validation loss: %.4f - GIoU Loss: %.4f - L1 Loss: %.4f - Presence Loss: %.4f - Contrastive Loss: %.4f" % (total_loss, total_bbox_loss, total_l1_loss, total_presence_loss, total_contrastive_loss))
                print("GIoU@0.5: %.4f - GIoU@0.75: %.4f - GIoU@0.90: %.4f | F1@0.5: %.4f - F1@0.75: %.4f - F1@0.90: %.4f" % (total_giou_50_acc, total_giou_75_acc, total_giou_90_acc, total_f1_50_acc, total_f1_75_acc, total_f1_90_acc))
                print("=" * 100)

            # Update the iteration.
            self.__current_iter += 1
            if self.__current_iter >= self.max_iter or self.__is_overfitting:
                break

    # Methods.
    def train(self, checkpoint_path=None):
        """
        It trains the model using the training and validation datasets.

        Args:
            checkpoint_path (str): The path to the checkpoint file to resume training. (Default: None)
        """
        print("=" * 100)
        print("🚀 Starting PromptableDeTR - %s training" % self.trainer_name)
        print("=" * 100)

        try:
            # Compile the model.
            self.__compile_model()
            if checkpoint_path is not None:
                self.resume_training(checkpoint_path=checkpoint_path)
            # Run the main loop.
            self.__main_loop()
        except KeyboardInterrupt:
            print("=" * 100)
            print("🛑 Training interrupted by the user.")
            print("=" * 100)
        print("🚀 Training finished.")

    def resume_training(self, checkpoint_path):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint file.
        """
        print("Resuming training from checkpoint from path: %s" % checkpoint_path)

        # Load the checkpoint.
        checkpoint = torch.load(f=checkpoint_path)

        # Load the model state.
        self.model.load_state_dict(checkpoint["model_state_dict"])
        for name, opt_data in checkpoint["optimizer_state_dict"].items():
            if name in self.optimizers:
                self.optimizers[name]["opt"].load_state_dict(opt_data["opt"])
                if self.optimizers[name]["scheduler"] is not None:
                    self.optimizers[name]["scheduler"].load_state_dict(opt_data["scheduler"])

        # Restore the training step.
        self.__current_iter = checkpoint["step"]
