"""
This module contains the main class used to train all models. It defines the training loop and the 
evaluation loop.
"""
import random
import time

import torch
import torch.optim as optim
from tqdm import tqdm

from data.loader import PromptableDeTRDataLoader
from utils.logger import Tensorboard


# Classes.
class Trainer:


    # Special methods.
    def __init__(
            self, 
            trainer_name, 
            model, 
            optimizer, 
            train_dataset, 
            valid_dataset, 
            max_caption_length, 
            lr,
            lr_factor,
            warmup_steps,
            frozen_steps,
            log_interval,
            eval_interval,
            max_iter, 
            overfit_threshold,
            overfit_patience,
            exp_dir,
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
            max_caption_length (int): The maximum caption length.
            lr (float): The learning rate.
            lr_factor (float): The factor to reduce the learning rate.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            frozen_steps (int): The number of steps to freeze the learning rate.
            log_interval (int): Period to log the training status.
            eval_interval (int): Period to evaluate the model.
            max_iter (int): The maximum number of iterations.
            overfit_threshold (float): The threshold to consider overfitting.
            overfit_patience (int): The number of iterations to wait before considering overfitting.
            exp_dir (str): The directory to save the experiment.
            device (torch.device): The device to use for training.
        """
        # Attributes.
        self.trainer_name = trainer_name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.max_caption_length = max_caption_length
        self.lr = lr
        self.lr_factor = lr_factor
        self.warmup_steps = warmup_steps
        self.frozen_steps = frozen_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.max_iter = max_iter
        self.overfit_threshold = overfit_threshold
        self.overfit_patience = overfit_patience
        self.exp_dir = exp_dir
        self.tokenizer = None
    
        self.device = device

        # Training attributes.
        self.__total_samples = 8
        self.__metric_window = 100
        self.__current_iter = 1
        self.__best_loss = float("inf")
        self.__is_overfitting = False
        self.__overfit_counter = 0
        self.__losses = []
        self.__bbox_losses = []
        self.__presence_losses = []
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
        # Compile the model.
        # self.model = torch.compile(model=self.model)

        # Define the optimizer.
        def lr_curve(step):
            if step < self.warmup_steps:
                return self.lr_factor + (1.0 - self.lr_factor) * step / self.warmup_steps
            elif step <= self.frozen_steps:
                return 1.0
            decay_progress = (step - self.frozen_steps) / (self.max_iter - self.frozen_steps)
            decay_factor = (1 - decay_progress) ** 2
            return self.lr_factor + (1.0 - self.lr_factor) * decay_factor
        self.optimizer = self.optimizer(params=self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lr_curve)

        # Move the model to the device.
        self.model.to(device=self.device)


    def __compute_current_training_metrics(self):
        """
        It computes the current training metrics.

        Returns:
            dict: A dictionary containing the current training metrics.
        """
        # Check the length of the metrics.
        if len(self.__losses) > self.__metric_window:
            self.__losses = self.__losses[-self.__metric_window:]
            self.__bbox_losses = self.__bbox_losses[-self.__metric_window:]
            self.__presence_losses = self.__presence_losses[-self.__metric_window:]
            self.__giou_50_accuracies = self.__giou_50_accuracies[-self.__metric_window:]
            self.__giou_75_accuracies = self.__giou_75_accuracies[-self.__metric_window:]
            self.__giou_90_accuracies = self.__giou_90_accuracies[-self.__metric_window:]
            self.__f1_50_accuracies = self.__f1_50_accuracies[-self.__metric_window:]
            self.__f1_75_accuracies = self.__f1_75_accuracies[-self.__metric_window:]
            self.__f1_90_accuracies = self.__f1_90_accuracies[-self.__metric_window:]

        # Compute mean loss.
        mean_loss = sum(self.__losses) / len(self.__losses)
        mean_bbox_loss = sum(self.__bbox_losses) / len(self.__bbox_losses)
        mean_presence_loss = sum(self.__presence_losses) / len(self.__presence_losses)

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
            "mean_presence_loss": mean_presence_loss,
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
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The images, labels and logits.
        """
        # Convert the batch into tensors.
        images, captions, mask, extra_data = PromptableDeTRDataLoader.convert_batch_into_tensor(batch=batch, max_len=self.max_caption_length)
        images = images.to(device=self.device)
        captions = captions.to(device=self.device)
        mask = mask.to(device=self.device)

        def run_forward(model, images, captions, extra_data):
            boxes = extra_data["boxes"].to(device=self.device)
            logits = model(images, captions, mask) # Input: Image, caption and the mask to occlude padded tokens.
            return logits, boxes # Output: Pred boxes and presences and the true boxes and presences.

        # Run the forward pass.
        if not is_training:
            model.eval()
            with torch.no_grad():
                logits, labels = run_forward(model, images, captions, extra_data)
        else:
            model.train()
            logits, labels = run_forward(model, images, captions, extra_data)
        
        return images, captions, labels, logits


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
        caption = caption[0]

        # Filter the objects.
        logits_presence = logits_objs[:, 2:].softmax(dim=1).detach().cpu().numpy()
        logits_boxes = logits_objs[:, :2].detach().cpu().numpy()

        y_objs = y_objs[y_objs[:, 2] == 1][:, :2]

        return image, caption, y_objs, logits_presence, logits_boxes


    def __save_model(self, valid_loss):
        """
        It saves the model if the validation loss is better than the previous one.

        Args:
            valid_loss (float): The validation loss.
        """
        # Save the model weights.
        metrics = self.__compute_current_training_metrics()
        current_train_loss = metrics["mean_loss"]
        if self.__best_loss > valid_loss:
            self.__best_loss = valid_loss
            self.__overfit_counter = 0 # Reset the overfit counter.
        
        # Check if it is overfitting.
        elif valid_loss - current_train_loss > self.overfit_threshold:
            self.__overfit_counter += 1
            if self.__overfit_counter >= self.overfit_patience:
                print("Overfitting detected. Stopping training.")
                self.__is_overfitting = True

        self.model.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dir_path=self.exp_dir,
            step=self.__current_iter
        )


    def __filter_samples_by_conf(self, sample, threshold):
        """
        It filters the samples by the confidence threshold.

        Args:
            sample (Tuple[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]): The sample containing the image, caption, true objects, and predicted objects.
            threshold (float): Presence threshold to filter the predicted objects.

        Returns:
            Tuple[np.ndarray, str, np.ndarray, list[np.ndarray]]: The filtered samples.
        """
        img, cap, y, confs, boxes = sample
        new_logits = []
        for conf, bbox in zip(confs, boxes):
            if len(conf.shape) == 1:
                conf = conf[None, :]
                bbox = bbox[None, :]
            filt_boxes = bbox[conf[:, 1] >= threshold]
            new_logits.append(filt_boxes)
        sample = (img, cap, y, new_logits)
        return sample


    def __main_loop(self):
        """
        It iterates over the training and validation datasets and trains the model.
        """
        print("Starting the training loop...")
        print("=" * 100)
        while self.__current_iter < self.max_iter and not self.__is_overfitting:

            # Loop over the training data loader.
            for training_batch in self.train_dataset:

                # Run the forward pass.
                images, captions, y, logits = self.__run_forward(model=self.model, batch=training_batch, is_training=True)

                # Compute the loss.
                metrics = self.model.compute_loss_and_accuracy(logits=logits, labels=y)
                loss = metrics["loss"]
                final_bbox_loss = metrics["bbox_loss"].cpu().detach().numpy().item()
                final_presence_loss = metrics["presence_loss"].cpu().detach().numpy().item()
                f1_50 = metrics["f1_50"].cpu().detach().numpy().item()
                f1_75 = metrics["f1_75"].cpu().detach().numpy().item()
                f1_90 = metrics["f1_90"].cpu().detach().numpy().item()
                giou_50 = metrics["giou_50"].cpu().detach().numpy().item()
                giou_75 = metrics["giou_75"].cpu().detach().numpy().item()
                giou_90 = metrics["giou_90"].cpu().detach().numpy().item()

                # Backward pass.
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=0.5)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Store current LR
                self.__tensorboard.add_current_lr(lr=self.scheduler.get_last_lr()[0], step=self.__current_iter)

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
                self.__presence_losses.append(final_presence_loss)

                # Check if it is time to log the loss.
                print("Iteration [%d/%d]" % (self.__current_iter, self.max_iter))
                if self.__current_iter % self.log_interval == 0:
                    metrics = self.__compute_current_training_metrics()
                    current_loss = metrics["mean_loss"]
                    current_bbox_loss = metrics["mean_bbox_loss"]
                    current_presence_loss = metrics["mean_presence_loss"]
                    current_giou_50_acc = metrics["mean_giou_50_acc"]
                    current_giou_75_acc = metrics["mean_giou_75_acc"]
                    current_giou_90_acc = metrics["mean_giou_90_acc"]
                    current_f1_50_acc = metrics["mean_f1_50_acc"]
                    current_f1_75_acc = metrics["mean_f1_75_acc"]
                    current_f1_90_acc = metrics["mean_f1_90_acc"]

                    # Log the training loss.
                    self.__tensorboard.add_train_losses(
                        loss=current_loss, 
                        bbox_loss=current_bbox_loss, 
                        presence_loss=current_presence_loss, 
                        step=self.__current_iter
                    )

                    # Log the training accuracy.
                    self.__tensorboard.add_train_giou_accuracy(acc=current_giou_50_acc, step=self.__current_iter, th=0.50)
                    self.__tensorboard.add_train_giou_accuracy(acc=current_giou_75_acc, step=self.__current_iter, th=0.75)
                    self.__tensorboard.add_train_giou_accuracy(acc=current_giou_90_acc, step=self.__current_iter, th=0.90)
                    self.__tensorboard.add_train_f1_accuracy(acc=current_f1_50_acc, step=self.__current_iter, th=0.50)
                    self.__tensorboard.add_train_f1_accuracy(acc=current_f1_75_acc, step=self.__current_iter, th=0.75)
                    self.__tensorboard.add_train_f1_accuracy(acc=current_f1_90_acc, step=self.__current_iter, th=0.90)

                    print("Loss: %.4f - L1 Loss: %.4f - Presence Loss: %.4f" % (current_loss, current_bbox_loss, current_presence_loss))
                    print("GIoU@0.50: %.4f - GIoU@0.75: %.4f - GIoU@0.90: %.4f | F1@0.50: %.4f - F1@0.75: %.4f - F1@0.90: %.4f" % (current_giou_50_acc, current_giou_75_acc, current_giou_90_acc, current_f1_50_acc, current_f1_75_acc, current_f1_90_acc))
                    print("-" * 100)

                # Check if it is time to validate the model.
                if self.__current_iter % self.eval_interval == 0:
                    print("=" * 100)
                    print("Validating the model...")

                    # Loop over the validation data loader.
                    total_loss = 0.0
                    total_bbox_loss = 0.0
                    total_presence_loss = 0.0
                    total_giou_50_acc = 0.0
                    total_giou_75_acc = 0.0
                    total_giou_90_acc = 0.0
                    total_f1_50_acc = 0.0
                    total_f1_75_acc = 0.0
                    total_f1_90_acc = 0.0
                    samples = []
                    init_time = time.time()
                    for validation_batch in tqdm(iterable=self.valid_dataset, desc="Validating", unit="batch"):
                        
                        # Run the forward pass.
                        images, captions, y, logits = self.__run_forward(model=self.model, batch=validation_batch, is_training=False)

                        # Compute the loss.
                        metrics = self.model.compute_loss_and_accuracy(logits=logits, labels=y)
                        total_loss += metrics["loss"]
                        total_bbox_loss += metrics["bbox_loss"]
                        total_presence_loss += metrics["presence_loss"]
                        total_f1_50_acc += metrics["f1_50"]
                        total_f1_75_acc += metrics["f1_75"]
                        total_f1_90_acc += metrics["f1_90"]
                        total_giou_50_acc += metrics["giou_50"]
                        total_giou_75_acc += metrics["giou_75"]
                        total_giou_90_acc += metrics["giou_90"]

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
                    total_bbox_loss /= len(self.valid_dataset)
                    total_presence_loss /= len(self.valid_dataset)
                    total_giou_50_acc /= len(self.valid_dataset)
                    total_giou_75_acc /= len(self.valid_dataset)
                    total_giou_90_acc /= len(self.valid_dataset)
                    total_f1_50_acc /= len(self.valid_dataset)
                    total_f1_75_acc /= len(self.valid_dataset)
                    total_f1_90_acc /= len(self.valid_dataset)
                    end_time = (time.time() - init_time) / 60.0

                    # Save the model weights.
                    self.__save_model(valid_loss=total_loss)

                    # Log the valid losses.
                    self.__tensorboard.add_valid_losses(
                        loss=total_loss, 
                        bbox_loss=total_bbox_loss, 
                        presence_loss=total_presence_loss, 
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
                    self.__tensorboard.add_image(samples=samples_giou_50_acc, step=self.__current_iter, bbox_loss_th=0.50)
                    self.__tensorboard.add_image(samples=samples_giou_75_acc, step=self.__current_iter, bbox_loss_th=0.75)
                    self.__tensorboard.add_image(samples=samples_giou_90_acc, step=self.__current_iter, bbox_loss_th=0.90)

                    print("Validation time: %.2f minutes" % end_time)
                    print("Overfit counter: %d" % self.__overfit_counter)
                    print("Validation loss: %.4f - L1 Loss: %.4f - Presence Loss: %.4f" % (total_loss, total_bbox_loss, total_presence_loss))
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
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore the training step.
        self.__current_iter = checkpoint["step"]
