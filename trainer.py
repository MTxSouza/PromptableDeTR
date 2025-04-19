"""
This module contains the main class used to train all models. It defines the training loop and the 
evaluation loop.
"""
import time

import torch

from data.loader import PromptableDeTRDataLoader


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
            lr,
            log_interval,
            eval_interval,
            max_iter, 
            overfit_threshold,
            overfit_patience,
            exp_dir
        ):
        """
        Initializes the Trainer class used to train models.

        Args:
            trainer_name (str): The name of the trainer.
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            train_dataset (PromptableDeTRDataLoader): The training dataset.
            valid_dataset (PromptableDeTRDataLoader): The validation dataset.
            lr (float): The learning rate.
            log_interval (int): Period to log the training status.
            eval_interval (int): Period to evaluate the model.
            max_iter (int): The maximum number of iterations.
            overfit_threshold (float): The threshold to consider overfitting.
            overfit_patience (int): The number of iterations to wait before considering overfitting.
            exp_dir (str): The directory to save the experiment.
        """
        # Attributes.
        self.trainer_name = trainer_name
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.max_iter = max_iter
        self.overfit_threshold = overfit_threshold
        self.overfit_patience = overfit_patience
        self.exp_dir = exp_dir
        self.tokenizer = None
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training attributes.
        self.__current_iter = 1
        self.__best_loss = float("inf")
        self.__is_overfitting = False
        self.__overfit_counter = 0
        self.__losses = []


    # Private methods.
    def __compile_model(self):
        """
        It initializes the model and the optimizer and compiles the model 
        for more efficient training.
        """
        # Compile the model.
        # self.model = torch.compile(model=self.model)

        # Define the optimizer.
        self.optimizer = self.optimizer(params=self.model.parameters(), lr=self.lr)

        # Move the model to the device.
        self.model.to(device=self.device)


    def __compute_current_training_loss(self, reset = True):
        """
        It computes the current training loss.

        Args:
            reset (bool): Whether to reset the loss or not. (Default: True)

        Returns:
            float: The current training loss.
        """
        # Compute mean loss.
        mean_loss = sum(self.__losses) / len(self.__losses)
        if reset:
            self.__losses.clear()
        return mean_loss
    

    def __run_forward(self, model, batch, is_training = True):
        """
        It runs the forward pass of the model.

        Args:
            model (nn.Module): The model to run the forward pass.
            batch (List[Sample]): The batch of data.
            is_training (bool): Whether the model is training or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The logits and the labels.
        """
        # Convert the batch into tensors.
        images, captions, mask, extra_data = PromptableDeTRDataLoader.convert_batch_into_tensor(batch=batch)
        images = images.to(device=self.device)
        captions = captions.to(device=self.device)
        mask = mask.to(device=self.device)

        def run_forward(model, images, captions, extra_data):
            bbox = extra_data["bbox"].to(device=self.device)
            logits = model(images, captions, mask) # Input: Image, caption and the mask to occlude padded tokens.
            return logits, bbox # Output: Pred boxes and presences and the true boxes and presences.

        # Run the forward pass.
        if not is_training:
            with torch.no_grad():
                logits, labels = run_forward(model, images, captions, extra_data)
        else:
            logits, labels = run_forward(model, images, captions, extra_data)
        
        return logits, labels


    def __get_random_sample(self, logits, y):
        """
        It gets a random sample from the logits and the true captions to be visualized further.

        Args:
            logits (torch.Tensor): The logits from the model.
            y (torch.Tensor): The true captions.

        Returns:
            Tuple[str, str] | Tuple[List[int], List[int]]: The true and predicted captions or objects.
        """
        # Get random batch index.
        batch_index = torch.randint(low=0, high=y.size(0), size=(1,)).item()

        # Retrieve samples.
        logits_objs = logits[batch_index].cpu()
        y_objs = y[batch_index].cpu().numpy()

        # Filter the objects.
        logits_max = logits_objs[:, 4:].argmax(dim=1)
        logits_objs = logits_objs[logits_max == 1]
        logits_objs[:, 4:] = logits_objs[:, 4:].softmax(dim=1)
        logits_objs = logits_objs.numpy().tolist()
        y_objs = y_objs[y_objs[:, 4] == 1][:, :4].tolist()

        return y_objs, logits_objs
    

    def __save_model(self, valid_loss, valid_time, samples):
        """
        It saves the model if the validation loss is better than the previous one.

        Args:
            valid_loss (float): The validation loss.
            valid_time (float): The evaluation time.
            samples (List[Tuple[str, str]]|List[Tuple[numpy.ndarray, numpy.ndarray]]): The samples to visualize.
        """
        # Save the model weights.
        is_best = False
        current_train_loss = self.__compute_current_training_loss()
        if self.__best_loss > valid_loss:
            self.__best_loss = valid_loss
            is_best = True
            self.__overfit_counter = 0 # Reset the overfit counter.
        
        # Check if it is overfitting.
        elif valid_loss - current_train_loss > self.overfit_threshold:
            self.__overfit_counter += 1
            if self.__overfit_counter >= self.overfit_patience:
                print("Overfitting detected. Stopping training.")
                self.__is_overfitting = True

        print("Validation time: %.2f minutes" % valid_time)
        print("Overfit counter: %d" % self.__overfit_counter)
        print("Validation loss: %.4f" % valid_loss)
        self.model.save_model(
            dir_path=self.exp_dir, 
            ckpt_step=self.__current_iter, 
            loss=valid_loss, 
            samples=samples, 
            is_best=is_best
        )
        print("=" * 100)


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
                logits, y = self.__run_forward(model=self.model, batch=training_batch, is_training=True)

                # Compute the loss.
                loss = self.model.compute_loss(logits=logits, labels=y)

                # Backward pass.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Check if it is time to log the loss.
                self.__losses.append(loss.cpu().detach().numpy().item())
                if self.__current_iter % self.log_interval == 0:
                    current_loss = self.__compute_current_training_loss(reset=False)
                    print("Iteration [%d/%d]" % (self.__current_iter, self.max_iter))
                    print("Loss: %.4f" % current_loss)
                    print("-" * 100)

                # Check if it is time to validate the model.
                if self.__current_iter % self.eval_interval == 0:
                    print("=" * 100)
                    print("Validating the model...")

                    # Loop over the validation data loader.
                    total_loss = 0.0
                    samples = []
                    init_time = time.time()
                    for validation_batch in self.valid_dataset:
                        
                        # Run the forward pass.
                        logits, y = self.__run_forward(model=self.model, batch=validation_batch, is_training=False)

                        # Compute the loss.
                        loss = self.model.compute_loss(logits=logits, labels=y)
                        total_loss += loss.cpu().numpy().item()

                        # Get a random sample.
                        y_sample, logits_sample = self.__get_random_sample(logits=logits, y=y)
                        samples.append((y_sample, logits_sample))
                    
                    total_loss /= len(self.valid_dataset)
                    end_time = (time.time() - init_time) / 60.0

                    # Save the model weights.
                    self.__save_model(valid_loss=total_loss, valid_time=end_time, samples=samples)
            
                # Update the iteration.
                self.__current_iter += 1
                if self.__current_iter >= self.max_iter or self.__is_overfitting:
                    break


    # Methods.
    def train(self):
        """
        It trains the model using the training and validation datasets.
        """
        print("=" * 100)
        print("🚀 Starting PromptableDeTR - %s training" % self.trainer_name)
        print("=" * 100)

        try:
            # Compile the model.
            self.__compile_model()
            # Run the main loop.
            self.__main_loop()
        except KeyboardInterrupt:
            print("=" * 100)
            print("🛑 Training interrupted by the user.")
            print("=" * 100)
        except Exception as e:
            print("😵 An error occurred during training.")
            print(e)
