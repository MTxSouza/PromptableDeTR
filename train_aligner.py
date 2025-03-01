"""
This script trains the Aligner model to optimize the Joiner block of the PromptableDeTR model.
"""
import time

import torch
import torch.optim as optim

from data.daug import MaskCaption, PrepareAlignerSample, ReshapeImage
from data.loader import PromptableDeTRDataLoader
from models.aligner import Aligner
from params import get_args


# Functions.
def get_data_loader(args):
    """
    Create the PromptableDeTRDataLoader object for Aligner training.

    Args:
        args (argparse.Namespace): The arguments from the command line.

    Returns:
        Tuple[PromptableDeTRDataLoader, PromptableDeTRDataLoader]: The training and validation data loaders.
    """
    # Split into training and validation.
    train_files, valid_files = PromptableDeTRDataLoader.get_train_val_split(
        sample_directory=args.dataset_dir,
        val_split=args.valid_split,
        shuffle_samples=args.shuffle,
        seed=args.seed
    )

    # Get the data loader.
    train_data_loader = PromptableDeTRDataLoader(
        sample_file_paths=train_files,
        image_directory=args.image_dir,
        batch_size=args.batch_size,
        transformations=[
            PrepareAlignerSample(vocab_file=args.vocab_file),
            ReshapeImage(image_size=args.image_size),
            MaskCaption(vocab_file=args.vocab_file, mask_ratio=args.mask_ratio),
        ],
        shuffle=args.shuffle,
        seed=args.seed,
        aligner=True # Mandatory for Aligner training.
    )

    valid_data_loader = PromptableDeTRDataLoader(
        sample_file_paths=valid_files,
        image_directory=args.image_dir,
        batch_size=args.batch_size,
        transformations=[
            PrepareAlignerSample(vocab_file=args.vocab_file),
            ReshapeImage(image_size=args.image_size),
            MaskCaption(vocab_file=args.vocab_file, mask_ratio=args.mask_ratio),
        ],
        shuffle=args.shuffle,
        seed=args.seed,
        aligner=True # Mandatory for Aligner training.
    )

    return train_data_loader, valid_data_loader


def get_model(args, data_loader):
    """
    Create the Aligner model.

    Args:
        args (argparse.Namespace): The arguments from the command line.
        data_loader (PromptableDeTRDataLoader): The data loader for the model.

    Returns:
        Aligner: The Aligner model.
    """
    # Get number of tokens in the vocabulary.
    tokenizer = data_loader.get_tokenizer()
    if tokenizer is None:
        raise ValueError("Could not find the tokenizer in the data loader.")
    vocab_size = len(tokenizer)

    # Get the model.
    model = Aligner(
        image_tokens=args.image_tokens,
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        emb_dropout_rate=args.emb_dropout_rate,
    )

    return model


def run_forward(model, batch, device, is_training = True):
    """
    Run the forward pass of the model.

    Args:
        model (Aligner): The Aligner model.
        batch (List[AlignerSample]): The batch of samples.
        device (torch.device): The device to run the model on.
        is_training (bool): Whether the model is in training mode. (Default: True)

    Returns:
        Dict[str, torch.Tensor]: The output of the model.
    """
    # Get tensors.
    images, captions, mask = PromptableDeTRDataLoader.convert_batch_into_tensor(batch=batch, aligner=True)
    images = images.to(device=device)
    captions = captions.to(device=device)
    mask = mask.to(device=device)

    # Run the forward pass.
    if not is_training:
        model.eval()
        with torch.no_grad():
            logits = model(images, captions, mask)
    else:
        model.train()
        logits = model(images, captions, mask)

    return logits, captions


def get_random_sample(y, logits, tokenizer):
    """
    Get a random sample from the logits and the true captions to 
    be visualized further.

    Args:
        y (torch.Tensor): The true captions.
        logits (torch.Tensor): The logits from the model.
        tokenizer (Tokenizer): The tokenizer object.

    Returns:
        Tuple[str, str]: The true and predicted captions.
    """
    # Get random bacth index.
    idx = torch.randint(low=0, high=y.size(0), size=(1,)).item()

    # Retrieve samples.
    y_sample = y[idx].cpu().detach().numpy().tolist()
    logits_sample = logits[idx].argmax(dim=1).cpu().detach().numpy().tolist()

    # Decode samples.
    y_caption = tokenizer.decode(indices=y_sample)[0]
    logits_caption = tokenizer.decode(indices=logits_sample)[0]

    return y_caption, logits_caption


def train(model, train_data_loader, valid_data_loader, args):
    """
    Function that deploy the training loop for the Aligner model.

    Args:
        model (Aligner): The Aligner model.
        train_data_loader (PromptableDeTRDataLoader): The training data loader.
        valid_data_loader (PromptableDeTRDataLoader): The validation data loader.
        args (argparse.Namespace): The arguments from the command line.
    """
    # Get the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    # Define optimizer.
    opt = optim.Adam(params=model.parameters(), lr=args.lr)

    # Define main training loop.
    tokenizer = train_data_loader.get_tokenizer()
    it = 1
    overfit_counter = 0
    is_overfitting = False
    best_loss = float("inf")
    current_train_loss = 0.0
    print("Starting the training loop...")
    print("=" * 100)
    while it < args.max_iter and not is_overfitting:

        # Loop over the training data loader.
        for training_batch in train_data_loader:

            # Run the forward pass.
            logits, y = run_forward(model=model, batch=training_batch, device=device)

            # Compute the loss.
            loss = model.compute_aligner_loss(y_pred=logits, y_true=y)

            # Backward pass.
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Check if it is time to log the loss.
            current_train_loss = loss.cpu().detach().numpy().item()
            if it % args.log_interval == 0:
                print("Iteration [%d/%d]" % (it, args.max_iter))
                print("Loss: %.4f" % current_train_loss)
                print("-" * 100)

            # Check if it is time to validate the model.
            if it % args.eval_interval == 0:
                print("=" * 100)
                print("Validating the model...")

                # Loop over the validation data loader.
                total_loss = 0.0
                samples = []
                init_time = time.time()
                for validation_batch in valid_data_loader:
                    
                    # Run the forward pass.
                    logits, y = run_forward(model=model, batch=validation_batch, device=device, is_training=False)

                    # Compute the loss.
                    loss = model.compute_aligner_loss(y_pred=logits, y_true=y)
                    total_loss += loss.cpu().numpy().item()

                    # Get a random sample.
                    y_caption, logits_caption = get_random_sample(y=y, logits=logits, tokenizer=tokenizer)
                    samples.append((y_caption, logits_caption))

                total_loss /= len(valid_data_loader)
                end_time = (time.time() - init_time) / 60.0
                print("Validation time: %.2f minutes" % end_time)
                print("Validation loss: %.4f" % total_loss)

                # Save the model weights.
                if best_loss > total_loss:
                    best_loss = total_loss
                    model.save_joiner_weights(dir_path=args.exp_dir, ckpt_step=it, loss=total_loss, samples=samples)
                    print("Model weights saved successfully.")
                
                # Check if it is overfitting.
                elif abs(current_train_loss - total_loss) > args.overfit_threshold:
                    overfit_counter += 1
                    if overfit_counter >= args.overfit_patience:
                        print("Overfitting detected. Stopping training.")
                        is_overfitting = True
                        break
                print("=" * 100)

            # Increment the iteration.
            it += 1
            if it >= args.max_iter:
                break


def main():

    # Get the arguments.
    args = get_args()
    for name, value in vars(args).items():
        print("[%s]: %s" % (name, value))
    print("=" * 100)
    print("🚀 Starting PromptableDeTR - Aligner training")
    print("=" * 100)

    # Prepare the data loader.
    print("Preparing the data loader...")
    train_data_loader, valid_data_loader = get_data_loader(args=args)

    # Create the model.
    print("Creating the model...")
    model = get_model(args=args, data_loader=train_data_loader)
    model.load_base_weights(image_encoder_weights=args.image_encoder_weights, text_encoder_weights=args.text_encoder_weights)
    model.freeze_encoder()

    # Train the model.
    train(model=model, train_data_loader=train_data_loader, valid_data_loader=valid_data_loader, args=args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by the user.")
    except Exception as e:
        print("😵 An error occurred during training.")
        print(e)
