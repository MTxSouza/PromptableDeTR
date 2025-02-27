"""
This script trains the Aligner model to optimize the Joiner block of the PromptableDeTR model.
"""
import os

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
    vocab_size = None
    for trans in data_loader.transformations:
        if isinstance(trans, PrepareAlignerSample):
            vocab_size = len(trans.caption_transform.tokenizer)
            break
    if vocab_size is None:
        raise ValueError("Could not find the tokenizer in the transformations of the data loader.")

    # Get the model.
    model = Aligner(
        image_tokens=args.image_tokens,
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        emb_dropout_rate=args.emb_dropout_rate,
    )

    return model


def run_forward(model, batch, is_training = True):
    """
    Run the forward pass of the model.

    Args:
        model (Aligner): The Aligner model.
        batch (List[AlignerSample]): The batch of samples.
        is_training (bool): Whether the model is in training mode. (Default: True)

    Returns:
        Dict[str, torch.Tensor]: The output of the model.
    """
    # Get tensors.
    images, captions, mask = PromptableDeTRDataLoader.convert_batch_into_tensor(batch=batch, aligner=True)

    # Run the forward pass.
    if not is_training:
        model.eval()
        with torch.no_grad():
            logits = model(images=images, captions=captions, mask=mask)
    else:
        model.train()
        logits = model(images=images, captions=captions, mask=mask)

    return logits, captions


def train(model, train_data_loader, valid_data_loader, args):
    """
    Function that deploy the training loop for the Aligner model.

    Args:
        model (Aligner): The Aligner model.
        train_data_loader (PromptableDeTRDataLoader): The training data loader.
        valid_data_loader (PromptableDeTRDataLoader): The validation data loader.
        args (argparse.Namespace): The arguments from the command line.
    """
    # Define optimizer.
    opt = optim.Adam(params=model.parameters(), lr=args.lr)

    # Define main training loop.
    it = 0
    best_loss = float("inf")
    while it < args.max_iter:

        # Loop over the training data loader.
        for training_batch in train_data_loader:

            # Check if it is time to validate the model.
            if it % args.eval_interval == 0:
                print("=" * 100)
                print("Validating the model...")

                # Loop over the validation data loader.
                total_loss = 0.0
                for validation_batch in valid_data_loader:
                    
                    # Run the forward pass.
                    logits, y = run_forward(model=model, batch=validation_batch, is_training=False)

                    # Compute the loss.
                    loss = model.compute_aligner_loss(y_pred=logits, y_true=y)
                    total_loss += loss.cpu().numpy().item()
                total_loss /= len(valid_data_loader)
                print("Validation loss: %.4f" % total_loss)

                # Save the model weights.
                if best_loss > total_loss:
                    best_loss = total_loss
                    model.save_joiner_weights(dir_path=args.model_dir, ckpt_step=it)
                    print("Model weights saved successfully.")
                print("=" * 100)
            
            else:
                print("-" * 100)

            # Run the forward pass.
            logits, y = run_forward(model=model, batch=training_batch)

            # Compute the loss.
            loss = model.compute_aligner_loss(y_pred=logits, y_true=y)

            # Backward pass.
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Increment the iteration.
            it += 1
            if it >= args.max_iter:
                break


def main():

    # Get the arguments.
    args = get_args()

    # Prepare the data loader.
    train_data_loader, valid_data_loader = get_data_loader(args=args)

    # Create the model.
    model = get_model(args=args, data_loader=train_data_loader)

    # Train the model.
    train(model=model, train_data_loader=train_data_loader, valid_data_loader=valid_data_loader, args=args)


if __name__ == "__main__":
    main()
