"""
This script trains the Aligner model to optimize the Joiner block of the PromptableDeTR model.
"""
import torch.optim as optim

from data.daug import MaskCaption, PrepareAlignerSample, ReshapeImage
from data.loader import PromptableDeTRDataLoader
from models.aligner import Aligner
from params import get_args
from trainer import Trainer


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
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        emb_dropout_rate=args.emb_dropout_rate,
        num_joiner_layers=args.num_joiner_layers
    )

    return model


def main():

    # Get the arguments.
    args = get_args()
    for name, value in vars(args).items():
        print("[%s]: %s" % (name, value))

    # Prepare the data loader.
    print("Preparing the data loader...")
    train_data_loader, valid_data_loader = get_data_loader(args=args)

    # Create the model.
    print("Creating the model...")
    model = get_model(args=args, data_loader=train_data_loader)
    model.load_base_weights(image_encoder_weights=args.image_encoder_weights, text_encoder_weights=args.text_encoder_weights)
    model.freeze_encoder()

    # Train the model.
    trainer = Trainer(
        trainer_name="Aligner",
        model=model,
        optimizer=optim.Adam,
        train_dataset=train_data_loader,
        valid_dataset=valid_data_loader,
        lr=args.lr,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        max_iter=args.max_iter,
        overfit_threshold=args.overfit_threshold,
        overfit_patience=args.overfit_patience,
        exp_dir=args.exp_dir,
        is_aligner=True
    )
    trainer.train()


if __name__ == "__main__":
    main()
