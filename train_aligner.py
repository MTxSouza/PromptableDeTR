"""
This script trains the Aligner model to optimize the Joiner block of the PromptableDeTR model.
"""
import os

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


def main():

    # Get the arguments.
    args = get_args()

    # Prepare the data loader.
    train_data_loader, valid_data_loader = get_data_loader(args=args)


if __name__ == "__main__":
    main()
