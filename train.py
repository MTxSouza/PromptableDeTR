"""
This script trains the Detector model to optimize the entire model to localize and classify objects 
in images based on the prompts.
"""
import torch.optim as optim

from data.daug import PrepareSample, ReshapeImage
from data.loader import PromptableDeTRDataLoader
from models.detector import PromptableDeTR
from params import get_args
from trainer import Trainer


# Functions.
def get_data_loader(args):
    """
    Create the PromptableDeTRDataLoader object for Detector training.

    Args:
        args (argparse.Namespace): The arguments from the command line.

    Returns:
        Tuple[PromptableDeTRDataLoader, PromptableDeTRDataLoader]: The training and validation data loaders.
    """
    # Get train and valid samples.
    train_files = PromptableDeTRDataLoader.get_samples_from_dir(dirpath=args.train_dataset_dir)
    valid_files = PromptableDeTRDataLoader.get_samples_from_dir(dirpath=args.valid_dataset_dir)
    if len(train_files) == 0:
        raise ValueError("No training samples found in the directory: %s" % args.train_dataset_dir)
    if len(valid_files) == 0:
        raise ValueError("No validation samples found in the directory: %s" % args.valid_dataset_dir)

    # Get the data loader.
    train_data_loader = PromptableDeTRDataLoader(
        sample_file_paths=train_files,
        image_directory=args.image_dir,
        batch_size=args.batch_size,
        transformations=[
            PrepareSample(vocab_file=args.vocab_file),
            ReshapeImage(image_size=args.image_size)
        ],
        shuffle=args.shuffle,
        seed=args.seed
    )

    valid_data_loader = PromptableDeTRDataLoader(
        sample_file_paths=valid_files,
        image_directory=args.image_dir,
        batch_size=args.batch_size,
        transformations=[
            PrepareSample(vocab_file=args.vocab_file),
            ReshapeImage(image_size=args.image_size)
        ],
        shuffle=args.shuffle,
        seed=args.seed
    )

    return train_data_loader, valid_data_loader


def get_model(args, data_loader):
    """
    Create the Dectector model.

    Args:
        args (argparse.Namespace): The arguments from the command line.
        data_loader (PromptableDeTRDataLoader): The data loader for the model.

    Returns:
        PromptableDeTR: The Detector model.
    """
    # Get number of tokens in the vocabulary.
    tokenizer = data_loader.get_tokenizer()
    if tokenizer is None:
        raise ValueError("Could not find the tokenizer in the data loader.")
    vocab_size = len(tokenizer)

    # Get the model.
    model = PromptableDeTR(
        image_tokens=args.image_tokens,
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        emb_dropout_rate=args.emb_dropout_rate,
        num_joiner_layers=args.num_joiner_layers
    )
    model.define_matcher(
        presence_loss_weight=args.presence_weight,
        l1_loss_weight=args.l1_weight,
        giou_loss_weight=args.giou_weight
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
    model.load_base_weights(
        image_encoder_weights=args.image_encoder_weights,
        text_encoder_weights=args.text_encoder_weights
    )

    # Train the model.
    trainer = Trainer(
        trainer_name="PromptableDeTR",
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
        exp_dir=args.exp_dir
    )
    trainer.train()


if __name__ == "__main__":
    main()
