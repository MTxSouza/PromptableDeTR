"""
This script trains the Detector model to optimize the entire model to localize and classify objects 
in images based on the prompts.
"""
import logging
import random

import torch
import torch.optim as optim

from data.daug import PrepareSample, ReshapeImage
from data.loader import PromptableDeTRDataLoader
from models.detector import PromptableDeTRTrainer
from params import get_args
from utils.trainer import Trainer


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
    train_files = []
    for dirpath in args.train_dataset_dir:
        train_files += PromptableDeTRDataLoader.get_samples_from_dir(dirpath=dirpath)
    valid_files = []
    for dirpath in args.valid_dataset_dir:
        valid_files += PromptableDeTRDataLoader.get_samples_from_dir(dirpath=dirpath)
    if len(train_files) == 0:
        raise ValueError("No training samples found in the directory: %s" % str(args.train_dataset_dir))
    if len(valid_files) == 0:
        raise ValueError("No validation samples found in the directory: %s" % str(args.valid_dataset_dir))

    # Get the data loader.
    train_data_loader = PromptableDeTRDataLoader(
        sample_file_paths=train_files,
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
    model = PromptableDeTRTrainer(
        image_size=args.image_size,
        vocab_size=vocab_size,
        emb_dim=args.proj_dim,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        emb_dropout_rate=args.emb_dropout_rate,
        num_joiner_layers=args.num_joiner_layers,
        use_focal_loss=args.use_focal_loss,
        presence_loss_weight=args.presence_weight,
        giou_loss_weight=args.giou_weight,
        l1_loss_weight=args.l1_weight,
        alpha=args.alpha,
        hm_presence_weight=args.hm_presence_weight,
        hm_giou_weight=args.hm_giou_weight,
        hm_l1_weight=args.hm_l1_weight
    )

    return model


def main(device=None):

    # Get the arguments.
    args = get_args()
    for name, value in vars(args).items():
        print("[%s]: %s" % (name, value))

    # Set the random seed for reproducibility.
    if args.seed is not None:
        print("Setting the random seed to %d" % args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Set logging level.
    if not args.save_logs:
        logging.basicConfig(level=logging.CRITICAL, force=True)

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
        max_caption_length=args.caption_length,
        lr=args.lr,
        lr_factor=args.lr_factor,
        warmup_steps=args.warmup_steps,
        frozen_steps=args.frozen_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        max_iter=args.max_iter,
        overfit_threshold=args.overfit_threshold,
        overfit_patience=args.overfit_patience,
        exp_dir=args.exp_dir,
        device=device
    )

    # Resume training if a checkpoint is provided.
    trainer.train(checkpoint_path=args.resume_checkpoint)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device=device)
