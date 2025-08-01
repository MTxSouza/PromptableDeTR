"""
This module contains the arguments used for most parts of the project, including training.
"""
import argparse


# Arguments.
def get_args():
    """
    Get the arguments from the command line.

    Returns:
        argparse.Namespace: The arguments from the command line.
    """
    # Define arguments.
    parser = argparse.ArgumentParser(description=__doc__)

    # General arguments.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for the random number generator."
    )

    # Dataset arguments.
    dataset_parser = parser.add_argument_group(title="Dataset")
    dataset_parser.add_argument(
        "--train-dataset-dir",
        "--train-dataset",
        "-td",
        type=str,
        required=True,
        help="The path to the train dataset directory."
    )
    dataset_parser.add_argument(
        "--valid-dataset-dir",
        "--valid-dataset",
        "-vd",
        type=str,
        required=True,
        help="The path to the valid dataset directory."
    )
    dataset_parser.add_argument(
        "--image-dir",
        "--image",
        "-i",
        type=str,
        required=True,
        help="The path to the image directory."
    )
    dataset_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset."
    )

    # Model arguments.
    model_parser = parser.add_argument_group(title="Model")
    model_parser.add_argument(
        "--vocab-file",
        "--vocab",
        "-v",
        type=str,
        required=True,
        help="The path to the vocabulary file."
    )
    model_parser.add_argument(
        "--image-encoder-weights",
        "--imgw",
        type=str,
        default=None,
        help="Path to the image encoder weights."
    )
    model_parser.add_argument(
        "--text-encoder-weights",
        "--txtw",
        type=str,
        default=None,
        help="Path to the text encoder weights."
    )
    model_parser.add_argument(
        "--base-model-weights",
        "--bmw",
        type=str,
        default=None,
        help="Path to the full base model weights."
    )
    model_parser.add_argument(
        "--image-size",
        "--img-size",
        type=int,
        default=640,
        help="The image size."
    )
    model_parser.add_argument(
        "--image-tokens",
        "--img-tk",
        type=int,
        nargs="+",
        default=[400, 100],
        help="The image tokens."
    )
    model_parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="The embedding dimension for MobileBERT."
    )
    model_parser.add_argument(
        "--proj-dim",
        type=int,
        default=512,
        help="The projection dimension for MobileBERT."
    )
    model_parser.add_argument(
        "--emb-dropout-rate",
        type=float,
        default=0.1,
        help="The dropout rate for the embeddings."
    )
    model_parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="The number of attention heads."
    )
    model_parser.add_argument(
        "--ff-dim",
        type=int,
        default=2048,
        help="The feedforward dimension."
    )
    model_parser.add_argument(
        "--num-joiner-layers",
        "-jl",
        type=int,
        default=4,
        help="The number of joiner layers."
    )

    # Training arguments.
    training_parser = parser.add_argument_group(title="Training")
    training_parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="The maximum number of iterations."
    )
    training_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="The batch size for the training process."
    )
    training_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="The learning rate for the optimizer."
    )
    training_parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.1,
        help="The factor to reduce the learning rate."
    )
    training_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="The number of warmup steps for the learning rate scheduler."
    )
    training_parser.add_argument(
        "--frozen-steps",
        type=int,
        default=2000,
        help="The number of frozen steps that the lr will not change."
    )
    training_parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="The interval for evaluation."
    )
    training_parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="The interval for logging."
    )
    training_parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=1e-3,
        help="The threshold for over fitting."
    )
    training_parser.add_argument(
        "--overfit-patience",
        type=int,
        default=5,
        help="The patience for over fitting."
    )
    training_parser.add_argument(
        "--exp-dir",
        "--exp",
        "-e",
        type=str,
        default="./exp",
        help="The path to the experiment directory."
    )
    training_parser.add_argument(
        "--giou-weight",
        type=float,
        default=1.0,
        help="The weight for the GIoU loss."
    )
    training_parser.add_argument(
        "--presence-weight",
        type=float,
        default=1.0,
        help="The weight for the presence loss."
    )
    training_parser.add_argument(
        "--l1-weight",
        type=float,
        default=1.0,
        help="The weight for the L1 loss."
    )

    # Parse the arguments.
    return parser.parse_args()
