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
        "--dataset-dir",
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="The path to the dataset directory."
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
        "--aligner",
        "-a",
        action="store_true",
        help="Whether to use the aligner samples."
    )
    dataset_parser.add_argument(
        "--valid-split",
        "--valid",
        "-vs",
        type=float,
        default=0.2,
        help="The validation split."
    )
    dataset_parser.add_argument(
        "--mask-ratio",
        "-mr",
        type=float,
        default=0.1,
        help="The ratio of masked tokens."
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
        "-imgw",
        type=str,
        default=None,
        help="Path to the image encoder weights."
    )
    model_parser.add_argument(
        "--text-encoder-weights",
        "-txtw",
        type=str,
        default=None,
        help="Path to the text encoder weights."
    )
    model_parser.add_argument(
        "--joiner-weights",
        "-jw",
        type=str,
        default=None,
        help="Path to the joiner weights."
    )
    model_parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="The number of attention heads."
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
        "--eval-interval",
        type=int,
        default=100,
        help="The interval for evaluation."
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
        "--bbox-weight",
        type=float,
        default=1.0,
        help="The weight for the bounding box loss."
    )
    training_parser.add_argument(
        "--presence-weight",
        type=float,
        default=1.0,
        help="The weight for the presence loss."
    )

    # Parse the arguments.
    return parser.parse_args()
