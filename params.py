"""
This module contains the arguments used for most parts of the project, including training.
"""
import argparse
import os


# Functions.
def get_dataset_args(value):
    """
    Get the dataset arguments from a comma-separated string.

    Args:
        value (str): The comma-separated string.

    Returns:
        tuple[str, float]: The dataset path and weight.
    """
    split = value.split(",")
    assert len(split) == 2, "The dataset argument must contain two paths separated by a comma."
    data_path, weight = split
    assert os.path.isdir(data_path), "The dataset path must be a valid directory."
    weight = float(weight)
    assert weight > 0, "The dataset weight must be a positive number."
    return data_path, weight

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
        type=get_dataset_args,
        nargs="+",
        required=True,
        help="List of paths to the train dataset directory."
    )
    dataset_parser.add_argument(
        "--valid-dataset-dir",
        "--valid-dataset",
        "-vd",
        type=get_dataset_args,
        nargs="+",
        required=True,
        help="List of paths to the valid dataset directory."
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
        "--model-weights",
        "--bmw",
        type=str,
        default=None,
        help="Path to the model weights to start training from."
    )
    model_parser.add_argument(
        "--image-size",
        "--img-size",
        type=int,
        default=320,
        choices=[640, 480, 320, 224],
        help="The image size."
    )
    model_parser.add_argument(
        "--num-queries",
        "--nq",
        type=int,
        default=10,
        choices=list(range(5, 51, 5)),
        help="The number of queries."
    )
    model_parser.add_argument(
        "--proj-dim",
        type=int,
        default=512,
        help="The projection dimension."
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
        default=1024,
        help="The feedforward dimension."
    )
    model_parser.add_argument(
        "--num-joiner-layers",
        "-jl",
        type=int,
        default=3,
        help="The number of joiner layers."
    )

    # Training arguments.
    training_parser = parser.add_argument_group(title="Training")
    training_parser.add_argument(
        "--resume-checkpoint",
        "--ckpt",
        type=str,
        default=None,
        help="The path to the checkpoint to resume training."
    )
    training_parser.add_argument(
        "--max-iter",
        type=int,
        default=80000,
        help="The maximum number of iterations."
    )
    training_parser.add_argument(
        "--curve-limit",
        type=int,
        default=55000,
        help="The maximum number of iterations for the LR curve."
    )
    training_parser.add_argument(
        "--disable-lr-curve",
        action="store_true",
        help="Whether to disable the learning rate curve. If it is True, the learning rate will be constant, using the `--min-lr` value."
    )
    training_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="The batch size for the training process."
    )
    training_parser.add_argument(
        "--max-lr",
        type=float,
        default=5e-5,
        help="Maximum learning rate for the Joiner optimizer."
    )
    training_parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for the Joiner optimizer."
    )
    training_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=250,
        help="The number of warmup steps for the learning rate scheduler."
    )
    training_parser.add_argument(
        "--frozen-steps",
        type=int,
        default=1000,
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
        default=10,
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
        "--use-focal-loss",
        action="store_true",
        help="Whether to use focal loss to compute the presence loss."
    )
    training_parser.add_argument(
        "--presence-weight",
        type=float,
        default=2.0,
        help="The weight for the presence loss. If using focal loss, this is the gamma parameter."
    )
    training_parser.add_argument(
        "--giou-weight",
        type=float,
        default=3.0,
        help="The weight for the GIoU loss."
    )
    training_parser.add_argument(
        "--l1-weight",
        type=float,
        default=5.0,
        help="The weight for the L1 loss."
    )
    training_parser.add_argument(
        "--local-contrastive-weight",
        type=float,
        default=1.0,
        help="The weight for the local contrastive loss."
    )
    training_parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="The alpha parameter for the focal loss."
    )
    training_parser.add_argument(
        "--hm-presence-weight",
        type=float,
        default=3.0,
        help="The presence loss weight for the Hungarian matcher."
    )
    training_parser.add_argument(
        "--hm-giou-weight",
        type=float,
        default=2.0,
        help="The GIoU loss weight for the Hungarian matcher."
    )
    training_parser.add_argument(
        "--hm-l1-weight",
        type=float,
        default=3.0,
        help="The L1 loss weight for the Hungarian matcher."
    )
    training_parser.add_argument(
        "--disable-caption-prob",
        type=float,
        default=0.3,
        help="The probability of disabling the caption during training."
    )
    training_parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Whether to save the logs."
    )
    training_parser.add_argument(
        "--log-grads",
        action="store_true",
        help="Whether to log the gradients."
    )

    # Parse the arguments.
    return parser.parse_args()
