"""
This script tries to find all the possible input sizes for a given image size.
"""
import argparse
from dataclasses import dataclass
from itertools import product

from tqdm import tqdm


# Structures.
@dataclass
class InputSizes:
    """
    Dataclass to store the input sizes.
    """
    image_size: int
    patch_size: int
    chunk_size: int
    embedding_size: int
    head_size: int
    context_window_size: int


# Functions.
def cli_args():
    """
    Get all CLI arguments to be used in script.
    """
    # Define parser.
    parser = argparse.ArgumentParser(usage=__doc__)
    # Arguments.
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Initial size for the image.")

    args = parser.parse_args()

    # Validate arguments.
    # --image-size
    if args.image_size <= 0:
        raise ValueError("Image size must be greater than 0.")
    elif args.image_size % 2 != 0:
        raise ValueError("Image size must be an even number to allow the network to downsample it properly.")

    return args


def main():
    """
    Main function to be executed.
    """
    # Get CLI arguments.
    args = cli_args()

    # Define initial variables.
    possible_patch_sizes = list(range(2, args.image_size + 1))
    possible_embedding_sizes = list(range(4, args.image_size ** 2 + 1))
    possible_heads = list(range(1, max(possible_patch_sizes) + 1))

    # Find optimal sizes.
    optimal_sizes = []
    for patch_size, embedding_size, head_size in tqdm(
        iterable=product(possible_patch_sizes, possible_embedding_sizes, possible_heads), 
        desc="Finding optimal sizes...",
        ):

        # Check if the patch size is divisible by the image size.
        if args.image_size % patch_size != 0:
            continue

        # Calculate chunk size.
        chunk_size = (args.image_size // patch_size) ** 2

        # Check if the embedding size is equal to the chunk size.
        if not embedding_size == chunk_size:
            continue

        # Check if the embedding size is divisible by the head size.
        if embedding_size % head_size != 0:
            continue

        # Compute final context window size.
        context_window_size = patch_size ** 2

        # Store optimal sizes.
        optimal_sizes.append(
            InputSizes(
                image_size=args.image_size,
                patch_size=patch_size,
                chunk_size=chunk_size,
                embedding_size=embedding_size,
                head_size=head_size,
                context_window_size=context_window_size
            )
        )

    # Print results.
    print("Optimal input sizes for PromptVision model:")
    for size in optimal_sizes:
        print("\t- Image size: %d" % size.image_size)
        print("\t- Patch size: %d" % size.patch_size)
        print("\t- Chunk size: %d" % size.chunk_size)
        print("\t- Embedding size: %d" % size.embedding_size)
        print("\t- Head size: %d" % size.head_size)
        print("\t- Context window size: %d" % size.context_window_size)
        print()


if __name__ == "__main__":
    main()
