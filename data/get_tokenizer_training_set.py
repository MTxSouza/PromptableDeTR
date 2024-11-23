"""
This script takes the JSON file from COCO dataset that contains the captions for each image. It generates 
a TXT file with all captions concatenated. This file will be used to train the Byte Pair Encoding (BPE) 
algorithm to generate the vocabulary used in the tokenizer for text encoder.

Before run this script, make sure to run the `download_coco_dataset.py` script using the `--annot` flag to 
download the annotations file. After it, extract the .zip file and set the path to the JSON file where the 
captions are stored.
"""
import argparse
import json
import os
import sys

# Add the project directory to the path.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import get_logger


def argument_parser():
    """
    Argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    # Create the parser.
    parser = argparse.ArgumentParser(description="Generate a TXT file with all captions concatenated.")

    # Add arguments.
    parser.add_argument(
        "--train-captions", 
        "-tc", 
        type=str, 
        required=False, 
        default=None, 
        help="The path to the JSON file with the training captions."
    )
    parser.add_argument(
        "--valid-captions", 
        "-vc", 
        type=str, 
        required=False, 
        default=None, 
        help="The path to the JSON file with the validation captions."
    )
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        default="./", 
        help="The path to the directory where the TXT file will be saved."
    )

    return parser.parse_args()


if __name__ == "__main__":

    # Get the logger.
    logger = get_logger(name="tokenizer_set", level="debug")

    # Parse the arguments.
    args = argument_parser()

    # Check if the output directory exists.
    if not os.path.isdir(args.output):
        logger.warning("Invalid output directory %s. Exiting..." % args.output)
        sys.exit(0)

    # Check if the paths are valid.
    if args.train_captions is None and args.valid_captions is None:
        logger.warning("No captions files provided. Exiting...")
        sys.exit(0)
    captions = {"train": args.train_captions, "valid": args.valid_captions}

    # Load the captions.
    content = {}
    for name, path in captions.items():

        # Check if the path was defined.
        if path is None:
            continue

        # Check if the path is valid.
        if not os.path.isfile(path=path):
            logger.warning("Invalid path %s. Skipping..." % path)
            continue

        # Check if the file is a JSON file.
        if not path.endswith(".json"):
            logger.warning("Invalid file format %s. Skipping..." % path)
            continue

        logger.info("Loading captions from %s..." % path)
        with open(file=path, mode="r", encoding="utf-8") as file_buffer:
            content[name] = json.load(fp=file_buffer).get("annotations", [])

    # Save the concatenated captions.
    for name, captions in content.items():

        # Check if there are captions.
        if not captions:
            logger.warning("No captions found for %s. Skipping..." % name)
            continue

        # Get captions.
        text = [caption.get("caption", "").strip() for caption in captions]
        text = list(filter(lambda x: x != "", text))
        text = [caption + "\n" for caption in text]
        logger.debug(msg="%d captions has been retrieved for %s." % (len(text), name))

        # Get the output file.
        output_file = os.path.join(args.output, "%s_captions.txt" % name)

        # Concatenate the captions.
        logger.info("Concatenating captions for %s..." % name)
        with open(file=output_file, mode="w", encoding="utf-8") as file_buffer:
            file_buffer.writelines(text)

        logger.info("Captions saved to %s." % output_file)
