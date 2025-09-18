"""
Script that generate samples from PhraseCut annotations.
"""
import argparse
import json
import os
import re
import uuid
from dataclasses import asdict

import tqdm
from PIL import Image

from data.schemas import Sample
from models.tokenizer import Tokenizer


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate samples from PhraseCut annotations.")
    parser.add_argument(
        "--annot_json_path",
        "--annot",
        "-a",
        type=str,
        required=True,
        help="Path to the annotation JSON file.",
    )
    parser.add_argument(
        "--image_dir",
        "-i",
        type=str,
        required=True,
        help="Directory where the images are stored.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save the generated samples.",
    )

    # Filters.
    parser.add_argument(
        "--vocab-path",
        "--vocab",
        type=str,
        required=True,
        help="Path to the vocabulary file.",
    )
    parser.add_argument(
        "--max-detections",
        "--max-det",
        type=int,
        default=5,
        help="Maximum number of detections per image."
    )
    parser.add_argument(
        "--max-tokens-per-caption",
        "--max-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens per caption."
    )
    parser.add_argument(
        "--target-words",
        type=str,
        default=None,
        help="Path to a TXT file with all target words to keep, the script will ignore any sample with words outside this list."
    )

    return parser.parse_args()

def load_annot_info(annot_json_path):
    """
    Load annotation information from a JSON file to retrieve
    the target image, its captions and object locations.

    Args:
        annot_json_path (str): Path to the annotation JSON file.
    
    Returns:
        dict: A dictionary containing all metadata information.
    """
    # Load the JSON file.
    with open(file=annot_json_path, mode="r") as file_buffer:
        annot_info = json.load(fp=file_buffer)

    return annot_info

def get_basic_annot(annot, image_dir):
    """
    Extract all basic annotation information from
    a single annotation entry.

    Args:
        annot (dict): A single annotation entry.
        image_dir (str): Directory where the images are stored.

    Returns:
        dict: A dictionary containing basic annotation information.
    """
    # Get image path.
    image_id = annot["image_id"]
    image_name = "%d.jpg" % image_id
    image_path = os.path.join(image_dir, image_name)
    assert os.path.exists(image_path), "Image path does not exist: %s" % image_path

    # Get image size.
    with Image.open(fp=image_path, mode="r") as pil_img:
        width, height = pil_img.width, pil_img.height
    assert width > 0 and height > 0, "Image width and height must be positive"
    del pil_img

    # Get caption and bounding box.
    white_space_pattern = re.compile(r"\s+")
    caption = annot["phrase"].strip().lower()
    caption = white_space_pattern.sub(" ", caption)

    bbox = annot["instance_boxes"]
    assert isinstance(bbox, list), "Bounding box must be a list"
    assert bbox, "Bounding box cannot be empty"

    # Get center boxes from bounding boxes.
    boxes = []
    for box in bbox:
        assert len(box) == 4, "Bounding box must have 4 elements"
        x1, y1, w, h = box
        cx = x1 + w / 2
        cy = y1 + h / 2
        cx = min(max(0.0, cx), width)
        cy = min(max(0.0, cy), height)
        w = min(max(0.0, w), width)
        h = min(max(0.0, h), height)
        cx /= width
        cy /= height
        w /= width
        h /= height
        assert cx <= 1.0, "Center x-coordinate must be <= 1.0"
        assert cy <= 1.0, "Center y-coordinate must be <= 1.0"
        assert w <= 1.0, "Width must be <= 1.0"
        assert h <= 1.0, "Height must be <= 1.0"
        boxes.append([cx, cy, w, h])

    return {
        "image_path": image_path,
        "caption": caption,
        "boxes": boxes
    }

def main():
    """
    Main function to execute the script.
    """
    # Parse command line arguments.
    args = parse_args()

    # Load word filter if provided.
    if args.target_words is not None:
        assert os.path.exists(args.target_words), "Target words file does not exist: %s" % args.target_words
        with open(file=args.target_words, mode="r") as file_buffer:
            target_words = set([line.strip().lower() for line in file_buffer.readlines()])
        print("Loaded %d target words from %s" % (len(target_words), args.target_words))
    else:
        target_words = None

    # Load annotation information.
    annot_info = load_annot_info(annot_json_path=args.annot_json_path)

    # Create output directory if it doesn't exist.
    os.makedirs(name=args.output_dir, exist_ok=True)

    # Instantiate tokenizer.
    tokenizer = Tokenizer(vocab_filepath=args.vocab_path)

    # Iterate through all images and labels.
    all_samples = []
    for annot in tqdm.tqdm(iterable=annot_info):

        # Get basic annotation information.
        main_annot_data = get_basic_annot(annot=annot, image_dir=args.image_dir)
        sample = Sample(**main_annot_data)
        all_samples.append(asdict(sample))

    # Save all samples to a JSON file.
    for sample in all_samples:
        
        # Filter sample.
        n_det = len(sample["boxes"])
        words = sample["caption"].split(" ")
        n_tokens = len(tokenizer.encode_str(sample["caption"]).pop(0))
        if n_det > args.max_detections or n_tokens > args.max_tokens_per_caption:
            continue
        if target_words is not None and any([w not in target_words for w in words]):
            continue

        while True:
            filename = str(uuid.uuid4()) + ".json"
            output_path = os.path.join(args.output_dir, filename)
            if not os.path.exists(output_path):
                break
        with open(file=output_path, mode="w") as file_buffer:
            json.dump(sample, file_buffer, indent=4)

if __name__=="__main__":
    main()
