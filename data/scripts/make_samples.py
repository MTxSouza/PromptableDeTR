"""
This script generates samples based on the Yolov12 label created by Roboflow platform.
"""
import argparse
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import tqdm
import yaml
from PIL import Image

from data.schemas import Sample


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate samples from Yolov12 labels.")
    parser.add_argument(
        "--annot_yaml_path",
        "--annot",
        "-a",
        type=str,
        required=True,
        help="Path to the annotation YAML file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save the generated samples.",
    )

    return parser.parse_args()


def load_annot_info(annot_yaml_path):
    """
    Load annotation information from a YAML file to retrieve
    the class names and their corresponding IDs.

    Args:
        annot_yaml_path (str): Path to the annotation YAML file.
    
    Returns:
        dict: A dictionary containing all metadata information.
    """
    # Load the YAML file.
    with open(file=annot_yaml_path, mode="r") as file_buffer:
        annot_info = yaml.safe_load(stream=file_buffer)
        annot_info["root"] = os.path.dirname(p=annot_yaml_path)

    return annot_info


def get_sample_files(annot_info):
    """
    Retrieve sample files from the annotation information.

    Args:
        annot_info (dict): Annotation information containing metadata.
    
    Returns:
        Tuple[str, str]: A tuple containing the paths to the image and label files.
    """
    # Get root path of yaml file.
    root_path = annot_info["root"]

    # Get training path only.
    train_path = annot_info["train"].replace("../", "").replace("images", "")
    train_path = os.path.join(root_path, train_path)

    image_dir = os.path.join(train_path, "images")
    label_dir = os.path.join(train_path, "labels")

    # Check if directories exist.
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    return image_dir, label_dir


def make_sample(image_file, annot_file, captions):
    """
    Generate a sample object.

    Args:
        image_file (str): Path to the image file.
        annot_file (str): Path to the annotation file.
        captions (list): List of captions for the image.

    Returns:
        List[Sample]: A list of Sample objects.
    """
    # Load the annotation file.
    with open(file=annot_file, mode="r") as file_buffer:
        lines = file_buffer.read().split(sep="\n")
        lines = list(filter(len, lines))
        annot = list(map(lambda line: line.split(sep=" "), lines))

    # Get image dimensions.
    with Image.open(fp=image_file) as img:
        img_width, img_height = img.size
    del img

    # Group boxes by class.
    mapper = {}
    for cat_idx, cx, cy, width, height in tqdm.tqdm(iterable=annot, desc="Mapping captions..."):
        caption = captions[int(cat_idx)]
        mapper[caption] = mapper.get(caption, [])

        # Convert to min and max.
        cx = float(cx) * img_width
        cy = float(cy) * img_height
        width = float(width) * img_width
        height = float(height) * img_height

        xmin = int(cx - width / 2)
        ymin = int(cy - height / 2)
        xmax = int(cx + width / 2)
        ymax = int(cy + height / 2)

        # Append to the list.
        mapper[caption].append([xmin, ymin, xmax, ymax])

    # Create a samples object.
    samples = []
    for caption, box_list in tqdm.tqdm(iterable=mapper.items(), desc="Creating samples..."):

        # Create a sample object.
        sample = Sample(
            image_path=image_file,
            caption=caption,
            bbox=box_list
        )
        samples.append(sample)
    
    return samples


def main():
    """
    Main function to execute the script.
    """
    # Parse command line arguments.
    args = parse_args()

    # Load annotation information.
    annot_info = load_annot_info(annot_yaml_path=args.annot_yaml_path)
    image_dir, label_dir = get_sample_files(annot_info=annot_info)

    # Create output directory if it doesn't exist.
    os.makedirs(name=args.output_dir, exist_ok=True)

    # Iterate through all images and labels.
    all_samples = []
    for annot_file in Path(label_dir).glob("*.txt"):
        
        # Get the corresponding image file.
        image_file = os.path.join(image_dir, annot_file.stem + ".jpg")

        # Check if the image file exists.
        if not os.path.exists(image_file):
            print(f"Image file does not exist: {image_file}")
            continue

        # Generate a samples.
        samples = make_sample(image_file=image_file, annot_file=annot_file, captions=annot_info["names"])
        all_samples.extend(samples)

    # Convert to JSON format.
    all_samples_json = list(map(asdict, all_samples))
    del all_samples

    # Save the samples to a JSON file.
    for sample in tqdm.tqdm(iterable=all_samples_json, desc="Saving samples..."):
        out_file = os.path.join(args.output_dir, str(uuid.uuid4()) + ".json")
        with open(file=out_file, mode="w") as file_buffer:
            json.dump(obj=sample, fp=file_buffer, indent=4)


if __name__=="__main__":
    main()
