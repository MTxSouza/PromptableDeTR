"""
This script loads the annotations of both train and validation set and creates new 
annotation files for the PromptVision project. It just simplify the annotation and 
saves only the necessary information from there.

The COCO dataset has the category classes for all 80 objects, this script will generate 
a .JSON file with all annotation for each image.
"""
import os
import sys

sys.path.append(os.path.abspath(path=os.path.dirname(p=__file__.split(sep=os.sep)[-2])))

import argparse
import json
import warnings
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from data.schemas import Annotation, Boxes, _ObjectCategory


def cli_args():
    """
    Get all CLI arguments to be used in script.
    """
    # Define the parser.
    parser = argparse.ArgumentParser(description="Format COCO annotations.", usage=__doc__)

    # Arguments.
    parser.add_argument(
        "--annotation-file", 
        "-a", 
        type=str, 
        required=True, 
        help="Path to the COCO annotation file. It expects a .JSON file."
    )
    parser.add_argument(
        "--image-dir", 
        "-i", 
        type=str, 
        required=True, 
        help="Path to the directory where the images are stored."
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        type=str, 
        required=True, 
        help="Path to save the formatted annotations."
    )

    args = parser.parse_args()

    # Validate arguments.
    # --annotation-file
    if not os.path.isfile(path=args.annotation_file):
        raise "Invalid value for --annotation-file. It must be a real file with the COCO annotations."
    elif Path(args.annotation_file).suffix != ".json":
        raise "Invalid value for --annotation-file. It must be a .JSON file."

    # --image-dir
    if not os.path.isdir(s=args.image_dir):
        raise "Invalid value for --image-dir. It must be a real directory to store the images."

    # --output-dir
    if not os.path.isdir(s=args.output_dir):
        raise "Invalid value for --output-dir. It must be a real directory to store the formatted annotations."

    return args


def get_annotation(annot, categories, images, image_dir):
    """
    Get the annotation from the COCO dataset and return a new annotation 
    with the necessary information.

    Args:
        annot (dict): The annotation from COCO dataset.
        categories (dict): The categories from COCO dataset.
        images (dict): The images from COCO dataset.
        image_dir (str): The directory where the images are stored.

    Returns:
        Tuple[Annotation, Annotation]: Two annotations, one for the object and another for the super category.
    """
    # Get IDs.
    category_id = annot.get("category_id", None)
    image_id = annot.get("image_id", None)
    if not category_id or not image_id:
        raise ValueError("Invalid annotation %s." % annot)

    # Get image path.
    image = images.get(image_id, None)
    if not image:
        raise ValueError("Invalid image ID %s." % image_id)
    image_path = os.path.join(image_dir, image.get("file_name", ""))
    if not os.path.isfile(path=image_path):
        raise ValueError("Invalid image path %s." % image_path)

    # Get categories.
    category, super_category = categories.get(category_id, (None, None))
    if category is None or super_category is None:
        raise ValueError("Invalid category ID %s." % category_id)

    # Get bounding box.
    bbox = annot.get("bbox", [])
    if not bbox:
        raise ValueError("Invalid bounding box %s." % bbox)
    
    x, y, width, height = bbox
    cx = x + width / 2
    cy = y + height / 2
    bbox = Boxes(
        cx=cx,
        cy=cy,
        width=width,
        height=height
    )

    # Get annotations.
    category_annot = Annotation(
        text=category,
        image_id=image_id,
        image_filepath=image_path,
        category="Object",
        annotations=[bbox]
    )
    super_category_annot = Annotation(
        text=super_category,
        image_id=image_id,
        image_filepath=image_path,
        category="Object",
        annotations=[bbox]
    )

    return category_annot, super_category_annot


def main():
    # Get CLI arguments.
    args = cli_args()

    # Load the annotations.
    with open(file=args.annotation_file, mode="r") as file_buffer:
        annotations = json.load(fp=file_buffer)

    # Get categories.
    categories = annotations.get("categories", [])
    if not categories:
        raise ValueError("Invalid COCO annotation file. It must have a list of categories.")

    custom_categories = {}
    for category in categories:

        category_id = category.get("id", "")
        category_name = category.get("name", "").replace(" ", "_")
        super_category = category.get("supercategory", "").replace(" ", "_")

        if not category_id or not category_name in _ObjectCategory or not super_category in _ObjectCategory:
            raise ValueError("Invalid category %s." % category)

        custom_categories[category_id] = (category_name, super_category)

    # Get images.
    images = annotations.get("images", [])
    if not images:
        raise ValueError("Invalid COCO annotation file. It must have a list of images.")
    images = {image.get("id", None): image for image in images}

    # Get new annotations.
    new_annotations = []
    for annotation in tqdm(iterable=annotations.get("annotations", []), desc="Getting annotations..."):

        new_annot = get_annotation(
            annot=annotation, 
            categories=custom_categories, 
            images=images, 
            image_dir=args.image_dir
        )

        new_annotations.extend(new_annot)

    # Unify annotations.
    unified_annotations = {}
    for annotation in tqdm(iterable=new_annotations, desc="Unifying annotations based on image and it text prompt..."):

        # Add image ID.
        unified_annotations[annotation.image_id] = unified_annotations.get(annotation.image_id, {})

        # Add text.
        unified_annotations[annotation.image_id][annotation.text] = unified_annotations[annotation.image_id].get(
            annotation.text, 
            Annotation(
                text=annotation.text,
                image_id=annotation.image_id,
                image_filepath=annotation.image_filepath,
                category=annotation.category,
                annotations=[]
            )
        )

        # Add annotation.
        unified_annotations[annotation.image_id][annotation.text].annotations.extend(annotation.annotations)

    # Save annotations.
    warnings.warn(message="The content of the output directory will be deleted before saving the new annotations.")
    for file in os.listdir(path=args.output_dir):
        os.remove(path=os.path.join(args.output_dir, file))

    tqdm_annot = tqdm(iterable=unified_annotations.items())
    for image_id, annotations in tqdm_annot:

        for text, annot in annotations.items():
            tqdm_annot.set_description(desc="Saving %s annotation for %d image..." % (text, image_id))

            # Generate annotation ID file.
            while True:
                annot_id = str(uuid4()) + ".json"
                output_file = os.path.join(args.output_dir, annot_id)
                if not os.path.isfile(path=output_file):
                    break

            # Save annotation.
            with open(file=output_file, mode="w") as file_buffer:
                json.dump(obj=annot.model_dump(), fp=file_buffer, indent=4)


if __name__ == "__main__":
    main()
