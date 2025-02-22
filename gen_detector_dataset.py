"""
This script creates the .JSON files with the input text and their corresponding bounding boxes 
for an image. This dataset is used to train the object detection model in the second training.
"""
import argparse
import json
import os
import shutil
import uuid
import xml.etree.ElementTree as ET

from data.schemas import Sample


# Arguments.
def get_args():
    """
    Get the arguments from the command line.

    Returns:
        argparse.Namespace: The arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Create the .JSON files with the input text and their corresponding bounding boxes for an image.")
    parser.add_argument(
        "--caption-folder",
        "--caption",
        "-c",
        type=str,
        required=True,
        help="The path where the caption files is located."
    )
    parser.add_argument(
        "--bboxes-folder",
        "--bboxes",
        "-b",
        type=str,
        required=True,
        help="The path where the bounding boxes files are located."
    )
    parser.add_argument(
        "--image-folder",
        "--image",
        "-i",
        type=str,
        required=True,
        help="The path where the images are located."
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        "-o",
        type=str,
        required=True,
        help="The path to the output directory."
    )

    return parser.parse_args()


# Functions.
def load_caption_file(file_path):
    """
    Load the caption file and return its content formatted as a list of 
    dictionaries.

    Args:
        file_path (str): The path to the caption file.

    Returns:
        list[dict]: The content of the caption file.
    """
    # Load the caption file.
    with open(file=file_path, mode="r") as file:
        raw_sentences_data = file.read().split(sep="\n")
    
    # Format the caption file.
    sentences = []
    for sentence in raw_sentences_data:
        
        # Ignore empty lines.
        if not sentence:
            continue

        text_id = []
        text_list = []
        current_text = []
        add_to_text = False
        for token in sentence.split():
            
            if add_to_text:
                
                if token[-1] == "]":
                    add_to_text = False
                    current_text.append(token[:-1])
                    text_list.append(" ".join(current_text))
                    current_text = []

                else:
                    current_text.append(token)

            elif token[0] == "[":
                add_to_text = True
                parts = token.split("/")
                text_id.append(parts[1][3:])
        
        current_data = []
        for idx, text in zip(text_id, text_list):
            current_data.append({"id": idx, "text": text})

        sentences.append(current_data)

    return sentences


def load_bboxes_file(file_path):
    """
    Load the bounding boxes file and return its content formatted as a dictionary.

    Args:
        file_path (str): The path to the bounding boxes file.

    Returns:
        dict: The content of the bounding boxes file.
    """
    # Load the bounding boxes file.
    tree = ET.parse(source=file_path)
    root = tree.getroot()

    # Get the bounding boxes.
    bboxes = {}
    for obj in root.findall(path="object"):
        for names in obj.findall(path="name"):
            
            obj_id = names.text
            box_content = obj.findall(path="bndbox")

            # Ignore empty boxes.
            if not len(box_content):
                continue

            bboxes[obj_id] = bboxes.get(obj_id, [])
            bboxes[obj_id].append([
                int(box_content[0].findall(path="xmin")[0].text) - 1,
                int(box_content[0].findall(path="ymin")[0].text) - 1,
                int(box_content[0].findall(path="xmax")[0].text) - 1,
                int(box_content[0].findall(path="ymax")[0].text) - 1
            ])
    
    return bboxes


if __name__=="__main__":
    
    # Arguments.
    args = get_args()

    # Get content of each folder.
    bbox_file_list = os.listdir(path=args.bboxes_folder)
    caption_file_list = os.listdir(path=args.caption_folder)
    image_file_list = os.listdir(path=args.image_folder)
    assert len(bbox_file_list) == len(caption_file_list) == len(image_file_list), "The number of files in each folder must be the same."

    # Sort the files by name.
    sort_by_name = lambda name: int(name.split(sep=".")[0])
    bbox_file_list = sorted(bbox_file_list, key=sort_by_name)
    caption_file_list = sorted(caption_file_list, key=sort_by_name)
    image_file_list = sorted(image_file_list, key=sort_by_name)

    # Create the samples.
    print("Creating samples...")
    samples = []
    for bbox_file, cap_file, img_file in zip(bbox_file_list, caption_file_list, image_file_list):

        # Get sample ID.
        bbox_sample_id, _ = os.path.splitext(p=os.path.basename(p=bbox_file))
        cap_sample_id, _ = os.path.splitext(p=os.path.basename(p=cap_file))
        img_sample_id, _ = os.path.splitext(p=os.path.basename(p=img_file))
        assert bbox_sample_id == cap_sample_id == img_sample_id, "The sample IDs must be the same."

        bbox_file = os.path.join(args.bboxes_folder, bbox_file)
        cap_file = os.path.join(args.caption_folder, cap_file)

        # Get annotations.
        bbox = load_bboxes_file(file_path=bbox_file)
        captions = load_caption_file(file_path=cap_file)

        # Create the samples.
        for caption in captions:
            for data in caption:
                
                idx, cap = data["id"], data["text"]

                # Check if the bounding box is present.
                boxes = bbox.get(str(idx), [])
                if not boxes:
                    continue

                samples.append(Sample(
                    image_path=os.path.join(args.image_folder, img_file),
                    caption=cap.strip().lower(),
                    bbox=boxes
                ))
    
    # Save the samples.
    print("Saving samples...")
    if os.path.exists(path=args.output_dir):
        shutil.rmtree(path=args.output_dir)
    os.mkdir(path=args.output_dir)
    for sample in samples:

        while True:
            sample_id = str(uuid.uuid4())
            sample_path = os.path.join(args.output_dir, "%s.json" % sample_id)
            if not os.path.exists(path=sample_path):
                break
        with open(file=sample_path, mode="w") as file:
            json.dump(sample.__dict__, file, indent=4)

    print("Done.")
