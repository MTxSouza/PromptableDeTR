"""
Utilities used to manipulate the raw data sample.
"""
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import tqdm


# Functions.
def save_sample(sample: dict, output_dir: str):
    """
    Save a sample in a directory.

    Args:
        sample (dict): Sample with the `Sample` structure.
        output_dir (str): Path to the directory to store the sample file.
    """
    while True:
        new_name = str(uuid.uuid4()) + ".json"
        out_path = os.path.join(output_dir, new_name)
        if not os.path.exists(path=out_path):
            break

    with open(file=out_path, mode="w") as f:
        json.dump(obj=sample, fp=f, indent=4)

def load_sample_directory(sample_directory: str) -> pd.DataFrame:
    """
    It loads all samples from a directory path and
    creates a pandas Dataframe of the entire content.

    Args:
        sample_directory (str): Path to the directory containing the sample files.

    Returns:
        pd.DataFrame: DataFrame containing all samples.
    """
    # Filter all samples.
    sample_fp_list = Path(sample_directory).glob(pattern="*.json")

    # Create Dataframe.
    db = []
    for sample_fp in tqdm.tqdm(iterable=sample_fp_list):

        with open(file=sample_fp, mode="r") as file_buffer:
            sample = json.load(fp=file_buffer)

        assert isinstance(sample["data"], list)

        # Check if the sample is reviewed. (More than one description per image)
        data = sample["data"]
        is_reviewed = len(data) > 1 and any([not len(sample_annot["boxes"]) for sample_annot in data])

        for sample_annot in data:

            # Compute boxes area.
            areas = [w * h for _, _, w, h in sample_annot["boxes"]]

            db.append({
                "sample_path": str(sample_fp),
                "image_path": sample["image_path"],
                "caption": sample_annot["caption"],
                "boxes": sample_annot["boxes"],
                "areas": areas,
                "num_objects": len(sample_annot["boxes"]),
                "is_reviewed": is_reviewed
            })

    db = pd.DataFrame(data=db)
    return db
