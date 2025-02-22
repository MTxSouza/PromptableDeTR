"""
This module contains the main data loader class used to load the data 
to be used in the training process.
"""
import json
import os

import numpy as np

from data.daug import PrepareAlignerSample, PrepareDetectionSample
from data.schemas import AlignerSample, DetectorSample, Sample


# Classes.
class PromptableDeTRDataLoader:
    """
    Data loader class for the Promptable DeTR model.
    """


    # Class methods.
    @classmethod
    def get_train_val_split(cls, sample_directory, val_split = 0.2, shuffle_samples = True, seed = 42):
        """
        Get the train and validation split from the sample directory.

        Args:
            sample_directory (str): The path to the sample directory.
            val_split (float): The validation split. (Default: 0.2)
            shuffle_samples (bool): Whether to shuffle the samples. (Default: True)
            seed (int): The seed for the random number generator. (Default: 42)

        Returns:
            list, list: The train and validation samples.
        """
        # Get the samples from the directory.
        samples = []
        for file in os.listdir(path=sample_directory):

            # Skip non-JSON files.
            if not file.endswith(".json"):
                continue

            # Load the samples.
            sample_file = os.path.join(sample_directory, file)
            with open(file=sample_file, mode="r") as f:
                raw_sample = json.load(fp=f)
                raw_sample["image_path"] = raw_sample.pop("image_name")

            # Check if the samples are valid.
            Sample(**raw_sample)
            del raw_sample

            # Append the samples.
            samples.append(sample_file)
        
        # Shuffle the samples.
        if shuffle_samples:
            np.random.seed(seed=seed)
            samples = np.random.permutation(x=samples).tolist()
        
        # Get the split index.
        split_index = int(len(samples) * val_split)
        val_samples = samples[:split_index]
        train_samples = samples[split_index:]

        return train_samples, val_samples


    # Special methods.
    def __init__(
            self, 
            sample_file_paths, 
            image_directory,
            batch_size, 
            transformations = None, 
            shuffle = True, 
            aligner = False, 
            seed = 42
        ):
        """
        Initialize the data loader class.

        Args:
            sample_file_paths (list): The list of sample file paths.
            image_directory (str): The path to the image directory where the images are stored.
            batch_size (int): The batch size.
            transformations (List[BaseTransform]): The transforms to apply to the data. (Default: None)
            shuffle (bool): Whether to shuffle the samples. (Default: True)
            aligner (bool): Whether to use the aligner model. (Default: False)
            seed (int): The seed for the random number generator. (Default: 42)
        """

        # Compute the number of batches.
        self.num_batches = len(sample_file_paths) // batch_size + (len(sample_file_paths) % batch_size)

        # Check transformations.
        if transformations is None:
            raise ValueError("Transformations must be specified.")

        if aligner:
            if not isinstance(transformations[0], PrepareAlignerSample):
                raise ValueError("Transformations must be a list containing the PrepareAlignerSample class.")
        elif not isinstance(transformations[0], PrepareDetectionSample):
            raise ValueError("Transformations must be a list containing the PrepareDetectionSample class.")

        # Shuffle the samples.
        if shuffle:
            np.random.seed(seed=seed)
            sample_file_paths = np.random.permutation(x=sample_file_paths).tolist()

        # Attributes.
        self.sample_file_paths = sample_file_paths
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformations = transformations
        self.aligner = aligner
        self.seed = seed


    def __len__(self):
        """
        Returns the number of batches in the data loader.
        """
        return self.num_batches


    def __iter__(self):
        """
        Returns the iterator object.
        """
        for i in range(0, len(self.sample_file_paths), self.batch_size):

            curr_sample_files = self.sample_file_paths[i:i + self.batch_size]
            curr_samples = []
            for file in curr_sample_files:

                # Load the samples.
                with open(file=file, mode="r") as f:
                    raw_sample = json.load(fp=f)
                    raw_sample["image_path"] = os.path.join(self.image_directory, raw_sample.pop("image_name"))

                # Load the samples for a specific model.
                if self.aligner:
                    # Get all captions.
                    raw_sample["caption_list"] = raw_sample.pop("captions")
                    del raw_sample["objects"]

                    # Define sample.
                    sample = AlignerSample(**raw_sample)

                else:
                    # Get random object.
                    n_objs = len(raw_sample["objects"])
                    curr_obj = raw_sample["objects"][np.random.randint(0, n_objs)]

                    # Get current caption.
                    raw_sample["caption"] = curr_obj["text"]
                    raw_sample["bbox"] = curr_obj["bbox"]
                    del raw_sample["objects"], raw_sample["captions"]

                    # Define sample.
                    sample = DetectorSample(**raw_sample)

                # Append the samples.
                curr_samples.append(sample)
            
            # Apply transformations.
            for transform in self.transformations:
                curr_samples = transform(samples=curr_samples)

            yield curr_samples
