"""
This module contains the main data loader class used to load the data 
to be used in the training process.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data.daug import PrepareSample
from data.schemas import Sample


# Classes.
class PromptableDeTRDataLoader:
    """
    Data loader class for the Promptable DeTR model.
    """


    # Class methods.
    @classmethod
    def get_samples_from_dir(cls, dirpath):
        """
        Get all valid samples from the directory.

        Args:
            dirpath (str): The path to the directory containing the samples.

        Returns:
            list: A list of valid sample file paths.
        """
        # Get the samples from the directory.
        samples = []
        for file in os.listdir(path=dirpath):

            # Skip non-JSON files.
            if not file.endswith(".json"):
                continue

            # Load the samples.
            sample_file = os.path.join(dirpath, file)
            with open(file=sample_file, mode="r") as f:
                raw_sample = json.load(fp=f)

            # Check if the samples are valid.
            Sample(**raw_sample)
            del raw_sample

            # Append the samples.
            samples.append(sample_file)
            
        return samples


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

            # Check if the samples are valid.
            Sample(**raw_sample)
            del raw_sample

            # Append the samples.
            samples.append(sample_file)
        
        # Shuffle the samples.
        if shuffle_samples:
            np.random.seed(seed=seed)
            samples = np.random.permutation(samples).tolist()
        
        # Get the split index.
        split_index = int(len(samples) * val_split)
        val_samples = samples[:split_index]
        train_samples = samples[split_index:]

        return train_samples, val_samples


    # Static methods.
    @staticmethod
    def convert_batch_into_tensor(batch, max_len = 500, pad_value = 0):
        """
        Convert a list of AlignerSample objects into tensors.

        Args:
            batch (List[AlignerSample]): The batch of samples.
            max_len (int): Maximum number of context length and object predictions. (Default: 100)
            pad_value (int): The padding value. (Default: 0)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: The image, caption, mask to occlude the caption and the extra data needed for the model.
        """
        # Standardize the captions length.
        tensor_captions = None
        masked_captions_tensor = None
        mask_tensors = None
        tensor_objects = None
        for sample in batch:

            caption_tokens = sample.caption_tokens

            if caption_tokens.size(0) != max_len:

                # Pad the caption tokens.
                pad_len = max_len - caption_tokens.size(0)
                caption_tokens = F.pad(input=caption_tokens, pad=(0, pad_len), value=pad_value)

            caption_tokens = caption_tokens.unsqueeze(dim=0)
            mask = torch.ones_like(input=caption_tokens, dtype=torch.int32)
            mask[caption_tokens == pad_value] = 0

            if tensor_captions is None:
                tensor_captions = caption_tokens
                mask_tensors = mask
            else:
                tensor_captions = torch.cat(tensors=(tensor_captions, caption_tokens), dim=0)
                mask_tensors = torch.cat(tensors=(mask_tensors, mask), dim=0)

        # Concatenate the image tensors.
        tensor_images = torch.cat(tensors=[sample.image.unsqueeze(dim=0) for sample in batch], dim=0)

        # Standardize the objects length.
        for sample in batch:
            
            bbox_tensor = sample.bbox_tensor

            # Add presence column.
            presence = torch.ones(size=(bbox_tensor.size(0), 1), dtype=torch.float32)
            bbox_tensor = torch.cat(tensors=[bbox_tensor, presence], dim=-1)

            if bbox_tensor.size(0) != max_len:

                # Pad the objects.
                pad_len = max_len - bbox_tensor.size(0)
                bbox_tensor = F.pad(input=bbox_tensor, pad=(0, 0, 0, pad_len), value=pad_value)

            bbox_tensor = bbox_tensor.unsqueeze(dim=0)
            if tensor_objects is None:
                tensor_objects = bbox_tensor
            else:
                tensor_objects = torch.cat(tensors=(tensor_objects, bbox_tensor), dim=0)
        
        # Create the extra data.
        extra_data = {
            "masked_caption": masked_captions_tensor, 
            "bbox": tensor_objects
        }
        
        return tensor_images, tensor_captions, mask_tensors, extra_data


    # Special methods.
    def __init__(
            self, 
            sample_file_paths, 
            image_directory,
            batch_size, 
            transformations = None, 
            shuffle = True, 
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
            seed (int): The seed for the random number generator. (Default: 42)
        """

        # Compute the number of batches.
        self.num_batches = len(sample_file_paths) // batch_size + (len(sample_file_paths) % batch_size)

        # Check transformations.
        if transformations is None:
            raise ValueError("Transformations must be specified.")

        if not isinstance(transformations[0], PrepareSample):
            raise ValueError("Transformations must be a list containing the PrepareSample class.")

        # Shuffle the samples.
        if shuffle:
            np.random.seed(seed=seed)
            sample_file_paths = np.random.permutation(sample_file_paths).tolist()

        # Attributes.
        self.sample_file_paths = sample_file_paths
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformations = transformations
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

                # Define sample.
                sample = Sample(**raw_sample)

                # Append the samples.
                curr_samples.append(sample)
            
            # Apply transformations.
            for transform in self.transformations:
                curr_samples = transform(samples=curr_samples)

            yield curr_samples


    # Methods.
    def get_tokenizer(self):
        """
        Tries to get the tokenizer from the transformations.

        Returns:
            Tokenizer | None: The tokenizer object.
        """
        for transform in self.transformations:
            if isinstance(transform, PrepareSample):
                return transform.caption_transform.tokenizer
        return None
