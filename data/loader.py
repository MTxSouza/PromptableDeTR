"""
This module contains the main data loader class used to load the data 
to be used in the training process.
"""
import itertools
import json
import multiprocessing
import os
import random

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
    def get_samples_from_dir(cls, dirpath, max_obj = None):
        """
        Get all valid samples from the directory.

        Args:
            dirpath (str): The path to the directory containing the samples.
            max_obj (int | None): Maximum number of objects to consider a sample valid. (Default: None)

        Returns:
            list: A list of valid sample file paths.
        """
        # Get contento of directory.
        dir_data = os.listdir(path=dirpath)
        dir_data = [os.path.join(dirpath, filename) for filename in dir_data]

        # Create chunk to load samples in parallel.
        n_cpu = os.cpu_count()
        dir_data = np.array_split(ary=dir_data, indices_or_sections=n_cpu)
        dir_data = [(chunk.tolist(), max_obj) for chunk in dir_data]

        # Create pool to load samples in parallel.
        pool = multiprocessing.Pool(processes=n_cpu)
        samples = pool.starmap(PromptableDeTRDataLoader.get_samples, dir_data)
        pool.close()
        pool.join()

        # Flatten the list of samples.
        samples = itertools.chain.from_iterable(samples)
        samples = list(samples)

        return samples

    # Static methods.
    @staticmethod
    def get_samples(data, max_obj = None):
        samples = []
        for sample_file in data:

            # Skip non-JSON files.
            if not sample_file.endswith(".json"):
                continue

            # Load the samples.
            with open(file=sample_file, mode="r") as f:
                raw_sample = json.load(fp=f)

            # Check if the samples are valid.
            obj = Sample(**raw_sample)
            del raw_sample

            # Filter samples.
            if max_obj is not None and len(obj.boxes) > max_obj:
                continue

            # Append the samples.
            samples.append(sample_file)
            
        return samples

    @staticmethod
    def convert_batch_into_tensor(batch, max_queries = 10, pad_value = 0):
        """
        Convert a list of Sample objects into tensors.

        Args:
            batch (List[Sample]): The batch of samples.
            max_queries (int): Maximum number of object predictions. (Default: 10)
            pad_value (int): The padding value. (Default: 0)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]: The image, caption, mask to occlude the caption and the extra data needed for the model.
        """
        # Get the maximum caption length.
        max_len = max(sample.caption_tokens.size(0) for sample in batch)

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

            boxes_tensor = sample.boxes_tensor

            # Check if there are no boxes.
            if boxes_tensor.size(0) == 0:
                boxes_tensor = torch.zeros(size=(0, 5), dtype=torch.float32)
            else:
                # Add presence column.
                presence = torch.ones(size=(boxes_tensor.size(0), 1), dtype=torch.float32)
                boxes_tensor = torch.cat(tensors=[boxes_tensor, presence], dim=-1)

            if boxes_tensor.size(0) != max_queries:

                # Pad the objects.
                pad_len = max_queries - boxes_tensor.size(0)
                boxes_tensor = F.pad(input=boxes_tensor, pad=(0, 0, 0, pad_len), value=pad_value)

            boxes_tensor = boxes_tensor.unsqueeze(dim=0)
            if tensor_objects is None:
                tensor_objects = boxes_tensor
            else:
                tensor_objects = torch.cat(tensors=(tensor_objects, boxes_tensor), dim=0)

        # Create the extra data.
        extra_data = {
            "masked_caption": masked_captions_tensor, 
            "boxes": tensor_objects
        }

        return tensor_images, tensor_captions, mask_tensors, extra_data

    # Special methods.
    def __init__(
            self, 
            sample_file_paths, 
            batch_size, 
            transformations = None, 
            use_sample_weights = False, 
            shuffle = True, 
            seed = 42
        ):
        """
        Initialize the data loader class.

        Args:
            sample_file_paths (list): The list of sample file paths.
            batch_size (int): The batch size.
            transformations (List[BaseTransform]): The transforms to apply to the data. (Default: None)
            use_sample_weights (bool): Whether to use sample weights. (Default: False)
            shuffle (bool): Whether to shuffle the samples. (Default: True)
            seed (int): The seed for the random number generator. (Default: 42)
        """

        # Check transformations.
        if transformations is None:
            raise ValueError("Transformations must be specified.")

        if not isinstance(transformations[0], PrepareSample):
            raise ValueError("Transformations must be a list containing the PrepareSample class.")

        # Organize sets.
        self.dataset = {dirpath: (samples, weight) for dirpath, samples, weight in sample_file_paths}

        # Normalize weights.
        total_weight = sum(weight for _, weight in self.dataset.values())
        self.dataset = {dirpath: (samples, int(batch_size * (weight / total_weight))) for dirpath, (samples, weight) in self.dataset.items()}
        self.full_dataset = []
        print("-" * 20)
        print("Datasets:")
        for dirpath, (samples, n_samples) in self.dataset.items():
            self.full_dataset.extend(samples)
            print(f"\t{dirpath} | Samples: {len(samples)} - Num Samples per batch: {n_samples}")
        print("-" * 20)

        # Shuffle dataset.
        if shuffle:
            random.seed(a=seed)
            random.shuffle(self.full_dataset)

        # Compute the number of batches.
        self.num_batches = sum(len(samples) for samples, _ in self.dataset.values()) // batch_size + 1

        # Attributes.
        self.current_index = 0
        self.max_index = len(self.full_dataset) // batch_size
        self.batch_size = batch_size
        self.use_sample_weights = use_sample_weights
        self.shuffle = shuffle
        self.transformations = transformations
        self.seed = seed

    def __len__(self):
        """
        Returns the number of batches in the data loader.
        """
        return self.num_batches

    def __next__(self):
        """
        Returns the next batch of samples.
        """
        return next(iter(self))

    def __iter__(self):
        """
        Returns the iterator object.
        """
        # Get samples based on weights.
        if self.use_sample_weights:
            sample_file_paths = []
            for samples, n_samples in self.dataset.values():
                if n_samples > len(samples):
                    n_samples = len(samples)
                selected_samples = random.sample(population=samples, k=n_samples)
                sample_file_paths += selected_samples

        # Get samples normally.
        else:
            sample_file_paths = self.full_dataset[self.current_index*self.batch_size:(self.current_index+1)*self.batch_size]
            self.current_index += 1
            if self.current_index >= self.max_index:
                self.current_index = 0

        curr_samples = []
        for curr_sample_file in sample_file_paths:

            # Load the samples.
            with open(file=curr_sample_file, mode="r") as f:
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
