"""
This module contains all data transformations and augmentations to be used in 
the training process.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image

from data.schemas import Sample
from models.tokenizer import Tokenizer


# Classes.
class BaseTransform(ABC):
    """
    Base class for all transformations.
    """


    # Special methods.
    @abstractmethod
    def __call__(self, samples):
        """
        Call the transformation.

        Args:
            samples (Sample | List[Sample]): The sample or list of samples to transform.
        """
        pass


    # Private methods.
    @abstractmethod
    def __transform(self, sample):
        """
        Method to transform the samples.

        Args:
            sample (Sample): The sample to transform.

        Returns:
            Sample: The transformed sample.
        """
        pass


    # Methods.
    def validate_samples(self, samples):
        """
        Validate the samples.

        Args:
            samples (Sample | List[Sample]): The sample or list of samples.

        Returns:
            List[Sample]: The list of samples.
        """
        # Check if the samples are a list.
        if not isinstance(samples, list):
            samples = [samples]

        # Check if all objects are samples.
        for sample in samples:
            if not isinstance(sample, Sample):
                raise ValueError("All objects in the list must be of type Sample.")

        return samples



class PrepareRawSample(BaseTransform):


    # Special methods.
    def __init__(self, vocab_file):
        """
        This class prepares the raw sample for training loading the 
        image and transforming the caption and bounding box.

        Args:
            vocab (Vocab): File path to the vocabulary of the model.
        """
        # Load the tokenizer.
        self.tokenizer = Tokenizer(vocab_filepath=vocab_file)


    def __call__(self, samples):
        
        # Validate the samples.
        samples = self.validate_samples(samples=samples)

        # Prepare the samples.
        samples = [self.__transform(sample=sample) for sample in samples]

        return samples


    # Private methods.
    def __transform(self, sample):
        
        # Load the image.
        with Image.open(fp=sample.image_path, mode="r") as pil_img:
            np_img = np.asarray(a=pil_img, dtype=np.float32)
        sample.image = torch.from_numpy(ndarray=np_img).permute(2, 0, 1)

        # Tokenize the caption.
        sample.caption_tokens = self.tokenizer.encode(text=sample.caption)[0]

        # Transform the bounding box.
        bbox = sample.bbox
        bbox = torch.tensor(data=bbox, dtype=torch.float32)
        
        _, h, w = sample.image.shape
        bbox[:, 0::2] /= w
        bbox[:, 1::2] /= h
        sample.bbox_tensor = bbox

        return sample
