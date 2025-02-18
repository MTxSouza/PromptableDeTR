"""
This module contains all data transformations and augmentations to be used in 
the training process.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
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


    # Methods.
    @abstractmethod
    def transform(self, sample):
        """
        Method to transform the samples.

        Args:
            sample (Sample): The sample to transform.

        Returns:
            Sample: The transformed sample.
        """
        pass


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
        samples = [self.transform(sample=sample) for sample in samples]

        return samples


    # Methods.
    def transform(self, sample):
        
        # Load the image.
        with Image.open(fp=sample.image_path, mode="r") as pil_img:
            np_img = np.asarray(a=pil_img, dtype=np.float32)
        sample.image = torch.from_numpy(np_img).permute(2, 0, 1)

        # Tokenize the caption.
        sample.caption_tokens = self.tokenizer.encode(texts=sample.caption)[0]

        # Transform the bounding box.
        bbox = sample.bbox
        bbox = torch.tensor(data=bbox, dtype=torch.float32)
        
        _, h, w = sample.image.shape
        bbox[:, 0::2] /= w
        bbox[:, 1::2] /= h
        sample.bbox_tensor = bbox

        return sample


class ReshapeImage(BaseTransform):


    # Special methods.
    def __init__(self, image_size):
        """
        This class reshapes the image to the desired size.

        Args:
            image_size (Tuple[int, int] | int): The size of the image.
        """
        # Check if the image size is an integer.
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size


    def __call__(self, samples):
        
        # Validate the samples.
        samples = self.validate_samples(samples=samples)

        # Prepare the samples.
        samples = [self.transform(sample=sample) for sample in samples]

        return samples


    # Methods.
    def transform(self, sample):

        # Check if the image has three dimensions.
        if len(sample.image.shape) == 3:
            sample.image = sample.image.unsqueeze(0)

        # Resize the image.
        sample.image = F.interpolate(
            input=sample.image, 
            size=self.image_size, 
            mode="bilinear", 
            align_corners=False
        ).squeeze(0)

        return sample
