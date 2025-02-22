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


class PrepareImage(BaseTransform):
    """
    This class prepares the image for training.
    """


    # Special methods.
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

        # Normalize the image.
        if sample.image.max() > 1:
            sample.image /= 255

        return sample


class PrepareCaption(BaseTransform):
    """
    This class prepares the caption for training.
    """


    # Special methods.
    def __init__(self, vocab_file):
        """
        Initialize the class.

        Args:
            vocab_file (str): The path to the vocabulary file.
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

        # Tokenize the caption.
        sample.caption_tokens = self.tokenizer.encode(texts=sample.caption)[0]

        return sample


class PrepareBBox(BaseTransform):
    """
    This class prepares the bounding box for training.
    """


    # Special methods.
    def __call__(self, samples):

        # Validate the samples.
        samples = self.validate_samples(samples=samples)

        # Prepare the samples.
        samples = [self.transform(sample=sample) for sample in samples]

        return samples


    # Methods.
    def transform(self, sample):

        # Transform the bounding box.
        bbox = sample.bbox
        bbox = torch.tensor(data=bbox, dtype=torch.float32)

        _, h, w = sample.image.shape
        bbox[:, 0::2] /= w
        bbox[:, 1::2] /= h
        sample.bbox_tensor = bbox

        return sample


class PrepareAlignerSample(BaseTransform):


    # Special methods.
    def __init__(self, vocab_file):
        """
        This class prepares the raw sample for training loading the 
        image and transforming the caption.

        Args:
            vocab (Vocab): File path to the vocabulary of the model.
        """
        
        # Define the transformations.
        self.image_transform = PrepareImage()
        self.caption_transform = PrepareCaption(vocab_file=vocab_file)


    def __call__(self, samples):
        
        # Validate the samples.
        samples = self.validate_samples(samples=samples)

        # Prepare the samples.
        samples = [self.transform(sample=sample) for sample in samples]

        return samples


    # Methods.
    def transform(self, sample):
        
        # Prepare the image.
        sample = self.image_transform.transform(sample=sample)

        # Prepare the caption.
        sample = self.caption_transform.transform(sample=sample)

        return sample


class PrepareDetectionSample(PrepareAlignerSample):


    # Special methods.
    def __init__(self, vocab_file):
        """
        This class prepares the raw sample for training loading the
        image, transforming the caption, and preparing the bounding box.

        Args:
            vocab_file (str): The path to the vocabulary file
        """
        super().__init__(vocab_file=vocab_file)

        # Define the transformations.
        self.bbox_transform = PrepareBBox()


    # Methods.
    def transform(self, sample):
        
        # Prepare the image and caption.
        sample = super().transform(sample=sample)

        # Prepare the bounding box.
        sample = self.bbox_transform.transform(sample=sample)

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
