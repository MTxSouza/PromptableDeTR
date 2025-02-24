"""
This module contains all data transformations and augmentations to be used in 
the training process.
"""
import random
from abc import ABC, abstractmethod

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.schemas import AlignerSample, DetectorSample
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
            if not isinstance(sample, (AlignerSample, DetectorSample)):
                raise ValueError("All objects in the list must be either AlignerSample or DetectorSample.")

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

        # Convert the tokens to tensor.
        sample.caption_tokens = torch.tensor(data=sample.caption_tokens, dtype=torch.int64)

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
        sample.masked_caption_tokens = sample.caption_tokens.clone()

        return sample


class PrepareDetectionSample(BaseTransform):


    # Special methods.
    def __init__(self, vocab_file):
        """
        This class prepares the raw sample for training loading the
        image, transforming the caption, and preparing the bounding box.

        Args:
            vocab_file (str): The path to the vocabulary file
        """

        # Define the transformations.
        self.image_transform = PrepareImage()
        self.caption_transform = PrepareCaption(vocab_file=vocab_file)
        self.bbox_transform = PrepareBBox()


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

        # Prepare the bounding box.
        sample = self.bbox_transform.transform(sample=sample)

        return sample


class MaskCaption(BaseTransform):


    # Special methods.
    def __init__(self, vocab_file, mask_token, mask_ratio):
        """
        This class masks the caption.

        Args:
            vocab_file (str): The path to the vocabulary file
            mask_token (int): The token to use for masking.
            mask_ratio (float): The ratio of the caption to mask.
        """
        # Load the tokenizer.
        self.tokenizer = Tokenizer(vocab_filepath=vocab_file)

        self.mask_token = mask_token
        self.mask_ratio = mask_ratio


    def __call__(self, samples):
        
        # Validate the samples.
        samples = self.validate_samples(samples=samples)

        # Prepare the samples.
        samples = [self.transform(sample=sample) for sample in samples]

        return samples


    # Private methods.
    def __get_tokens_to_mask(self, sample):
        """
        This method finds all nouns and verbs in the caption to randomly 
        mask them.

        Args:
            sample (AlignerSample): The sample to mask.

        Returns:
            List[List[int]]: The indices of the tokens to mask.
        """
        # Tokenize the caption.
        caption = sample.caption
        str_tokens = self.tokenizer.encode_str(text=caption)
        nltk_tokens = nltk.word_tokenize(text=caption)
        nltk_tokens = nltk.pos_tag(tokens=nltk_tokens)

        # Get nouns and verbs.
        tgt_words = set()
        for token, tag in nltk_tokens:
            if tag in ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                tgt_words.add(token)
        
        # Get the indices of the tokens to mask.
        mask_indices = []
        max_size = sample.caption_tokens.size(0) - 1
        start_index = 1
        end_index = max_size
        while start_index < max_size:

            token = "".join(str_tokens[start_index:end_index])
            if token in tgt_words:
                mask_indices.append(sample.caption_tokens[start_index:end_index].tolist())
                start_index = end_index
                end_index = max_size
                continue

            else:
                end_index -= 1

            if end_index == start_index:
                start_index += 1
                end_index = max_size

        return mask_indices


    # Methods.
    def transform(self, sample):

        # Get the tokens to mask.
        mask_indices = self.__get_tokens_to_mask(sample=sample)

        # Get the number of tokens to mask.
        num_tokens = int(self.mask_ratio * len(mask_indices))
        random_tokens = random.choices(population=mask_indices, k=num_tokens)

        # Mask the caption.
        for tokens in random_tokens:
            if isinstance(tokens, int):
                tokens = [tokens]
            for token in tokens:
                mask_filter = sample.masked_caption_tokens == token
                sample.masked_caption_tokens[mask_filter] = self.mask_token

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
