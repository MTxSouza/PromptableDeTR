"""
This module stores all data schemas.
"""
from dataclasses import dataclass
from typing import List, Optional

try:
    from typing import LiteralString
except ImportError:
    from typing_extensions import LiteralString

import numpy as np
import torch


# Structures.
@dataclass
class ObjectAnnotation:
    """
    Object structure that stores all annotation of a specific object.
    """
    caption: LiteralString
    bbox: List[List[int]]


@dataclass
class Sample:
    """
    Sample structure that stores all annotation of a specific image.
    """
    image_path: LiteralString
    captions: List
    objects: List[ObjectAnnotation]


@dataclass
class DetectorSample:
    """
    Sample structure.
    """
    image_path: str
    caption: str
    bbox: list

    image: Optional[torch.FloatTensor | np.ndarray] = None
    caption_tokens: Optional[torch.IntTensor | np.ndarray] = None
    bbox_tensor: Optional[torch.FloatTensor | np.ndarray] = None


@dataclass
class AlignerSample:
    """
    Sample structure.
    """
    image_path: str
    caption: str

    image: Optional[torch.FloatTensor | np.ndarray] = None
    caption_tokens: Optional[torch.IntTensor | np.ndarray] = None
    masked_caption_tokens: Optional[torch.IntTensor | np.ndarray] = None
