"""
This module stores all data schemas.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# Structures.
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
    caption_list: str

    image: Optional[torch.FloatTensor | np.ndarray] = None
    caption_tokens: Optional[torch.IntTensor | np.ndarray] = None
