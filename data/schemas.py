"""
This module stores all data schemas.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# Structures.
@dataclass
class Sample:
    """
    Sample structure.
    """
    image_path: str
    caption: str
    boxes: list

    image: Optional[torch.FloatTensor | np.ndarray] = None
    caption_tokens: Optional[torch.IntTensor | np.ndarray] = None
    boxes_tensor: Optional[torch.FloatTensor | np.ndarray] = None

@dataclass
class Boxes:
    """
    Structure for bounding boxes.
    """
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class Output:
    """
    Struture output for inference.
    """
    description: str
    boxes: list[Boxes]
