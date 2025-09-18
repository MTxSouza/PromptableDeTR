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
