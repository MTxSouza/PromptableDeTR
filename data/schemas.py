"""
This module stores all data schemas.
"""
from dataclasses import dataclass


# Structures.
@dataclass
class Sample:
    """
    Sample structure.
    """
    sample_id: str
    image_path: str
    caption: str
    bbox: list
