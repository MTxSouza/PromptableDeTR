"""
This module tests the Embedding class.
"""
from unittest import TestCase

import torch

from models.embedding import ImageEmbedding


# Classes.
class TestImageEmbedding(TestCase):
    """
    This class tests the ImageEmbedding class.
    """

    # Tests.
    def test_init(self):
        """
        Tests the __init__ method.
        """
        # Create an ImageEmbedding object.
        image_embedding = ImageEmbedding(
            image_size=224,
            num_patches=14,
            padding_idx=0
        )

        # Assertions.
        self.assertEqual(image_embedding.num_embedding, 196)
        self.assertEqual(image_embedding.embedding_dim, 768)
        self.assertEqual(image_embedding.padding_idx, 0)


    def test_image_embedding_size(self):
        """
        Tests the size of image embedding.
        """
        # Mock image.
        image = torch.randn(1, 3, 224, 224)

        # Create an ImageEmbedding object.
        image_embedding = ImageEmbedding(
            image_size=image.size(2),
            num_patches=14,
            padding_idx=0
        )

        # Get the image embedding.
        with torch.no_grad():
            emb = image_embedding(image)

        # Check the size of the embedding.
        self.assertEqual(emb.size(), (1, 196, 768))


    def test_multi_image_embedding_size(self):
        """
        Tests the size of image embedding.
        """
        # Mock image.
        image = torch.randn(8, 3, 224, 224)

        # Create an ImageEmbedding object.
        image_embedding = ImageEmbedding(
            image_size=image.size(2),
            num_patches=14,
            padding_idx=0
        )

        # Get the image embedding.
        with torch.no_grad():
            emb = image_embedding(image)

        # Check the size of the embedding.
        self.assertEqual(emb.size(), (8, 196, 768))
