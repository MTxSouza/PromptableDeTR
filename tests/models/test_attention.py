"""
This module tests the Attention class.
"""
from unittest import TestCase

import torch

from models.decoder import Attention, MultiHeadAttention


# Classes.
class TestAttetion(TestCase):


    # Tests.
    def test_init(self):
        """
        Tests the __init__ method.
        """
        # Create an Attention object.
        attention = Attention(
            base_embedding_dim=768,
            embedding_dim=768
        )

        # Assertions.
        self.assertEqual(attention._Attention__n_dim, 768)


    def test_attention(self):
        """
        Tests the forward method.
        """
        # Create an Attention object.
        attention = Attention(
            base_embedding_dim=768,
            embedding_dim=768
        )

        # Mock tensors.
        query = torch.randn(1, 196, 768)
        key = torch.randn(1, 196, 768)
        value = torch.randn(1, 196, 768)

        # Get the attention output.
        with torch.no_grad():
            output = attention(query, key, value)

        # Check the size of the output.
        self.assertEqual(output.size(), (1, 196, 768))


    def test_attention_with_different_dimensions(self):
        """
        Tests the forward method with different dimensions.
        """
        # Create an Attention object.
        attention = Attention(
            base_embedding_dim=768,
            embedding_dim=512
        )

        # Mock tensors.
        query = torch.randn(1, 196, 768)
        key = torch.randn(1, 196, 768)
        value = torch.randn(1, 196, 768)

        # Get the attention output.
        with torch.no_grad():
            output = attention(query, key, value)

        # Check the size of the output.
        self.assertEqual(output.size(), (1, 196, 512))


    def test_attention_distribution(self):
        """
        Tests the distribution of attention distribuition before 
        the softmax function. It expects a mean closes to zero and 
        a low variance, to avoid either the vanishing gradient problem 
        and one-hot encoding problem in attention matrix.
        """
        # Create an Attention object.
        attention = Attention(
            base_embedding_dim=768,
            embedding_dim=768
        )

        # Mock tensors.
        query = torch.randn(1, 196, 768)
        key = torch.randn(1, 196, 768)
        value = torch.randn(1, 196, 768)

        # Get the attention output.
        with torch.no_grad():
            _ = attention(query, key, value)

        # Check the attention distribution.
        assert_mean = torch.isclose(attention.score.mean(), torch.tensor(0.0), atol=1e-2)
        self.assertTrue(assert_mean)
        assert_var = torch.isclose(attention.score.var(), torch.tensor(0.1), atol=1e-1)
        self.assertTrue(assert_var)


    def test_attention_shape(self):
        """
        Tests the shape of the attention matrix.
        """
        # Create an Attention object.
        attention = Attention(
            base_embedding_dim=768,
            embedding_dim=768
        )

        # Mock tensors.
        query = torch.randn(1, 196, 768)
        key = torch.randn(1, 196, 768)
        value = torch.randn(1, 196, 768)

        # Get the attention output.
        with torch.no_grad():
            _ = attention(query, key, value)

        # Check the attention shape.
        self.assertEqual(attention.attention.size(), (1, 196, 196))


class TestMultiHeadAttention(TestCase):


    # Tests.
    def test_init(self):
        """
        Tests the __init__ method.
        """
        # Create a MultiHeadAttention object.
        MultiHeadAttention(
            embedding_dim=768,
            num_heads=8
        )


    def test_multi_head_attention(self):
        """
        Tests the forward method.
        """
        # Create a MultiHeadAttention object.
        attention = MultiHeadAttention(
            embedding_dim=768,
            num_heads=8
        )

        # Mock tensors.
        query = torch.randn(1, 196, 768)
        key = torch.randn(1, 196, 768)
        value = torch.randn(1, 196, 768)

        # Get the attention output.
        with torch.no_grad():
            output = attention(query, key, value)

        # Check the size of the output.
        self.assertEqual(output.size(), (1, 196, 768))
