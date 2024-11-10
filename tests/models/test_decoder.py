"""
This module tests the Decoder class.
"""
from unittest import TestCase

import torch

from models.decoder import Decoder, DecoderBlock


# Classes.
class TestDecoderBlock(TestCase):


    # Tests.
    def test_init(self):
        """
        Case: test the initialization of the DecoderBlock class.
        """
        # Initialization.
        DecoderBlock(
            num_heads=8,
            hidden_dim=2048,
            embedding_dim=256
        )


    def test_forward(self):
        """
        Case: test the forward method of the DecoderBlock class.
        """
        # Create a DecoderBlock object.
        decoder_block = DecoderBlock(
            num_heads=8,
            hidden_dim=2048,
            embedding_dim=768
        )

        # Mock tensors.
        image_embedding = torch.randn(1, 256, 768)
        text_embedding = torch.randn(1, 256, 768)

        # Get the embedding output.
        with torch.no_grad():
            output = decoder_block(image_embedding, text_embedding)

        # Assertions.
        self.assertEqual(output.shape, (1, 256, 768))


class TestDecoder(TestCase):


    # Tests.
    def test_init(self):
        """
        Case: test the initialization of the Decoder class.
        """
        # Initialization.
        Decoder(
            num_blocks=6,
            num_heads=8,
            hidden_dim=2048,
            embedding_dim=256
        )


    def test_forward(self):
        """
        Tests the forward method.
        """
        # Create a Decoder object.
        decoder = Decoder(
            num_blocks=1,
            num_heads=8,
            hidden_dim=2048,
            embedding_dim=768
        )

        # Mock tensors.
        image_embedding = torch.randn(1, 256, 768)
        text_embedding = torch.randn(1, 256, 768)

        # Get the embedding output.
        with torch.no_grad():
            output = decoder(image_embedding, text_embedding)

        # Assertions.
        self.assertEqual(output.shape, (1, 256, 768))


    def test_inf_number_output(self):
        """
        Tests the infinite gradient problem.
        """
        # Create a Decoder object.
        decoder = Decoder(
            num_blocks=8,
            num_heads=8,
            hidden_dim=2048,
            embedding_dim=768
        )

        # Mock tensors.
        image_embedding = torch.randn(1, 256, 768)
        text_embedding = torch.randn(1, 256, 768)

        # Get the embedding output.
        with torch.no_grad():
            output = decoder(image_embedding, text_embedding)

        # Assertions.
        self.assertFalse(torch.isinf(output).any())
