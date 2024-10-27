"""
This module contains the Encoder class of the model. It uses the BERT model from 
the Hugging Face library, retrived with `torch.hub.load` to process the input text.
"""
import torch
import torch.nn as nn


# Classes.
class TextEncoder(nn.Module):


    # Special methods.
    def __init__(self, model_name):
        """
        Initializes the TextEncoder class.

        Args:
            model_name (str): The name of the model to be used.
        """
        super(TextEncoder, self).__init__()

        # Load the model.
        self.__bert_model = torch.hub.load("huggingface/pytorch-transformers", "model", model_name)

        # Freeze the model.
        for param in self.__bert_model.parameters():
            param.requires_grad = False


    # Methods.
    def forward(self, tokens):
        """
        Encodes the input tokens.

        Args:
            tokens (torch.Tensor): The input tokens.

        Returns:
            torch.Tensor: The encoded tokens.
        """
        # Get the output of the model.
        with torch.no_grad():
            output = self.__bert_model(tokens)

        # Get the last hidden state.
        return output.last_hidden_state
