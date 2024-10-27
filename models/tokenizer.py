"""
This module contains the main tokenizer class for the project. It uses the BERT 
Tokenizer from the Hugging Face library, retrived with `torch.hub.load`.
"""
from abc import ABC, abstractmethod

import torch


# Classes.
class _BaseTokenizer(ABC):


    # Special methods.
    def __init__(self):
        """
        Initializes the _BaseTokenizer class. This is an abstract class and 
        should not be instantiated.
        """
        super(_BaseTokenizer, self).__init__()


    # Methods.
    @abstractmethod
    def tokenize(self, texts):
        """
        Tokenizes the input text.

        Args:
            text (List[str] | str): Text to be tokenized.

        Returns:
            List[List[str]] | List[str]: The tokenized text.
        """
        pass


    @abstractmethod
    def encode(self, texts):
        """
        Encodes the input text.

        Args:
            text (List[str] | str): Text to be encoded.

        Returns:
            torch.Tensor: The encoded text with dimension (batch_size, seq_len).
        """
        pass


class BertTokenizer(_BaseTokenizer):


    # Special methods.
    def __init__(self, model_name):
        """
        Initializes the BertTokenizer class.

        Args:
            model_name (str): The name of the BERT model.
        """
        super(BertTokenizer, self).__init__()

        # Load the tokenizer.
        self.__tokenizer = torch.hub.load(
            "huggingface/pytorch-transformers", 
            "tokenizer", 
            model_name
        )
    

    def __len__(self):
        """
        Returns the vocabulary size.

        Returns:
            int: The vocabulary size.
        """
        return self.vocab_size


    # Methods.
    def tokenize(self, texts):
        """
        Tokenizes the input text.

        Args:
            text (List[str] | str): Text to be tokenized.

        Returns:
            List[List[str]] | List[str]: The tokenized text.
        """
        # Check if it is a list of texts.
        if not isinstance(texts, list):
            texts = [texts]

        # Tokenize the text.
        str_tokens = [self.__tokenizer.tokenize(text=text) for text in texts]

        if len(str_tokens) == 1:
            return str_tokens[0]
        return str_tokens


    def encode(self, texts):
        """
        Encodes the input text.

        Args:
            text (List[str] | str): Text to be encoded.

        Returns:
            torch.Tensor: The encoded text with dimension (batch_size, seq_len).
        """
        return self.__tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True, 
            return_tensors="pt"
        )


    # Properties.
    @property
    def vocab_size(self):
        """
        Returns the vocabulary size.

        Returns:
            int: The vocabulary size.
        """
        return self.__tokenizer.vocab_size


    @property
    def pad_token(self):
        """
        Returns the padding token.

        Returns:
            str: The padding token.
        """
        return self.__tokenizer.pad_token


    @property
    def pad_token_id(self):
        """
        Returns the padding token id.

        Returns:
            int: The padding token id.
        """
        return self.__tokenizer.pad_token_id
