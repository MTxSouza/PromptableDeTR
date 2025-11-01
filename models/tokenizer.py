"""
Main module for the Tokenizer class that is used to encode and decode text data. It uses 
the MobileBERT vocabulary to encode and decode text data.
"""
import argparse
import os
import re
import string
import unicodedata
import warnings
from enum import Enum


# Functions.
def get_args():
    """
    Get the arguments passed to the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    # Define arguments.
    parser = argparse.ArgumentParser(description="Encode and decode text data.")
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to the vocabulary file."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, World!",
        help="Text data to encode."
    )

    # Parse arguments.
    return parser.parse_args()


# Enums.
class SpecialTokens(Enum):
    """
    Enum class that contains the special tokens used by the MobileBERT model.
    """
    PAD = ("[PAD]", 0)
    UNK = ("[UNK]", 100)
    CLS = ("[CLS]", 101)
    SEP = ("[SEP]", 102)
    MASK = ("[MASK]", 103)


    # Methods.
    @classmethod
    def get_special_tokens_identifier(cls):
        """
        Get a regex pattern that tries to find if a text already contains special tokens.

        Returns:
            re.Pattern: Compiled regex pattern that tries to find special tokens in a text.
        """
        # Get special tokens.
        special_tokens = "|".join([token.value[0] for token in cls])
        special_tokens = special_tokens.replace("[", r"\[").replace("]", r"\]")

        # Create regex pattern.
        return re.compile(pattern=r"%s" % special_tokens)


# Classes.
class Tokenizer:


    # Special methods.
    def __init__(self, vocab_filepath):
        """
        Main class used to encode and decode text data based on the MobileBERT vocabulary.

        Args:
            vocab_filepath (str): Path to the vocabulary file.
        """

        # Check if the vocabulary file exists.
        if not os.path.exists(path=vocab_filepath):
            raise FileNotFoundError("Vocabulary file not found.")

        # Check if the vocabulary file is a file.
        if not os.path.isfile(path=vocab_filepath):
            raise FileNotFoundError("Vocabulary file is not a file.")

        # Load the vocabulary file.
        self.token_to_index = self.__load_vocab_file(vocab_filepath=vocab_filepath)
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}

        self.special_tokens_regex = SpecialTokens.get_special_tokens_identifier()


    def __len__(self):
        """
        Get the number of tokens in the vocabulary.

        Returns:
            int: Number of tokens in the vocabulary.
        """
        return len(self.token_to_index) - 1 # BUG: The number of tokens in vocabulary is 30522, but the length of the tokenizer is 30523.


    # Private methods.
    def __load_vocab_file(self, vocab_filepath):
        """
        Load the vocabulary file and map the tokens to their respective indices.

        Args:
            vocab_filepath (str): Path to the vocabulary file.

        Returns:
            dict: Dictionary mapping tokens to their respective indices.
        """
        # Load file.
        with open(file=vocab_filepath, mode="r", encoding="utf-8") as file_buffer:
            raw_tokens = file_buffer.read().split(sep="\n")

        # Map tokens to indices.
        token_to_index = {token.strip(): index for index, token in enumerate(iterable=raw_tokens)}

        # Check if special tokens are in the vocabulary
        # and if they are in the correct index.
        for special_token in SpecialTokens:

            if not special_token.value[0] in token_to_index:
                raise ValueError("Special token %s not found in the vocabulary." % special_token.value[0])

            elif token_to_index[special_token.value[0]] != special_token.value[1]:
                raise ValueError("Special token %s is not in the correct index." % special_token.value[0])

        return token_to_index


    def __remove_accents(self, text):
        """
        Remove accents from a text.

        Args:
            text (str): Text to remove accents from.

        Returns:
            str: Text without accents.
        """
        return "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")


    # Methods.
    def encode(self, texts):
        """
        Encode a text into a list of token indices.

        Args:
            texts (str|List[str]): Text or list of texts to encode.

        Returns:
            list: List of token indices.
        """
        # Check if the input is a string.
        if isinstance(texts, str):
            texts = [texts]

        # Encode text.
        indices = []
        for text in texts:

            # Replace `#` with `spaces`.
            if "#" in text:
                warnings.warn(message="The character `#` is in the text. Replacing it with `space`.")
            text = text.replace("#", " ")
            text = text.strip()

            # Remove special tokens.
            text = text.upper()
            text = self.special_tokens_regex.sub(repl="", string=text)

            # Normalize text.
            text = text.lower()
            text = self.__remove_accents(text=text)

            start_index = 0
            final_index = len(text)
            indice = []
            add_double_hash = False
            while start_index < final_index:

                token = text[start_index:final_index]
                if add_double_hash:
                    token = "##" + token

                # Check if the token is in the vocabulary.
                if token in self.token_to_index:
                    indice.append(self.token_to_index[token])

                    # Update indices.
                    start_index = final_index
                    final_index = len(text)

                    # Check if the word was split.
                    if start_index < final_index and text[start_index] in string.ascii_lowercase + string.digits and token not in string.punctuation:
                        add_double_hash = True
                    else:
                        add_double_hash = False

                    continue

                # Update final index.
                final_index -= 1

                # Check `start_index` and `final_index` are the same.
                if start_index == final_index:
                    indice.append(SpecialTokens.UNK.value[1])
                    start_index += 1
                    final_index = len(text)

            # Append special tokens.
            indice = [SpecialTokens.CLS.value[1]] + indice + [SpecialTokens.SEP.value[1]]

            # Append indice.
            indices.append(indice)
        
        # Remove unknown tokens.
        clean_indices = []
        for indice in indices:
            clean_indice = list(filter(lambda token: token != SpecialTokens.UNK.value[1], indice))
            clean_indices.append(clean_indice)

        return clean_indices


    def decode(self, indices, remove_special_tokens = True):
        """
        Decode a list of token indices into text.

        Args:
            indices (List[int]|List[List[int]]): List or list of lists of token indices.
            remove_special_tokens (bool): Whether to remove special tokens. (Default: True)

        Returns:
            list: List of texts.
        """
        # Check if the input is a list of integers.
        if isinstance(indices, list) and all(isinstance(index, int) for index in indices):
            indices = [indices]

        # Decode indices.
        texts = []
        for indice in indices:

            text = ""
            for index in indice:

                # Get current token.
                token = self.index_to_token.get(index, SpecialTokens.UNK.name[0])

                # Check if the token is a special token.
                if token == "[PAD]":
                    continue
                
                if token.startswith("##") or token in string.punctuation:
                    text = text.strip() + token + " "
                else:
                    text += token + " "

            # Replace `##` with `space`.
            text = text.replace("##", "")

            # Remove special tokens.
            if remove_special_tokens:
                text = self.special_tokens_regex.sub(repl="", string=text)
            text = text.strip()
            if not text:
                continue

            # Append text.
            texts.append(text)

        return texts


    def encode_str(self, text):
        """
        Encode a text into a list string tokens.

        Args:
            text (str): Text to encode.

        Returns:
            list: List of string tokens.
        """
        # Encode text.
        indices = self.encode(texts=text)

        # Decode indices.
        str_tokens = []
        for indice in indices:
            curr_str_tokens = []
            for index in indice:
                str_token = self.decode(indices=[index])
                if str_token:
                    curr_str_tokens.append(str_token.pop(0))
            str_tokens.append(curr_str_tokens)
        
        return str_tokens


if __name__=="__main__":

    # Get arguments.
    args = get_args()

    # Create the tokenizer object.
    tokenizer = Tokenizer(vocab_filepath=args.vocab)
    print("Tokenizer has been created with a vocabulary of %d tokens." % len(tokenizer))

    # Encode text.
    print("Encoding text: %s" % args.text)
    tokens = tokenizer.encode(texts=args.text)
    print("Encoded tokens: %s" % tokens)
    print("String tokens: %s" % tokenizer.encode_str(text=args.text))

    # Decode tokens.
    decoded_text = tokenizer.decode(indices=tokens)
    print("Decoded text: %s" % decoded_text)
