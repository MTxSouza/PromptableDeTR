"""
Script that runs the Byte Pair Encoding (BPE) algorithm on a given text file and saves the vocabulary to a JSON 
file.
"""
import argparse
import json
import re
from dataclasses import dataclass, field
from enum import Enum


# Enums.
class SpecialTokens(Enum):
    """
    Enum class for special tokens.
    """
    PAD = ("<PAD>", 0) # Used for padding sequences.
    UNK = ("<UNK>", 1) # Used for unknown tokens.
    SOS = ("<SOS>", 2) # Used for start of sequence.
    EOS = ("<EOS>", 3) # Used for end of sequence.


    # Methods.
    @classmethod
    def get_special_tokens(cls):
        """
        Get the list of special tokens.

        Returns:
            Dict[str, int]: The special tokens with string tokens as keys and integer indices as values.
        """
        return {data.value[0]: data.value[1] for data in cls._member_map_.values()}


def argument_parser():
    """
    Argument parser for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    # Create the parser.
    parser = argparse.ArgumentParser(description="Run the Byte Pair Encoding (BPE) algorithm on a text file.")

    # Add arguments.
    parser.add_argument(
        "--training-files", 
        "-tf", 
        type=str, 
        nargs="+", 
        required=True, 
        help="List of paths to the text file to be used in training."
    )
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        default="./vocab.json", 
        help="The path to the output JSON file."
    )
    parser.add_argument(
        "--max-vocab-size", 
        "-mvs", 
        type=int, 
        default=1000, 
        help="The maximum size of the vocabulary."
    )

    return parser.parse_args()


if __name__ == "__main__":


    # Structures.
    @dataclass
    class TextContent:
        """
        Simple structure used to store the content of a file and it 
        tokenized version.
        """
        raw: str = ""
        normalized: str = ""
        regex: list[str] = field(default_factory=lambda: [])
        tokens: list[str] = field(default_factory=lambda: [])
        indices: list[int] = field(default_factory=lambda: [])


    # Functions.
    def get_base_vocab():
        """
        Construct the base vocabulary from the ASCII characters and special tokens. It will only 
        consider lowercase characters, numbers and space for the vocabulary.

        Returns:
            Dict[str, int]: The base vocabulary with string tokens as keys and integer indices as values.
        """
        # Define the special tokens.
        special_tokens = SpecialTokens.get_special_tokens()

        # Define the ASCII characters.
        valid_characters = re.compile(pattern=r"[a-z0-9 ']") # Only lowercase characters and numbers.
        printable_characters = [chr(i) for i in range(32, 127)]
        printable_characters = list(filter(valid_characters.match, printable_characters))
        printable_characters = {token: idx for idx, token in enumerate(printable_characters, start=len(special_tokens))}

        # Combine the special tokens and ASCII characters.
        base_vocab = {**special_tokens, **printable_characters}

        return base_vocab


    def load_tokenzer_training_set(*filepath):
        """
        Load the training set from a text file. It will split the text file by a simple regex 
        pattern to get the initial tokens to be passed to the BPE algorithm.

        Args:
            filepath (str): The path to the text file.

        Returns:
            List[TextContent]: List with text content.
        """
        # Load files.
        full_content = []
        for path in filepath:

            with open(file=path, mode="r", encoding="utf-8") as file_buffer:
                content = file_buffer.read().split(sep="\n")
                content = [TextContent(raw=text) for text in content]
                full_content.extend(content)

        # Tokenize the text.
        special_tokens = r"|".join(list(SpecialTokens.get_special_tokens()))
        split_pattern = re.compile(pattern=special_tokens + r"|'m|'t|'s|'re|'ll|'d|'ve| ?[a-z]+| ?[0-9]{1,3}|\s+(?!\S)|\s+")
        def get_tokens(text_content):

            regex = split_pattern.findall(string=text_content.raw.lower())
            normalized = re.sub(pattern=r"\s{2,}", repl=" ", string=" ".join(regex))

            text_content.regex = regex
            text_content.normalized = normalized

            return text_content

        tokens = list(map(get_tokens, full_content))

        # Filter empty text.
        tokens = list(filter(lambda text_content: text_content if text_content.normalized else None, tokens))

        return tokens


    def get_current_indices(text_content_list, vocab, vocab_reverse):
        """
        Get the current indices of the tokens in the vocabulary.

        Args:
            text_content_list (List[TextContent]): List with text content.
            vocab (Dict[str, int]): The vocabulary with string tokens as keys and integer indices as values.
            vocab_reverse (Dict[int, str]): The vocabulary with integer indices as keys and string tokens as values.

        Returns:
            List[TextContent]: List with text content with the tokens converted to indices.
        """
        # Convert tokens into indices.
        for text_content in text_content_list:

            text_content.tokens.clear()
            text_content.indices.clear()

            # Convert to indices.
            for token in text_content.regex:

                init_indice = 0
                last_indice = len(token)
                new_token = []
                while init_indice < len(token):

                    # Get potential token.
                    current_token = token[init_indice:last_indice]

                    # Check if token is in the vocabulary.
                    if current_token in vocab:
                        new_token.append(vocab[current_token])
                        init_indice = last_indice
                        last_indice = len(token)
                        continue

                    # Check if token is a single character.
                    if len(current_token) == 1:
                        new_token.append(SpecialTokens.UNK.value[1])
                        init_indice += 1
                        last_indice = len(token)
                        continue

                    last_indice -= 1

                text_content.indices.append(new_token)
                text_content.tokens.append([vocab_reverse[indice] for indice in new_token])

        return text_content_list


    def count_pairs_of_tokens(text_content_list):
        """
        Count the frequency of pairs of tokens for each text.

        Args:
            text_content_list (List[TextContent]): List with text content.

        Returns:
            Dict[tuple[int, int], int]: All indice pairs in training set with their frequencies.
        """
        # Count frequency.
        frequency = {}
        for text_content in text_content_list:

            for indices in text_content.indices:

                # Check if the text has less than 2 tokens.
                if len(indices) < 2:
                    continue

                for tk1, tk2 in zip(indices, indices[1:]):
                    current_pair = (tk1, tk2)
                    frequency[current_pair] = frequency.get(current_pair, 0) + 1

        # Sort by frequency.
        frequency = dict(sorted(frequency.items(), key=lambda item: item[1], reverse=True))

        return frequency


    # Parse arguments.
    args = argument_parser()

    # Load training set.
    training_set = load_tokenzer_training_set(*args.training_files)

    # Get base vocabulary.
    base_vocab = get_base_vocab()
    base_vocab_reverse = {indice: token for token, indice in base_vocab.items()}

    # BPE algorithm.
    while len(base_vocab) < args.max_vocab_size:

        # Get current indices.
        current_indices = get_current_indices(text_content_list=training_set, vocab=base_vocab, vocab_reverse=base_vocab_reverse)

        # Count pairs of tokens.
        pairs = count_pairs_of_tokens(text_content_list=current_indices)

        # Get the most frequent pair.
        most_frequent_pair = max(pairs, key=pairs.get)
        if pairs[most_frequent_pair] == 1:
            break

        # Merge tokens.
        indice1, indice2 = most_frequent_pair
        new_indice = len(base_vocab)
        new_token = base_vocab_reverse[indice1] + base_vocab_reverse[indice2]

        base_vocab[new_token] = new_indice
        base_vocab_reverse[new_indice] = new_token

    # Save vocabulary.
    with open(file=args.output, mode="w", encoding="utf-8") as file_buffer:
        json.dump(base_vocab, file_buffer, indent=4)
