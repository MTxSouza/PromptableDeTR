"""
Module that stores the custom tokenizer class that will be used to tokenize the 
text data. Different than other tokenizers from Bert and GPT, this tokenizer needs 
to be trained before use it.
"""
import os
import re


# Classes.
class CustomTokenizer:


    # Class attributes.
    __SOS = "<sos>"
    __EOS = "<eos>"
    __PAD = "<pad>"
    __UNK = "<unk>"
    __special_tokens = [__SOS, __EOS, __PAD, __UNK]

    __token_pattern = r"'s|'d|'t|'ll|'re|'ve|'m|'em| ?[a-z]+| ?[a-zA-Z]+| ?\d{1,3}| ?[^a-zA-Z\d\s]{1,3}|\s+(?!\S)|\s+"
    __process_final_token_pattern = r"^Ġ"

    # Regex.
    __pattern = "|".join(__special_tokens) + "|" + __token_pattern
    __token_regex = re.compile(pattern=r"%s" % __pattern)


    # Static methods.
    @staticmethod
    def __count_pair_of_tokens(tokens):
        """
        Count the number of pairs of tokens that appear in the text dataset.

        Args:
            tokens (list[int]): List with the tokens that compose the text dataset.

        Returns:
            dict[(int, int), int]: Dictionary that stores the number of pairs of tokens \
                that appear in the text dataset.
        """
        # Count the number of pairs of tokens that appear in the text dataset.
        pairs = {}
        for pair in zip(tokens, tokens[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1

        return pairs


    @staticmethod
    def __merge_pairs(tokens, pair, pair_index):
        """
        Merge the pair of tokens in the text dataset.

        Args:
            tokens (list[int]): List with the tokens that compose the text dataset.
            pair (tuple[int, int]): Pair of tokens that will be merged.
            pair_index (int): Index of the pair of tokens in the text dataset.

        Returns:
            list[int]: List with the tokens that compose the text dataset after \
                merging the pair of tokens.
        """
        # Merge the pair of tokens in the text dataset.
        new_tokens = []
        i = 0
        while i < len(tokens):

            # Check if the pair of tokens is in the text dataset.
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(pair_index)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens


    # Class methods.
    @classmethod
    def get_base_vocab(cls):
        """
        Return the base vocabulary that will be used to train the custom tokenizer.

        Returns:
            dict[str, int]: Base vocabulary that will be used to train the custom tokenizer.
        """
        # Define valid bytes.
        valid_bytes = list(range(ord("!"), ord("~") + 1)) + \
                        list(range(ord("¡"), ord("¬") + 1)) + \
                        list(range(ord("®"), ord("ÿ") + 1))
        valid_chars = valid_bytes[:] # Copy the list of valid bytes.

        # Check if there are missing bytes.
        n = 0
        for byte in range(256):
            if byte not in valid_bytes:
                valid_bytes.append(byte)
                valid_chars.append(256 + n)
                n += 1

        # Convert bytes into characters.
        valid_chars = [chr(byte) for byte in valid_chars]

        # Define the base vocabulary.
        vocab = dict(zip(valid_bytes, valid_chars))

        return vocab


    @classmethod
    def apply_regex(cls, text):
        """
        Split the text into tokens using the regex pattern defined in the class.

        Args:
            text (str): Text that will be tokenized.
        
        Returns:
            list[str]: List of tokens that compose the text.
        """
        return cls.__token_regex.findall(string=text)


    @classmethod
    def train(cls, text_dataset, max_vocab_size = None, output_path = None):
        """
        Train the custom tokenizer with the text dataset provided and return the 
        vocabulary that will be used to tokenize the text data.

        It uses the BPE (Byte Pair Encoding) algorithm to create the vocabulary.

        Args:
            text_dataset (list[str]): Text dataset that will be used to train the tokenizer.
            max_vocab_size (int): Maximum number of tokens that the vocabulary can have. If \
                it is None, the BPE algorithm will stop when there are no more tokens to \
                merge. (Default: None)
            output_path (str): Path where the vocabulary will be saved. If it is None, the \
                vocabulary will not be saved and it will returned by the method. (Default: None)
        
        Returns:
            dict[str, int] | None: Vocabulary that will be used to tokenize the text data or \
                None if the vocabulary is saved in the output path.
        """
        # Define maximum vocabulary size.
        if max_vocab_size is None:
            max_vocab_size = float("inf")

        # Tokenize the text dataset using the regex pattern.
        full_text_dataset = " ".join(text_dataset)
        initial_tokens = cls.apply_regex(text=full_text_dataset)

        # Convert tokens into bytes.
        initial_tokens = [list(token.encode(encoding="utf-8")) for token in initial_tokens]

        # Loop until the vocabulary has the maximum size.
        vocab = cls.get_base_vocab()
        while len(vocab) < max_vocab_size:

            # Count the number of pairs of tokens that appear in the text dataset.
            pairs = {}
            for tokens in initial_tokens:

                pairs_count = cls.__count_pair_of_tokens(tokens=tokens)

                # Update the pairs dictionary.
                for pair, count in pairs_count.items():
                    pairs[pair] = pairs.get(pair, 0) + count
            
            # Get the most frequent pair of tokens.
            most_frequent_pair = max(pairs, key=pairs.get)

            # Check if the pair of tokens is frequent enough.
            if pairs[most_frequent_pair] == 1:
                break

            # Merge the most frequent pair of tokens.
            pair_index = len(vocab)
            vocab[pair_index] = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]

            # Merge the most frequent pair of tokens in the text dataset.
            initial_tokens = [
                cls.__merge_pairs(tokens=tokens, pair=most_frequent_pair, pair_index=pair_index)
                for tokens in initial_tokens
            ]

        # Add special tokens to the vocabulary.
        for token in cls.__special_tokens:
            vocab[len(vocab)] = token

        # Return the vocabulary.
        if output_path is None:
            return vocab
            
        # Check if the output path is directory.
        if not os.path.isdir(output_path):
            raise ValueError("The output path must be a directory.")
        
        # Save the vocabulary in the output path.
        filename = "prompt_vision_%d.vocab" % len(vocab)
        with open(file=os.path.join(output_path, filename), mode="w") as file_buffer:
            for index, token in vocab.items():
                file_buffer.write("%s %d\n" % (token, index))


    # Instance methods.
    def __init__(self, vocab_filepath):
        """
        Initialize the custom tokenizer with the vocabulary that will be used to 
        tokenize the text data.

        Args:
            vocab_filepath (str): Path where the vocabulary is stored.
        """
        # Load the vocabulary.
        with open(file=vocab_filepath, mode="r") as file_buffer:
            self.__token_to_index = {}
            self.__index_to_token = {}
            content = file_buffer.read().splitlines()
            for line in content:
                token, index = line.split(sep=" ")
                self.__token_to_index[token] = int(index)
                self.__index_to_token[int(index)] = token


    def __len__(self):
        """
        Return the number of tokens in the vocabulary.

        Returns:
            int: Number of tokens in the vocabulary.
        """
        return len(self.__token_to_index)
    

    def tokenize(self, text):
        """
        Tokenize the text data into tokens using the vocabulary.

        Args:
            text (str): Text data that will be tokenized.
        
        Returns:
            list[int]: List of tokens that compose the text data.
        """
        # Split text with regex pattern.
        initial_tokens = self.__token_regex.findall(string=text)

        # Convert tokens into bytes.
        initial_tokens = [list(token.encode(encoding="utf-8")) for token in initial_tokens]

        # Tokenize the text data using the vocabulary.
        def get_final_tokens(tokens):
            """
            Convert the token into the index using the vocabulary.

            Args:
                tokens (list[int]): List of bytes that compose the token.

            Returns:
                list[int]: List of indexes that compose the token.
            """
            # Get string token.
            str_token = "".join([self.__index_to_token.get(byte, self.__token_to_index[self.__UNK]) for byte in tokens])

            # Get index token.
            new_tokens = []
            initial_index = 0
            max_index = len(str_token)
            while initial_index < max_index:
                
                end_index = max_index
                while end_index > initial_index:

                    # Get current token.
                    current_token = str_token[initial_index:end_index]
                    current_index = self.__token_to_index.get(current_token)
                    if current_index is not None:
                        new_tokens.append(current_index)
                        initial_index = end_index
                        break
                        
                    end_index -= 1
                
                    # Check if the token was not found.
                    if end_index == initial_index:
                        new_tokens.append(self.__token_to_index[self.__UNK])
                        initial_index += 1
            
            return new_tokens

        tokens = [get_final_tokens(tokens=tokens) for tokens in initial_tokens]
        final_tokens = []
        for token in tokens:
            final_tokens.extend(token)
        
        return final_tokens


    def get_token_division(self, text):
        """
        Get the division of the tokens in the text data.

        Args:
            text (str): Text data that will be tokenized.

        Returns:
            list[str]: List of tokens that compose the text data.
        """
        # Get tokens from the text data.
        tokens = self.tokenize(text=text)

        # Get the division of the tokens.
        division = [self.__index_to_token.get(token) for token in tokens]
        division = [
            re.sub(pattern=self.__process_final_token_pattern, repl=" ", string=token) 
            for token in division
        ]

        return division


    def detokenize(self, tokens):
        """
        Detokenize the tokens into text data using the vocabulary.

        Args:
            tokens (list[int]): List of tokens that compose the text data.
        
        Returns:
            str: Text data that is composed by the tokens.
        """
        # Detokenize the tokens using the vocabulary.
        text = ""
        for token in tokens:

            # Process token.
            processed_token = self.__index_to_token.get(token, self.__UNK)
            processed_token = re.sub(pattern=self.__process_final_token_pattern, repl=" ", string=processed_token)

            text += processed_token

        return text
