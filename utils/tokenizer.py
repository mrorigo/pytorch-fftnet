import tiktoken
import torch
from typing import List, Union

class TiktokenTokenizer:
    """
    Wrapper class for the tiktoken tokenizer to use with the FFTNet model.
    """
    def __init__(self, encoding_name="cl100k_base"):
        """
        Initialize the tokenizer with the specified encoding.

        Args:
            encoding_name (str): Name of the tiktoken encoding to use.
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        self.bos_token_id = None
        self.eos_token_id = None

        # Check if encoding has special tokens
        try:
            self.bos_token_id = self.encoding.encode_single_token(self.encoding.special_tokens_map["<|endoftext|>"])
        except:
            pass

        try:
            self.eos_token_id = self.encoding.encode_single_token(self.encoding.special_tokens_map["<|endoftext|>"])
        except:
            pass

    def encode(self, text: Union[str, List[str]], return_tensors=None):
        """
        Tokenize the input text(s).

        Args:
            text (str or List[str]): Input text or list of texts to tokenize.
            return_tensors (str, optional): If "pt", returns PyTorch tensors.

        Returns:
            List[int] or List[List[int]] or torch.Tensor: Encoded tokens.
        """
        if isinstance(text, str):
            tokens = self.encoding.encode(text)
            if return_tensors == "pt":
                return torch.tensor(tokens).unsqueeze(0)
            return tokens
        else:
            tokens = [self.encoding.encode(t) for t in text]
            if return_tensors == "pt":
                # Pad sequences to the same length
                max_len = max(len(t) for t in tokens)
                padded_tokens = [t + [0] * (max_len - len(t)) for t in tokens]
                return torch.tensor(padded_tokens)
            return tokens

    def decode(self, tokens):
        """
        Convert token IDs back to text.

        Args:
            tokens (List[int] or torch.Tensor): Token IDs to decode.

        Returns:
            str: Decoded text.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()

        if isinstance(tokens[0], list) or (isinstance(tokens[0], torch.Tensor) and tokens[0].dim() > 0):
            return [self.encoding.decode(t) for t in tokens]
        else:
            return self.encoding.decode(tokens)
