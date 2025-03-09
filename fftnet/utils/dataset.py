import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    """
    Dataset for text corpus.
    """
    def __init__(self,
                 text_file,
                 tokenizer,
                 seq_length,
                 vocab_size,
                 seed=42):
        """
        Initialize the TextDataset.

        Args:
            text_file (str): Path to the text file.
            tokenizer (TiktokenTokenizer): Tokenizer instance.
            seq_length (int): Length of each sequence.
            vocab_size (int): Size of the vocabulary.
            seed (int): Random seed for reproducibility.
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed
        self.tokenizer = tokenizer

        # Load text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the text
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        """
        Returns the length of the tokenized corpus minus 1.
        """
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        """
        Get input sequence and target sequence for next-token prediction.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_sequence, target_sequence)
        """
        # Input sequence: tokens from idx to idx + seq_length
        input_seq = torch.tensor(self.tokens[idx:idx+self.seq_length], dtype=torch.long)
        # Target sequence: tokens from idx+1 to idx + seq_length + 1
        target_seq = torch.tensor(self.tokens[idx+1:idx+self.seq_length+1], dtype=torch.long)

        return input_seq, target_seq

class SyntheticSequenceDataset(Dataset):
    """
    A synthetic dataset for sequence modeling tasks.
    Generates sequences with some learnable patterns.
    """
    def __init__(self,
                 num_samples=1000,
                 seq_length=128,
                 vocab_size=1000,
                 pattern_complexity=0.3,
                 seed=42):
        """
        Initialize the synthetic dataset.

        Args:
            num_samples (int): Number of sequences to generate.
            seq_length (int): Length of each sequence.
            vocab_size (int): Size of the vocabulary.
            pattern_complexity (float): Determines the complexity of patterns (0.0-1.0).
            seed (int): Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Set seed for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(seed)

        # Generate synthetic data
        self.data = self._generate_data(pattern_complexity)

    def _generate_data(self, pattern_complexity):
        """
        Generate synthetic sequences with patterns.

        Args:
            pattern_complexity (float): Higher values create more complex patterns.

        Returns:
            torch.Tensor: Generated sequences of shape (num_samples, seq_length).
        """
        sequences = []

        for _ in range(self.num_samples):
            # Start with random sequence
            seq = self.rng.integers(0, self.vocab_size, size=self.seq_length)

            # Add some patterns based on complexity
            num_patterns = int(pattern_complexity * 10)

            for _ in range(num_patterns):
                # Random pattern length between 2 and 5
                pattern_length = np.random.randint(2, 6)

                # Random start positions
                start_pos1 = self.rng.integers(0, self.seq_length - 2*pattern_length)
                start_pos2 = self.rng.integers(start_pos1 + pattern_length, self.seq_length - pattern_length)

                # Copy pattern
                seq[start_pos2:start_pos2+pattern_length] = seq[start_pos1:start_pos1+pattern_length]

                # Add some repetitions
                if self.rng.random() < 0.5:
                    rep_start = self.rng.integers(0, self.seq_length - 3)
                    seq[rep_start+1:rep_start+3] = seq[rep_start:rep_start+2]

            sequences.append(seq)

        return torch.tensor(np.array(sequences), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get input sequence and target sequence for next-token prediction.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_sequence, target_sequence)
        """
        sequence = self.data[idx]

        # For next token prediction: input is all tokens except the last one
        # target is all tokens except the first one
        input_seq = sequence[:-1]
        target_seq = sequence[1:]

        return input_seq, target_seq


def create_dataloaders(dataset, batch_size=32, val_split=0.1, test_split=0.1):
    """
    Create train, validation, and test dataloaders from a dataset.

    Args:
        dataset (Dataset): The dataset to split.
        batch_size (int): Batch size for the dataloaders.
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
