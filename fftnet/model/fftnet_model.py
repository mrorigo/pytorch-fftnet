import torch
import torch.nn as nn
import math
from .fftnet_layer import FFTNetLayer

class FFTNet(nn.Module):
    """
    Complete FFTNet model for sequence modeling tasks.
    """
    def __init__(self,
                vocab_size,
                d_model=512,
                num_layers=6,
                mlp_hidden_dim=2048,
                max_seq_length=1024,
                dropout=0.1,
                layer_norm=True):
        super(FFTNet, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # Stack of FFTNet layers
        self.layers = nn.ModuleList([
            FFTNetLayer(
                embedding_dim=d_model,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                layer_norm=layer_norm
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize positional encodings
        # self._init_positional_encoding(max_seq_length, d_model)

    # New helper method to compute positional encodings dynamically for any sequence length
    def get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)  # Shape: (1, seq_len, d_model)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input token indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """

        # print(f"FFTNetModel Input shape: {x.shape}") # Debugging - can be removed

        seq_len = x.size(1)

        # Embedding and dynamic positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        # Compute positional encodings dynamically based on current sequence length
        pos_encoding = self.get_positional_encoding(seq_len, self.d_model).to(x.device)
        x = x + pos_encoding

        # Apply FFTNet layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)

        # print(f"FFTNetModel Output shape: {logits.shape}") # Debugging - can be removed
        return logits
