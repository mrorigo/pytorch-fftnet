import torch.nn as nn
from .components import FourierTransform, AdaptiveSpectralFilter, ModReLU, InverseFourierTransform

class FFTNetLayer(nn.Module):
    """
    A complete FFTNet layer that combines Fourier Transform, Adaptive Spectral Filtering,
    modReLU activation, and Inverse Fourier Transform.
    """
    def __init__(self, embedding_dim, mlp_hidden_dim=128, dropout=0.1, layer_norm=True):
        super(FFTNetLayer, self).__init__()
        self.layer_norm = layer_norm

        if layer_norm:
            self.input_norm = nn.LayerNorm(embedding_dim)
            self.output_norm = nn.LayerNorm(embedding_dim)

        # Core FFTNet components
        self.fourier_transform = FourierTransform()
        self.adaptive_filter = AdaptiveSpectralFilter(embedding_dim, mlp_hidden_dim, layer_norm=False)
        self.modrelu = ModReLU(embedding_dim)
        self.inverse_fourier = InverseFourierTransform()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Output after FFTNet processing of same shape.
        """
        # Apply layer norm if enabled
        if self.layer_norm:
            normed_x = self.input_norm(x)
        else:
            normed_x = x

        # Store original input for residual connection
        residual = x

        # Apply FFTNet processing
        # 1. Fourier Transform
        fourier_x = self.fourier_transform(normed_x)

        # 2. Adaptive Spectral Filtering (uses original input for context)
        filtered_x = self.adaptive_filter(fourier_x, normed_x)

        # 3. modReLU Activation
        activated_x = self.modrelu(filtered_x)

        # 4. Inverse Fourier Transform
        output = self.inverse_fourier(activated_x)

        # Apply dropout
        output = self.dropout(output)

        # Add residual connection
        output = output + residual

        # Apply output normalization if enabled
        if self.layer_norm:
            output = self.output_norm(output)

        return output
