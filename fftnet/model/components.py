import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierTransform(nn.Module):
    """
    Applies the Discrete Fourier Transform to the input tensor along the token dimension.
    """
    def __init__(self):
        super(FourierTransform, self).__init__()
        self.global_step = 0 # placeholder for global step

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, embedding_dim).
        Returns:
            torch.Tensor: Fourier transformed tensor of shape (batch_size, seq_len, embedding_dim) with complex dtype.
        """

        # Log input tensor shape
        # print(f"FT Input shape: {x.shape}") # Debugging - can be removed

        # Apply FFT along the sequence length dimension (dim=1)
        output = torch.fft.fft(x, dim=1)
        # Log output tensor shape
        # print(f"FT Output shape: {output.shape}") # Debugging - can be removed
        return output


class AdaptiveSpectralFilter(nn.Module):
    """
    Applies adaptive spectral filtering to the Fourier representation based on a global context.
    """
    def __init__(self, embedding_dim, mlp_hidden_dim=128, layer_norm=True):
        super(AdaptiveSpectralFilter, self).__init__()
        self.global_step = 0 # placeholder for global step
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, embedding_dim)
        )
        # Fixed base filter initialized to all ones
        self.W_base = nn.Parameter(torch.ones(1, 1, embedding_dim))

    def forward(self, fourier_representation, original_input):
        """
        Args:
            fourier_representation (torch.Tensor): Complex-valued Fourier transformed tensor
                of shape (batch_size, seq_len, embedding_dim).
            original_input (torch.Tensor): Original input sequence before Fourier transform
                of shape (batch_size, seq_len, embedding_dim).
        Returns:
            torch.Tensor: Filtered Fourier representation of same shape and dtype.
        """

        # Log input tensor shapes
        # print(f"ASF Fourier Input shape: {fourier_representation.shape}") # Debugging - can be removed
        # print(f"ASF Original Input shape: {original_input.shape}") # Debugging - can be removed

        # 1. Compute global context vector from original input
        if self.layer_norm:
            context_input = self.norm(original_input)
        else:
            context_input = original_input

        context_vector = torch.mean(context_input, dim=1)  # Average along token dimension

        # 2. Generate modulation tensor via MLP
        delta_W = self.mlp(context_vector).unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)

        # 3. Define the final filter
        W = self.W_base + delta_W

        # 4. Apply filter element-wise to Fourier coefficients
        filtered_fourier = fourier_representation * W

        # Log output tensor shape
        filtered_fourier = filtered_fourier
        # print(f"ASF Output shape: {filtered_fourier.shape}") # Debugging - can be removed
        return filtered_fourier


class ModReLU(nn.Module):
    """
    Implements the modReLU activation function for complex numbers.
    modReLU(z) = (|z| + b)*(z/|z|) if |z| + b > 0, else 0
    """
    def __init__(self, embedding_dim):
        super(ModReLU, self).__init__()
        self.global_step = 0 # placeholder for global step
        # Learnable bias parameter
        self.bias = nn.Parameter(torch.zeros(1, 1, embedding_dim))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Complex-valued input tensor.

        Returns:
            torch.Tensor: Complex-valued output tensor after modReLU activation.
        """
        # Log input tensor shape
        # print(f"ModReLU Input shape: {x.shape}") # Debugging - can be removed
        magnitude = torch.abs(x)
        # Handle zero magnitude to avoid division by zero
        phase_factor = x / (magnitude + 1e-10)

        # Apply ReLU to magnitude after adding bias
        relu_magnitude = F.relu(magnitude + self.bias)

        # Reconstruct complex numbers
        output = relu_magnitude * phase_factor
        # Log output tensor shape
        # print(f"ModReLU Output shape: {output.shape}") # Debugging - can be removed
        return output


class InverseFourierTransform(nn.Module):
    """
    Applies the Inverse Discrete Fourier Transform and returns the real component.
    """
    def __init__(self):
        super(InverseFourierTransform, self).__init__()
        self.global_step = 0 # placeholder for global step

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Complex-valued Fourier representation.

        Returns:
            torch.Tensor: Real-valued tensor after inverse FFT.
        """

        # Log input tensor shape
        # print(f"IFT Input shape: {x.shape}") # Debugging - can be removed

        # Apply inverse FFT along sequence length dimension
        inverse_output = torch.fft.ifft(x, dim=1)
        # Return the real component
        inverse_output = torch.fft.ifft(x, dim=1)
        # Log output tensor shape
        # print(f"IFT Output shape: {inverse_output.real.shape}") # Debugging - can be removed
        return inverse_output.real
