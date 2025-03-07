# FFTNet Implementation

This repository contains a PyTorch implementation of FFTNet, an efficient alternative to self-attention as described in the paper "The FFT Strikes Back: An Efficient Alternative to Self-Attention".

## Overview

FFTNet replaces the traditional self-attention mechanism with an adaptive spectral filtering framework based on the Fast Fourier Transform (FFT). This approach achieves global token mixing in O(n log n) time, providing a more efficient alternative to self-attention's O(n²) complexity.

## Key Components

1. **Fourier Transform**: Converts input sequences to the frequency domain
2. **Adaptive Spectral Filtering**: Applies a learnable filter to frequency components based on input context
3. **modReLU Activation**: A nonlinear activation designed for complex numbers
4. **Inverse Fourier Transform**: Converts processed sequences back to the token domain

## Repository Structure

```
fftnet/
├── model/
│   ├── components.py     # Core FFTNet components
│   ├── fftnet_layer.py   # Complete FFTNet layer
│   └── fftnet_model.py   # Full FFTNet model
├── utils/
│   ├── tokenizer.py      # TikToken wrapper
│   ├── dataset.py        # Synthetic dataset generation
│   └── trainer.py        # Training and evaluation utilities
├── main.py               # Training script
├── example.py            # Text generation example
└── README.md             # This file
```

## Requirements

- PyTorch >= 1.10.0
- TikToken
- Matplotlib
- NumPy
- tqdm

## Usage

### Training

To train the model on synthetic data:

```bash
python main.py --d_model 128 --num_layers 4 --num_epochs 30 --save_model
```

### Text Generation

To generate text using a trained model:

```bash
python example.py
```

## Model Architecture

FFTNet's architecture is designed to efficiently capture long-range dependencies in sequences:

1. Input embeddings are transformed into the frequency domain using FFT
2. A global context vector is computed from the input sequence
3. This context vector is used to generate a modulation tensor via an MLP
4. The modulation is added to a base filter to create the final adaptive filter
5. The filter is applied to the frequency components
6. modReLU activation introduces non-linearity while preserving phase information
7. The processed data is transformed back to the token domain using IFFT

This approach achieves global token mixing with O(n log n) complexity instead of O(n²), making it more efficient for long sequences.

## References

- "The FFT Strikes Back: An Efficient Alternative to Self-Attention" (arXiv preprint v2)
