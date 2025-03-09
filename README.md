# FFTNet Implementation

This repository contains a PyTorch implementation of FFTNet, an efficient alternative to self-attention as described in the paper ["The FFT Strikes Back: An Efficient Alternative to Self-Attention"](https://arxiv.org/abs/2502.18394).

## Overview

FFTNet replaces the traditional self-attention mechanism with an adaptive spectral filtering framework based on the Fast Fourier Transform (FFT). This approach achieves global token mixing in O(n log n) time, providing a more efficient alternative to self-attention's O(n²) complexity, especially beneficial for long sequences.

This implementation provides a modular and understandable codebase for researchers and practitioners interested in exploring and utilizing FFTNet. It includes:

*   **Core FFTNet Components:** Implementations of Fourier Transform, Adaptive Spectral Filtering, modReLU activation, and Inverse Fourier Transform.
*   **Complete FFTNet Layer & Model:** Ready-to-use FFTNet Layer and full FFTNet model for sequence modeling tasks.
*   **Utilities:** Tools for tokenization (TikToken wrapper), dataset generation, and model training.
*   **Example Training & Generation Scripts:** Scripts to train the model and generate text examples.

## Key Components

The FFTNet architecture is built upon the following core components, each implemented as a separate PyTorch module in this repository:

1.  **Fourier Transform (FT):**
    *   Converts the input sequence from the time domain to the frequency domain using the Fast Fourier Transform.
    *   Implemented in `fftnet/model/components.py` as `FourierTransform`.
    *   Achieves global token mixing in O(n log n) complexity.

2.  **Adaptive Spectral Filter (ASF):**
    *   Applies a learnable filter in the frequency domain to emphasize salient frequency components dynamically.
    *   Implemented in `fftnet/model/components.py` as `AdaptiveSpectralFilter`.
    *   Uses an MLP to generate an adaptive filter based on the sequence context.

3.  **modReLU Activation:**
    *   Introduces non-linearity in the complex domain, enhancing model expressivity.
    *   Applies a ReLU-like function to the magnitude of complex numbers while preserving phase information.
    *   Implemented in `fftnet/model/components.py` as `ModReLU`.

4.  **Inverse Fourier Transform (IFT):**
    *   Transforms the filtered frequency domain representation back to the time domain.
    *   Implemented in `fftnet/model/components.py` as `InverseFourierTransform`.
    *   Returns the real component of the inverse transformed signal.

These components are combined within the `FFTNetLayer` in `fftnet/model/fftnet_layer.py` to form a complete FFTNet processing layer.

## Repository Structure

```
fftnet/
├── fftnet/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── components.py     # Core FFTNet components (FT, ASF, ModReLU, IFT)
│   │   ├── fftnet_layer.py   # Complete FFTNet layer
│   │   └── fftnet_model.py   # Full FFTNet model
│   └── utils/
│       ├── __init__.py
│       ├── dataset.py        # Dataset classes (TextDataset, Synthetic) and dataloader creation
│       ├── tokenizer.py      # TikToken wrapper for tokenization
│       └── trainer.py        # Trainer class for model training and evaluation
├── example.py                # Example script for text generation
├── main.py                   # Main training script
├── README.md                 # This file
├── requirements.txt          # Project dependencies
├── setup.py                  # Installation script
└── .gitignore                # Git ignore file
```

## Requirements

Before running the code, ensure you have the following dependencies installed. You can install them using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file lists the necessary packages:

-   `torch>=1.10.0`: PyTorch deep learning framework.
-   `tiktoken`:  Fast BPE tokenizer, compatible with OpenAI models.
-   `matplotlib`: Plotting library for visualization (e.g., loss plots).
-   `numpy`: Numerical computation library.
-   `tqdm`: Progress bar library for training loops.

## Installation

You can install the package directly from the source:

```bash
pip install -e .
```

## Usage

### Training FFTNet

To train the FFTNet model, use the `main.py` script. You can customize training and model parameters using command-line arguments.

```bash
python main.py --d_model 128 --num_layers 4 --num_epochs 30 --batch_size 64 --lr 1e-4 --validate_every 5 --save_model --output_dir ./outputs
```

**Available arguments:**

*   `--d_model`:  Model dimension (default: 128).
*   `--num_layers`: Number of FFTNet layers (default: 4).
*   `--mlp_hidden_dim`: MLP hidden dimension in Adaptive Spectral Filter (default: 512).
*   `--max_seq_length`: Maximum sequence length the model can handle (default: 128).
*   `--dropout`: Dropout rate (default: 0.1).
*   `--tokenizer`: Tokenizer encoding name (default: "cl100k_base").
*   `--seq_length`: Sequence length for the dataset (default: 128).
*   `--text_file`: Path to the text corpus file (default: `tinyshakespeare.txt`). The script will download `tinyshakespeare.txt` from a public source if not found.
*   `--seed`: Random seed for reproducibility (default: 42).
*   `--batch_size`: Batch size for training (default: 64).
*   `--lr`: Learning rate for the optimizer (default: 1e-4).
*   `--weight_decay`: Weight decay for the optimizer (default: 0.01).
*   `--num_epochs`: Number of training epochs (default: 30).
*   `--validate_every`:  Validation frequency in epochs (default: 1).
*   `--device`:  Device to use for training (`cpu`, `cuda`, or `mps`, default: `cuda` if available).
*   `--output_dir`: Directory to save outputs (models, logs, plots) (default: `./outputs`).
*   `--save_model`: Flag to save the best model during training.
*   `--resume`: Path to a checkpoint file to resume training from.

**Outputs of training:**

*   **Model Checkpoints:** If `--save_model` is used, the best model (based on validation loss) is saved as `best_model.pth` in the `--output_dir`. Checkpoints are also saved periodically during training, including optimizer state and training progress, to allow for resuming training.
*   **Loss Plots:** Training and validation loss curves are saved as `loss_plot.png` in the `--output_dir`.
*   **TensorBoard Logs:** Training progress, including losses, learning rate, and optionally histograms of gradients and parameters, are logged to TensorBoard in the `logs` subdirectory of `--output_dir`. You can visualize these logs by running `tensorboard --logdir=outputs/logs` (or your specified `output_dir`).
*   **Test Metrics:** After training, the model is evaluated on the test set, and the test loss, perplexity, and accuracy are saved to `test_metrics.txt` in the `--output_dir`.

### Text Generation Example

After training, or using a pre-trained model, you can generate text using the `example.py` script. This script loads a trained model (if available) and generates text based on a prompt.

```bash
python example.py --prompt "Once upon a time" --temperature 0.7 --max_length 100
```

**Available arguments:**

*   `--d_model`:  Model dimension (default: 128).
*   `--num_layers`: Number of FFTNet layers (default: 4).
*   `--mlp_hidden_dim`: MLP hidden dimension (default: 512).
*   `--max_seq_length`: Maximum sequence length (default: 1024).
*   `--temperature`: Sampling temperature - higher values create more diverse outputs (default: 1.0).
*   `--max_length`: Maximum generation length (default: 50).
*   `--prompt`: Text prompt to continue from (default: "Hello").
*   `--device`: Device to run inference on (default: "mps" or available device).
*   `--tokenizer`: Tokenizer encoding name (default: "cl100k_base").
*   `--top_k`: Top-k sampling parameter (default: 0, disabled).
*   `--model_path`: Path to model checkpoint (default: "outputs/best_model.pth").

The script automatically attempts to load a trained model from the specified path. If a trained model is not found, it will use an untrained model for demonstration purposes (which will produce random text).

**Example Output:**

```
Loaded trained model from outputs/best_model.pth
Model has 189,923 parameters
Generating text with:
- Temperature: 0.7
- Top-k: 40
- Max length: 50

==================================================
Prompt: Once upon a time
Generated:  there lived a king and queen, who had a fair daughter. And the king said unto the kingdom, "Whosoever shall bring me the head of the dragon shall have my daughter to wife."
==================================================
```

## Model Architecture

The FFTNet model architecture is designed for efficient sequence modeling, especially for long sequences. It comprises the following key layers:

1.  **Embedding Layer:**
    *   Maps input token indices to dense vector representations.
    *   Initialized using `nn.Embedding` from PyTorch.
    *   Scales embeddings by √d_model as per Transformer conventions.
    *   Uses sinusoidal positional encodings to incorporate sequence order information.

2.  **FFTNet Layers (Stack):**
    *   A stack of `FFTNetLayer` modules, where each layer consists of the core FFTNet components.
    *   Number of layers is configurable via `num_layers` argument.
    *   Each `FFTNetLayer` sequentially applies:
        *   Layer Normalization (optional)
        *   Fourier Transform
        *   Adaptive Spectral Filter
        *   ModReLU activation
        *   Inverse Fourier Transform
        *   Dropout
        *   Residual Connection
        *   Output Layer Normalization (optional)

3.  **Output Projection:**
    *   A linear layer (`nn.Linear`) followed by Layer Normalization to project the final hidden state to logits over the vocabulary.
    *   Used for next-token prediction tasks.

This architecture effectively replaces the self-attention layers of a standard Transformer with the FFT-based spectral filtering framework, providing a more computationally efficient approach for processing sequential data.

## References

*   **The FFT Strikes Back: An Efficient Alternative to Self-Attention**
    *   arXiv preprint: [https://arxiv.org/abs/2307.00302v2](https://arxiv.org/abs/2307.00302v2)
    *   Author: Jacob Fein-Ashley (University of Southern California)

## License

This project is released under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
