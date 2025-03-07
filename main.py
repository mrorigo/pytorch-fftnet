import argparse
import torch
import os
import matplotlib.pyplot as plt
from utils.tokenizer import TiktokenTokenizer
from utils.dataset import SyntheticSequenceDataset, create_dataloaders
from model.fftnet_model import FFTNet
from utils.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate FFTNet model")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of FFTNet layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512, help="MLP hidden dimension")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Dataset parameters
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--pattern_complexity", type=float, default=0.5, help="Pattern complexity (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--validate_every", type=int, default=1, help="Validate every N epochs")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_model", action="store_true", help="Save the best model")

    return parser.parse_args()

def plot_losses(train_losses, val_losses, output_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Creating synthetic dataset with {args.num_samples} samples...")
    # Create synthetic dataset
    dataset = SyntheticSequenceDataset(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        pattern_complexity=args.pattern_complexity,
        seed=args.seed
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size
    )

    print(f"Initializing FFTNet model...")
    # Initialize model
    model = FFTNet(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device
    )

    print(f"Training FFTNet model for {args.num_epochs} epochs...")
    # Train model
    save_path = os.path.join(args.output_dir, "best_model.pth") if args.save_model else None
    history = trainer.train(
        num_epochs=args.num_epochs,
        validate_every=args.validate_every,
        save_path=save_path
    )

    # Plot and save losses
    plot_losses(
        history["train_losses"],
        history["val_losses"],
        os.path.join(args.output_dir, "loss_plot.png")
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate()

    # Save test metrics
    if test_metrics:
        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
            for key, value in test_metrics.items():
                f.write(f"{key}: {value}\n")

    print("Done!")

if __name__ == "__main__":
    main()
