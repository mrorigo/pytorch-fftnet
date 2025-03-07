import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm

class Trainer:
    """
    Trainer class for handling the training and evaluation of the FFTNet model.
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None,
                 lr=1e-4,
                 weight_decay=0.01,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The FFTNet model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader, optional): DataLoader for validation data.
            test_loader (DataLoader, optional): DataLoader for test data.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            device (str): Device to run training on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Set up optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.test_metrics = None

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Reshape outputs and targets for loss calculation
                # outputs: [batch_size, seq_len, vocab_size] -> [batch_size*seq_len, vocab_size]
                # targets: [batch_size, seq_len] -> [batch_size*seq_len]
                batch_size, seq_len = targets.size()
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """
        Evaluate the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Reshape outputs and targets for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self, num_epochs, validate_every=1, save_path=None):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs (int): Number of epochs to train.
            validate_every (int): Validate every N epochs.
            save_path (str, optional): Path to save the best model.

        Returns:
            dict: Training history.
        """
        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Validate if specified
            val_loss = None
            if self.val_loader and epoch % validate_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save best model
                if save_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "time": total_time
        }

    def evaluate(self):
        """
        Evaluate the model on the test set.

        Returns:
            dict: Test metrics including loss, perplexity, and accuracy.
        """
        if self.test_loader is None:
            return None

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Get predictions
                _, predicted = torch.max(outputs, dim=-1)

                # Reshape for metrics calculation
                flat_targets = targets.view(-1)
                flat_predicted = predicted.view(-1)

                # Calculate accuracy
                correct += (flat_predicted == flat_targets).sum().item()
                total += flat_targets.size(0)

                # Calculate loss
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = targets.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()

        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct / total

        self.test_metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy
        }

        print(f"Test Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.4f}")

        return self.test_metrics
