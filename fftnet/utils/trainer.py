import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

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
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 start_epoch=0,
                 args=None):
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
            start_epoch (int): The epoch to start training from.
            args: Command line arguments
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.start_epoch = start_epoch
        self.args = args
        # TensorBoard writer
        datestring=time.strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join(args.output_dir, "logs", datestring) if args and args.output_dir else "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Set up optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Initialize an ExponentialLR scheduler with a decay factor of 0.99
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.criterion = nn.CrossEntropyLoss()

        print(f"training data size: {len(train_loader)}")
        print(f"validation data size: {len(val_loader)}")

        # Load optimizer state if resuming
        if self.start_epoch > 0 and self.args and self.args.resume:
            checkpoint_path = self.args.resume
            if os.path.isfile(checkpoint_path):
                # Load the checkpoint with weights_only=False to avoid UnpicklingError.
                # This is safe as we trust the source of the checkpoint (our own training script).
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if 'optimizer_state_dict' in checkpoint:
                     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                     print("Loaded optimizer state from checkpoint.")
                     self.optimizer.param_groups[0]['lr'] = lr # Reset learning rate to the specified value
                else:
                    print("Optimizer state not found in checkpoint.")
            else:
                print("Checkpoint not found or resume argument not provided.")
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint.")
            if 'rng_state' in checkpoint:
                np.random.set_state(checkpoint['rng_state'])
                print("Loaded numpy random state from checkpoint.")
            else:
                print("Numpy random state not found in checkpoint.")
                if 'torch_rng_state' in checkpoint:
                    # Ensure torch_rng_state is a ByteTensor to avoid TypeError
                    torch_rng_state = checkpoint['torch_rng_state']
                    if not isinstance(torch_rng_state, torch.ByteTensor):
                        torch_rng_state = torch.ByteTensor(torch_rng_state) # Convert to ByteTensor if necessary
                    torch.set_rng_state(torch_rng_state)
                    print("Loaded torch random state from checkpoint.")
                else:
                    print("Torch random state not found in checkpoint.")
        else:
            print("Checkpoint not found or resume argument not provided.")

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
                # Set global step for layer logging
                global_step = epoch * len(self.train_loader) + batch_idx

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
                # Log training loss and learning rate to TensorBoard
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train', loss.item(), global_step)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], global_step)

                if global_step % 100 == 0:
                    # Log gradients and parameters histograms
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(f'Parameters/{name}', param, global_step)
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad, global_step)

                    # Log layer activations
                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Linear):
                            self.writer.add_histogram(f'LayerActivations/{name}', module.weight, global_step)
                            if module.bias is not None:
                                self.writer.add_histogram(f'LayerActivations/{name}_bias', module.bias, global_step)
                        elif isinstance(module, nn.LayerNorm):
                            self.writer.add_histogram(f'LayerActivations/{name}', module.weight, global_step)
                            self.writer.add_histogram(f'LayerActivations/{name}_bias', module.bias, global_step)

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        # Log epoch-level training loss
        self.val_losses = []
        self.test_metrics = None
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)
        return avg_loss

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.

        Args:
            epoch (int): Current epoch number.

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

        # Log epoch-level validation loss - now using epoch argument passed to validate()
        self.writer.add_scalar('Loss/validation_epoch', avg_loss, epoch + 1)
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
        start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': train_loss,
                'rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                }, save_path)

            # Validate if specified
            if self.val_loader and epoch % validate_every == 0:
                val_loss = None

                if self.val_loader and epoch % validate_every == 0:
                    val_loss = self.validate(epoch) # Pass epoch number to validate
                    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}, Val Loss: {val_loss_str}")

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}")

            # Log model parameters and gradients histograms
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Parameters/{name}', param, epoch + 1)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch + 1)
            # Update the learning rate using the scheduler after each epoch
            self.scheduler.step()

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
            pbar = tqdm(self.test_loader, desc="Testing")
            for inputs, targets in pbar:
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
