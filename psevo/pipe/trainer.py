
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Dict, Optional


class PSVAETrainer:
    """ Complete PyTorch trainer for PS-VAE model implementing the original training logic.
    This trainer replicates the PyTorch Lightning training loop with:
    - KL annealing schedule from original code
    - Weighted loss combination (α * rec_loss + (1-α) * pred_loss + β * kl_loss)
    - Learning rate scheduling
    - Early stopping based on validation loss
    - Comprehensive logging and checkpointing
    """

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.total_time = 0

        # Optimizer (from original code)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-3)
        )

        # Learning rate scheduler (from original code)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 1 / (epoch + 1)
        )

        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        epoch_metrics = []
        self.model.train()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Training step
            start_time = time.time()
            metrics = self.model.training_step(batch, self.optimizer, self.global_step)
            step_time = time.time() - start_time

            # Update global step
            self.global_step += 1
            self.total_time += step_time

            # Add timing info
            metrics['step_time'] = step_time
            metrics['total_time'] = self.total_time
            metrics['lr'] = self.optimizer.param_groups[0]['lr']

            epoch_metrics.append(metrics)

            # Log progress
            if batch_idx % 100 == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                      f"Loss = {metrics['total_loss']:.4f}, "
                      f"Beta = {metrics['beta']:.6f}, "
                      f"Encoder = {metrics['encoder_type']}")

        # Average metrics over epoch
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if key not in ['step_time', 'total_time', 'encoder_type']:
                avg_metrics[f'train_{key}'] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)

        return avg_metrics

    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        epoch_metrics = []
        self.model.eval()

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Validation step
                metrics = self.model.validation_step(batch, self.global_step)
                epoch_metrics.append(metrics)

        # Average metrics over epoch
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if key != 'encoder_type':
                avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)

        return avg_metrics

    def fit(self, train_loader, val_loader, num_epochs, save_dir=None):
        """
        Complete training loop implementing original PS-VAE training.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        patience = self.config.get('patience', 3)
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)

            # Learning rate scheduling
            self.scheduler.step()

            # Print epoch summary
            print(f"Train Loss: {train_metrics['train_total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_total_loss']:.4f}")
            print(f"Piece Acc: {val_metrics['val_piece_accuracy']:.4f}")
            print(f"Edge Acc: {val_metrics['val_edge_accuracy']:.4f}")
            print(f"Beta: {val_metrics['val_beta']:.6f}")

            # Early stopping and checkpointing
            current_val_loss = val_metrics['val_total_loss']

            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                patience_counter = 0

                # Save best model
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': self.best_val_loss,
                        'config': self.config,
                        'global_step': self.global_step
                    }, f"{save_dir}/best_model.pt")
                    print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

            # Store metrics
            self.train_losses.append(train_metrics)
            self.val_losses.append(val_metrics)
