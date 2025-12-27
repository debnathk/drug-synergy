"""
Pretraining script for MoleculeVAE model using Guacamol dataset.

This script trains the MoleculeVAE model with:
- Training and validation loops
- Learning rate scheduling (ReduceLROnPlateau)
- Checkpoint saving for best validation loss
- Progress tracking and metrics logging

Author: Kusal Debnath
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
import json
from tqdm import tqdm

from src.models.model_zinc import MoleculeVAE
from src.training.dataloader import train_loader, val_loader
import src.zinc_grammar as G

# Configuration
ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "moleculevae"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
CONFIG = {
    "latent_dim": 256,
    "learning_rate": 1e-5, # 1e-4 causes converges very fast and results in loss = 0 (overfitting) 
    "num_epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "epsilon_std": 0.01,
    "scheduler_patience": 5,
    "scheduler_factor": 0.2,
    "scheduler_min_lr": 1e-8,
    "grad_clip": 1.0,
}

# Save config
with open(CHECKPOINT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=4)

print("="*60)
print("MoleculeVAE Pretraining")
print("="*60)
print(f"Device: {CONFIG['device']}")
print(f"Latent dimension: {CONFIG['latent_dim']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"Number of epochs: {CONFIG['num_epochs']}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print("="*60)


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model: MoleculeVAE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device
        data = data.to(device)
        target = target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        x_recon, z_mean, z_log_var = model(data, epsilon_std=CONFIG["epsilon_std"])

        # Compute loss
        loss = model.vae_loss(target, x_recon, z_mean, z_log_var)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

        # Optimizer step
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, dataloader, device, epoch):
    """
    Validate the model for one epoch.

    Args:
        model: MoleculeVAE model
        dataloader: Validation data loader
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ")
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            x_recon, z_mean, z_log_var = model(data, epsilon_std=CONFIG["epsilon_std"])

            # Compute loss
            loss = model.vae_loss(target, x_recon, z_mean, z_log_var)

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, is_best=False):
    """
    Save model checkpoint.

    Args:
        model: MoleculeVAE model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": CONFIG,
    }

    # Save latest checkpoint
    checkpoint_path = CHECKPOINT_DIR / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = CHECKPOINT_DIR / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    # Save epoch checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        epoch_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, epoch_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load model checkpoint.

    Args:
        model: MoleculeVAE model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to checkpoint file

    Returns:
        start_epoch, best_val_loss
    """
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Resuming from epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")
        return start_epoch, best_val_loss
    else:
        return 0, float('inf')


def main():
    """Main training loop."""

    # Set device
    device = torch.device(CONFIG["device"])
    print(f"\nUsing device: {device}")

    # Initialize model
    charset = list(range(G.D))  # Grammar charset
    model = MoleculeVAE(charset, CONFIG["latent_dim"])
    model = model.to(device)

    print(f"\nModel initialized:")
    print(f"  - Latent dimension: {CONFIG['latent_dim']}")
    print(f"  - Charset length: {len(charset)}")
    print(f"  - Max sequence length: {model.max_len}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["scheduler_min_lr"]
        # verbose parameter removed (deprecated)
    )

    # Load checkpoint if exists
    checkpoint_path = CHECKPOINT_DIR / "checkpoint_latest.pt"
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "epoch_time": []
    }

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    # Training loop
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate_epoch(model, val_loader, device, epoch)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Time:       {epoch_time:.2f}s")

        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, is_best)

        # Save training history
        with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        print("-" * 60)

        # Early stopping check (if LR is too small)
        if current_lr < CONFIG["scheduler_min_lr"]:
            print(f"\nLearning rate reached minimum ({CONFIG['scheduler_min_lr']:.2e}). Stopping training.")
            break

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {CHECKPOINT_DIR / 'checkpoint_best.pt'}")
    print(f"Training history saved at: {CHECKPOINT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()
