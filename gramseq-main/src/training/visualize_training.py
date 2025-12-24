"""
Visualize MoleculeVAE training progress.

This script loads the training history and creates visualizations
of loss curves, learning rate schedule, and training time.

Usage:
    python -m src.training.visualize_training
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "moleculevae"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"
OUTPUT_DIR = CHECKPOINT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_history():
    """Load training history from JSON file."""
    if not HISTORY_FILE.exists():
        print(f"Error: History file not found at {HISTORY_FILE}")
        print("Run the training script first!")
        return None

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    return history


def plot_losses(history):
    """Plot training and validation loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Find best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val_loss = min(history["val_loss"])
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
    plt.legend(fontsize=10)

    # Log scale
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.title("Training and Validation Loss (Log Scale)", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "loss_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved loss curves to: {output_file}")
    plt.close()

    return best_epoch, best_val_loss


def plot_learning_rate(history):
    """Plot learning rate schedule."""
    epochs = range(1, len(history["learning_rate"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["learning_rate"], 'g-', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "learning_rate.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved learning rate plot to: {output_file}")
    plt.close()


def plot_epoch_time(history):
    """Plot time per epoch."""
    epochs = range(1, len(history["epoch_time"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["epoch_time"], 'purple', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Training Time per Epoch", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add average line
    avg_time = np.mean(history["epoch_time"])
    plt.axhline(y=avg_time, color='r', linestyle='--', alpha=0.5, label=f'Average: {avg_time:.2f}s')
    plt.legend(fontsize=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "epoch_time.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved epoch time plot to: {output_file}")
    plt.close()

    return avg_time


def print_statistics(history, best_epoch, best_val_loss, avg_time):
    """Print training statistics."""
    print("\n" + "="*60)
    print("Training Statistics")
    print("="*60)
    print(f"Total epochs trained:     {len(history['train_loss'])}")
    print(f"Best epoch:               {best_epoch}")
    print(f"Best validation loss:     {best_val_loss:.4f}")
    print(f"Final training loss:      {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss:    {history['val_loss'][-1]:.4f}")
    print(f"Final learning rate:      {history['learning_rate'][-1]:.2e}")
    print(f"Average time per epoch:   {avg_time:.2f}s")
    print(f"Total training time:      {sum(history['epoch_time'])/3600:.2f} hours")
    print("="*60)


def create_combined_plot(history, best_epoch):
    """Create a combined plot with all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss Curves", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Loss curves (log)
    ax = axes[0, 1]
    ax.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Loss Curves (Log Scale)", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history["learning_rate"], 'g-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Epoch time
    ax = axes[1, 1]
    ax.plot(epochs, history["epoch_time"], 'purple', linewidth=2)
    avg_time = np.mean(history["epoch_time"])
    ax.axhline(y=avg_time, color='r', linestyle='--', alpha=0.5, label=f'Avg: {avg_time:.2f}s')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title("Training Time per Epoch", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "training_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot to: {output_file}")
    plt.close()


def main():
    """Main function."""
    print("="*60)
    print("MoleculeVAE Training Visualization")
    print("="*60)

    # Load history
    history = load_history()
    if history is None:
        return

    print(f"\nLoaded history with {len(history['train_loss'])} epochs")

    # Create plots
    print("\nGenerating plots...")
    best_epoch, best_val_loss = plot_losses(history)
    plot_learning_rate(history)
    avg_time = plot_epoch_time(history)
    create_combined_plot(history, best_epoch)

    # Print statistics
    print_statistics(history, best_epoch, best_val_loss, avg_time)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
