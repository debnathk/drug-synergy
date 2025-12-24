"""
Test script for pretrained MoleculeVAE model.

This script loads a pretrained model and performs various tests:
- Reconstruction quality
- Latent space interpolation
- Random generation

Usage:
    python -m src.training.test_pretrained
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.model_zinc import MoleculeVAE
from src.training.dataloader import val_loader
import src.zinc_grammar as G

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "moleculevae"
BEST_MODEL_PATH = CHECKPOINT_DIR / "checkpoint_best.pt"


def load_pretrained_model(checkpoint_path, device):
    """
    Load pretrained MoleculeVAE model.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model
        config: Training configuration
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint["config"]

    # Initialize model
    charset = list(range(G.D))
    model = MoleculeVAE(charset, config["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model, config


def test_reconstruction(model, dataloader, device, num_samples=100):
    """
    Test reconstruction quality on validation set.

    Args:
        model: MoleculeVAE model
        dataloader: Validation data loader
        device: Device
        num_samples: Number of samples to test

    Returns:
        avg_recon_loss: Average reconstruction loss
    """
    print("\n" + "="*60)
    print("Testing Reconstruction Quality")
    print("="*60)

    model.eval()
    total_loss = 0.0
    num_batches = 0
    samples_processed = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing reconstruction")
        for data, target in pbar:
            if samples_processed >= num_samples:
                break

            data = data.to(device)
            target = target.to(device)

            # Forward pass
            x_recon, z_mean, z_log_var = model(data, epsilon_std=0.0)  # No noise for testing

            # Compute loss
            loss = model.vae_loss(target, x_recon, z_mean, z_log_var)

            total_loss += loss.item()
            num_batches += 1
            samples_processed += data.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    print(f"\nAverage reconstruction loss: {avg_loss:.4f}")

    return avg_loss


def test_latent_space(model, dataloader, device, num_samples=10):
    """
    Analyze latent space properties.

    Args:
        model: MoleculeVAE model
        dataloader: Validation data loader
        device: Device
        num_samples: Number of samples to analyze

    Returns:
        latent_stats: Dictionary of latent space statistics
    """
    print("\n" + "="*60)
    print("Analyzing Latent Space")
    print("="*60)

    model.eval()
    all_z_mean = []
    all_z_log_var = []

    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i >= num_samples:
                break

            data = data.to(device)

            # Encode
            z_mean, z_log_var = model._encoderMeanVar(data)

            all_z_mean.append(z_mean.cpu().numpy())
            all_z_log_var.append(z_log_var.cpu().numpy())

    # Concatenate all samples
    all_z_mean = np.concatenate(all_z_mean, axis=0)
    all_z_log_var = np.concatenate(all_z_log_var, axis=0)

    # Compute statistics
    stats = {
        "z_mean_mean": np.mean(all_z_mean, axis=0),
        "z_mean_std": np.std(all_z_mean, axis=0),
        "z_log_var_mean": np.mean(all_z_log_var, axis=0),
        "z_log_var_std": np.std(all_z_log_var, axis=0),
    }

    print(f"\nLatent space statistics (n={all_z_mean.shape[0]} samples):")
    print(f"  z_mean:    mean={np.mean(stats['z_mean_mean']):.4f}, std={np.mean(stats['z_mean_std']):.4f}")
    print(f"  z_log_var: mean={np.mean(stats['z_log_var_mean']):.4f}, std={np.mean(stats['z_log_var_std']):.4f}")

    # Check if latent space is well-regularized (should be close to N(0,1))
    mean_dist_from_zero = np.mean(np.abs(stats['z_mean_mean']))
    mean_std = np.mean(stats['z_mean_std'])

    print(f"\nRegularization check:")
    print(f"  Mean distance from 0: {mean_dist_from_zero:.4f} (should be close to 0)")
    print(f"  Mean std:             {mean_std:.4f} (should be close to 1)")

    return stats


def test_interpolation(model, dataloader, device, num_steps=10):
    """
    Test latent space interpolation between two molecules.

    Args:
        model: MoleculeVAE model
        dataloader: Validation data loader
        device: Device
        num_steps: Number of interpolation steps

    Returns:
        interpolated_sequences: List of interpolated sequences
    """
    print("\n" + "="*60)
    print("Testing Latent Space Interpolation")
    print("="*60)

    model.eval()

    # Get two samples
    data_iter = iter(dataloader)
    data1, _ = next(data_iter)
    data2, _ = next(data_iter)

    # Take first sample from each batch
    mol1 = data1[0:1].to(device)
    mol2 = data2[0:1].to(device)

    with torch.no_grad():
        # Encode both molecules
        z1 = model.encode(mol1)
        z2 = model.encode(mol2)

        print(f"\nInterpolating between two molecules...")
        print(f"  Latent vector 1 shape: {z1.shape}")
        print(f"  Latent vector 2 shape: {z2.shape}")

        # Interpolate
        interpolated_sequences = []
        alphas = np.linspace(0, 1, num_steps)

        for alpha in alphas:
            # Linear interpolation
            z_interp = (1 - alpha) * z1 + alpha * z2

            # Decode
            x_decoded = model.decode(z_interp)
            interpolated_sequences.append(x_decoded.cpu().numpy())

        print(f"  Generated {len(interpolated_sequences)} interpolated sequences")

    return interpolated_sequences


def test_random_generation(model, device, num_samples=10):
    """
    Test random generation from prior.

    Args:
        model: MoleculeVAE model
        device: Device
        num_samples: Number of samples to generate

    Returns:
        generated_sequences: List of generated sequences
    """
    print("\n" + "="*60)
    print("Testing Random Generation")
    print("="*60)

    model.eval()

    with torch.no_grad():
        # Sample from standard normal prior
        z = torch.randn(num_samples, model.latent_dim).to(device)

        print(f"\nGenerating {num_samples} molecules from random latent vectors...")
        print(f"  Latent vector shape: {z.shape}")

        # Decode
        x_decoded = model.decode(z)

        print(f"  Generated sequence shape: {x_decoded.shape}")

    return x_decoded.cpu().numpy()


def main():
    """Main test function."""
    print("="*60)
    print("MoleculeVAE Pretrained Model Testing")
    print("="*60)

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load model
    try:
        model, config = load_pretrained_model(BEST_MODEL_PATH, device)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train the model first using pretrain_moleculevae.py")
        return

    # Run tests
    print("\n" + "="*60)
    print("Running Tests")
    print("="*60)

    # Test 1: Reconstruction
    recon_loss = test_reconstruction(model, val_loader, device, num_samples=100)

    # Test 2: Latent space analysis
    latent_stats = test_latent_space(model, val_loader, device, num_samples=10)

    # Test 3: Interpolation
    interpolated = test_interpolation(model, val_loader, device, num_steps=10)

    # Test 4: Random generation
    generated = test_random_generation(model, device, num_samples=10)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✓ Reconstruction test:     avg_loss={recon_loss:.4f}")
    print(f"✓ Latent space analysis:   completed")
    print(f"✓ Interpolation test:      {len(interpolated)} sequences generated")
    print(f"✓ Random generation test:  {generated.shape[0]} molecules generated")
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
