# MoleculeVAE Pretraining Guide

## Overview
The `pretrain_moleculevae.py` script trains the MoleculeVAE model on the Guacamol dataset with full training pipeline including validation, learning rate scheduling, and checkpoint management.

## Features

### 1. Training & Validation
- Complete training loop with progress bars (tqdm)
- Validation after each epoch
- Automatic metric tracking

### 2. Learning Rate Optimization
- **Scheduler**: ReduceLROnPlateau
  - Reduces LR when validation loss plateaus
  - Patience: 5 epochs (configurable)
  - Reduction factor: 0.5 (halves the LR)
  - Minimum LR: 1e-6

### 3. Checkpoint Management
- **Best model**: Saved when validation loss improves
- **Latest checkpoint**: Saved after every epoch
- **Periodic checkpoints**: Saved every 10 epochs
- **Resume training**: Automatically resumes from latest checkpoint if exists

### 4. Monitoring & Logging
- Training/validation loss per epoch
- Learning rate tracking
- Epoch timing
- All metrics saved to JSON file

## Usage

### Basic Training
```bash
# From project root
cd /Users/debnathk/Documents/drug-synergy/gramseq-main
python -m src.training.pretrain_moleculevae
```

### Configuration
Edit the `CONFIG` dictionary in the script to modify hyperparameters:

```python
CONFIG = {
    "latent_dim": 56,              # VAE latent space dimension
    "learning_rate": 1e-3,         # Initial learning rate
    "num_epochs": 100,             # Total training epochs
    "epsilon_std": 0.01,           # Sampling noise std
    "scheduler_patience": 5,       # Epochs before LR reduction
    "scheduler_factor": 0.5,       # LR reduction factor
    "scheduler_min_lr": 1e-6,      # Minimum learning rate
    "grad_clip": 1.0,              # Gradient clipping threshold
}
```

## Output Structure

```
checkpoints/moleculevae/
├── config.json                  # Training configuration
├── checkpoint_best.pt           # Best model (lowest val loss)
├── checkpoint_latest.pt         # Most recent checkpoint
├── checkpoint_epoch_10.pt       # Periodic checkpoints
├── checkpoint_epoch_20.pt
├── ...
└── training_history.json        # Loss/metrics history
```

## Checkpoint Contents
Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current epoch
- Training/validation loss
- Configuration

## Using Trained Model

### Load Best Model
```python
import torch
from src.models.model_zinc import MoleculeVAE
import src.zinc_grammar as G

# Initialize model
charset = list(range(G.D))
model = MoleculeVAE(charset, latent_dim=56)

# Load checkpoint
checkpoint = torch.load("checkpoints/moleculevae/checkpoint_best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use model for inference
with torch.no_grad():
    z_mean = model.encode(input_data)
    reconstructed = model.decode(z_mean)
```

### Resume Training
The script automatically detects and resumes from the latest checkpoint if it exists. To start fresh, delete or rename the checkpoint files.

## Device Selection
The script automatically selects the best available device:
1. CUDA (if available)
2. MPS (Apple Silicon)
3. CPU (fallback)

You can manually set the device in the CONFIG dictionary.

## Monitoring Training

### Real-time Progress
The script shows progress bars with current loss for both training and validation.

### Training History
View the `training_history.json` file for:
- Loss curves (train/val)
- Learning rate schedule
- Epoch durations

### Example Visualization
```python
import json
import matplotlib.pyplot as plt

# Load history
with open("checkpoints/moleculevae/training_history.json", "r") as f:
    history = json.load(f)

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_curves.png")
```

## Troubleshooting

### Out of Memory
- Reduce batch size in `dataloader.py`
- Reduce `latent_dim` in CONFIG
- Use gradient accumulation

### Training Instability
- Reduce learning rate
- Increase `grad_clip` value
- Adjust `epsilon_std`

### Slow Training
- Increase batch size (if memory allows)
- Reduce `num_workers` in dataloader if CPU bottleneck
- Use GPU if available

## Notes
- The model uses grammar-constrained loss (see `model_zinc.py:224`)
- VAE loss = Reconstruction Loss (BCE) + KL Divergence
- Best checkpoint is determined by lowest validation loss
