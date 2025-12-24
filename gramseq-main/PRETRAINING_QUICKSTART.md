# MoleculeVAE Pretraining - Quick Start Guide

## Overview
Complete pretraining pipeline for MoleculeVAE model using Guacamol dataset with:
- ✓ Training and validation loops
- ✓ Learning rate scheduler (ReduceLROnPlateau)
- ✓ Checkpoint management (best weights saved)
- ✓ Progress tracking and visualization
- ✓ Testing utilities

## Files Created

### Training Scripts
- [`src/training/pretrain_moleculevae.py`](src/training/pretrain_moleculevae.py) - Main pretraining script
- [`src/training/visualize_training.py`](src/training/visualize_training.py) - Training visualization
- [`src/training/test_pretrained.py`](src/training/test_pretrained.py) - Model testing utilities

### Documentation
- [`src/training/README_PRETRAIN.md`](src/training/README_PRETRAIN.md) - Detailed documentation

## Quick Start

### 1. Start Training
```bash
cd /Users/debnathk/Documents/drug-synergy/gramseq-main
python -m src.training.pretrain_moleculevae
```

The script will:
- Load Guacamol train/val dataloaders
- Initialize MoleculeVAE model (latent_dim=56)
- Train with Adam optimizer (lr=1e-3)
- Apply ReduceLROnPlateau scheduler
- Save checkpoints to `checkpoints/moleculevae/`
- Track metrics in JSON format

### 2. Monitor Progress
During training, you'll see:
```
Epoch 1/100 [Train]: 100%|████████| 1234/1234 [02:15<00:00, loss=125.34]
Epoch 1/100 [Val]:   100%|████████| 156/156 [00:15<00:00, loss=118.76]

Epoch 1/100:
  Train Loss: 125.3421
  Val Loss:   118.7634
  LR:         1.00e-03
  Time:       150.23s
  → Saved best model (val_loss: 118.7634)
```

### 3. Visualize Training
```bash
python -m src.training.visualize_training
```

Creates plots:
- Loss curves (linear and log scale)
- Learning rate schedule
- Training time per epoch
- Combined summary plot

Saved to: `checkpoints/moleculevae/plots/`

### 4. Test Pretrained Model
```bash
python -m src.training.test_pretrained
```

Runs tests:
- Reconstruction quality on validation set
- Latent space analysis and regularization check
- Interpolation between molecules
- Random generation from prior

## Key Features

### Learning Rate Scheduler
```python
ReduceLROnPlateau(
    mode='min',                    # Minimize validation loss
    factor=0.5,                    # Halve LR on plateau
    patience=5,                    # Wait 5 epochs before reducing
    min_lr=1e-6                    # Minimum learning rate
)
```

### Checkpoint Management
```
checkpoints/moleculevae/
├── checkpoint_best.pt          # Best validation loss
├── checkpoint_latest.pt        # Resume training
├── checkpoint_epoch_10.pt      # Periodic saves
└── training_history.json       # All metrics
```

### Model Architecture
```
MoleculeVAE(
  Encoder: Conv1D(3 layers) → Dense → (z_mean, z_log_var)
  Latent:  56-dimensional space
  Decoder: Dense → GRU(3 layers) → Dense
  Loss:    BCE reconstruction + KL divergence + Grammar constraints
)
```

## Configuration

Edit `CONFIG` in [`pretrain_moleculevae.py`](src/training/pretrain_moleculevae.py):

```python
CONFIG = {
    "latent_dim": 56,              # Latent space dimension
    "learning_rate": 1e-3,         # Initial learning rate
    "num_epochs": 100,             # Total epochs
    "epsilon_std": 0.01,           # Sampling noise
    "scheduler_patience": 5,       # LR scheduler patience
    "scheduler_factor": 0.5,       # LR reduction factor
    "scheduler_min_lr": 1e-6,      # Minimum learning rate
    "grad_clip": 1.0,              # Gradient clipping
}
```

## Loading Pretrained Model

```python
import torch
from src.models.model_zinc import MoleculeVAE
import src.zinc_grammar as G

# Initialize
charset = list(range(G.D))
model = MoleculeVAE(charset, latent_dim=56)

# Load checkpoint
checkpoint = torch.load("checkpoints/moleculevae/checkpoint_best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use for inference
with torch.no_grad():
    z_mean = model.encode(input_data)
    reconstructed = model.decode(z_mean)
```

## Resume Training

The script automatically resumes from the latest checkpoint. To start fresh:
```bash
rm checkpoints/moleculevae/checkpoint_latest.pt
```

## Device Selection

Automatic device selection:
1. **CUDA** (NVIDIA GPU) - if available
2. **MPS** (Apple Silicon) - if available
3. **CPU** - fallback

## Monitoring & Debugging

### View Training History
```python
import json

with open("checkpoints/moleculevae/training_history.json", "r") as f:
    history = json.load(f)

print(f"Best val loss: {min(history['val_loss'])}")
print(f"Final LR: {history['learning_rate'][-1]}")
```

### Check Checkpoint Info
```python
import torch

checkpoint = torch.load("checkpoints/moleculevae/checkpoint_best.pt")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Config: {checkpoint['config']}")
```

## Troubleshooting

### Out of Memory
- Reduce batch size in [`dataloader.py`](src/training/dataloader.py:36):
  ```python
  batch_size=32  # instead of 64
  ```

### Slow Training
- Ensure GPU is being used (check device in output)
- Reduce `num_workers` in dataloader if CPU bottleneck
- Increase batch size if memory allows

### Training Divergence
- Reduce learning rate: `learning_rate=5e-4`
- Increase gradient clipping: `grad_clip=2.0`
- Adjust sampling noise: `epsilon_std=0.005`

## Next Steps

After pretraining:
1. Test reconstruction quality
2. Visualize latent space embeddings
3. Generate new molecules
4. Fine-tune for specific tasks
5. Use encoder for downstream applications

## References

- Model: [`src/models/model_zinc.py`](src/models/model_zinc.py)
- Grammar: [`src/zinc_grammar.py`](src/zinc_grammar.py)
- Data: Guacamol dataset at `data/guacamol/`
- Original paper: Grammar Variational Autoencoder (Kusner et al., 2017)

---

**Questions or issues?** Check [`README_PRETRAIN.md`](src/training/README_PRETRAIN.md) for detailed documentation.
