# MoleculeVAE Pretraining Pipeline Implementation Plan

## Goal
Set up a complete pretraining pipeline for the MoleculeVAE model. **START with 1M molecules for quick results**, with infrastructure designed to scale to millions/billions later.

## User Requirements
- **Initial Dataset**: Start with 1M molecules for quick proof-of-concept
- **Scalability**: Design pipeline to easily scale to 10M → 100M → 1B+ molecules
- **Hardware**: Single GPU with CUDA
- **Training**: Basic training loop with checkpointing and logging
- **Quick Results**: Get initial trained model within hours, not days

## Current State
The PyTorch MoleculeVAE model is **fully implemented** at `src/models/model_zinc.py` with:
- Complete encoder/decoder architecture
- VAE loss with grammar constraints
- Forward pass and inference methods
- Existing preprocessing function `smiles2onehot()` in `src/utils.py`

## 🚀 Quick Start Summary

**Goal**: Get a trained MoleculeVAE model in **~3 hours total time**

**Steps**:
1. **Download** GuacaMol dataset (1M molecules) - 1 minute
2. **Preprocess** SMILES to one-hot tensors - 12 minutes
3. **Train** VAE for 50 epochs - 2-3 hours
4. **Result**: Trained model capable of encoding/generating drug-like molecules

**Requirements**:
- **Storage**: 13 GB free space
- **GPU**: 8GB+ VRAM recommended
- **Time**: ~3 hours start to finish

**Scalability**: Infrastructure designed to scale to billions of molecules later!

---

## Implementation Plan

### Phase 1: Data Download Script
**File**: `src/data/download_data.py`

**Purpose**: Download large-scale SMILES datasets (millions to billions of molecules)

**Recommended Datasets (Ordered by Scale)**:

**Phase 1: START HERE - GuacaMol (1M molecules) ✓ RECOMMENDED FOR QUICK START**
- **Size**: 1.6M total (use 1M for quick start)
- **Access**: https://ndownloader.figshare.com/files/13612760 (training set)
- **Download Time**: ~1 minute
- **Quality**: Pre-filtered, drug-like molecules from ChEMBL
- **Storage**: ~100 MB compressed, ~12 GB preprocessed
- **Training Time**: ~2-3 hours for 50 epochs
- **Perfect for**: Proof-of-concept, testing pipeline, quick results

**Phase 2: Scale to 10M - ZINC250K + ChEMBL**
- **Size**: ~2-10M molecules
- **Access**: Kaggle ZINC250K + ChEMBL downloads
- **Storage**: ~1-2 GB raw, ~120 GB preprocessed
- **Training Time**: ~10-15 hours

**Phase 3: Scale to 100M - ZINC20**
- **Size**: 100M+ molecules
- **Access**: https://zinc20.docking.org/
- **Storage**: ~20 GB raw, ~200 GB cache (streaming mode)
- **Training Time**: 6-12 days

**Phase 4: Scale to 1B+ - Enamine REAL**
- **Size**: 1.4B+ molecules
- **Access**: https://enamine.net/compound-collections/real-compounds/real-database
- **Storage**: ~200 GB raw, ~200 GB cache (streaming mode)
- **Training Time**: ~10 epochs = 6-12 days

**Key Features**:
```python
class DatasetDownloader:
    - download_file(url) -> downloads with progress bar
    - handle_large_files() -> streaming download for multi-GB files
    - convert_sdf_to_smiles() -> for PubChem SDF files
    - deduplicate_smiles() -> removes duplicates using RDKit canonicalization
    - split_train_val() -> 95/5 split (more training data for large datasets)
    - incremental_processing() -> process in chunks to avoid memory issues
```

**Scalability Features**:
- Support for streaming downloads (don't load entire dataset into memory)
- Parallel processing of multiple source files
- Resume capability for interrupted downloads
- Automatic sharding for datasets >1B molecules

**Output**:
- `data/raw/train_smiles_part*.txt` (sharded for large datasets, e.g., 100M per file)
- `data/raw/val_smiles.txt` (5% of total)
- `data/raw/download_stats.json` (metadata including total size)

### Phase 2: Preprocessing Script
**File**: `src/data/preprocess_smiles.py`

**Purpose**: Convert raw SMILES to one-hot tensors using existing `smiles2onehot()` function

**Key Implementation Details**:
```python
class SMILESPreprocessor:
    - Use multiprocessing (8 cores) for parallel parsing
    - Process in batches of 1000 SMILES
    - Use existing utils.smiles2onehot() function
    - Handle invalid SMILES gracefully (skip and log)
    - Save to HDF5 with gzip compression (level 4)
```

**HDF5 Storage Format**:
```python
with h5py.File('smiles_train.h5', 'w') as f:
    dset = f.create_dataset(
        'smiles_onehot',
        shape=(n_samples, 277, 111),  # MAX_LEN=277, n_productions=111
        dtype='float32',
        chunks=(1000, 277, 111),
        compression='gzip',
        compression_opts=4
    )
```

**Scalability Approach for Large Datasets**:

**For 10M-100M molecules**:
- Preprocess all and store in sharded HDF5 files
- Storage: ~12 GB per 1M molecules compressed
- Total for 100M: ~1.2 TB

**For 100M-1B+ molecules**:
- **On-the-fly encoding** (recommended for billion-scale)
- Preprocess in streaming fashion during training
- Cache preprocessed batches to disk (LRU cache)
- Storage: Only store raw SMILES (~10-20 GB per 100M)

**Hybrid Approach** (RECOMMENDED):
```python
class StreamingSMILESPreprocessor:
    - Preprocess validation set fully (smaller, fixed)
    - Stream training set with disk caching
    - Use multiprocessing pool for parallel parsing
    - Implement LRU cache for recently used batches
```

**Performance Estimates**:
- **Small scale (10M)**: ~2 hours preprocessing with 8 cores, ~120GB storage
- **Medium scale (100M)**: ~20 hours preprocessing OR streaming with caching
- **Large scale (1B)**: Streaming only, ~200GB cache storage

**Output**:
- **Small/Medium datasets**: `data/processed/smiles_train_part*.h5` (sharded)
- **Large datasets**: `data/processed/cache/` (LRU cache directory)
- `data/processed/smiles_val.h5` (always fully preprocessed)
- `data/processed/metadata.json` (statistics)
- `data/logs/invalid_smiles.txt` (failed SMILES)

### Phase 3: PyTorch Dataset Class
**File**: `src/data/dataset.py`

**Purpose**: Memory-efficient loading with support for both preprocessed and streaming modes

**Key Implementation**:
```python
class MoleculeVAEDataset(Dataset):
    """Supports both preprocessed HDF5 and streaming SMILES"""

    def __init__(self, data_source, mode='hdf5', cache_dir=None):
        self.mode = mode  # 'hdf5' or 'streaming'

        if mode == 'hdf5':
            # For preprocessed data (small/medium scale)
            self.h5_files = glob.glob(f"{data_source}/smiles_train_part*.h5")
            self.load_metadata()

        elif mode == 'streaming':
            # For large-scale datasets (100M+ molecules)
            self.smiles_files = glob.glob(f"{data_source}/train_smiles_part*.txt")
            self.cache = DiskCache(cache_dir, max_size_gb=200)

    def __getitem__(self, idx):
        if self.mode == 'hdf5':
            # Load from HDF5
            file_idx, local_idx = self._get_file_and_index(idx)
            return self._load_from_hdf5(file_idx, local_idx)

        elif self.mode == 'streaming':
            # Check cache first
            if idx in self.cache:
                return self.cache[idx]

            # Otherwise, parse on-the-fly
            smiles = self._get_smiles(idx)
            tensor = smiles2onehot(smiles)
            self.cache[idx] = tensor  # Cache for future
            return torch.from_numpy(tensor).float()
```

**Streaming with Disk Cache**:
```python
class DiskCache:
    """LRU cache for preprocessed tensors"""
    def __init__(self, cache_dir, max_size_gb=200):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_size_bytes = max_size_gb * 1024**3
        self.index = {}  # idx -> (filepath, timestamp)

    def __contains__(self, idx):
        return idx in self.index

    def __getitem__(self, idx):
        # Load from cache file
        filepath = self.cache_dir / f"{idx}.npy"
        return torch.from_numpy(np.load(filepath))

    def __setitem__(self, idx, tensor):
        # Save to cache with LRU eviction
        self._evict_if_needed()
        filepath = self.cache_dir / f"{idx}.npy"
        np.save(filepath, tensor.numpy())
        self.index[idx] = (filepath, time.time())

    def _evict_if_needed(self):
        # Remove oldest cached items if size exceeds limit
        current_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npy"))
        if current_size > self.max_size_bytes:
            # Remove 20% oldest items
            ...
```

**DataLoader Configuration**:
```python
DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

### Phase 4: Training Configuration
**File**: `src/training/config.py`

**Purpose**: Centralized hyperparameter management

**Key Hyperparameters**:
```python
@dataclass
class TrainingConfig:
    # Model
    latent_dim: int = 56

    # Training
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Optimization
    grad_clip: float = 1.0
    lr_scheduler: str = 'reduce_on_plateau'
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Checkpointing
    save_every: int = 5
    keep_last_n: int = 3

    # Early stopping
    early_stopping_patience: int = 20
```

### Phase 5: Main Training Script
**File**: `src/training/train_vae.py`

**Purpose**: Complete training loop with validation, checkpointing, and logging

**Key Components**:
```python
class VAETrainer:
    def __init__(self, model_config, train_config):
        # Initialize model from model_zinc.py
        self.model = MoleculeVAE(charset, latent_dim)
        self.optimizer = torch.optim.Adam(...)
        self.scheduler = ReduceLROnPlateau(...)

    def train_epoch(self):
        # Training loop
        for batch in train_loader:
            x_recon, z_mean, z_log_var = model(x)
            loss = model.vae_loss(x, x_recon, z_mean, z_log_var)
            loss.backward()
            optimizer.step()

    def validate(self):
        # Validation loop

    def save_checkpoint(self, is_best=False):
        # Save model state, optimizer, scheduler

    def train(self):
        # Main training loop
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.save_checkpoint(is_best=...)
```

**Training Features**:
- Forward pass using existing `model.forward(x)`
- Loss computation using existing `model.vae_loss(x, x_recon, z_mean, z_log_var)`
- Gradient clipping for stability
- ReduceLROnPlateau scheduler
- Checkpointing every 5 epochs
- Early stopping after 20 epochs without improvement
- TensorBoard logging

### Phase 6: Command-Line Interface
Add argparse to `train_vae.py`:
```bash
# From scratch
python src/training/train_vae.py \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.001

# Resume from checkpoint
python src/training/train_vae.py \
    --resume checkpoints/vae_pretrain/epoch_50.pth
```

## File Structure (After Implementation)

```
gramseq-main/
├── data/
│   ├── raw/                      # Raw SMILES files
│   │   ├── train_smiles.txt     # ~2.4M molecules
│   │   └── val_smiles.txt       # ~300K molecules
│   ├── processed/                # Preprocessed tensors
│   │   ├── smiles_train.h5      # ~28 GB
│   │   ├── smiles_val.h5        # ~3 GB
│   │   └── metadata.json
│   └── logs/
│       └── invalid_smiles.txt
├── src/
│   ├── data/                     # NEW
│   │   ├── __init__.py
│   │   ├── download_data.py      # NEW: Download datasets
│   │   ├── preprocess_smiles.py  # NEW: Batch preprocessing
│   │   └── dataset.py            # NEW: PyTorch Dataset
│   ├── training/                 # NEW
│   │   ├── __init__.py
│   │   ├── config.py             # NEW: Configuration
│   │   └── train_vae.py          # NEW: Training script
│   ├── models/
│   │   └── model_zinc.py         # EXISTING: Use as-is
│   ├── utils.py                  # EXISTING: Use smiles2onehot()
│   └── zinc_grammar.py           # EXISTING: Grammar masks
├── checkpoints/
│   └── vae_pretrain/
│       ├── config.yaml
│       ├── epoch_*.pth
│       └── best_model.pth
└── logs/
    └── training/
        └── tensorboard/
```

## Critical Files (No Modifications Needed)

These files are already complete and will be **imported/used as-is**:

1. **`src/models/model_zinc.py`**
   - Fully implemented MoleculeVAE
   - Use: `model = MoleculeVAE(charset, latent_dim)`
   - Use: `x_recon, z_mean, z_log_var = model(x)`
   - Use: `loss = model.vae_loss(x, x_recon, z_mean, z_log_var)`

2. **`src/utils.py`**
   - Contains `smiles2onehot()` function
   - Use: `tensor = utils.smiles2onehot(smiles_string)`
   - Returns: numpy array of shape (277, 111)

3. **`src/zinc_grammar.py`**
   - Grammar masks and production rules
   - Imported automatically by model_zinc.py
   - No direct usage needed

## Quick Start Workflow (1M Molecules - Get Results in Hours!)

### Step 1: Download GuacaMol (1M molecules)
```bash
python src/data/download_data.py \
    --source guacamol \
    --output_dir data/raw \
    --num_molecules 1000000
# Download time: ~1 minute
# Output: data/raw/train_smiles.txt (~900K), data/raw/val_smiles.txt (~100K)
```

### Step 2: Preprocess 1M Molecules
```bash
python src/data/preprocess_smiles.py \
    --input data/raw/train_smiles.txt \
    --output data/processed/smiles_train.h5 \
    --num_workers 8
# Preprocessing time: ~12 minutes (8 cores)
# Output: data/processed/smiles_train.h5 (~11 GB)

python src/data/preprocess_smiles.py \
    --input data/raw/val_smiles.txt \
    --output data/processed/smiles_val.h5 \
    --num_workers 8
# Preprocessing time: ~1 minute
# Output: data/processed/smiles_val.h5 (~1.2 GB)
```

### Step 3: Train Model (1M Molecules)
```bash
python src/training/train_vae.py \
    --train_data data/processed/smiles_train.h5 \
    --val_data data/processed/smiles_val.h5 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.001
# Training time: ~2-3 hours (RTX 3080)
# Output: checkpoints/vae_pretrain/best_model.pth
```

### Step 4: Monitor Training
```bash
tensorboard --logdir logs/training
```

## Resource Requirements by Scale

### Quick Start: 1M Molecules ✓ START HERE
- **Raw SMILES**: ~100 MB compressed
- **Preprocessed HDF5**: ~12 GB (train ~11 GB + val ~1.2 GB)
- **Checkpoints**: ~500 MB
- **Total**: ~13 GB
- **Training Time**: 2-3 hours (50 epochs, RTX 3080)
- **Perfect for**: Testing pipeline, quick results

### Scale Up: 10M Molecules
- **Raw SMILES**: ~2 GB
- **Preprocessed HDF5**: ~120 GB compressed
- **Checkpoints**: ~1.5 GB
- **Total**: ~125 GB
- **Training Time**: ~15-20 hours

### Large Scale: 100M Molecules
- **Raw SMILES**: ~20 GB
- **Option A (Full preprocessing)**: ~1.2 TB compressed
- **Option B (Streaming + cache)**: ~20 GB raw + ~200 GB cache = ~220 GB (RECOMMENDED)
- **Checkpoints**: ~1.5 GB
- **Total**: ~225 GB (streaming)
- **Training Time**: 6-12 days

### Billion Scale: 1B+ Molecules
- **Raw SMILES**: ~200 GB
- **Streaming only**: ~200 GB raw + ~200 GB cache = ~400 GB
- **Checkpoints**: ~1.5 GB
- **Total**: ~405 GB
- **Training Time**: ~10 epochs = 6-12 days

### GPU Memory
- Batch size 128: ~4-6 GB VRAM
- Batch size 256: ~8-10 GB VRAM (recommended for 1M molecules)
- Model parameters: ~30 MB
- **Recommended: GPU with 8GB+ VRAM (1M molecules), 16GB+ for larger datasets**

## Validation Metrics

During training, monitor:
1. **Total Loss**: Should decrease steadily
2. **Reconstruction Loss**: Converges to ~50-100
3. **KL Divergence**: Stabilizes around 5-10
4. **Validation Loss**: Should track training loss

After training, evaluate:
1. **Reconstruction Accuracy**: % correctly reconstructed SMILES
2. **Validity**: % of decoded SMILES that are chemically valid
3. **Uniqueness**: % of unique molecules generated
4. **Novelty**: % of molecules not in training set

## Implementation Order

1. **`src/data/download_data.py`** - Download and merge datasets
2. **`src/data/preprocess_smiles.py`** - Convert SMILES to tensors
3. **`src/data/dataset.py`** - PyTorch Dataset class
4. **`src/training/config.py`** - Configuration dataclasses
5. **`src/training/train_vae.py`** - Main training loop
6. **Create directory structure** - data/, checkpoints/, logs/
7. **Test with small dataset** - Verify pipeline with 1000 samples
8. **Run full preprocessing** - Process all 2.7M molecules
9. **Start training** - Train for 100 epochs with early stopping

## Dependencies

Ensure these packages are installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py tqdm rdkit tensorboard pyyaml requests
```

## Success Criteria

✓ Download and preprocess 2.7M SMILES molecules
✓ Create efficient HDF5 dataset (~30-40GB)
✓ Implement training loop with checkpointing
✓ Train for 50-100 epochs with validation
✓ Achieve stable VAE loss convergence
✓ Save best model checkpoint
✓ Generate valid SMILES from trained model
