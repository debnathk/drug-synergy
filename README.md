# Drug Synergy Prediction

End-to-end deep learning pipeline for predicting drug combination synergy scores using drug molecular structure (SMILES) and multi-omics cell-line features (mRNA, miRNA, proteomics) from the DrugComb dataset (via TDC).

---

## Project Structure

```
drug-synergy/
├── main.py                     # Main training script
├── evaluate.py                 # Model evaluation script
├── analyze_attention.py        # Attention weight analysis and visualization
├── ablation.py                 # Omics leave-one-out ablation study
├── data_scaling.py             # Data scaling study (25%, 50%, 75%, 100%)
├── run_main.sh                 # SLURM batch script for cluster experiments
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Dataset classes and TDC data splitting
│   ├── utils.py                # Metrics, bootstrap, PrettyTable formatting
│   ├── drugs_embeddings.py     # Drug embedding generation (MolFormer, ChemBERTa)
│   ├── omics_embeddings.py     # Pre-computed omics embedding generation (legacy)
│   └── models/
│       ├── __init__.py         # Exports: SynergyMLP, SynergyKAN, KANLinear, SynergyModel
│       ├── synergy_model.py    # End-to-end model with omics encoder + attention fusion
│       ├── mlp.py              # MLP prediction head
│       ├── kan.py              # KAN prediction head (B-spline basis)
│       └── language_model.py   # MolFormer-XL and ChemBERTa wrappers for SMILES embeddings
├── data/
│   ├── raw/<split_type>/       # Train/val/test splits (random/, cold_drug/)
│   ├── embeddings/             # Drug embeddings (.pt) per split type and model
│   └── models/                 # Saved omics encoder weights
└── docs/                       # Documentation (excluded from repo)
```

---

## Models

### SynergyModel (`src/models/synergy_model.py`)

The main model combines frozen drug embeddings with a trainable omics encoder and predicts a synergy score.

| Component | Description |
|---|---|
| `OmicsEncoder` | Per-modality encoder: `Linear(in, 1024) -> BatchNorm -> ReLU -> Linear(1024, embed_dim)` |
| `OmicsAttentionFusion` | Multi-head self-attention (4 heads) with learnable CLS token over [CLS, mRNA, miRNA, Proteomics] -> CLS output as fused representation |
| `OmicsFusionModel` | Chains the above: raw omics -> embed_dim fused embedding |
| `SynergyModel` | Concatenates `[drug1_emb, drug2_emb, omics_emb]` (3 × embed_dim) and passes through a prediction head |

### Prediction Heads

| Head | Architecture | Module |
|---|---|---|
| **MLP** | `3*embed_dim -> 1024 -> ReLU -> Dropout -> 256 -> ReLU -> 1` | `SynergyMLP` (`src/models/mlp.py`) |
| **KAN** | `3*embed_dim -> 128 -> 32 -> 1` (B-spline basis functions) | `SynergyKAN` (`src/models/kan.py`) |

### Drug Embedding Models (`src/models/language_model.py`)

| Model | Class | Output Dim | Description |
|---|---|---|---|
| **MolFormer-XL** | `MolFormerMLM` | 768 | IBM's MolFormer language model for SMILES |
| **ChemBERTa** | `ChemBERTaMLM` | 384 | DeepChem's ChemBERTa model for SMILES |

Drug embeddings are frozen during training.

---

## Dataset and Splits (`src/dataset.py`)

Data comes from the **DrugComb** dataset via TDC (`tdc.multi_pred.DrugSyn`).

| Class | Input | Usage |
|---|---|---|
| `DrugSynergyRawOmicsDataset` | Raw omics + drug embeddings | Used by `SynergyModel` (end-to-end training) |
| `DrugSynergyDataset` | Pre-computed omics embeddings + drug embeddings | Legacy, used by standalone `SynergyMLP` |

### Split Types

| Split | TDC Method | Description |
|---|---|---|
| `random` | `data.get_split(method='random')` | Random train/val/test split (default) |
| `cold_drug` | `data.get_split(method='cold_split', column_name='Drug1')` | Test set contains Drug1 entities unseen in training |

Splits are stored under `data/raw/<split_type>/` (e.g., `data/raw/random/train_split.pkl`).

Generate splits:

```bash
python -m src.dataset --split_type all      # both random and cold_drug
python -m src.dataset --split_type random   # only random
```

---

## Drug Embeddings (`src/drugs_embeddings.py`)

Generates drug embeddings for all unique drugs in a given split type. Supports multiple embedding models.

**Output format:** `data/embeddings/drug_embeddings_<split_type>_<model>.pt`

| Argument | Default | Description |
|---|---|---|
| `--split_type` | `random` | Data split type: `random` or `cold_drug` |
| `--model` | `molformer` | Embedding model: `molformer` (768-dim) or `chemberta` (384-dim) |

```bash
# MolFormer embeddings (768-dim)
python -m src.drugs_embeddings --split_type random --model molformer
python -m src.drugs_embeddings --split_type cold_drug --model molformer

# ChemBERTa embeddings (384-dim)
python -m src.drugs_embeddings --split_type random --model chemberta
python -m src.drugs_embeddings --split_type cold_drug --model chemberta
```

---

## Training (`main.py`)

Trains a `SynergyModel` with either an MLP or KAN prediction head. Uses MSE loss with AdamW optimizer. Reports RMSE, SCC (Spearman), PCC (Pearson) with bootstrap std in a PrettyTable at the end.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--exp_name` | auto-generated | Experiment name |
| `--standardize` | `False` | Standardize targets (mean=0, std=1) |
| `--epochs` | `100` | Number of epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--target` | `Synergy_ZIP` | Target column (`Synergy_ZIP`, `Synergy_Bliss`, `Synergy_Loewe`, `Synergy_HSA`) |
| `--head_type` | `kan` | Prediction head (`mlp` or `kan`) |
| `--grid_size` | `5` | KAN grid size (only if `head_type=kan`) |
| `--split_type` | `random` | Data split (`random` or `cold_drug`) |
| `--drug_emb_dim` | `768` | Drug embedding dimension (768 for MolFormer, 384 for ChemBERTa) |
| `--drug_model` | `None` | Drug embedding model (`molformer` or `chemberta`) for loading embeddings |

### Example

```bash
# Train with MolFormer embeddings (default)
python main.py --exp_name kan_std --standardize --head_type kan --epochs 50

# Train with ChemBERTa embeddings
python main.py --exp_name mlp_chemberta --standardize --head_type mlp --epochs 50 \
    --drug_model chemberta --drug_emb_dim 384

# Cold drug split
python main.py --exp_name mlp_cold --standardize --head_type mlp --split_type cold_drug
```

### Outputs (per run in `logs/<exp_name>_<timestamp>/`)

- `training.log` -- full training log
- `config.json` -- hyperparameters and config
- `metrics.json` -- per-epoch losses + best RMSE/SCC/PCC with std
- `checkpoint_best.pt` -- best model checkpoint
- `checkpoint_latest.pt` -- latest checkpoint
- `omics_encoder.pt` -- trained omics encoder weights
- `loss_plot_std.png` -- train/val loss curve
- `main.py` -- copy of the training script (for reproducibility)

---

## Evaluation (`evaluate.py`)

Evaluates a trained model on a chosen split. Reports MSE, RMSE, MAE, R-squared, Pearson, Spearman and a PrettyTable with RMSE, SCC, PCC (with bootstrap std).

The script automatically reads `drug_model` and `drug_emb_dim` from the checkpoint config to load the correct embeddings.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `attention` | Model type: `attention` (SynergyModel), `mlp` (legacy SynergyMLP), or `both` |
| `--checkpoint` | auto-detect | Path to checkpoint (auto-detects latest if omitted) |
| `--split` | `test` | Split to evaluate: `train`, `val`, `test` |
| `--split_type` | `random` | Data split type: `random` or `cold_drug` |
| `--target` | `Synergy_ZIP` | Target column (must match training) |
| `--batch_size` | `64` | Batch size |
| `--output_dir` | auto-generated | Output directory |

### Example

```bash
python evaluate.py --split_type random --split test
python evaluate.py --checkpoint logs/kan_std_random_*/checkpoint_best.pt --split test
```

---

## Ablation Study (`ablation.py`)

Omics leave-one-out ablation study. Trains SynergyModel under four conditions (drug embeddings always active):

| Condition | Description |
|---|---|
| `no_mrna` | mRNA zeroed, miRNA + Proteomics active |
| `no_mirna` | miRNA zeroed, mRNA + Proteomics active |
| `no_prot` | Proteomics zeroed, mRNA + miRNA active |
| `no_omics` | All omics zeroed (drug embeddings only baseline) |

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--exp_name` | auto-generated | Experiment name prefix |
| `--conditions` | all | Conditions to run (space-separated) |
| `--standardize` | `False` | Standardize targets |
| `--epochs` | `50` | Number of epochs |
| `--head_type` | `mlp` | Prediction head (`mlp` or `kan`) |
| `--split_type` | `random` | Data split type |
| `--target` | `Synergy_ZIP` | Target column |

### Example

```bash
# Run all ablation conditions
python ablation.py --standardize --epochs 50 --head_type mlp --split_type random

# Run specific conditions
python ablation.py --conditions no_mrna no_mirna --standardize --epochs 50
```

---

## Data Scaling Study (`data_scaling.py`)

Trains SynergyModel on progressively larger fractions of training data (25%, 50%, 75%, 100%) while keeping validation and test sets fixed. Reveals whether the model would benefit from more data.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--exp_name` | auto-generated | Experiment name prefix |
| `--fractions` | `0.25 0.50 0.75 1.00` | Training data fractions to evaluate |
| `--standardize` | `False` | Standardize targets |
| `--epochs` | `50` | Number of epochs |
| `--head_type` | `mlp` | Prediction head |
| `--target` | `Synergy_ZIP` | Target column |
| `--seed` | `42` | Random seed for reproducible subsampling |

### Example

```bash
python data_scaling.py --standardize --epochs 50 --head_type mlp
```

### Outputs

- Per-fraction training logs and checkpoints
- `scaling_curves.png` -- Performance vs. data fraction plot
- `summary.json` -- Aggregated metrics across fractions

---

## Attention Analysis (`analyze_attention.py`)

Extracts and visualizes omics fusion attention weights from a trained SynergyModel.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | auto-detect | Path to checkpoint |
| `--split` | `test` | Split to analyze |
| `--split_type` | `random` | Data split type |
| `--target` | `Synergy_ZIP` | Target column |
| `--num_samples` | all | Number of samples to analyze |
| `--batch_size` | `64` | Batch size |
| `--output_dir` | auto-generated | Output directory |

### Outputs

- CLS-to-modality attention weights (which omics the model focuses on)
- Full self-attention matrix heatmaps
- Per-head attention analysis
- Attention patterns by synergy level (high vs low)
- `attention_report.txt` -- Summary statistics

---

## Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error |
| **SCC** | Spearman Correlation Coefficient (rank correlation) |
| **PCC** | Pearson Correlation Coefficient (linear correlation) |

All three are reported with bootstrap standard deviation (200 samples) in a PrettyTable:

```
+--------+-----------------+
| Metric |      Value      |
+--------+-----------------+
|  RMSE  | 4.0005 (0.0288) |
|  SCC   | 0.6184 (0.0041) |
|  PCC   | 0.6681 (0.0050) |
+--------+-----------------+
```

---

## Quickstart

```bash
# 1. Generate data splits
python -m src.dataset --split_type all

# 2. Generate drug embeddings (choose one or both)
python -m src.drugs_embeddings --split_type random --model molformer
python -m src.drugs_embeddings --split_type random --model chemberta

# 3. Train (MLP head, standardized, random split, 50 epochs)
python main.py --exp_name mlp_std --standardize --head_type mlp --epochs 50

# 4. Evaluate on test set
python evaluate.py --split_type random --split test

# 5. Analyze attention weights
python analyze_attention.py --split_type random --split test

# 6. (Optional) Run ablation study
python ablation.py --standardize --epochs 50 --head_type mlp

# 7. (Optional) Run data scaling study
python data_scaling.py --standardize --epochs 50 --head_type mlp
```

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- transformers
- tdc (Therapeutics Data Commons)
- scikit-learn
- scipy
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- prettytable
