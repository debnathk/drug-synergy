# Drug Synergy Prediction

End-to-end deep learning pipeline for predicting drug combination synergy scores using drug molecular structure (SMILES) and multi-omics cell-line features (mRNA, miRNA, proteomics) from the DrugComb dataset (via TDC).

---

## Project Structure

```
drug-synergy/
├── main.py                     # Training script
├── evaluate.py                 # Evaluation script
├── analyze_attention.py        # Attention weight analysis and visualization
├── run_main.sh                 # SLURM batch script for cluster experiments
├── requirements.txt            # Python dependencies
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Dataset classes and TDC data splitting
│   ├── utils.py                # Metrics, bootstrap, PrettyTable formatting
│   ├── drugs_embeddings.py     # Drug embedding generation (MolFormer-XL)
│   ├── omics_embeddings.py     # Pre-computed omics embedding generation (legacy)
│   └── models/
│       ├── __init__.py         # Exports: SynergyMLP, SynergyKAN, KANLinear, SynergyModel
│       ├── synergy_model.py    # End-to-end model with omics encoder + attention fusion
│       ├── mlp.py              # MLP prediction head
│       ├── kan.py              # KAN prediction head (B-spline basis)
│       └── language_model.py   # MolFormer-XL wrapper for SMILES embeddings
├── data/
│   ├── raw/<split_type>/       # Train/val/test splits (random/, cold_drug/)
│   ├── embeddings/             # Drug embeddings (.pt) per split type
│   └── models/                 # Saved omics encoder weights
├── logs/                       # Experiment logs, checkpoints, plots
├── assets/                     # Saved figures
└── docs/                       # Problem statement, implementation plan, report
```

---

## Models

### SynergyModel (`src/models/synergy_model.py`)

The main model. Combines frozen drug embeddings with a trainable omics encoder and predicts a synergy score.

| Component | Description |
|---|---|
| `OmicsEncoder` | Per-modality encoder: `BatchNorm -> Linear(in, 512) -> BatchNorm -> ReLU -> Linear(512, 256)` |
| `OmicsAttentionFusion` | Multi-head self-attention (4 heads) over the 3 modality embeddings (mRNA, miRNA, proteomics) -> mean-pooled 256-dim |
| `ProjectionAttention` | Attention-based projection from 256-dim to 768-dim |
| `OmicsFusionModel` | Chains the above: raw omics -> 768-dim fused embedding |
| `SynergyModel` | Concatenates `[drug1_emb, drug2_emb, omics_emb]` (2304-dim) and passes through a prediction head |

### Prediction Heads

| Head | Architecture | Module |
|---|---|---|
| **MLP** | `2304 -> 1024 -> ReLU -> Dropout -> 256 -> ReLU -> 1` | `SynergyMLP` (`src/models/mlp.py`) |
| **KAN** | `2304 -> 128 -> 32 -> 1` (B-spline basis functions) | `SynergyKAN` (`src/models/kan.py`) |

### Drug Embedding Model (`src/models/language_model.py`)

`MolFormerMLM`: Wraps the MolFormer-XL language model. Takes SMILES strings and returns 768-dim molecular embeddings. Used in the preprocessing step; embeddings are frozen during training.

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

Splits are stored under `data/raw/<split_type>/` (e.g. `data/raw/random/train_split.pkl`).

Generate splits:

```bash
python -m src.dataset --split_type all      # both random and cold_drug
python -m src.dataset --split_type random    # only random
```

---

## Drug Embeddings (`src/drugs_embeddings.py`)

Generates MolFormer-XL embeddings for all unique drugs in a given split type. Saved as `data/embeddings/drug_embeddings_<split_type>.pt`.

```bash
python -m src.drugs_embeddings --split_type random
python -m src.drugs_embeddings --split_type cold_drug
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

### Example

```bash
python main.py --exp_name kan_std --standardize --head_type kan --epochs 10
python main.py --exp_name mlp_std --standardize --head_type mlp --epochs 10 --split_type cold_drug
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
python evaluate.py --checkpoint logs/kan_std_random_20260129/checkpoint_best.pt --split test
```

---

## Attention Analysis (`analyze_attention.py`)

Extracts and visualizes omics fusion attention weights from a trained SynergyModel.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | auto-detect | Path to checkpoint |
| `--split` | `val` | Split to analyze |
| `--split_type` | `random` | Data split type |
| `--num_samples` | `1000` | Number of samples to analyze |
| `--batch_size` | `64` | Batch size |
| `--output_dir` | auto-generated | Output directory |

### Outputs

- Average attention matrices (omics fusion)
- Per-head attention heatmaps
- Attention weight distributions
- High vs low synergy attention comparison

---

## Utility Functions (`src/utils.py`)

| Function | Description |
|---|---|
| `compute_regression_metrics(y_true, y_pred)` | Returns `{RMSE, SCC, PCC}` |
| `compute_metrics_with_std(y_true, y_pred, n_bootstrap=200)` | Returns `{RMSE: (mean, std), SCC: (mean, std), PCC: (mean, std)}` via bootstrap |
| `format_metrics_table_pretty(metrics_with_std, title)` | Returns PrettyTable string with `value (std)` format |
| `standardize(target_col)` | Standardizes a column using `StandardScaler` |
| `visualize_omics_data()` | Box plots for mRNA, miRNA, proteomics distributions |
| `visualize_target_col_distribution()` | Box plot for synergy score distribution |

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

# 2. Generate drug embeddings
python -m src.drugs_embeddings --split_type random
python -m src.drugs_embeddings --split_type cold_drug

# 3. Train (KAN head, standardized, random split, 10 epochs)
python main.py --exp_name kan_std --standardize --head_type kan --epochs 10

# 4. Evaluate
python evaluate.py --split_type random --split test

# 5. Analyze attention weights
python analyze_attention.py --split_type random --split val
```
