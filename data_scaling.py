"""
Data Scaling Study for Drug Synergy Prediction.

Trains SynergyModel on progressively larger fractions of training data
(25%, 50%, 75%, 100%) while keeping validation and test sets fixed.
This analysis reveals whether the model would benefit from more data or has saturated.

Key features:
  - Only training data is subsampled; val/test remain fixed
  - Random split type only (as specified)
  - Target: Synergy_ZIP (default)
  - Reproducible subsampling via fixed seed
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import shutil

from pathlib import Path
from src.dataset import DrugSynergyRawOmicsDataset
from src.models.synergy_model import SynergyModel
from src.utils import compute_metrics_with_std, format_metrics_table_pretty
from tqdm import tqdm
import logging
from datetime import datetime
import json
import argparse

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"
LOG_PATH = ROOT / "logs"
ASSETS_PATH = ROOT / "assets"
MODELS_PATH = ROOT / "data/models"

DEFAULT_FRACTIONS = (0.25, 0.50, 0.75, 1.0)

FRACTION_LABELS = {
    0.25: "25%",
    0.50: "50%",
    0.75: "75%",
    1.0: "100%",
}


# ---------------------------------------------------------------------------
# TargetScaler (identical to main.py)
# ---------------------------------------------------------------------------

class TargetScaler:
    """Standardizes targets to mean=0, std=1 for better training stability."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, targets):
        targets = np.array(targets).flatten()
        self.mean = float(np.mean(targets))
        self.std = float(np.std(targets))
        self.fitted = True
        return self

    def transform(self, targets):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        targets = np.array(targets).flatten()
        return (targets - self.mean) / self.std

    def fit_transform(self, targets):
        self.fit(targets)
        return self.transform(targets)

    def inverse_transform(self, standardized_targets):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        standardized_targets = np.array(standardized_targets).flatten()
        return standardized_targets * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean, "std": self.std, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, params):
        scaler = cls()
        scaler.mean = params["mean"]
        scaler.std = params["std"]
        scaler.fitted = params["fitted"]
        return scaler


# ---------------------------------------------------------------------------
# Path helpers (identical to main.py)
# ---------------------------------------------------------------------------

def get_raw_path(split_type: str) -> Path:
    subdir = ROOT / "data" / "raw" / split_type
    flat_train = ROOT / "data" / "raw" / "train_split.pkl"
    if subdir.exists() and (subdir / "train_split.pkl").exists():
        return subdir
    if split_type == "random" and flat_train.exists():
        return ROOT / "data" / "raw"
    return subdir


def get_drug_emb_path(split_type: str, drug_model: str = None) -> Path:
    """Resolve drug embeddings path; fallback to older formats for backward compat.
    
    Priority order:
    1. drug_embeddings_{split_type}_{model}.pt (new format with model name)
    2. drug_embeddings_{split_type}.pt (intermediate format)
    3. drug_embeddings.pt (legacy format)
    """
    if drug_model:
        new_format = EMBEDDINGS_PATH / f"drug_embeddings_{split_type}_{drug_model}.pt"
        if new_format.exists():
            return new_format
    
    specific = EMBEDDINGS_PATH / f"drug_embeddings_{split_type}.pt"
    default = EMBEDDINGS_PATH / "drug_embeddings.pt"
    return specific if specific.exists() else default


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_dataframe(df, fraction, seed=42):
    """
    Randomly subsample a DataFrame while maintaining reproducibility.
    
    Args:
        df: pandas DataFrame to subsample
        fraction: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Subsampled DataFrame with reset index
    """
    if fraction >= 1.0:
        return df.copy().reset_index(drop=True)
    n_samples = int(len(df) * fraction)
    return df.sample(n=n_samples, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Logging / saving helpers (identical to main.py/ablation.py)
# ---------------------------------------------------------------------------

def setup_logging(experiment_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_PATH / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    return run_dir, logger


def save_config(run_dir: Path, config: dict, logger: logging.Logger):
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_file}")


def save_metrics(run_dir: Path, train_losses: list, val_losses: list,
                 logger: logging.Logger, best_val_metrics_with_std=None,
                 test_metrics_with_std=None):
    metrics_file = run_dir / "metrics.json"
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_train_loss": min(train_losses),
        "best_val_loss": min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }
    if best_val_metrics_with_std:
        for name in ("RMSE", "SCC", "PCC"):
            mean, std = best_val_metrics_with_std.get(name, (None, None))
            metrics[f"best_val_{name}"] = mean
            metrics[f"best_val_{name}_std"] = std
    if test_metrics_with_std:
        for name in ("RMSE", "SCC", "PCC"):
            mean, std = test_metrics_with_std.get(name, (None, None))
            metrics[f"test_{name}"] = mean
            metrics[f"test_{name}_std"] = std
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, run_dir,
                    is_best, logger, scaler=None, config=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.to_dict()
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, run_dir / "checkpoint_latest.pt")
    if is_best:
        torch.save(checkpoint, run_dir / "checkpoint_best.pt")
        logger.info(f"Best model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}")


def copy_script(run_dir: Path, logger: logging.Logger):
    dst = run_dir / "data_scaling.py"
    shutil.copy2(Path(__file__).resolve(), dst)
    logger.info(f"Script copied to {dst}")


def plot_losses(train_losses, val_losses, num_epochs, run_dir, fraction, standardized, logger):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    epochs = range(1, num_epochs + 1)
    df = pd.DataFrame({
        "Epoch": list(epochs) * 2,
        "Loss": train_losses + val_losses,
        "Type": ["Train Loss"] * num_epochs + ["Validation Loss"] * num_epochs,
    })
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Epoch", y="Loss", hue="Type", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    std_text = "with" if standardized else "without"
    frac_label = FRACTION_LABELS.get(fraction, f"{fraction*100:.0f}%")
    plt.title(f"Data Scaling: {frac_label} training data — Loss ({std_text} standardized targets)")
    plt.legend(title="")
    filename = f"loss_plot_{'std' if standardized else 'wo_std'}.png"
    run_path = run_dir / filename
    plt.savefig(run_path, dpi=300, bbox_inches="tight")
    logger.info(f"Loss plot saved to {run_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Train / eval loops (identical to main.py)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(loader, model, optimizer, criterion, logger):
    model.train()
    total_loss = 0
    batch_losses = []
    for d1, d2, mrna, mirna, prot, y in loader:
        d1, d2 = d1.to(device), d2.to(device)
        mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(d1, d2, mrna, mirna, prot)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss * d1.size(0)
        batch_losses.append(batch_loss)
    avg_loss = total_loss / len(loader.dataset)
    if batch_losses:
        logger.debug(
            f"Train batch losses - Min: {min(batch_losses):.4f}, "
            f"Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}"
        )
    return avg_loss


def eval_epoch(loader, model, criterion, logger, collect_predictions=False):
    model.eval()
    total_loss = 0
    batch_losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for d1, d2, mrna, mirna, prot, y in loader:
            d1, d2 = d1.to(device), d2.to(device)
            mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
            y = y.to(device)
            preds = model(d1, d2, mrna, mirna, prot)
            loss = criterion(preds, y)
            batch_loss = loss.item()
            total_loss += batch_loss * d1.size(0)
            batch_losses.append(batch_loss)
            if collect_predictions:
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    if batch_losses:
        logger.debug(
            f"Val batch losses - Min: {min(batch_losses):.4f}, "
            f"Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}"
        )
    if collect_predictions and all_preds:
        y_true = np.concatenate(all_targets, axis=0).flatten()
        y_pred = np.concatenate(all_preds, axis=0).flatten()
        return avg_loss, y_true, y_pred
    return avg_loss, None, None


# ---------------------------------------------------------------------------
# Single-fraction training run
# ---------------------------------------------------------------------------

def run_fraction(fraction: float, train_df_full, val_df_orig, test_df_orig, args,
                 mrna_dim, mirna_dim, prot_dim, drug_emb_path):
    """
    Train and evaluate for one data fraction.

    After training, loads the best checkpoint and evaluates on the held-out
    test split. Test metrics (RMSE, SCC, PCC with bootstrap std) are used
    in the cross-fraction summary table.

    Returns:
        test_metrics_with_std: dict {RMSE: (mean, std), SCC: ..., PCC: ...}
        run_dir: Path to the fraction's log directory
        best_val_loss: Best validation loss achieved
        train_samples: Number of training samples used
    """
    frac_label = FRACTION_LABELS.get(fraction, f"{fraction*100:.0f}%")
    frac_str = f"frac_{fraction:.2f}".replace(".", "_")
    exp_name = f"{args.exp_name}_{frac_str}"
    run_dir, logger = setup_logging(exp_name)
    copy_script(run_dir, logger)

    logger.info("=" * 80)
    logger.info(f"Data Scaling: {frac_label} of training data (fraction={fraction})")
    logger.info("=" * 80)

    # Subsample training data
    train_df_sub = subsample_dataframe(train_df_full, fraction, seed=args.seed)
    logger.info(f"Subsampled training data: {len(train_df_sub)}/{len(train_df_full)} samples ({frac_label})")

    # Target standardization — fit on subsampled train targets
    target_scaler = None
    if args.standardize:
        logger.info("Applying target standardization...")
        target_scaler = TargetScaler()
        original_train_targets = train_df_sub[args.target].values.copy()
        original_val_targets = val_df_orig[args.target].values.copy()
        original_test_targets = test_df_orig[args.target].values.copy()
        target_scaler.fit(original_train_targets)
        
        train_df_std = train_df_sub.copy()
        val_df_std = val_df_orig.copy()
        test_df_std = test_df_orig.copy()
        train_df_std[args.target] = target_scaler.transform(original_train_targets)
        val_df_std[args.target] = target_scaler.transform(original_val_targets)
        test_df_std[args.target] = target_scaler.transform(original_test_targets)
        
        train_dataset = DrugSynergyRawOmicsDataset(train_df_std, drug_emb_path, args.target)
        val_dataset = DrugSynergyRawOmicsDataset(val_df_std, drug_emb_path, args.target)
        test_dataset = DrugSynergyRawOmicsDataset(test_df_std, drug_emb_path, args.target)
        
        logger.info(f"Target scaler fitted - Mean: {target_scaler.mean:.4f}, Std: {target_scaler.std:.4f}")
        logger.info(f"Original target range: [{original_train_targets.min():.2f}, {original_train_targets.max():.2f}]")
        logger.info(f"Standardized target range: [{train_df_std[args.target].min():.2f}, {train_df_std[args.target].max():.2f}]")
    else:
        train_dataset = DrugSynergyRawOmicsDataset(train_df_sub, drug_emb_path, args.target)
        val_dataset = DrugSynergyRawOmicsDataset(val_df_orig, drug_emb_path, args.target)
        test_dataset = DrugSynergyRawOmicsDataset(test_df_orig, drug_emb_path, args.target)

    logger.info(
        f"Train samples: {len(train_dataset)}, "
        f"Val samples: {len(val_dataset)}, "
        f"Test samples: {len(test_dataset)}"
    )

    config = {
        "data_fraction": fraction,
        "data_fraction_label": frac_label,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "target_column": args.target,
        "standardize_targets": args.standardize,
        "split_type": "random",
        "subsample_seed": args.seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "train_samples_full": len(train_df_full),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "model": f"SynergyModel ({args.head_type.upper()} Head)",
        "head_type": args.head_type,
        "grid_size": args.grid_size if args.head_type == "kan" else None,
        "drug_emb_dim": args.drug_emb_dim,
        "drug_model": args.drug_model,
        "mrna_dim": mrna_dim,
        "mirna_dim": mirna_dim,
        "prot_dim": prot_dim,
    }
    if target_scaler is not None:
        config["scaler_mean"] = target_scaler.mean
        config["scaler_std"] = target_scaler.std
    save_config(run_dir, config, logger)

    logger.info(f"Device: {device}")
    logger.info(f"Target column: {args.target}")
    logger.info(f"Head type: {args.head_type.upper()}")
    logger.info(f"Omics dimensions - mRNA: {mrna_dim}, miRNA: {mirna_dim}, Proteomics: {prot_dim}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(
        f"Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, "
        f"Test batches: {len(test_loader)}"
    )

    model = SynergyModel(
        mrna_dim=mrna_dim,
        mirna_dim=mirna_dim,
        prot_dim=prot_dim,
        embed_dim=args.drug_emb_dim,
        head_type=args.head_type,
        grid_size=args.grid_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    num_epochs = args.epochs
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_val_metrics_with_std = None
    test_metrics_with_std = None

    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    pbar = tqdm(range(num_epochs), desc=f"[{frac_label}]")
    for epoch in pbar:
        train_loss = train_epoch(train_loader, model, optimizer, criterion, logger)
        val_loss, y_true, y_pred = eval_epoch(
            val_loader, model, criterion, logger, collect_predictions=True
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics_with_std = None
        if y_true is not None and y_pred is not None:
            if target_scaler is not None:
                y_true = target_scaler.inverse_transform(y_true)
                y_pred = target_scaler.inverse_transform(y_pred)
            metrics_with_std = compute_metrics_with_std(y_true, y_pred, n_bootstrap=200)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            if metrics_with_std is not None:
                best_val_metrics_with_std = metrics_with_std

        if (epoch + 1) % 10 == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                            run_dir, is_best, logger, scaler=target_scaler, config=config)

        pbar.set_postfix({
            "Epoch": f"{epoch + 1:02d}",
            "Train": f"{train_loss:.4f}",
            "Val": f"{val_loss:.4f}",
            "Best": f"{best_val_loss:.4f}",
        })

        if (epoch + 1) % 10 == 0:
            msg = (
                f"Epoch {epoch + 1:03d}/{num_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best Val: {best_val_loss:.4f}"
            )
            if metrics_with_std is not None:
                m = metrics_with_std
                msg += f" | RMSE: {m['RMSE'][0]:.4f} | SCC: {m['SCC'][0]:.4f} | PCC: {m['PCC'][0]:.4f}"
            logger.info(msg)

    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    if target_scaler is not None:
        original_scale_best_val = best_val_loss * (target_scaler.std ** 2)
        logger.info(f"Best validation loss (original scale): {original_scale_best_val:.4f}")

    if best_val_metrics_with_std is not None:
        table_str = format_metrics_table_pretty(
            best_val_metrics_with_std,
            title=f"Best validation metrics — {frac_label} training data (RMSE, SCC, PCC)",
        )
        logger.info(table_str)
        print(table_str)

    # ------------------------------------------------------------------
    # Test evaluation: reload best checkpoint and evaluate on test split
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Test Evaluation (best checkpoint)")
    logger.info("=" * 80)

    best_ckpt_path = run_dir / "checkpoint_best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {best_ckpt['epoch'] + 1}")

        test_loss, y_true_test, y_pred_test = eval_epoch(
            test_loader, model, criterion, logger, collect_predictions=True
        )
        logger.info(f"Test loss (MSE): {test_loss:.4f}")

        if y_true_test is not None and y_pred_test is not None:
            if target_scaler is not None:
                y_true_test = target_scaler.inverse_transform(y_true_test)
                y_pred_test = target_scaler.inverse_transform(y_pred_test)
            test_metrics_with_std = compute_metrics_with_std(
                y_true_test, y_pred_test, n_bootstrap=200
            )
            table_str = format_metrics_table_pretty(
                test_metrics_with_std,
                title=f"Test metrics — {frac_label} training data (RMSE, SCC, PCC)",
            )
            logger.info(table_str)
            print(table_str)
    else:
        logger.warning(
            "No best checkpoint found — test evaluation skipped. "
            "This can happen if training was too short for any checkpoint to be saved."
        )

    save_metrics(run_dir, train_losses, val_losses, logger,
                 best_val_metrics_with_std=best_val_metrics_with_std,
                 test_metrics_with_std=test_metrics_with_std)
    plot_losses(train_losses, val_losses, num_epochs, run_dir, fraction, args.standardize, logger)

    logger.info(f"All outputs saved to: {run_dir}")
    return test_metrics_with_std, run_dir, best_val_loss, len(train_dataset)


# ---------------------------------------------------------------------------
# Summary table and plotting across all fractions
# ---------------------------------------------------------------------------

def print_summary_table(results: dict):
    """
    Print and return a PrettyTable comparing test-set RMSE, SCC, PCC across fractions.

    Args:
        results: {fraction: {"metrics": test_metrics_with_std_or_None, "train_samples": int, "best_val_loss": float}}
    """
    try:
        from prettytable import PrettyTable
        pt = PrettyTable(["Fraction", "Train Samples", "Best Val Loss", "Test RMSE", "Test SCC", "Test PCC"])
        for fraction in sorted(results.keys()):
            data = results[fraction]
            m = data.get("metrics")
            frac_label = FRACTION_LABELS.get(fraction, f"{fraction*100:.0f}%")
            train_samples = data.get("train_samples", "N/A")
            best_val_loss = data.get("best_val_loss")
            val_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A"
            if m is None:
                pt.add_row([frac_label, train_samples, val_str, "N/A", "N/A", "N/A"])
            else:
                rmse = f"{m['RMSE'][0]:.4f} ({m['RMSE'][1]:.4f})"
                scc = f"{m['SCC'][0]:.4f} ({m['SCC'][1]:.4f})"
                pcc = f"{m['PCC'][0]:.4f} ({m['PCC'][1]:.4f})"
                pt.add_row([frac_label, train_samples, val_str, rmse, scc, pcc])
        table_str = f"\nData Scaling Study Summary — Test Set (mean (std))\n{pt.get_string()}\n"
    except ImportError:
        lines = ["\nData Scaling Study Summary — Test Set (mean (std))", "-" * 100]
        header = f"{'Fraction':<12} {'Train Samples':<15} {'Best Val Loss':<15} {'Test RMSE':<22} {'Test SCC':<22} {'Test PCC':<22}"
        lines.append(header)
        lines.append("-" * 100)
        for fraction in sorted(results.keys()):
            data = results[fraction]
            m = data.get("metrics")
            frac_label = FRACTION_LABELS.get(fraction, f"{fraction*100:.0f}%")
            train_samples = data.get("train_samples", "N/A")
            best_val_loss = data.get("best_val_loss")
            val_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A"
            if m is None:
                lines.append(f"{frac_label:<12} {train_samples:<15} {val_str:<15} {'N/A':<22} {'N/A':<22} {'N/A':<22}")
            else:
                rmse = f"{m['RMSE'][0]:.4f} ({m['RMSE'][1]:.4f})"
                scc = f"{m['SCC'][0]:.4f} ({m['SCC'][1]:.4f})"
                pcc = f"{m['PCC'][0]:.4f} ({m['PCC'][1]:.4f})"
                lines.append(f"{frac_label:<12} {train_samples:<15} {val_str:<15} {rmse:<22} {scc:<22} {pcc:<22}")
        table_str = "\n".join(lines) + "\n"
    print(table_str)
    return table_str


def plot_scaling_curves(results: dict, output_dir: Path):
    """
    Generate scaling curve plots showing how metrics change with data size.
    
    Creates two plots:
    1. scaling_curves.png - RMSE, SCC, PCC vs training data fraction
    2. loss_vs_data.png - Best validation loss vs training data fraction
    
    Args:
        results: {fraction: {"metrics": {...}, "train_samples": int, "best_val_loss": float}}
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract data for plotting
    fractions = sorted(results.keys())
    train_samples = [results[f]["train_samples"] for f in fractions]
    
    # Check if we have metrics
    valid_fractions = [f for f in fractions if results[f].get("metrics") is not None]
    if not valid_fractions:
        print("No valid metrics to plot scaling curves.")
        return
    
    # Extract metrics
    rmse_means = [results[f]["metrics"]["RMSE"][0] for f in valid_fractions]
    rmse_stds = [results[f]["metrics"]["RMSE"][1] for f in valid_fractions]
    scc_means = [results[f]["metrics"]["SCC"][0] for f in valid_fractions]
    scc_stds = [results[f]["metrics"]["SCC"][1] for f in valid_fractions]
    pcc_means = [results[f]["metrics"]["PCC"][0] for f in valid_fractions]
    pcc_stds = [results[f]["metrics"]["PCC"][1] for f in valid_fractions]
    
    valid_train_samples = [results[f]["train_samples"] for f in valid_fractions]
    valid_frac_labels = [FRACTION_LABELS.get(f, f"{f*100:.0f}%") for f in valid_fractions]
    
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Scaling curves with all metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE (lower is better)
    ax1 = axes[0]
    ax1.errorbar(valid_train_samples, rmse_means, yerr=rmse_stds, marker='o', capsize=5, 
                 linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel("Training Samples", fontsize=12)
    ax1.set_ylabel("Test RMSE", fontsize=12)
    ax1.set_title("RMSE vs Training Data Size", fontsize=14)
    ax1.grid(True, alpha=0.3)
    for i, (x, y, label) in enumerate(zip(valid_train_samples, rmse_means, valid_frac_labels)):
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    # SCC (higher is better)
    ax2 = axes[1]
    ax2.errorbar(valid_train_samples, scc_means, yerr=scc_stds, marker='s', capsize=5,
                 linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel("Training Samples", fontsize=12)
    ax2.set_ylabel("Test SCC", fontsize=12)
    ax2.set_title("Spearman Correlation vs Training Data Size", fontsize=14)
    ax2.grid(True, alpha=0.3)
    for i, (x, y, label) in enumerate(zip(valid_train_samples, scc_means, valid_frac_labels)):
        ax2.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    # PCC (higher is better)
    ax3 = axes[2]
    ax3.errorbar(valid_train_samples, pcc_means, yerr=pcc_stds, marker='^', capsize=5,
                 linewidth=2, markersize=8, color='#2ecc71')
    ax3.set_xlabel("Training Samples", fontsize=12)
    ax3.set_ylabel("Test PCC", fontsize=12)
    ax3.set_title("Pearson Correlation vs Training Data Size", fontsize=14)
    ax3.grid(True, alpha=0.3)
    for i, (x, y, label) in enumerate(zip(valid_train_samples, pcc_means, valid_frac_labels)):
        ax3.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    scaling_plot_path = output_dir / "scaling_curves.png"
    plt.savefig(scaling_plot_path, dpi=300, bbox_inches="tight")
    print(f"Scaling curves plot saved to {scaling_plot_path}")
    plt.close()
    
    # Also save to assets
    assets_path = ASSETS_PATH / "scaling_curves.png"
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1 = axes[0]
    ax1.errorbar(valid_train_samples, rmse_means, yerr=rmse_stds, marker='o', capsize=5, 
                 linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel("Training Samples", fontsize=12)
    ax1.set_ylabel("Test RMSE", fontsize=12)
    ax1.set_title("RMSE vs Training Data Size", fontsize=14)
    ax1.grid(True, alpha=0.3)
    for i, (x, y, label) in enumerate(zip(valid_train_samples, rmse_means, valid_frac_labels)):
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax2 = axes[1]
    ax2.errorbar(valid_train_samples, scc_means, yerr=scc_stds, marker='s', capsize=5,
                 linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel("Training Samples", fontsize=12)
    ax2.set_ylabel("Test SCC", fontsize=12)
    ax2.set_title("Spearman Correlation vs Training Data Size", fontsize=14)
    ax2.grid(True, alpha=0.3)
    for i, (x, y, label) in enumerate(zip(valid_train_samples, scc_means, valid_frac_labels)):
        ax2.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax3 = axes[2]
    ax3.errorbar(valid_train_samples, pcc_means, yerr=pcc_stds, marker='^', capsize=5,
                 linewidth=2, markersize=8, color='#2ecc71')
    ax3.set_xlabel("Training Samples", fontsize=12)
    ax3.set_ylabel("Test PCC", fontsize=12)
    ax3.set_title("Pearson Correlation vs Training Data Size", fontsize=14)
    ax3.grid(True, alpha=0.3)
    for i, (x, y, label) in enumerate(zip(valid_train_samples, pcc_means, valid_frac_labels)):
        ax3.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(assets_path, dpi=300, bbox_inches="tight")
    print(f"Scaling curves also saved to {assets_path}")
    plt.close()
    
    # Plot 2: Best validation loss vs data size
    val_losses = [results[f].get("best_val_loss") for f in fractions]
    valid_val_fractions = [(f, v, s) for f, v, s in zip(fractions, val_losses, train_samples) if v is not None]
    
    if valid_val_fractions:
        vf, vv, vs = zip(*valid_val_fractions)
        vf_labels = [FRACTION_LABELS.get(f, f"{f*100:.0f}%") for f in vf]
        
        plt.figure(figsize=(8, 6))
        plt.plot(vs, vv, marker='o', linewidth=2, markersize=10, color='#9b59b6')
        plt.xlabel("Training Samples", fontsize=12)
        plt.ylabel("Best Validation Loss (MSE)", fontsize=12)
        plt.title("Validation Loss vs Training Data Size", fontsize=14)
        plt.grid(True, alpha=0.3)
        for x, y, label in zip(vs, vv, vf_labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
        
        loss_plot_path = output_dir / "loss_vs_data.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        print(f"Loss vs data plot saved to {loss_plot_path}")
        plt.close()


def save_summary(results: dict, args, output_dir: Path):
    """Save scaling study summary as JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / "scaling_summary.json"
    summary = {
        "timestamp": timestamp,
        "exp_name": args.exp_name,
        "target": args.target,
        "split_type": "random",
        "head_type": args.head_type,
        "epochs": args.epochs,
        "standardize": args.standardize,
        "subsample_seed": args.seed,
        "fractions": {},
    }
    for fraction, data in results.items():
        m = data.get("metrics")
        summary["fractions"][str(fraction)] = {
            "fraction_label": FRACTION_LABELS.get(fraction, f"{fraction*100:.0f}%"),
            "train_samples": data.get("train_samples"),
            "best_val_loss": data.get("best_val_loss"),
            "run_dir": str(data.get("run_dir", "")),
            "test_metrics": {
                name: {"mean": v[0], "std": v[1]}
                for name, v in m.items()
            } if m else None
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Data scaling study for drug synergy prediction — train on different data fractions"
    )
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS),
        help="List of training data fractions to test (default: 0.25 0.50 0.75 1.0)",
    )
    parser.add_argument(
        "--exp_name", type=str, default="data_scaling",
        help="Experiment name prefix (default: data_scaling)",
    )
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="Standardize target values (mean=0, std=1)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs per fraction (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument(
        "--target", type=str, default="Synergy_ZIP",
        choices=["Synergy_ZIP", "Synergy_Bliss", "Synergy_Loewe", "Synergy_HSA"],
        help="Target synergy column (default: Synergy_ZIP)",
    )
    parser.add_argument(
        "--head_type", type=str, default="kan", choices=["mlp", "kan"],
        help="Prediction head: mlp or kan (default: kan)",
    )
    parser.add_argument("--grid_size", type=int, default=5,
                        help="Grid size for KAN layers (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling reproducibility (default: 42)")
    parser.add_argument(
        "--drug_emb_dim", type=int, default=768,
        help="Drug embedding dimension (768 for MolFormer, 384 for ChemBERTa)",
    )
    parser.add_argument(
        "--drug_model", type=str, default=None, choices=["molformer", "chemberta"],
        help="Drug embedding model to use (for loading embeddings)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Sort fractions for consistent output
    fractions_to_run = sorted(args.fractions)
    
    print("=" * 80)
    print("Data Scaling Study for Drug Synergy Prediction")
    print(f"Fractions: {', '.join([f'{f*100:.0f}%' for f in fractions_to_run])}")
    print(f"Target: {args.target} | Head: {args.head_type.upper()} | "
          f"Epochs: {args.epochs} | Seed: {args.seed}")
    print("=" * 80)

    # Load splits once (random split only)
    split_type = "random"
    raw_path = get_raw_path(split_type)
    drug_emb_path = get_drug_emb_path(split_type, args.drug_model)
    print(f"Raw data path: {raw_path}")
    print(f"Drug embeddings: {drug_emb_path}")
    print(f"Drug embedding dimension: {args.drug_emb_dim}")

    with open(raw_path / "train_split.pkl", "rb") as f:
        train_df = pickle.load(f)
    with open(raw_path / "val_split.pkl", "rb") as f:
        val_df = pickle.load(f)
    with open(raw_path / "test_split.pkl", "rb") as f:
        test_df = pickle.load(f)
    print(f"Loaded {len(train_df)} train / {len(val_df)} val / {len(test_df)} test samples (full dataset)")

    # Get omics dimensions from full training set
    base_train_dataset = DrugSynergyRawOmicsDataset(train_df, drug_emb_path, args.target)
    mrna_dim, mirna_dim, prot_dim = base_train_dataset.get_omics_dims()
    print(f"Omics dimensions — mRNA: {mrna_dim}, miRNA: {mirna_dim}, Proteomics: {prot_dim}")

    # Create a parent directory for this scaling study run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = LOG_PATH / f"{args.exp_name}_study_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nStudy output directory: {study_dir}")

    # Run each fraction and collect results
    all_results = {}
    for fraction in fractions_to_run:
        frac_label = FRACTION_LABELS.get(fraction, f"{fraction*100:.0f}%")
        print(f"\n{'=' * 80}")
        print(f"Running fraction: {frac_label} ({fraction})")
        print(f"{'=' * 80}")
        metrics, run_dir, best_val_loss, train_samples = run_fraction(
            fraction=fraction,
            train_df_full=train_df,
            val_df_orig=val_df,
            test_df_orig=test_df,
            args=args,
            mrna_dim=mrna_dim,
            mirna_dim=mirna_dim,
            prot_dim=prot_dim,
            drug_emb_path=drug_emb_path,
        )
        all_results[fraction] = {
            "metrics": metrics,
            "run_dir": run_dir,
            "best_val_loss": best_val_loss,
            "train_samples": train_samples,
        }

    # Summary across all fractions
    print("\n" + "=" * 80)
    print_summary_table(all_results)
    
    # Generate scaling curve plots
    print("\nGenerating scaling curve plots...")
    plot_scaling_curves(all_results, study_dir)
    
    # Save summary JSON
    save_summary(all_results, args, study_dir)
    
    # Copy script to study directory
    shutil.copy2(Path(__file__).resolve(), study_dir / "data_scaling.py")
    print(f"\nAll study outputs saved to: {study_dir}")
    print("=" * 80)
