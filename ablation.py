"""
Omics Leave-One-Out Ablation Study for Drug Synergy Prediction.

Trains SynergyModel under four conditions (drug embeddings always active):
  - no_mrna:   mRNA zeroed, miRNA + Proteomics active
  - no_mirna:  miRNA zeroed, mRNA + Proteomics active
  - no_prot:   Proteomics zeroed, mRNA + miRNA active
  - no_omics:  all omics zeroed (drug embeddings only baseline)

Follows the same training/evaluation schema as main.py.
A summary PrettyTable is printed after all conditions complete.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

ABLATION_CONDITIONS = ("no_mrna", "no_mirna", "no_prot", "no_omics")

CONDITION_LABELS = {
    "no_mrna":  "No mRNA",
    "no_mirna": "No miRNA",
    "no_prot":  "No Proteomics",
    "no_omics": "No Omics (drug only)",
}

# ---------------------------------------------------------------------------
# AblationDataset
# ---------------------------------------------------------------------------

class AblationDataset(Dataset):
    """
    Wraps DrugSynergyRawOmicsDataset and zeroes out one or all omics modalities.
    Drug embeddings are always returned unchanged.

    Args:
        base_dataset: DrugSynergyRawOmicsDataset instance.
        ablate: Which modality to zero out:
            'mrna'  — zero mRNA only
            'mirna' — zero miRNA only
            'prot'  — zero proteomics only
            'none'  — zero all omics (drug-embedding-only baseline)
    """

    def __init__(self, base_dataset: DrugSynergyRawOmicsDataset, ablate: str):
        valid = ("mrna", "mirna", "prot", "none")
        if ablate not in valid:
            raise ValueError(f"ablate must be one of {valid}, got {ablate!r}")
        self.base = base_dataset
        self.ablate = ablate

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        d1, d2, mrna, mirna, prot, y = self.base[idx]
        if self.ablate == "mrna":
            mrna = torch.zeros_like(mrna)
        elif self.ablate == "mirna":
            mirna = torch.zeros_like(mirna)
        elif self.ablate == "prot":
            prot = torch.zeros_like(prot)
        elif self.ablate == "none":
            mrna = torch.zeros_like(mrna)
            mirna = torch.zeros_like(mirna)
            prot = torch.zeros_like(prot)
        return d1, d2, mrna, mirna, prot, y


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
# Logging / saving helpers (identical to main.py)
# ---------------------------------------------------------------------------

def setup_logging(experiment_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_PATH / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training.log"

    # Each condition gets its own logger instance to avoid handler accumulation
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
    dst = run_dir / "ablation.py"
    shutil.copy2(Path(__file__).resolve(), dst)
    logger.info(f"Script copied to {dst}")


def plot_losses(train_losses, val_losses, num_epochs, run_dir, condition, standardized, logger):
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
    plt.title(f"Ablation: {CONDITION_LABELS[condition]} — Loss ({std_text} standardized targets)")
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
# Single-condition training run
# ---------------------------------------------------------------------------

def run_condition(condition: str, base_train_dataset, base_val_dataset,
                  train_df_orig, val_df_orig, test_df_orig, args,
                  mrna_dim, mirna_dim, prot_dim):
    """
    Train and evaluate for one ablation condition.

    After training, loads the best checkpoint and evaluates on the held-out
    test split.  Test metrics (RMSE, SCC, PCC with bootstrap std) are used
    in the cross-condition summary table.

    Returns:
        test_metrics_with_std: dict {RMSE: (mean, std), SCC: ..., PCC: ...}
        run_dir: Path to the condition's log directory
    """
    # Map condition name to AblationDataset ablate key
    # "no_omics" -> "none" (all omics zeroed), others strip "no_" prefix
    ablate_key = "none" if condition == "no_omics" else condition.replace("no_", "")
    exp_name = f"{args.exp_name}_{condition}"
    run_dir, logger = setup_logging(exp_name)
    copy_script(run_dir, logger)

    logger.info("=" * 80)
    logger.info(f"Ablation condition: {CONDITION_LABELS[condition]} ({condition})")
    logger.info("=" * 80)

    drug_emb_path = get_drug_emb_path(args.split_type, args.drug_model)

    # Target standardization — re-fit on unmodified train targets for each condition
    target_scaler = None
    if args.standardize:
        logger.info("Applying target standardization...")
        target_scaler = TargetScaler()
        original_train_targets = train_df_orig[args.target].values.copy()
        original_val_targets = val_df_orig[args.target].values.copy()
        original_test_targets = test_df_orig[args.target].values.copy()
        target_scaler.fit(original_train_targets)
        # Re-build base datasets with standardized targets so AblationDataset sees them.
        train_df_std = train_df_orig.copy()
        val_df_std = val_df_orig.copy()
        test_df_std = test_df_orig.copy()
        train_df_std[args.target] = target_scaler.transform(original_train_targets)
        val_df_std[args.target] = target_scaler.transform(original_val_targets)
        test_df_std[args.target] = target_scaler.transform(original_test_targets)
        base_train = DrugSynergyRawOmicsDataset(train_df_std, drug_emb_path, args.target)
        base_val = DrugSynergyRawOmicsDataset(val_df_std, drug_emb_path, args.target)
        base_test = DrugSynergyRawOmicsDataset(test_df_std, drug_emb_path, args.target)
        logger.info(f"Target scaler fitted - Mean: {target_scaler.mean:.4f}, Std: {target_scaler.std:.4f}")
        logger.info(f"Original target range: [{original_train_targets.min():.2f}, {original_train_targets.max():.2f}]")
        logger.info(f"Standardized target range: [{train_df_std[args.target].min():.2f}, {train_df_std[args.target].max():.2f}]")
    else:
        base_train = base_train_dataset
        base_val = base_val_dataset
        base_test = DrugSynergyRawOmicsDataset(test_df_orig, drug_emb_path, args.target)

    train_dataset = AblationDataset(base_train, ablate=ablate_key)
    val_dataset = AblationDataset(base_val, ablate=ablate_key)
    test_dataset = AblationDataset(base_test, ablate=ablate_key)
    logger.info(f"AblationDataset: zeroing '{ablate_key}' modality")
    logger.info(
        f"Train samples: {len(train_dataset)}, "
        f"Val samples: {len(val_dataset)}, "
        f"Test samples: {len(test_dataset)}"
    )

    config = {
        "ablation_condition": condition,
        "ablated_modality": ablate_key,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "target_column": args.target,
        "standardize_targets": args.standardize,
        "split_type": args.split_type,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "model": f"SynergyModel ({args.head_type.upper()} Head, {args.fusion_type} fusion)",
        "head_type": args.head_type,
        "fusion_type": args.fusion_type,
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
        fusion_type=args.fusion_type,
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

    pbar = tqdm(range(num_epochs), desc=f"[{CONDITION_LABELS[condition]}]")
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
            title=f"Best validation metrics — {CONDITION_LABELS[condition]} (RMSE, SCC, PCC)",
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
                title=f"Test metrics — {CONDITION_LABELS[condition]} (RMSE, SCC, PCC)",
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
    plot_losses(train_losses, val_losses, num_epochs, run_dir, condition, args.standardize, logger)

    logger.info(f"All outputs saved to: {run_dir}")
    return test_metrics_with_std, run_dir


# ---------------------------------------------------------------------------
# Summary table across all conditions
# ---------------------------------------------------------------------------

def print_summary_table(results: dict):
    """
    Print and return a PrettyTable comparing test-set RMSE, SCC, PCC across conditions.

    Args:
        results: {condition: test_metrics_with_std_or_None}
    """
    try:
        from prettytable import PrettyTable
        pt = PrettyTable(["Condition", "Test RMSE", "Test SCC", "Test PCC"])
        for condition, m in results.items():
            if m is None:
                pt.add_row([CONDITION_LABELS[condition], "N/A", "N/A", "N/A"])
            else:
                rmse = f"{m['RMSE'][0]:.4f} ({m['RMSE'][1]:.4f})"
                scc = f"{m['SCC'][0]:.4f} ({m['SCC'][1]:.4f})"
                pcc = f"{m['PCC'][0]:.4f} ({m['PCC'][1]:.4f})"
                pt.add_row([CONDITION_LABELS[condition], rmse, scc, pcc])
        table_str = f"\nAblation Study Summary — Test Set (mean (std))\n{pt.get_string()}\n"
    except ImportError:
        lines = ["\nAblation Study Summary — Test Set (mean (std))", "-" * 70]
        header = f"{'Condition':<20} {'Test RMSE':<22} {'Test SCC':<22} {'Test PCC':<22}"
        lines.append(header)
        lines.append("-" * 70)
        for condition, m in results.items():
            if m is None:
                lines.append(f"{CONDITION_LABELS[condition]:<20} {'N/A':<22} {'N/A':<22} {'N/A':<22}")
            else:
                rmse = f"{m['RMSE'][0]:.4f} ({m['RMSE'][1]:.4f})"
                scc = f"{m['SCC'][0]:.4f} ({m['SCC'][1]:.4f})"
                pcc = f"{m['PCC'][0]:.4f} ({m['PCC'][1]:.4f})"
                lines.append(f"{CONDITION_LABELS[condition]:<20} {rmse:<22} {scc:<22} {pcc:<22}")
        table_str = "\n".join(lines) + "\n"
    print(table_str)
    return table_str


def save_summary(results: dict, args):
    """Save ablation summary (test set metrics) as JSON to the logs directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = LOG_PATH / f"{args.exp_name}_ablation_summary_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "exp_name": args.exp_name,
        "target": args.target,
        "split_type": args.split_type,
        "head_type": args.head_type,
        "fusion_type": args.fusion_type,
        "epochs": args.epochs,
        "standardize": args.standardize,
        "metric_source": "test_set",
        "conditions": {},
    }
    for condition, m in results.items():
        if m is None:
            summary["conditions"][condition] = None
        else:
            summary["conditions"][condition] = {
                "test_metrics": {
                    name: {"mean": v[0], "std": v[1]}
                    for name, v in m.items()
                }
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
        description="Omics leave-one-out ablation study for drug synergy prediction"
    )
    parser.add_argument(
        "--ablation", type=str, default="all",
        choices=["no_mrna", "no_mirna", "no_prot", "no_omics", "all"],
        help="Which omics modality to ablate, or 'all' to run all four conditions (default: all)",
    )
    parser.add_argument(
        "--exp_name", type=str, default="ablation",
        help="Experiment name (default: ablation)",
    )
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="Standardize target values (mean=0, std=1)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs per condition (default: 100)")
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
    parser.add_argument(
        "--fusion_type", type=str, default="attention",
        choices=["attention", "concat", "product", "mean_pool", "max_pool"],
        help="Omics fusion strategy (default: attention)",
    )
    parser.add_argument("--grid_size", type=int, default=5,
                        help="Grid size for KAN layers (default: 5)")
    parser.add_argument(
        "--split_type", type=str, default="random", choices=["random", "cold_drug"],
        help="Data split type: random or cold_drug (default: random)",
    )
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

    conditions_to_run = ABLATION_CONDITIONS if args.ablation == "all" else (args.ablation,)

    print("=" * 80)
    print(f"Omics Leave-One-Out Ablation Study")
    print(f"Conditions: {', '.join(conditions_to_run)}")
    print(f"Target: {args.target} | Head: {args.head_type.upper()} | "
          f"Epochs: {args.epochs} | Split: {args.split_type}")
    print("=" * 80)

    # Load splits once (shared across conditions)
    raw_path = get_raw_path(args.split_type)
    drug_emb_path = get_drug_emb_path(args.split_type, args.drug_model)
    print(f"Raw data path: {raw_path}")
    print(f"Drug embeddings: {drug_emb_path}")
    print(f"Drug embedding dimension: {args.drug_emb_dim}")

    with open(raw_path / "train_split.pkl", "rb") as f:
        train_df = pickle.load(f)
    with open(raw_path / "val_split.pkl", "rb") as f:
        val_df = pickle.load(f)
    with open(raw_path / "test_split.pkl", "rb") as f:
        test_df = pickle.load(f)
    print(f"Loaded {len(train_df)} train / {len(val_df)} val / {len(test_df)} test samples")

    # Build base datasets (unstandardized; each condition handles standardization)
    base_train_dataset = DrugSynergyRawOmicsDataset(train_df, drug_emb_path, args.target)
    base_val_dataset = DrugSynergyRawOmicsDataset(val_df, drug_emb_path, args.target)
    mrna_dim, mirna_dim, prot_dim = base_train_dataset.get_omics_dims()
    print(f"Omics dimensions — mRNA: {mrna_dim}, miRNA: {mirna_dim}, Proteomics: {prot_dim}")

    # Run each condition and collect test results
    all_results = {}
    for condition in conditions_to_run:
        print(f"\n{'=' * 80}")
        print(f"Running condition: {CONDITION_LABELS[condition]}")
        print(f"{'=' * 80}")
        metrics, run_dir = run_condition(
            condition=condition,
            base_train_dataset=base_train_dataset,
            base_val_dataset=base_val_dataset,
            train_df_orig=train_df,
            val_df_orig=val_df,
            test_df_orig=test_df,
            args=args,
            mrna_dim=mrna_dim,
            mirna_dim=mirna_dim,
            prot_dim=prot_dim,
        )
        all_results[condition] = metrics

    # Summary across all completed conditions
    print("\n" + "=" * 80)
    print_summary_table(all_results)
    save_summary(all_results, args)
