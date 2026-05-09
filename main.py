import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import shutil
import pandas as pd

from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dataset import DrugSynergyRawOmicsDataset, DrugSynergyL1000Dataset
from src.models.synergy_model import SynergyModel, SynergyModelL1000
from src.utils import compute_metrics_with_std, format_metrics_table_pretty
from tqdm import tqdm
import logging
from datetime import datetime
import json
import argparse

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR  # drug-synergy/
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"
LOG_PATH = ROOT / "logs"
ASSETS_PATH = ROOT / "assets"


class TargetScaler:
    """
    Target standardization scaler that can be saved/loaded with checkpoints.
    Standardizes targets to have mean=0, std=1 for better training stability.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, targets):
        """Fit scaler on training targets."""
        targets = np.array(targets).flatten()
        self.mean = float(np.mean(targets))
        self.std = float(np.std(targets))
        self.fitted = True
        return self
    
    def transform(self, targets):
        """Transform targets to standardized form."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        targets = np.array(targets).flatten()
        return (targets - self.mean) / self.std
    
    def fit_transform(self, targets):
        """Fit and transform in one step."""
        self.fit(targets)
        return self.transform(targets)
    
    def inverse_transform(self, standardized_targets):
        """Convert standardized targets back to original scale."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        standardized_targets = np.array(standardized_targets).flatten()
        return standardized_targets * self.std + self.mean
    
    def to_dict(self):
        """Export scaler parameters for saving."""
        return {
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }
    
    @classmethod
    def from_dict(cls, params):
        """Load scaler from saved parameters."""
        scaler = cls()
        scaler.mean = params['mean']
        scaler.std = params['std']
        scaler.fitted = params['fitted']
        return scaler

def setup_logging(experiment_name="mlp"):
    """
    Creates a timestamped log directory and configures logging.
    Returns the run directory path and logger.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_PATH / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = run_dir / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    
    return run_dir, logger

def save_config(run_dir: Path, config: dict, logger: logging.Logger):
    """Save training configuration to JSON"""
    config_file = run_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_file}")

def save_metrics(run_dir: Path, train_losses: list, val_losses: list, logger: logging.Logger,
                 best_val_metrics_with_std=None):
    """Save training metrics to file (includes RMSE, SCC, PCC with std when provided)."""
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
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

# Paths to embeddings (set per run using split_type)
MODELS_PATH = ROOT / "data/models"


def get_raw_path(split_type: str) -> Path:
    """Resolve raw data directory: raw/<split_type>/ or flat raw/ for backward compat."""
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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _unpack_batch(batch, device, use_l1000=False):
    """Move a batch to *device* and return (inputs_tuple, y)."""
    if use_l1000:
        d1, d2, mrna, l1000_d1, l1000_d2, y = batch
        d1, d2 = d1.to(device), d2.to(device)
        mrna = mrna.to(device)
        l1000_d1, l1000_d2 = l1000_d1.to(device), l1000_d2.to(device)
        y = y.to(device)
        return (d1, d2, mrna, l1000_d1, l1000_d2), y
    else:
        d1, d2, mrna, mirna, prot, y = batch
        d1, d2 = d1.to(device), d2.to(device)
        mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
        y = y.to(device)
        return (d1, d2, mrna, mirna, prot), y


def train_epoch(loader, model, optimizer, criterion, device, logger,
                use_l1000=False):
    model.train()
    total_loss = 0
    batch_losses = []

    for batch_idx, batch in enumerate(loader):
        inputs, y = _unpack_batch(batch, device, use_l1000)

        optimizer.zero_grad()
        preds = model(*inputs)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * y.size(0)
        batch_losses.append(batch_loss)

    avg_loss = total_loss / len(loader.dataset)

    if len(batch_losses) > 0:
        logger.debug(f"Train batch losses - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}")

    return avg_loss


def eval_epoch(loader, model, criterion, device, logger,
               collect_predictions=False, use_l1000=False):
    model.eval()
    total_loss = 0
    batch_losses = []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            inputs, y = _unpack_batch(batch, device, use_l1000)

            preds = model(*inputs)
            loss = criterion(preds, y)

            batch_loss = loss.item()
            total_loss += batch_loss * y.size(0)
            batch_losses.append(batch_loss)
            if collect_predictions:
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    if len(batch_losses) > 0:
        logger.debug(f"Val batch losses - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}")

    if collect_predictions and all_preds:
        y_true = np.concatenate(all_targets, axis=0).flatten()
        y_pred = np.concatenate(all_preds, axis=0).flatten()
        return avg_loss, y_true, y_pred
    return avg_loss, None, None
    
def sanity_check(dataset, logger, use_l1000=False):
    sample = dataset[0]
    if use_l1000:
        d1, d2, mrna, l1000_d1, l1000_d2, y = sample
        logger.info(f"Drug1 embedding shape: {d1.shape}")
        logger.info(f"Drug2 embedding shape: {d2.shape}")
        logger.info(f"mRNA shape: {mrna.shape}")
        logger.info(f"L1000 drug1 shape: {l1000_d1.shape}")
        logger.info(f"L1000 drug2 shape: {l1000_d2.shape}")
    else:
        d1, d2, mrna, mirna, prot, y = sample
        logger.info(f"Drug1 embedding shape: {d1.shape}")
        logger.info(f"Drug2 embedding shape: {d2.shape}")
        logger.info(f"mRNA shape: {mrna.shape}")
        logger.info(f"miRNA shape: {mirna.shape}")
        logger.info(f"Proteomics shape: {prot.shape}")
        y = sample[-1]
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target sample: {y.item()}")
    
def plot_losses(train_losses, val_losses, num_epochs, run_dir, standardized, logger):
    """Generate and save loss plot"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    epochs = range(1, num_epochs + 1)

    df = pd.DataFrame({
        "Epoch": list(epochs) * 2,
        "Loss": train_losses + val_losses,
        "Type": ["Train Loss"] * num_epochs + ["Validation Loss"] * num_epochs
    })

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Loss",
        hue="Type",
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    std_text = "with" if standardized else "without"
    plt.title(f"Training and Validation Loss - {std_text} Standardized Targets")
    plt.legend(title="")
    
    # Save to run directory and assets
    plot_filename = f"loss_plot_{'std' if standardized else 'wo_std'}.png"
    plot_path_run = run_dir / plot_filename
    
    plt.savefig(plot_path_run, dpi=300, bbox_inches="tight")
    logger.info(f"Loss plot saved to {plot_path_run}")
    
    # Also save to assets
    plot_path_assets = ASSETS_PATH / plot_filename
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path_assets, dpi=300, bbox_inches="tight")
    logger.info(f"Loss plot also saved to {plot_path_assets}")
    
    plt.close()

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, run_dir, is_best, logger, 
                    scaler=None, config=None):
    """Save model checkpoint with optional scaler parameters."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Save scaler parameters if standardization is used
    if scaler is not None:
        checkpoint['scaler'] = scaler.to_dict()
    
    # Save config for reference
    if config is not None:
        checkpoint['config'] = config
    
    # Save latest checkpoint
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = run_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")


def save_encoders(model, run_dir, logger):
    """Save trained encoder weights separately for future use."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    if isinstance(model, SynergyModelL1000):
        for name, module in [("mrna_encoder", model.mrna_encoder),
                             ("l1000_encoder", model.l1000_encoder)]:
            state = module.state_dict()
            run_path = run_dir / f"{name}.pt"
            torch.save(state, run_path)
            logger.info(f"{name} weights saved to {run_path}")

            models_path = MODELS_PATH / f"{name}_best.pt"
            torch.save(state, models_path)
            logger.info(f"{name} weights also saved to {models_path}")
    else:
        omics_encoder_state = model.omics_encoder.state_dict()
        run_path = run_dir / "omics_encoder.pt"
        torch.save(omics_encoder_state, run_path)
        logger.info(f"OmicsFusionModel weights saved to {run_path}")

        models_path = MODELS_PATH / "omics_fusion_model_best.pt"
        torch.save(omics_encoder_state, models_path)
        logger.info(f"OmicsFusionModel weights also saved to {models_path}")

def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics: MSE, RMSE, MAE, R², Pearson r, Spearman rho."""
    mse = mean_squared_error(y_true, y_pred)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearson_r),
        "Pearson_pval": float(pearson_p),
        "Spearman_r": float(spearman_r),
        "Spearman_pval": float(spearman_p),
    }


def log_full_metrics(metrics: dict, label: str, logger: logging.Logger):
    """Log a full metrics dict in a readable table."""
    logger.info(f"\n{'='*60}")
    logger.info(f" {label}")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<20} {'Value':<20}")
    logger.info(f"{'-'*40}")
    for key in ("MSE", "RMSE", "MAE", "R2", "Pearson_r", "Spearman_r"):
        logger.info(f"{key:<20} {metrics[key]:<20.6f}")
    logger.info(f"{'='*60}")


def copy_main_script(run_dir: Path, logger: logging.Logger):
    """Copy main.py to the run directory for reproducibility."""
    src = Path(__file__).resolve()
    dst = run_dir / "main.py"
    shutil.copy2(src, dst)
    logger.info(f"Script copied to {dst}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Drug Synergy Prediction Model')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Custom experiment name (default: auto-generated from head_type and standardize)')
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Standardize target values (mean=0, std=1)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--target', type=str, default='Synergy_ZIP',
                        choices=['Synergy_ZIP', 'Synergy_Bliss', 'Synergy_Loewe', 'Synergy_HSA'],
                        help='Target column to predict')
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'kan'],
                        help='Prediction head type: mlp or kan (default: mlp)')
    parser.add_argument('--fusion_type', type=str, default='attention',
                        choices=['attention', 'concat', 'product', 'mean_pool', 'max_pool'],
                        help='Omics fusion strategy (default: attention)')
    parser.add_argument('--grid_size', type=int, default=5,
                        help='Grid size for KAN layers (only used if head_type=kan)')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'cold_drug'],
                        help='Data split type: random or cold_drug (default: random)')
    parser.add_argument('--drug_emb_dim', type=int, default=768,
                        help='Drug embedding dimension (768 for MolFormer, 384 for ChemBERTa)')
    parser.add_argument('--drug_model', type=str, default=None,
                        choices=['molformer', 'chemberta'],
                        help='Drug embedding model to use (for loading embeddings)')
    parser.add_argument('--l1000_path', type=str, default=None,
                        help='Path to L1000 representative profiles CSV (enables L1000 mode)')
    parser.add_argument('--l1000_genes', type=str, default='landmark',
                        choices=['landmark', 'all'],
                        help='L1000 gene set: landmark (978) or all (12328)')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    use_l1000 = args.l1000_path is not None

    # Determine experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        head_suffix = "kan" if args.head_type == "kan" else "mlp"
        std_suffix = "_std" if args.standardize else ""
        l1000_suffix = "_l1000" if use_l1000 else ""
        fusion_suffix = f"_{args.fusion_type}" if args.fusion_type != "attention" else ""
        exp_name = f"synergy_{head_suffix}{std_suffix}{fusion_suffix}{l1000_suffix}_{args.split_type}"

    # Setup logging
    run_dir, logger = setup_logging(experiment_name=exp_name)

    # Copy main.py to output folder for reproducibility
    copy_main_script(run_dir, logger)

    logger.info("=" * 80)
    mode_label = "L1000-augmented" if use_l1000 else "Original"
    fusion_label = f", {args.fusion_type} fusion" if not use_l1000 else ""
    logger.info(f"Starting Drug Synergy Prediction Training ({mode_label}, {args.head_type.upper()} Head{fusion_label})")
    logger.info("=" * 80)

    # Load data splits
    raw_path = get_raw_path(args.split_type)
    drug_emb_path = get_drug_emb_path(args.split_type, args.drug_model)
    logger.info(f"Split type: {args.split_type}, raw path: {raw_path}, drug embeddings: {drug_emb_path}")
    logger.info(f"Drug embedding dimension: {args.drug_emb_dim}")
    logger.info("Loading datasets...")
    with open(raw_path / "train_split.pkl", "rb") as file:
        train_df = pickle.load(file)
    with open(raw_path / "val_split.pkl", "rb") as file:
        val_df = pickle.load(file)

    # Target standardization
    target_scaler = None
    if args.standardize:
        logger.info("Applying target standardization...")
        target_scaler = TargetScaler()

        original_train_targets = train_df[args.target].values.copy()
        original_val_targets = val_df[args.target].values.copy()

        target_scaler.fit(original_train_targets)
        train_df[args.target] = target_scaler.transform(original_train_targets)
        val_df[args.target] = target_scaler.transform(original_val_targets)

        logger.info(f"Target scaler fitted - Mean: {target_scaler.mean:.4f}, Std: {target_scaler.std:.4f}")
        logger.info(f"Original target range: [{original_train_targets.min():.2f}, {original_train_targets.max():.2f}]")
        logger.info(f"Standardized target range: [{train_df[args.target].min():.2f}, {train_df[args.target].max():.2f}]")

    # ---- Create datasets & model (branching on L1000 mode) ----
    if use_l1000:
        l1000_path = Path(args.l1000_path)
        if not l1000_path.is_absolute():
            l1000_path = ROOT / l1000_path
        logger.info(f"L1000 mode enabled — profiles: {l1000_path} ({args.l1000_genes} genes)")

        train_dataset = DrugSynergyL1000Dataset(train_df, drug_emb_path, l1000_path, args.target)
        val_dataset = DrugSynergyL1000Dataset(val_df, drug_emb_path, l1000_path, args.target)

        mrna_dim, l1000_dim = train_dataset.get_dims()
        logger.info(f"Feature dimensions — mRNA: {mrna_dim}, L1000: {l1000_dim}")

        model = SynergyModelL1000(
            mrna_dim=mrna_dim,
            l1000_dim=l1000_dim,
            embed_dim=args.drug_emb_dim,
            head_type=args.head_type,
            grid_size=args.grid_size,
        ).to(device)

        config = {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "criterion": "MSELoss",
            "target_column": args.target,
            "standardize_targets": args.standardize,
            "split_type": args.split_type,
            "device": str(device),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "model": f"SynergyModelL1000 ({args.head_type.upper()} Head)",
            "head_type": args.head_type,
            "grid_size": args.grid_size if args.head_type == "kan" else None,
            "drug_emb_dim": args.drug_emb_dim,
            "drug_model": args.drug_model,
            "mrna_dim": mrna_dim,
            "l1000_dim": l1000_dim,
            "l1000_path": str(l1000_path),
            "l1000_genes": args.l1000_genes,
        }
    else:
        train_dataset = DrugSynergyRawOmicsDataset(train_df, drug_emb_path, args.target)
        val_dataset = DrugSynergyRawOmicsDataset(val_df, drug_emb_path, args.target)

        mrna_dim, mirna_dim, prot_dim = train_dataset.get_omics_dims()
        logger.info(f"Omics dimensions — mRNA: {mrna_dim}, miRNA: {mirna_dim}, Proteomics: {prot_dim}")

        model = SynergyModel(
            mrna_dim=mrna_dim,
            mirna_dim=mirna_dim,
            prot_dim=prot_dim,
            embed_dim=args.drug_emb_dim,
            head_type=args.head_type,
            grid_size=args.grid_size,
            fusion_type=args.fusion_type,
        ).to(device)

        config = {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "criterion": "MSELoss",
            "target_column": args.target,
            "standardize_targets": args.standardize,
            "split_type": args.split_type,
            "device": str(device),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
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

    logger.info("Datasets loaded successfully")

    if target_scaler is not None:
        config["scaler_mean"] = target_scaler.mean
        config["scaler_std"] = target_scaler.std

    save_config(run_dir, config, logger)

    logger.info(f"Device: {device}")
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Target column: {config['target_column']}")
    logger.info(f"Target standardization: {config['standardize_targets']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Number of epochs: {config['num_epochs']}")

    # Sanity check
    logger.info("Running sanity check...")
    sanity_check(train_dataset, logger, use_l1000=use_l1000)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    num_epochs = config['num_epochs']
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_metrics_with_std = None

    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        train_loss = train_epoch(
            train_loader, model, optimizer, criterion, device, logger,
            use_l1000=use_l1000,
        )
        val_loss, y_true, y_pred = eval_epoch(
            val_loader, model, criterion, device, logger,
            collect_predictions=True, use_l1000=use_l1000,
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
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, run_dir, is_best, logger,
                            scaler=target_scaler, config=config)

        pbar.set_postfix({
            "Epoch": f"{epoch+1:02d}",
            "Train": f"{train_loss:.4f}",
            "Val": f"{val_loss:.4f}",
            "Best": f"{best_val_loss:.4f}",
        })

        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1:03d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best Val: {best_val_loss:.4f}"
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
            best_val_metrics_with_std, title="Best validation metrics (RMSE, SCC, PCC)"
        )
        logger.info(table_str)
        print(table_str)

    save_metrics(run_dir, train_losses, val_losses, logger,
                 best_val_metrics_with_std=best_val_metrics_with_std)

    logger.info("Saving trained encoder weights...")
    save_encoders(model, run_dir, logger)

    logger.info("Generating loss plot...")
    plot_losses(train_losses, val_losses, num_epochs, run_dir, config['standardize_targets'], logger)

    # =====================================================================
    # Test Evaluation — load best checkpoint and evaluate on test split
    # =====================================================================
    test_pkl = raw_path / "test_split.pkl"
    best_ckpt = run_dir / "checkpoint_best.pt"

    if test_pkl.exists() and best_ckpt.exists():
        logger.info("=" * 80)
        logger.info("Starting Test Evaluation (best checkpoint)")
        logger.info("=" * 80)

        with open(test_pkl, "rb") as f:
            test_df = pickle.load(f)
        logger.info(f"Test samples: {len(test_df)}")

        if use_l1000:
            test_dataset = DrugSynergyL1000Dataset(test_df, drug_emb_path, l1000_path, args.target)
        else:
            test_dataset = DrugSynergyRawOmicsDataset(test_df, drug_emb_path, args.target)

        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        # Reload best model weights
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?') + 1}, val_loss {ckpt.get('val_loss', 0):.4f})")

        _, y_true_test, y_pred_test = eval_epoch(
            test_loader, model, criterion, device, logger,
            collect_predictions=True, use_l1000=use_l1000,
        )

        if y_true_test is not None:
            if target_scaler is not None:
                y_pred_test = target_scaler.inverse_transform(y_pred_test)

            test_metrics = compute_full_metrics(y_true_test, y_pred_test)
            test_metrics_with_std = compute_metrics_with_std(y_true_test, y_pred_test, n_bootstrap=200)

            log_full_metrics(test_metrics, "Test Set Metrics", logger)

            table_str = format_metrics_table_pretty(
                test_metrics_with_std, title="Test metrics (RMSE, SCC, PCC)"
            )
            logger.info(table_str)
            print(table_str)

            # Save test predictions
            pd.DataFrame({"y_true": y_true_test, "y_pred": y_pred_test}).to_csv(
                run_dir / "test_predictions.csv", index=False
            )
            logger.info(f"Test predictions saved to {run_dir / 'test_predictions.csv'}")

            # Append test metrics to metrics.json
            metrics_file = run_dir / "metrics.json"
            with open(metrics_file) as f:
                all_metrics = json.load(f)
            all_metrics["test"] = test_metrics
            all_metrics["test_with_std"] = {k: list(v) for k, v in test_metrics_with_std.items()}
            with open(metrics_file, "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info("Test metrics appended to metrics.json")
    else:
        if not test_pkl.exists():
            logger.info("No test_split.pkl found — skipping test evaluation.")
        if not best_ckpt.exists():
            logger.info("No best checkpoint found — skipping test evaluation.")

    logger.info("=" * 80)
    logger.info(f"All outputs saved to: {run_dir}")
    logger.info("=" * 80)