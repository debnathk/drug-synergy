import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

from pathlib import Path
from src.dataset import DrugSynergyRawOmicsDataset
from src.models.synergy_model import SynergyModel
from tqdm import tqdm
import logging
from datetime import datetime
import json
import argparse

ROOT = Path(__file__).resolve().parent
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

def save_metrics(run_dir: Path, train_losses: list, val_losses: list, logger: logging.Logger):
    """Save training metrics to file"""
    metrics_file = run_dir / "metrics.json"
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_train_loss": min(train_losses),
        "best_val_loss": min(val_losses),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1]
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

# Paths to embeddings
drug_emb_path = EMBEDDINGS_PATH / 'drug_embeddings.pt'
MODELS_PATH = ROOT / "data/models"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(loader, model, optimizer, criterion, device, logger):
    model.train()
    total_loss = 0
    batch_losses = []

    for batch_idx, (d1, d2, mrna, mirna, prot, y) in enumerate(loader):
        # Move all inputs to device
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
    
    # Log statistics every N batches
    if len(batch_losses) > 0:
        logger.debug(f"Train batch losses - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}")

    return avg_loss


def eval_epoch(loader, model, criterion, device, logger):
    model.eval()
    total_loss = 0
    batch_losses = []

    with torch.no_grad():
        for d1, d2, mrna, mirna, prot, y in loader:
            # Move all inputs to device
            d1, d2 = d1.to(device), d2.to(device)
            mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
            y = y.to(device)
            
            preds = model(d1, d2, mrna, mirna, prot)
            loss = criterion(preds, y)
            
            batch_loss = loss.item()
            total_loss += batch_loss * d1.size(0)
            batch_losses.append(batch_loss)

    avg_loss = total_loss / len(loader.dataset)
    
    if len(batch_losses) > 0:
        logger.debug(f"Val batch losses - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}")

    return avg_loss
    
def sanity_check(dataset, logger):
    d1, d2, mrna, mirna, prot, y = dataset[0]
    logger.info(f"Drug1 embedding shape: {d1.shape}")
    logger.info(f"Drug2 embedding shape: {d2.shape}")
    logger.info(f"mRNA shape: {mrna.shape}")
    logger.info(f"miRNA shape: {mirna.shape}")
    logger.info(f"Proteomics shape: {prot.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target sample: {y.item()}")
    
def plot_losses(train_losses, val_losses, num_epochs, run_dir, standardized, logger):
    """Generate and save loss plot"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
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
    
    # Save to both run directory and assets
    plot_filename = f"loss_plot_{'std' if standardized else 'wo_std'}.png"
    plot_path_run = run_dir / plot_filename
    plot_path_assets = ASSETS_PATH / plot_filename
    
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(plot_path_run, dpi=300, bbox_inches="tight")
    plt.savefig(plot_path_assets, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Loss plot saved to {plot_path_run}")
    logger.info(f"Loss plot also saved to {plot_path_assets}")

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


def save_omics_encoder(model, run_dir, logger):
    """Save trained OmicsFusionModel weights separately for future use"""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save to both run directory and models directory
    omics_encoder_state = model.omics_encoder.state_dict()
    
    # Save to run directory
    run_path = run_dir / "omics_encoder.pt"
    torch.save(omics_encoder_state, run_path)
    
    # Save to models directory (overwrite best)
    models_path = MODELS_PATH / "omics_fusion_model_best.pt"
    torch.save(omics_encoder_state, models_path)
    
    logger.info(f"OmicsFusionModel weights saved to {run_path}")
    logger.info(f"OmicsFusionModel weights also saved to {models_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Drug Synergy Prediction Model')
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Standardize target values (mean=0, std=1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--target', type=str, default='Synergy_ZIP',
                        choices=['Synergy_ZIP', 'Synergy_Bliss', 'Synergy_Loewe', 'Synergy_HSA'],
                        help='Target column to predict')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Setup logging with experiment name reflecting standardization
    exp_name = "synergy_attn_std" if args.standardize else "synergy_attn"
    run_dir, logger = setup_logging(experiment_name=exp_name)
    
    logger.info("="*80)
    logger.info("Starting Drug Synergy Prediction Training (End-to-End with Attention)")
    logger.info("="*80)
    
    # Load data
    logger.info("Loading datasets...")
    with open(RAW_DATA_PATH / "train_split.pkl", 'rb') as file:
        train_df = pickle.load(file)
    with open(RAW_DATA_PATH / "val_split.pkl", 'rb') as file:
        val_df = pickle.load(file)
    
    # Target standardization
    target_scaler = None
    if args.standardize:
        logger.info("Applying target standardization...")
        target_scaler = TargetScaler()
        
        # Fit on training data
        original_train_targets = train_df[args.target].values.copy()
        original_val_targets = val_df[args.target].values.copy()
        
        # Fit scaler on training data only
        target_scaler.fit(original_train_targets)
        
        # Transform both train and val targets
        train_df[args.target] = target_scaler.transform(original_train_targets)
        val_df[args.target] = target_scaler.transform(original_val_targets)
        
        logger.info(f"Target scaler fitted - Mean: {target_scaler.mean:.4f}, Std: {target_scaler.std:.4f}")
        logger.info(f"Original target range: [{original_train_targets.min():.2f}, {original_train_targets.max():.2f}]")
        logger.info(f"Standardized target range: [{train_df[args.target].min():.2f}, {train_df[args.target].max():.2f}]")
    
    # Create datasets
    train_dataset = DrugSynergyRawOmicsDataset(train_df, drug_emb_path, args.target)
    val_dataset = DrugSynergyRawOmicsDataset(val_df, drug_emb_path, args.target)
    logger.info("Datasets loaded successfully")
    
    # Get omics dimensions from dataset
    mrna_dim, mirna_dim, prot_dim = train_dataset.get_omics_dims()
    
    # Configuration
    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "target_column": args.target,
        "standardize_targets": args.standardize,
        "device": str(device),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "model": "SynergyModel (End-to-End with Attention)",
        "mrna_dim": mrna_dim,
        "mirna_dim": mirna_dim,
        "prot_dim": prot_dim,
    }
    
    # Add scaler params to config if standardizing
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
    logger.info(f"Omics dimensions - mRNA: {mrna_dim}, miRNA: {mirna_dim}, Proteomics: {prot_dim}")

    # Sanity check
    logger.info("Running sanity check...")
    sanity_check(train_dataset, logger)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Model setup
    logger.info("Initializing SynergyModel (End-to-End with Attention)...")
    model = SynergyModel(
        mrna_dim=mrna_dim,
        mirna_dim=mirna_dim,
        prot_dim=prot_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    num_epochs = config['num_epochs']
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    logger.info("="*80)
    logger.info("Starting Training")
    logger.info("="*80)

    # Run training
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        train_loss = train_epoch(train_loader, model, optimizer, criterion, device, logger)
        val_loss = eval_epoch(val_loader, model, criterion, device, logger)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint every 10 epochs or if best
        if (epoch + 1) % 10 == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, run_dir, is_best, logger,
                          scaler=target_scaler, config=config)

        pbar.set_postfix({
            "Epoch": f"{epoch+1:02d}", 
            "Train": f"{train_loss:.4f}", 
            "Val": f"{val_loss:.4f}",
            "Best": f"{best_val_loss:.4f}"
        })
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:03d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best Val: {best_val_loss:.4f}")

    logger.info("="*80)
    logger.info("Training Complete")
    logger.info("="*80)
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # If standardized, also report in original scale
    if target_scaler is not None:
        # For MSE, need to multiply by std^2 to get original scale MSE
        original_scale_best_val = best_val_loss * (target_scaler.std ** 2)
        logger.info(f"Best validation loss (original scale): {original_scale_best_val:.4f}")
    
    # Save final metrics
    save_metrics(run_dir, train_losses, val_losses, logger)
    
    # Save trained OmicsFusionModel weights separately
    logger.info("Saving trained OmicsFusionModel weights...")
    save_omics_encoder(model, run_dir, logger)
    
    # Plot losses
    logger.info("Generating loss plot...")
    plot_losses(train_losses, val_losses, num_epochs, run_dir, config['standardize_targets'], logger)
    
    logger.info("="*80)
    logger.info(f"All outputs saved to: {run_dir}")
    logger.info("="*80)