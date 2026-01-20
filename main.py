import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from src.dataset import DrugSynergyDataset
from src.models.mlp import SynergyMLP
from tqdm import tqdm
import logging
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"
LOG_PATH = ROOT / "logs"
ASSETS_PATH = ROOT / "assets"

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

# Load your dataset
with open(RAW_DATA_PATH / "train_split.pkl", 'rb') as file:
    train_df = pickle.load(file)

with open(RAW_DATA_PATH / "val_split.pkl", 'rb') as file:
    val_df = pickle.load(file)

# Define the target columns
target_cols = ['Synergy_ZIP', 'Synergy_Bliss', 'Synergy_Loewe', 'Synergy_HSA']

# Paths to embeddings
drug_emb_path = EMBEDDINGS_PATH / 'drug_embeddings.pt'
train_omics_emb_path = EMBEDDINGS_PATH / 'train_omics.pt'
val_omics_emb_path = EMBEDDINGS_PATH / 'val_omics.pt'

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(loader, model, optimizer, criterion, device, logger):
    model.train()
    total_loss = 0
    batch_losses = []

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * x.size(0)
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            
            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            batch_losses.append(batch_loss)

    avg_loss = total_loss / len(loader.dataset)
    
    if len(batch_losses) > 0:
        logger.debug(f"Val batch losses - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, Avg: {avg_loss:.4f}")

    return avg_loss
    
def sanity_check(dataset, logger):
    x, y = dataset[0]
    logger.info(f"Sample input shape: {x.shape}")
    logger.info(f"Sample target shape: {y.shape}")
    logger.info(f"Input sample (first 10): {x[:10].tolist()}")
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

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, run_dir, is_best, logger):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Save latest checkpoint
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = run_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Setup logging
    run_dir, logger = setup_logging(experiment_name="mlp")
    
    logger.info("="*80)
    logger.info("Starting Drug Synergy Prediction Training")
    logger.info("="*80)
    
    # Configuration
    config = {
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "target_column": target_cols[0],
        "standardize_targets": False,
        "device": str(device),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }
    
    save_config(run_dir, config, logger)
    
    logger.info(f"Device: {device}")
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Target column: {config['target_column']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Number of epochs: {config['num_epochs']}")
    
    # Standardize target col (optional)
    # scaler = StandardScaler()
    # train_df['Synergy_ZIP'] = scaler.fit_transform(train_df['Synergy_ZIP'].values.reshape(-1, 1))
    # val_df['Synergy_ZIP'] = scaler.transform(val_df['Synergy_ZIP'].values.reshape(-1, 1))

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = DrugSynergyDataset(train_df, drug_emb_path, train_omics_emb_path, target_cols[0])
    val_dataset = DrugSynergyDataset(val_df, drug_emb_path, val_omics_emb_path, target_cols[0])
    logger.info("Datasets loaded successfully")

    # Sanity check
    logger.info("Running sanity check...")
    sanity_check(train_dataset, logger)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Model setup
    logger.info("Initializing model...")
    model = SynergyMLP().to(device)
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
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, run_dir, is_best, logger)

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
    
    # Save final metrics
    save_metrics(run_dir, train_losses, val_losses, logger)
    
    # Plot losses
    logger.info("Generating loss plot...")
    plot_losses(train_losses, val_losses, num_epochs, run_dir, config['standardize_targets'], logger)
    
    logger.info("="*80)
    logger.info(f"All outputs saved to: {run_dir}")
    logger.info("="*80)