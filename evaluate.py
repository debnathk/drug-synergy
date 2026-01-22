"""
Evaluation script for Drug Synergy Prediction models.
Computes comprehensive metrics: MSE, MAE, R², Pearson, Spearman correlations.
Supports both SynergyModel (attention-based) and SynergyMLP (baseline).

Author: Kusal Debnath
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import argparse
from datetime import datetime
import pandas as pd

from src.dataset import DrugSynergyRawOmicsDataset, DrugSynergyDataset
from src.models.synergy_model import SynergyModel
from src.models.mlp import SynergyMLP

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"
LOG_PATH = ROOT / "logs"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary with MSE, RMSE, MAE, R², Pearson, and Spearman correlations
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation metrics
    pearson_corr, pearson_pval = stats.pearsonr(y_true, y_pred)
    spearman_corr, spearman_pval = stats.spearmanr(y_true, y_pred)
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'Pearson_r': float(pearson_corr),
        'Pearson_pval': float(pearson_pval),
        'Spearman_r': float(spearman_corr),
        'Spearman_pval': float(spearman_pval)
    }


def evaluate_synergy_model(model, dataloader, device):
    """
    Evaluate SynergyModel on a dataset.
    
    Args:
        model: SynergyModel instance
        dataloader: DataLoader for evaluation
        device: torch device
        
    Returns:
        y_true, y_pred as numpy arrays
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for d1, d2, mrna, mirna, prot, y in dataloader:
            d1, d2 = d1.to(device), d2.to(device)
            mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
            y = y.to(device)
            
            preds = model(d1, d2, mrna, mirna, prot)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()
    
    return y_true, y_pred


def evaluate_mlp_model(model, dataloader, device):
    """
    Evaluate SynergyMLP on a dataset.
    
    Args:
        model: SynergyMLP instance
        dataloader: DataLoader for evaluation
        device: torch device
        
    Returns:
        y_true, y_pred as numpy arrays
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()
    
    return y_true, y_pred


class TargetScaler:
    """
    Target standardization scaler that can be loaded from checkpoints.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.fitted = mean is not None and std is not None
    
    def inverse_transform(self, standardized_targets):
        """Convert standardized targets back to original scale."""
        if not self.fitted:
            return standardized_targets
        standardized_targets = np.array(standardized_targets).flatten()
        return standardized_targets * self.std + self.mean
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Load scaler from checkpoint if available."""
        if 'scaler' in checkpoint:
            params = checkpoint['scaler']
            return cls(mean=params['mean'], std=params['std'])
        elif 'config' in checkpoint and checkpoint['config'].get('standardize_targets'):
            # Try to get from config
            config = checkpoint['config']
            if 'scaler_mean' in config and 'scaler_std' in config:
                return cls(mean=config['scaler_mean'], std=config['scaler_std'])
        return cls()  # Return unfitted scaler


def load_synergy_model(checkpoint_path: Path, mrna_dim: int, mirna_dim: int, prot_dim: int):
    """Load SynergyModel from checkpoint. Returns model and scaler."""
    model = SynergyModel(
        mrna_dim=mrna_dim,
        mirna_dim=mirna_dim,
        prot_dim=prot_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load scaler if available
    scaler = TargetScaler.from_checkpoint(checkpoint)
    
    print(f"Loaded SynergyModel from {checkpoint_path}")
    print(f"  - Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    if scaler.fitted:
        print(f"  - Target standardization: Yes (mean={scaler.mean:.4f}, std={scaler.std:.4f})")
    else:
        print(f"  - Target standardization: No")
    
    return model, scaler


def load_mlp_model(checkpoint_path: Path, input_dim: int = 2304):
    """Load SynergyMLP from checkpoint."""
    model = SynergyMLP(input_dim=input_dim).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded SynergyMLP from {checkpoint_path}")
    
    return model


def print_metrics_table(metrics: dict, model_name: str = "Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f" {model_name} - Evaluation Metrics")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Value':<20}")
    print(f"{'-'*40}")
    print(f"{'MSE':<20} {metrics['MSE']:<20.4f}")
    print(f"{'RMSE':<20} {metrics['RMSE']:<20.4f}")
    print(f"{'MAE':<20} {metrics['MAE']:<20.4f}")
    print(f"{'R²':<20} {metrics['R2']:<20.4f}")
    print(f"{'Pearson r':<20} {metrics['Pearson_r']:<20.4f}")
    print(f"{'Pearson p-value':<20} {metrics['Pearson_pval']:<20.2e}")
    print(f"{'Spearman ρ':<20} {metrics['Spearman_r']:<20.4f}")
    print(f"{'Spearman p-value':<20} {metrics['Spearman_pval']:<20.2e}")
    print(f"{'='*60}\n")


def save_evaluation_results(results: dict, output_path: Path):
    """Save evaluation results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def create_comparison_table(attn_metrics: dict, mlp_metrics: dict = None):
    """Create a comparison table between attention model and baseline."""
    print(f"\n{'='*80}")
    print(f" Model Comparison")
    print(f"{'='*80}")
    
    metrics_to_show = ['MSE', 'RMSE', 'MAE', 'R2', 'Pearson_r', 'Spearman_r']
    metric_names = ['MSE ↓', 'RMSE ↓', 'MAE ↓', 'R² ↑', 'Pearson r ↑', 'Spearman ρ ↑']
    
    if mlp_metrics:
        print(f"{'Metric':<20} {'Attention Model':<20} {'MLP Baseline':<20} {'Difference':<15}")
        print(f"{'-'*75}")
        
        for metric, name in zip(metrics_to_show, metric_names):
            attn_val = attn_metrics[metric]
            mlp_val = mlp_metrics[metric]
            diff = attn_val - mlp_val
            
            # Format difference with sign
            diff_str = f"{diff:+.4f}"
            print(f"{name:<20} {attn_val:<20.4f} {mlp_val:<20.4f} {diff_str:<15}")
    else:
        print(f"{'Metric':<20} {'Attention Model':<20}")
        print(f"{'-'*40}")
        for metric, name in zip(metrics_to_show, metric_names):
            print(f"{name:<20} {attn_metrics[metric]:<20.4f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Drug Synergy Models')
    parser.add_argument('--model', type=str, default='attention', 
                        choices=['attention', 'mlp', 'both'],
                        help='Model to evaluate: attention, mlp, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = LOG_PATH / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Evaluation split: {args.split}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    data_path = RAW_DATA_PATH / f"{args.split}_split.pkl"
    with open(data_path, 'rb') as f:
        eval_df = pickle.load(f)
    print(f"Loaded {len(eval_df)} samples from {args.split} split")
    
    # Paths
    drug_emb_path = EMBEDDINGS_PATH / 'drug_embeddings.pt'
    omics_emb_path = EMBEDDINGS_PATH / f'{args.split}_omics.pt'
    target_col = 'Synergy_ZIP'
    
    results = {
        'split': args.split,
        'num_samples': len(eval_df),
        'timestamp': datetime.now().isoformat()
    }
    
    # Evaluate Attention Model
    if args.model in ['attention', 'both']:
        print("\n" + "="*60)
        print(" Evaluating SynergyModel (Attention-based)")
        print("="*60)
        
        # Create dataset
        attn_dataset = DrugSynergyRawOmicsDataset(eval_df, drug_emb_path, target_col)
        attn_loader = DataLoader(attn_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Get omics dimensions
        mrna_dim, mirna_dim, prot_dim = attn_dataset.get_omics_dims()
        print(f"Omics dimensions - mRNA: {mrna_dim}, miRNA: {mirna_dim}, Proteomics: {prot_dim}")
        
        # Load model
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            # Find the latest attention model checkpoint
            attn_dirs = sorted(LOG_PATH.glob("synergy_attn_*"), reverse=True)
            if not attn_dirs:
                raise FileNotFoundError("No attention model checkpoint found!")
            checkpoint_path = attn_dirs[0] / "checkpoint_best.pt"
        
        attn_model, attn_scaler = load_synergy_model(checkpoint_path, mrna_dim, mirna_dim, prot_dim)
        
        # Evaluate
        y_true, y_pred = evaluate_synergy_model(attn_model, attn_loader, device)
        
        # If model was trained with standardization, inverse transform predictions
        # Note: y_true comes from dataset which uses original values, but y_pred is in standardized space
        if attn_scaler.fitted:
            print("\nInverse transforming predictions from standardized space...")
            y_pred = attn_scaler.inverse_transform(y_pred)
        
        attn_metrics = compute_metrics(y_true, y_pred)
        
        print_metrics_table(attn_metrics, "SynergyModel (Attention)")
        results['attention_model'] = {
            'metrics': attn_metrics,
            'checkpoint': str(checkpoint_path),
            'standardized': attn_scaler.fitted
        }
        
        # Save predictions
        attn_preds_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred
        })
        attn_preds_df.to_csv(output_dir / 'attention_predictions.csv', index=False)
    
    # Evaluate MLP Baseline
    if args.model in ['mlp', 'both']:
        print("\n" + "="*60)
        print(" Evaluating SynergyMLP (Baseline)")
        print("="*60)
        
        # Check if omics embeddings exist for MLP
        if not omics_emb_path.exists():
            print(f"WARNING: Omics embeddings not found at {omics_emb_path}")
            print("MLP evaluation requires pre-computed omics embeddings.")
            print("Please run omics_embeddings.py first to generate embeddings.")
            if args.model == 'mlp':
                return
            else:
                print("Skipping MLP evaluation...")
                mlp_metrics = None
        else:
            # Create dataset
            mlp_dataset = DrugSynergyDataset(eval_df, drug_emb_path, omics_emb_path, target_col)
            mlp_loader = DataLoader(mlp_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Find MLP checkpoint
            mlp_dirs = sorted(LOG_PATH.glob("synergy_mlp_*"), reverse=True)
            if not mlp_dirs:
                print("No MLP checkpoint found. Train MLP first with train_mlp.py")
                mlp_metrics = None
            else:
                mlp_checkpoint_path = mlp_dirs[0] / "checkpoint_best.pt"
                
                if not mlp_checkpoint_path.exists():
                    print(f"MLP checkpoint not found at {mlp_checkpoint_path}")
                    mlp_metrics = None
                else:
                    mlp_model = load_mlp_model(mlp_checkpoint_path)
                    
                    # Evaluate
                    y_true_mlp, y_pred_mlp = evaluate_mlp_model(mlp_model, mlp_loader, device)
                    mlp_metrics = compute_metrics(y_true_mlp, y_pred_mlp)
                    
                    print_metrics_table(mlp_metrics, "SynergyMLP (Baseline)")
                    results['mlp_baseline'] = {
                        'metrics': mlp_metrics,
                        'checkpoint': str(mlp_checkpoint_path)
                    }
                    
                    # Save predictions
                    mlp_preds_df = pd.DataFrame({
                        'y_true': y_true_mlp,
                        'y_pred': y_pred_mlp
                    })
                    mlp_preds_df.to_csv(output_dir / 'mlp_predictions.csv', index=False)
    
    # Create comparison table
    if args.model == 'both':
        attn_metrics = results.get('attention_model', {}).get('metrics')
        mlp_metrics = results.get('mlp_baseline', {}).get('metrics')
        if attn_metrics:
            create_comparison_table(attn_metrics, mlp_metrics)
    
    # Save all results
    save_evaluation_results(results, output_dir / 'evaluation_results.json')
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
