"""
Attention Analysis for Drug Synergy Prediction Model.
Extracts and visualizes attention weights from the SynergyModel.

Author: Kusal Debnath
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm

from src.dataset import DrugSynergyRawOmicsDataset
from src.models.synergy_model import SynergyModel

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"
LOG_PATH = ROOT / "logs"
ASSETS_PATH = ROOT / "assets"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, mrna_dim: int, mirna_dim: int, prot_dim: int):
    """Load SynergyModel from checkpoint."""
    model = SynergyModel(
        mrna_dim=mrna_dim,
        mirna_dim=mirna_dim,
        prot_dim=prot_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    
    return model


def extract_attention_weights(model, dataloader, device, num_samples=None):
    """
    Extract attention weights from the model.
    
    Returns:
        dict with:
            - omics_attention: [N, num_heads, 3, 3] - omics fusion attention
            - cross_attention: [N, num_heads, 3, 3] - drug-omics cross attention
            - predictions: [N] - model predictions
            - targets: [N] - ground truth values
    """
    model.eval()
    
    all_omics_attn = []
    all_cross_attn = []
    all_preds = []
    all_targets = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch_idx, (d1, d2, mrna, mirna, prot, y) in enumerate(tqdm(dataloader, desc="Extracting attention")):
            d1, d2 = d1.to(device), d2.to(device)
            mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
            y = y.to(device)
            
            # Forward pass with attention
            preds, attention_weights = model(d1, d2, mrna, mirna, prot, return_attention=True)
            
            all_omics_attn.append(attention_weights['omics_fusion'].cpu().numpy())
            all_cross_attn.append(attention_weights['cross_attention'].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            samples_collected += d1.size(0)
            if num_samples and samples_collected >= num_samples:
                break
    
    return {
        'omics_attention': np.concatenate(all_omics_attn, axis=0),
        'cross_attention': np.concatenate(all_cross_attn, axis=0),
        'predictions': np.concatenate(all_preds, axis=0).flatten(),
        'targets': np.concatenate(all_targets, axis=0).flatten()
    }


def plot_average_attention_matrix(attention_weights, labels, title, output_path):
    """
    Plot average attention matrix across all samples and heads.
    
    Args:
        attention_weights: [N, num_heads, seq_len, seq_len]
        labels: List of labels for attention positions
        title: Plot title
        output_path: Path to save the plot
    """
    # Average across samples and heads
    avg_attn = attention_weights.mean(axis=(0, 1))  # [seq_len, seq_len]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        avg_attn,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.title(title)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_per_head(attention_weights, labels, title_prefix, output_path):
    """
    Plot attention matrices for each head.
    
    Args:
        attention_weights: [N, num_heads, seq_len, seq_len]
        labels: List of labels for attention positions
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
    """
    # Average across samples
    avg_attn = attention_weights.mean(axis=0)  # [num_heads, seq_len, seq_len]
    num_heads = avg_attn.shape[0]
    
    # Create subplot grid
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            avg_attn[head_idx],
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            vmin=0,
            vmax=1,
            square=True,
            ax=ax,
            cbar=False
        )
        ax.set_title(f'{title_prefix} - Head {head_idx+1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    # Hide unused axes
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_distribution(attention_weights, labels, title, output_path):
    """
    Plot distribution of attention weights.
    
    Shows how much attention each position receives on average.
    """
    # Average across samples and heads: [seq_len, seq_len]
    avg_attn = attention_weights.mean(axis=(0, 1))
    
    # Sum attention received by each position (column sum)
    attention_received = avg_attn.sum(axis=0)
    
    # Normalize to percentages
    attention_pct = attention_received / attention_received.sum() * 100
    
    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("Blues_d", len(labels))
    bars = plt.bar(labels, attention_pct, color=colors)
    
    # Add value labels on bars
    for bar, pct in zip(bars, attention_pct):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    plt.title(title)
    plt.ylabel('Attention Received (%)')
    plt.ylim(0, max(attention_pct) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_by_synergy(attention_data, output_dir):
    """
    Analyze how attention patterns differ based on synergy score.
    
    Compares attention for high vs low synergy samples.
    """
    targets = attention_data['targets']
    cross_attn = attention_data['cross_attention']
    
    # Split into high and low synergy (above/below median)
    median_synergy = np.median(targets)
    high_synergy_mask = targets > median_synergy
    low_synergy_mask = targets <= median_synergy
    
    high_synergy_attn = cross_attn[high_synergy_mask].mean(axis=(0, 1))
    low_synergy_attn = cross_attn[low_synergy_mask].mean(axis=(0, 1))
    
    labels = ['Drug 1', 'Drug 2', 'Omics']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # High synergy attention
    sns.heatmap(
        high_synergy_attn,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.3f',
        cmap='Reds',
        vmin=0,
        vmax=1,
        square=True,
        ax=axes[0]
    )
    axes[0].set_title(f'High Synergy (>{median_synergy:.1f})')
    axes[0].set_xlabel('Key')
    axes[0].set_ylabel('Query')
    
    # Low synergy attention
    sns.heatmap(
        low_synergy_attn,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=0,
        vmax=1,
        square=True,
        ax=axes[1]
    )
    axes[1].set_title(f'Low Synergy (≤{median_synergy:.1f})')
    axes[1].set_xlabel('Key')
    axes[1].set_ylabel('Query')
    
    # Difference (High - Low)
    diff_attn = high_synergy_attn - low_synergy_attn
    max_diff = max(abs(diff_attn.min()), abs(diff_attn.max()))
    sns.heatmap(
        diff_attn,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        vmin=-max_diff,
        vmax=max_diff,
        square=True,
        ax=axes[2],
        center=0
    )
    axes[2].set_title('Difference (High - Low)')
    axes[2].set_xlabel('Key')
    axes[2].set_ylabel('Query')
    
    plt.suptitle('Cross-Attention Patterns by Synergy Level', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_by_synergy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'attention_by_synergy.png'}")


def generate_attention_report(attention_data, output_dir):
    """Generate a text report summarizing attention patterns."""
    
    omics_attn = attention_data['omics_attention']
    cross_attn = attention_data['cross_attention']
    
    omics_labels = ['mRNA', 'miRNA', 'Proteomics']
    cross_labels = ['Drug 1', 'Drug 2', 'Omics']
    
    report_lines = [
        "=" * 60,
        " Attention Analysis Report",
        "=" * 60,
        f"\nNumber of samples analyzed: {len(attention_data['targets'])}",
        f"Synergy score range: [{attention_data['targets'].min():.2f}, {attention_data['targets'].max():.2f}]",
        f"Mean synergy score: {attention_data['targets'].mean():.2f}",
        "",
        "-" * 60,
        " Omics Fusion Attention (mRNA, miRNA, Proteomics)",
        "-" * 60,
    ]
    
    # Omics attention analysis
    avg_omics = omics_attn.mean(axis=(0, 1))
    omics_received = avg_omics.sum(axis=0) / avg_omics.sum(axis=0).sum() * 100
    
    report_lines.append("\nAttention received by each modality:")
    for label, pct in zip(omics_labels, omics_received):
        report_lines.append(f"  {label}: {pct:.1f}%")
    
    report_lines.extend([
        "",
        "-" * 60,
        " Drug-Omics Cross-Attention (Drug1, Drug2, Omics)",
        "-" * 60,
    ])
    
    # Cross attention analysis
    avg_cross = cross_attn.mean(axis=(0, 1))
    cross_received = avg_cross.sum(axis=0) / avg_cross.sum(axis=0).sum() * 100
    
    report_lines.append("\nAttention received by each component:")
    for label, pct in zip(cross_labels, cross_received):
        report_lines.append(f"  {label}: {pct:.1f}%")
    
    # Analysis by synergy level
    targets = attention_data['targets']
    median_synergy = np.median(targets)
    high_mask = targets > median_synergy
    
    high_cross = cross_attn[high_mask].mean(axis=(0, 1))
    low_cross = cross_attn[~high_mask].mean(axis=(0, 1))
    
    report_lines.extend([
        "",
        "-" * 60,
        " Attention Patterns by Synergy Level",
        "-" * 60,
        f"\nMedian synergy threshold: {median_synergy:.2f}",
        f"High synergy samples: {high_mask.sum()}",
        f"Low synergy samples: {(~high_mask).sum()}",
        "",
        "Cross-attention differences (High - Low):",
    ])
    
    diff = high_cross - low_cross
    for i, q_label in enumerate(cross_labels):
        for j, k_label in enumerate(cross_labels):
            if abs(diff[i, j]) > 0.01:  # Only show significant differences
                report_lines.append(f"  {q_label} → {k_label}: {diff[i,j]:+.3f}")
    
    report_lines.extend([
        "",
        "=" * 60,
        " Key Findings",
        "=" * 60,
    ])
    
    # Key findings
    omics_max_idx = omics_received.argmax()
    cross_max_idx = cross_received.argmax()
    
    report_lines.append(f"\n1. Most attended omics modality: {omics_labels[omics_max_idx]} ({omics_received[omics_max_idx]:.1f}%)")
    report_lines.append(f"2. Most attended in cross-attention: {cross_labels[cross_max_idx]} ({cross_received[cross_max_idx]:.1f}%)")
    
    report_lines.append("")
    
    # Save report
    report_path = output_dir / "attention_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved: {report_path}")
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze Attention Weights')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to analyze')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to analyze (None for all)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = LOG_PATH / f"attention_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Analysis split: {args.split}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    data_path = RAW_DATA_PATH / f"{args.split}_split.pkl"
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    print(f"Loaded {len(df)} samples")
    
    # Create dataset
    drug_emb_path = EMBEDDINGS_PATH / 'drug_embeddings.pt'
    target_col = 'Synergy_ZIP'
    
    dataset = DrugSynergyRawOmicsDataset(df, drug_emb_path, target_col)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    mrna_dim, mirna_dim, prot_dim = dataset.get_omics_dims()
    
    # Load model
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        attn_dirs = sorted(LOG_PATH.glob("synergy_attn_*"), reverse=True)
        if not attn_dirs:
            raise FileNotFoundError("No attention model checkpoint found!")
        checkpoint_path = attn_dirs[0] / "checkpoint_best.pt"
    
    model = load_model(checkpoint_path, mrna_dim, mirna_dim, prot_dim)
    
    # Extract attention weights
    print("\nExtracting attention weights...")
    attention_data = extract_attention_weights(
        model, dataloader, device,
        num_samples=args.num_samples
    )
    
    print(f"\nAnalyzing {len(attention_data['targets'])} samples...")
    print(f"Omics attention shape: {attention_data['omics_attention'].shape}")
    print(f"Cross attention shape: {attention_data['cross_attention'].shape}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Omics fusion attention
    omics_labels = ['mRNA', 'miRNA', 'Proteomics']
    plot_average_attention_matrix(
        attention_data['omics_attention'],
        omics_labels,
        'Omics Fusion - Average Attention',
        output_dir / 'omics_fusion_attention.png'
    )
    
    plot_attention_per_head(
        attention_data['omics_attention'],
        omics_labels,
        'Omics Fusion',
        output_dir / 'omics_fusion_per_head.png'
    )
    
    plot_attention_distribution(
        attention_data['omics_attention'],
        omics_labels,
        'Attention Received by Omics Modality',
        output_dir / 'omics_attention_distribution.png'
    )
    
    # Cross attention
    cross_labels = ['Drug 1', 'Drug 2', 'Omics']
    plot_average_attention_matrix(
        attention_data['cross_attention'],
        cross_labels,
        'Drug-Omics Cross-Attention - Average',
        output_dir / 'cross_attention.png'
    )
    
    plot_attention_per_head(
        attention_data['cross_attention'],
        cross_labels,
        'Cross-Attention',
        output_dir / 'cross_attention_per_head.png'
    )
    
    plot_attention_distribution(
        attention_data['cross_attention'],
        cross_labels,
        'Attention Received by Component',
        output_dir / 'cross_attention_distribution.png'
    )
    
    # Attention by synergy level
    plot_attention_by_synergy(attention_data, output_dir)
    
    # Generate report
    print("\nGenerating attention report...")
    generate_attention_report(attention_data, output_dir)
    
    # Also save to assets folder for easy access
    print("\nCopying key visualizations to assets folder...")
    import shutil
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    
    key_files = [
        'omics_fusion_attention.png',
        'cross_attention.png',
        'attention_by_synergy.png'
    ]
    for filename in key_files:
        src = output_dir / filename
        if src.exists():
            shutil.copy(src, ASSETS_PATH / filename)
            print(f"  Copied: {filename}")
    
    print(f"\nAttention analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
