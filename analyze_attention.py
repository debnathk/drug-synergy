"""
Attention Analysis for Drug Synergy Prediction Model.
Extracts and visualizes omics self-attention weights from OmicsAttentionFusion.

The model uses a CLS token that attends to mRNA, miRNA, and Proteomics embeddings.
This script visualizes:
  1. CLS-to-modality attention (which omics the model focuses on)
  2. Full self-attention matrix (including CLS and modality-to-modality)
  3. Attention patterns by synergy level (high vs low)
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
from torch.utils.data import Dataset

from src.dataset import DrugSynergyRawOmicsDataset
from src.models.synergy_model import SynergyModel

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"
LOG_PATH = ROOT / "logs"


# ---------------------------------------------------------------------------
# AblationDataset (for compatibility with ablation checkpoints)
# ---------------------------------------------------------------------------

class AblationDataset(Dataset):
    """
    Wraps DrugSynergyRawOmicsDataset and zeroes out one or all omics modalities.
    Used to ensure consistent data during attention visualization for ablation models.
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

    def get_omics_dims(self):
        return self.base.get_omics_dims()
ASSETS_PATH = ROOT / "assets"

# Labels for attention positions (CLS token + 3 omics modalities)
FULL_LABELS = ['CLS', 'mRNA', 'miRNA', 'Proteomics']
MODALITY_LABELS = ['mRNA', 'miRNA', 'Proteomics']


def get_raw_path(split_type: str) -> Path:
    """Resolve raw data directory: raw/<split_type>/ or flat raw/ for backward compat."""
    subdir = RAW_DATA_PATH / split_type
    flat_train = RAW_DATA_PATH / "train_split.pkl"
    if subdir.exists() and (subdir / "train_split.pkl").exists():
        return subdir
    if split_type == "random" and flat_train.exists():
        return RAW_DATA_PATH
    return subdir


def get_drug_emb_path(split_type: str) -> Path:
    """Resolve drug embeddings path; fallback to drug_embeddings.pt for backward compat."""
    specific = EMBEDDINGS_PATH / f"drug_embeddings_{split_type}.pt"
    default = EMBEDDINGS_PATH / "drug_embeddings.pt"
    return specific if specific.exists() else default


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_checkpoint_architecture(state_dict):
    """
    Detect architecture version from checkpoint state_dict.
    
    Returns:
        dict with architecture info, or raises error if incompatible.
    """
    has_project = any('omics_encoder.project.' in k for k in state_dict.keys())
    
    # Check CLS token shape to determine emb_dim
    cls_key = 'omics_encoder.fusion.cls_token'
    if cls_key in state_dict:
        cls_shape = state_dict[cls_key].shape
        emb_dim = cls_shape[-1]
    else:
        emb_dim = None
    
    # Check encoder hidden dim from first linear layer bias
    hidden_key = 'omics_encoder.mrna_encoder.net.1.bias'
    if hidden_key in state_dict:
        hidden_dim = state_dict[hidden_key].shape[0]
    else:
        hidden_dim = None
    
    # Check encoder output dim
    out_key = 'omics_encoder.mrna_encoder.net.4.bias'
    if out_key in state_dict:
        encoder_out_dim = state_dict[out_key].shape[0]
    else:
        encoder_out_dim = None
    
    return {
        'has_project': has_project,
        'emb_dim': emb_dim,
        'hidden_dim': hidden_dim,
        'encoder_out_dim': encoder_out_dim,
    }


def load_model(checkpoint_path: Path, mrna_dim: int, mirna_dim: int, prot_dim: int):
    """Load SynergyModel from checkpoint with architecture compatibility check."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    head_type = config.get('head_type', 'mlp')
    grid_size = config.get('grid_size', 5)
    
    # Detect checkpoint architecture
    state_dict = checkpoint['model_state_dict']
    arch_info = detect_checkpoint_architecture(state_dict)
    
    # Current architecture: hidden=1024, out=768, no project layer
    current_hidden = 1024
    current_emb = 768
    
    if arch_info['has_project'] or arch_info['emb_dim'] != current_emb or arch_info['hidden_dim'] != current_hidden:
        error_msg = f"""
================================================================================
ARCHITECTURE MISMATCH: Checkpoint incompatible with current model
================================================================================

Checkpoint: {checkpoint_path}
  - OmicsEncoder hidden_dim: {arch_info['hidden_dim']} (current model: {current_hidden})
  - OmicsEncoder out_dim: {arch_info['encoder_out_dim']} (current model: {current_emb})
  - OmicsAttentionFusion emb_dim: {arch_info['emb_dim']} (current model: {current_emb})
  - Has project layer: {arch_info['has_project']} (current model: False)

This checkpoint was trained with an older version of SynergyModel.

SOLUTIONS:
1. Use a checkpoint trained with the current architecture (after recent model changes)
2. Train a new model with: python main.py --head_type {head_type} --standardize --epochs 50

To find compatible checkpoints, look for recent experiment folders in logs/
================================================================================
"""
        raise RuntimeError(error_msg)

    model = SynergyModel(
        mrna_dim=mrna_dim,
        mirna_dim=mirna_dim,
        prot_dim=prot_dim,
        head_type=head_type,
        grid_size=grid_size
    ).to(device)

    model.load_state_dict(state_dict)
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Head type: {head_type}, Grid size: {grid_size}")
    print(f"  Architecture: hidden={arch_info['hidden_dim']}, emb={arch_info['emb_dim']}")

    # Check if this is an ablation checkpoint
    ablation_info = {
        'is_ablation': 'ablation_condition' in config or 'ablated_modality' in config,
        'condition': config.get('ablation_condition'),
        'ablate_key': config.get('ablated_modality'),
    }
    if ablation_info['is_ablation']:
        print(f"  Ablation checkpoint: {ablation_info['condition']} (ablate_key: {ablation_info['ablate_key']})")

    return model, ablation_info


def extract_attention_weights(model, dataloader, device, num_samples=None):
    """
    Extract omics self-attention weights from the model.

    Returns:
        dict with:
            - omics_attention: [N, num_heads, 4, 4] - full attention (CLS + modalities)
            - predictions: [N] - model predictions
            - targets: [N] - ground truth values
    """
    model.eval()

    all_omics_attn = []
    all_preds = []
    all_targets = []

    samples_collected = 0

    with torch.no_grad():
        for batch_idx, (d1, d2, mrna, mirna, prot, y) in enumerate(tqdm(dataloader, desc="Extracting attention")):
            d1, d2 = d1.to(device), d2.to(device)
            mrna, mirna, prot = mrna.to(device), mirna.to(device), prot.to(device)
            y = y.to(device)

            preds, attention_weights = model(d1, d2, mrna, mirna, prot, return_attention=True)

            all_omics_attn.append(attention_weights['omics_fusion'].cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

            samples_collected += d1.size(0)
            if num_samples and samples_collected >= num_samples:
                break

    return {
        'omics_attention': np.concatenate(all_omics_attn, axis=0),
        'predictions': np.concatenate(all_preds, axis=0).flatten(),
        'targets': np.concatenate(all_targets, axis=0).flatten()
    }


def plot_full_attention_matrix(attention_weights, output_path):
    """
    Plot full 4x4 attention matrix (CLS + mRNA + miRNA + Proteomics).

    Args:
        attention_weights: [N, num_heads, 4, 4]
        output_path: Path to save the plot
    """
    avg_attn = attention_weights.mean(axis=(0, 1))  # [4, 4]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        avg_attn,
        xticklabels=FULL_LABELS,
        yticklabels=FULL_LABELS,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=0,
        vmax=avg_attn.max(),
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    # plt.title('Omics Self-Attention (Average across samples and heads)')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cls_to_modality_attention(attention_weights, output_path):
    """
    Plot how the CLS token attends to each omics modality.
    This is the most interpretable attention - shows which modalities
    the model focuses on when forming the fused representation.

    Args:
        attention_weights: [N, num_heads, 4, 4]
        output_path: Path to save the plot
    """
    # CLS is row 0, modalities are columns 1, 2, 3
    cls_to_modality = attention_weights[:, :, 0, 1:4]  # [N, num_heads, 3]

    # Average across samples and heads
    avg_attn = cls_to_modality.mean(axis=(0, 1))  # [3]
    std_attn = cls_to_modality.mean(axis=1).std(axis=0)  # std across samples

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("Blues_d", 3)
    bars = plt.bar(MODALITY_LABELS, avg_attn, yerr=std_attn, capsize=5, color=colors)

    for bar, val in zip(bars, avg_attn):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_attn.max() + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # plt.title('CLS Token Attention to Omics Modalities\n(How much the model attends to each omics type)')
    plt.ylabel('Attention Weight')
    plt.ylim(0, min(1.0, avg_attn.max() + std_attn.max() + 0.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_per_head(attention_weights, output_path):
    """
    Plot attention matrices for each head separately.

    Args:
        attention_weights: [N, num_heads, 4, 4]
        output_path: Path to save the plot
    """
    avg_attn = attention_weights.mean(axis=0)  # [num_heads, 4, 4]
    num_heads = avg_attn.shape[0]

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            avg_attn[head_idx],
            xticklabels=FULL_LABELS,
            yticklabels=FULL_LABELS,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            vmin=0,
            vmax=avg_attn[head_idx].max(),
            square=True,
            ax=ax,
            cbar=False
        )
        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    # plt.suptitle('Omics Self-Attention Per Head', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cls_attention_per_head(attention_weights, output_path):
    """
    Plot CLS-to-modality attention for each head as grouped bar chart.

    Args:
        attention_weights: [N, num_heads, 4, 4]
        output_path: Path to save the plot
    """
    cls_to_modality = attention_weights[:, :, 0, 1:4]  # [N, num_heads, 3]
    avg_per_head = cls_to_modality.mean(axis=0)  # [num_heads, 3]
    num_heads = avg_per_head.shape[0]

    x = np.arange(len(MODALITY_LABELS))
    width = 0.8 / num_heads

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", num_heads)

    for i in range(num_heads):
        offset = (i - num_heads / 2 + 0.5) * width
        ax.bar(x + offset, avg_per_head[i], width, label=f'Head {i + 1}', color=colors[i])

    ax.set_ylabel('Attention Weight')
    # ax.set_title('CLS Attention to Modalities by Head')
    ax.set_xticks(x)
    ax.set_xticklabels(MODALITY_LABELS)
    ax.legend(title='Attention Head')
    ax.set_ylim(0, min(1.0, avg_per_head.max() + 0.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_modality_attention_only(attention_weights, output_path):
    """
    Plot modality-to-modality attention (excluding CLS).
    Shows how mRNA, miRNA, and Proteomics attend to each other.

    Args:
        attention_weights: [N, num_heads, 4, 4]
        output_path: Path to save the plot
    """
    # Extract modality-to-modality block (rows 1-3, cols 1-3)
    mod_to_mod = attention_weights[:, :, 1:4, 1:4]  # [N, num_heads, 3, 3]
    avg_attn = mod_to_mod.mean(axis=(0, 1))  # [3, 3]

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        avg_attn,
        xticklabels=MODALITY_LABELS,
        yticklabels=MODALITY_LABELS,
        annot=True,
        fmt='.3f',
        cmap='Greens',
        vmin=0,
        vmax=avg_attn.max(),
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    # plt.title('Modality-to-Modality Attention\n(How omics modalities attend to each other)')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_attention_by_synergy(attention_data, output_dir):
    """
    Compare CLS-to-modality attention patterns for high vs low synergy samples.
    """
    targets = attention_data['targets']
    omics_attn = attention_data['omics_attention']

    # CLS-to-modality attention
    cls_to_mod = omics_attn[:, :, 0, 1:4]  # [N, num_heads, 3]

    median_synergy = np.median(targets)
    high_mask = targets > median_synergy
    low_mask = targets <= median_synergy

    high_attn = cls_to_mod[high_mask].mean(axis=(0, 1))  # [3]
    low_attn = cls_to_mod[low_mask].mean(axis=(0, 1))  # [3]

    high_std = cls_to_mod[high_mask].mean(axis=1).std(axis=0)
    low_std = cls_to_mod[low_mask].mean(axis=1).std(axis=0)

    x = np.arange(len(MODALITY_LABELS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparison
    ax1 = axes[0]
    ax1.bar(x - width / 2, high_attn, width, yerr=high_std, capsize=4,
            label=f'High Synergy (>{median_synergy:.1f})', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width / 2, low_attn, width, yerr=low_std, capsize=4,
            label=f'Low Synergy (≤{median_synergy:.1f})', color='#3498db', alpha=0.8)

    ax1.set_ylabel('CLS Attention Weight')
    # ax1.set_title('CLS Attention to Modalities by Synergy Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODALITY_LABELS)
    ax1.legend()
    ax1.set_ylim(0, min(1.0, max(high_attn.max(), low_attn.max()) + 0.15))

    # Difference plot
    ax2 = axes[1]
    diff = high_attn - low_attn
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
    ax2.bar(MODALITY_LABELS, diff, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Attention Difference (High - Low)')
    ax2.set_title('Attention Shift: High vs Low Synergy')

    for i, (label, d) in enumerate(zip(MODALITY_LABELS, diff)):
        ax2.text(i, d + 0.005 if d > 0 else d - 0.015,
                 f'{d:+.3f}', ha='center', fontsize=10)

    # plt.suptitle(f'Omics Attention by Synergy Level (median threshold: {median_synergy:.2f})',
                #  fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_by_synergy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'attention_by_synergy.png'}")


def generate_attention_report(attention_data, output_dir, ablation_info=None, ablation_applied=False):
    """Generate a text report summarizing attention patterns."""

    omics_attn = attention_data['omics_attention']
    targets = attention_data['targets']

    cls_to_mod = omics_attn[:, :, 0, 1:4]  # [N, num_heads, 3]
    avg_cls_attn = cls_to_mod.mean(axis=(0, 1))

    report_lines = [
        "=" * 60,
        " Omics Self-Attention Analysis Report",
        "=" * 60,
        f"\nNumber of samples analyzed: {len(targets)}",
        f"Synergy score range: [{targets.min():.2f}, {targets.max():.2f}]",
        f"Mean synergy score: {targets.mean():.2f}",
        f"Attention shape: {omics_attn.shape} (samples, heads, 4, 4)",
    ]

    # Add ablation info if relevant
    if ablation_info and ablation_info.get('is_ablation'):
        report_lines.extend([
            "",
            "-" * 60,
            " Ablation Information",
            "-" * 60,
            f"\nCheckpoint from ablation training: {ablation_info.get('condition')}",
            f"Ablated modality: {ablation_info.get('ablate_key')}",
            f"Ablation applied to viz data: {'Yes' if ablation_applied else 'No (--no_ablation)'}",
        ])
        if ablation_applied:
            report_lines.append(
                "  → Attention shown is on ABLATED data (matching training conditions)"
            )
        else:
            report_lines.append(
                "  → Attention shown is on FULL data (may differ from training conditions)"
            )

    report_lines.extend([
        "",
        "-" * 60,
        " CLS Token Attention to Omics Modalities",
        "-" * 60,
        "\nThe CLS token aggregates information from all modalities.",
        "Higher attention = model focuses more on that modality.\n",
    ])

    total_attn = avg_cls_attn.sum()
    for label, attn in zip(MODALITY_LABELS, avg_cls_attn):
        pct = attn / total_attn * 100
        report_lines.append(f"  {label}: {attn:.4f} ({pct:.1f}%)")

    most_attended_idx = avg_cls_attn.argmax()
    report_lines.append(f"\n  → Most attended: {MODALITY_LABELS[most_attended_idx]}")

    # Per-head analysis
    avg_per_head = cls_to_mod.mean(axis=0)  # [num_heads, 3]
    num_heads = avg_per_head.shape[0]

    report_lines.extend([
        "",
        "-" * 60,
        " Per-Head Attention Analysis",
        "-" * 60,
    ])

    for h in range(num_heads):
        head_attn = avg_per_head[h]
        most_idx = head_attn.argmax()
        report_lines.append(
            f"  Head {h + 1}: mRNA={head_attn[0]:.3f}, miRNA={head_attn[1]:.3f}, "
            f"Prot={head_attn[2]:.3f} → focuses on {MODALITY_LABELS[most_idx]}"
        )

    # Synergy-based analysis
    median_synergy = np.median(targets)
    high_mask = targets > median_synergy

    high_attn = cls_to_mod[high_mask].mean(axis=(0, 1))
    low_attn = cls_to_mod[~high_mask].mean(axis=(0, 1))
    diff = high_attn - low_attn

    report_lines.extend([
        "",
        "-" * 60,
        " Attention by Synergy Level",
        "-" * 60,
        f"\nMedian synergy threshold: {median_synergy:.2f}",
        f"High synergy samples: {high_mask.sum()}",
        f"Low synergy samples: {(~high_mask).sum()}",
        "",
        "CLS attention differences (High - Low):",
    ])

    for label, d in zip(MODALITY_LABELS, diff):
        direction = "↑" if d > 0.001 else ("↓" if d < -0.001 else "≈")
        report_lines.append(f"  {label}: {d:+.4f} {direction}")

    report_lines.extend([
        "",
        "=" * 60,
        " Key Findings",
        "=" * 60,
        f"\n1. Most attended modality overall: {MODALITY_LABELS[avg_cls_attn.argmax()]} "
        f"({avg_cls_attn.max():.3f})",
    ])

    max_diff_idx = np.abs(diff).argmax()
    if abs(diff[max_diff_idx]) > 0.01:
        direction = "more" if diff[max_diff_idx] > 0 else "less"
        report_lines.append(
            f"2. Largest synergy-dependent shift: {MODALITY_LABELS[max_diff_idx]} "
            f"(high synergy attends {direction})"
        )
    else:
        report_lines.append("2. Attention patterns are similar for high and low synergy samples")

    report_lines.append("")

    report_path = output_dir / "attention_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Saved: {report_path}")
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze Omics Self-Attention Weights')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to analyze')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'cold_drug'],
                        help='Data split type (must match training)')
    parser.add_argument('--target', type=str, default='Synergy_ZIP',
                        choices=['Synergy_ZIP', 'Synergy_Bliss', 'Synergy_Loewe', 'Synergy_HSA'],
                        help='Target column')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to analyze (default: all samples)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results')
    parser.add_argument('--no_ablation', action='store_true', default=False,
                        help='Skip ablation even for ablation checkpoints (see attention on full data)')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = LOG_PATH / f"{args.target}_{args.split}_{args.split_type}_attention_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = get_raw_path(args.split_type)
    drug_emb_path = get_drug_emb_path(args.split_type)

    print(f"Device: {device}")
    print(f"Split type: {args.split_type}, analysis split: {args.split}")
    print(f"Target: {args.target}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading data...")
    data_path = raw_path / f"{args.split}_split.pkl"
    with open(data_path, "rb") as f:
        df = pickle.load(f)
    print(f"Loaded {len(df)} samples (raw path: {raw_path})")

    base_dataset = DrugSynergyRawOmicsDataset(df, drug_emb_path, args.target)
    mrna_dim, mirna_dim, prot_dim = base_dataset.get_omics_dims()

    # Load model
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Try to find a recent checkpoint
        patterns = ["kan_*", "mlp_*", "synergy_kan_*", "synergy_mlp_*", "synergy_attn_*"]
        candidates = []
        for pattern in patterns:
            candidates.extend(LOG_PATH.glob(pattern))
        if not candidates:
            raise FileNotFoundError("No model checkpoint found! Specify --checkpoint")
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        checkpoint_path = candidates[0] / "checkpoint_best.pt"

    print(f"\nLoading model from: {checkpoint_path}")
    model, ablation_info = load_model(checkpoint_path, mrna_dim, mirna_dim, prot_dim)

    # Apply ablation to dataset if checkpoint is from ablation training
    ablation_applied = False
    if ablation_info['is_ablation'] and ablation_info['ablate_key'] and not args.no_ablation:
        ablate_key = ablation_info['ablate_key']
        print(f"\nAblation checkpoint detected! Applying ablation: {ablation_info['condition']}")
        print(f"  Zeroing modality: '{ablate_key}'")
        print(f"  (Use --no_ablation to visualize attention on full data instead)")
        dataset = AblationDataset(base_dataset, ablate=ablate_key)
        ablation_applied = True
    elif ablation_info['is_ablation'] and args.no_ablation:
        print(f"\nAblation checkpoint detected but --no_ablation specified.")
        print(f"  Visualizing attention on FULL data (not ablated).")
        dataset = base_dataset
    else:
        dataset = base_dataset

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Extract attention weights
    print("\nExtracting attention weights...")
    attention_data = extract_attention_weights(
        model, dataloader, device,
        num_samples=args.num_samples
    )

    print(f"\nAnalyzing {len(attention_data['targets'])} samples...")
    print(f"Omics attention shape: {attention_data['omics_attention'].shape}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Full attention matrix (CLS + modalities)
    plot_full_attention_matrix(
        attention_data['omics_attention'],
        output_dir / 'omics_full_attention.png'
    )

    # 2. CLS-to-modality attention (most interpretable)
    plot_cls_to_modality_attention(
        attention_data['omics_attention'],
        output_dir / 'cls_to_modality_attention.png'
    )

    # 3. Per-head full attention
    plot_attention_per_head(
        attention_data['omics_attention'],
        output_dir / 'omics_attention_per_head.png'
    )

    # 4. CLS attention per head (grouped bar)
    plot_cls_attention_per_head(
        attention_data['omics_attention'],
        output_dir / 'cls_attention_per_head.png'
    )

    # 5. Modality-to-modality attention (excluding CLS)
    plot_modality_attention_only(
        attention_data['omics_attention'],
        output_dir / 'modality_to_modality_attention.png'
    )

    # 6. Attention by synergy level
    plot_attention_by_synergy(attention_data, output_dir)

    # Generate report
    print("\nGenerating attention report...")
    generate_attention_report(attention_data, output_dir, ablation_info, ablation_applied)

    # Copy key visualizations to assets folder
    print("\nCopying key visualizations to assets folder...")
    import shutil
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)

    key_files = [
        'cls_to_modality_attention.png',
        'omics_full_attention.png',
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
