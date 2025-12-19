"""
Download SMILES datasets for MoleculeVAE pretraining.

Supports multiple sources:
- GuacaMol (ChEMBL): 1.6M drug-like molecules (recommended for quick start)
- ZINC250K: 250K molecules (benchmark dataset)
- Future: ZINC20, PubChem, Enamine REAL for larger scale

Author: Kusal Debnath
"""

import argparse
import gzip
import json
import os
import requests
from pathlib import Path
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "guacamol"


class DatasetDownloader:
    """Download and prepare SMILES datasets"""

    # Dataset URLs
    DATASETS = {
        'guacamol': {
            'train': 'https://ndownloader.figshare.com/files/13612760',
            'valid': 'https://ndownloader.figshare.com/files/13612766',
            'test': 'https://ndownloader.figshare.com/files/13612745',
            'description': 'GuacaMol ChEMBL dataset (~1.6M drug-like molecules)'
        },
        'zinc250k': {
            'train': 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv',
            'description': 'ZINC250K dataset (250K molecules)'
        }
    }

    def __init__(self, output_dir=DATA_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename, chunk_size=8192):
        """Download file with progress bar"""
        filepath = self.output_dir / filename

        print(f"Downloading {filename}...")
        print(f"  URL: {url}")
        print(f"  Output: {filepath}")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✓ Downloaded: {filepath} ({total_size / 1024 / 1024:.2f} MB)\n")
        return filepath

    def download_guacamol(self, use_train_only=True, num_molecules=None):
        """
        Download GuacaMol dataset

        Args:
            use_train_only: If True, only download training set (faster)
            num_molecules: Limit to N molecules (None = all)

        Returns:
            train_path, val_path
        """
        print("="*80)
        print("Downloading GuacaMol Dataset")
        print("="*80)
        print(self.DATASETS['guacamol']['description'])
        print()

        # Download files
        train_path = self.download_file(
            self.DATASETS['guacamol']['train'],
            'guacamol_train.smiles'
        )

        if not use_train_only:
            valid_path = self.download_file(
                self.DATASETS['guacamol']['valid'],
                'guacamol_valid.smiles'
            )
        else:
            valid_path = None

        # Read and process
        print("Processing SMILES...")
        with open(train_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        print(f"  Total SMILES: {len(smiles_list):,}")

        # Limit if requested
        if num_molecules and num_molecules < len(smiles_list):
            print(f"  Limiting to {num_molecules:,} molecules")
            random.seed(42)
            smiles_list = random.sample(smiles_list, num_molecules)

        # Split into train/val (95/5 split)
        n_train = int(len(smiles_list) * 0.95)
        train_smiles = smiles_list[:n_train]
        val_smiles = smiles_list[n_train:]

        print(f"  Train: {len(train_smiles):,}")
        print(f"  Val:   {len(val_smiles):,}")

        # Save splits
        train_output = self.output_dir / 'train_smiles.txt'
        val_output = self.output_dir / 'val_smiles.txt'

        print(f"\nSaving splits...")
        with open(train_output, 'w') as f:
            f.write('\n'.join(train_smiles))
        print(f"  ✓ Train: {train_output}")

        with open(val_output, 'w') as f:
            f.write('\n'.join(val_smiles))
        print(f"  ✓ Val:   {val_output}")

        # Save metadata
        metadata = {
            'source': 'guacamol',
            'total_molecules': len(smiles_list),
            'train_molecules': len(train_smiles),
            'val_molecules': len(val_smiles),
            'train_file': str(train_output),
            'val_file': str(val_output)
        }

        metadata_path = self.output_dir / 'download_stats.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata: {metadata_path}")

        print("\n" + "="*80)
        print("✓ Download Complete!")
        print("="*80)
        print(f"Train: {train_output} ({len(train_smiles):,} molecules)")
        print(f"Val:   {val_output} ({len(val_smiles):,} molecules)")
        print("="*80)

        return train_output, val_output


def main():
    parser = argparse.ArgumentParser(description='Download SMILES datasets for MoleculeVAE')
    parser.add_argument('--source', type=str, default='guacamol',
                       choices=['guacamol', 'zinc250k'],
                       help='Dataset source (default: guacamol)')
    # parser.add_argument('--output_dir', type=str, default='data/raw',
    #                    help='Output directory (default: data/raw)')
    parser.add_argument('--num_molecules', type=int, default=1000000,
                       help='Limit to N molecules (default: 1M for quick start)')

    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.source == 'guacamol':
        downloader.download_guacamol(num_molecules=args.num_molecules)
    else:
        print(f"Dataset '{args.source}' not yet implemented.")
        print("Currently supported: guacamol")


if __name__ == '__main__':
    main()
