"""
Description: Split synergy dataset into train, val and test and save splits as pkl (CSV is not recommended because of the heterogenous schema of the data)
Outcome: Train-val-test data as .pkl
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tdc.multi_pred import DrugSyn
from pathlib import Path

from .utils import standardize

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"

RAW_DATA_PATH = DATA_PATH / "raw"
# Legacy flat paths (for backward compat when split_type subdirs not used)
TRAIN_DATA_PATH = RAW_DATA_PATH / "train_split.pkl"
VAL_DATA_PATH = RAW_DATA_PATH / "val_split.pkl"
TEST_DATA_PATH = RAW_DATA_PATH / "test_split.pkl"

SPLIT_TYPES = ("random", "cold_drug")

# Dataset class
class DrugSynergyDataset(Dataset):
    def __init__(self, df, drug_emb_path, omics_emb_path, target_col):
        self.df = df.reset_index(drop=True)
        self.drug_emb = torch.load(drug_emb_path)
        self.omics_emb = torch.load(omics_emb_path)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        d1 = self.drug_emb["embeddings"][row['Drug1']]
        d2 = self.drug_emb["embeddings"][row['Drug2']]

        sample_id = f"{row['Drug1']}__{row['Drug2']}__{row['Cell_Line_ID']}"
        omics = self.omics_emb[sample_id]

        x = torch.cat([d1, d2, omics], dim=-1)   # [2304]
        y = torch.tensor([row[self.target_col]], dtype=torch.float32) # [1]

        return x, y


class DrugSynergyRawOmicsDataset(Dataset):
    """
    Dataset for end-to-end training with raw omics features.
    Returns drug embeddings and raw omics separately for the SynergyModel.
    """
    def __init__(self, df, drug_emb_path, target_col):
        self.df = df.reset_index(drop=True)
        self.drug_emb = torch.load(drug_emb_path)
        self.target_col = target_col

        # Get omics dimensions from first sample
        first_cell_line = self.df.iloc[0]['CellLine']
        self.mrna_dim = len(first_cell_line[0])
        self.mirna_dim = len(first_cell_line[2])
        self.prot_dim = len(first_cell_line[1])

    def __len__(self):
        return len(self.df)

    def get_omics_dims(self):
        """Returns dimensions of omics features: (mrna_dim, mirna_dim, prot_dim)"""
        return self.mrna_dim, self.mirna_dim, self.prot_dim

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Drug embeddings (frozen, pre-computed)
        d1 = self.drug_emb["embeddings"][row['Drug1']]
        d2 = self.drug_emb["embeddings"][row['Drug2']]

        # Raw omics features
        cell_line = row['CellLine']
        mrna = torch.tensor(cell_line[0], dtype=torch.float32)
        prot = torch.tensor(cell_line[1], dtype=torch.float32)
        mirna = torch.tensor(cell_line[2], dtype=torch.float32)

        # Target
        y = torch.tensor([row[self.target_col]], dtype=torch.float32)

        return d1, d2, mrna, mirna, prot, y


class DrugSynergyL1000Dataset(Dataset):
    """
    Dataset for training with mRNA + per-drug L1000 perturbation profiles.

    Returns (drug1_emb, drug2_emb, mrna, l1000_d1, l1000_d2, y).
    """

    def __init__(self, df, drug_emb_path, l1000_path, target_col):
        self.df = df.reset_index(drop=True)
        self.drug_emb = torch.load(drug_emb_path)
        self.target_col = target_col

        # mRNA dimension from first sample
        first_cell_line = self.df.iloc[0]['CellLine']
        self.mrna_dim = len(first_cell_line[0])

        # Load L1000 representative profiles and build SMILES -> tensor lookup
        l1000_df = pd.read_csv(l1000_path)
        gene_cols = [c for c in l1000_df.columns if c not in ("Drug_ID", "Drug")]
        self.l1000_dim = len(gene_cols)
        gene_values = l1000_df[gene_cols].values.astype(np.float32)

        self._l1000_lookup: dict[str, torch.Tensor] = {}
        for i, smiles in enumerate(l1000_df["Drug"]):
            self._l1000_lookup[smiles] = torch.from_numpy(gene_values[i])

        self._l1000_zero = torch.zeros(self.l1000_dim, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def get_dims(self):
        """Returns (mrna_dim, l1000_dim)."""
        return self.mrna_dim, self.l1000_dim

    def _get_l1000(self, smiles: str) -> torch.Tensor:
        return self._l1000_lookup.get(smiles, self._l1000_zero)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        d1 = self.drug_emb["embeddings"][row['Drug1']]
        d2 = self.drug_emb["embeddings"][row['Drug2']]

        mrna = torch.tensor(row['CellLine'][0], dtype=torch.float32)

        l1000_d1 = self._get_l1000(row['Drug1'])
        l1000_d2 = self._get_l1000(row['Drug2'])

        y = torch.tensor([row[self.target_col]], dtype=torch.float32)

        return d1, d2, mrna, l1000_d1, l1000_d2, y


def split_train_val_test(split_type="random"):
    """
    Generate train/val/test splits for DrugComb and save as pkl.

    Args:
        split_type: One of 'random', 'cold_drug'. Determines TDC split method and output subdir.
    """
    if split_type not in SPLIT_TYPES:
        raise ValueError(f"split_type must be one of {SPLIT_TYPES}, got {split_type!r}")

    data = DrugSyn(name="DrugComb")
    if split_type == "random":
        split = data.get_split(method="random")
    else:
        # cold_drug: hold out Drug1 entities so test has unseen drugs
        split = data.get_split(method="cold_split", column_name="Drug1")

    train_split, val_split, test_split = split["train"], split["valid"], split["test"]

    out_dir = RAW_DATA_PATH / split_type
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_split.pkl"
    val_path = out_dir / "val_split.pkl"
    test_path = out_dir / "test_split.pkl"

    if not train_path.exists():
        train_split.to_pickle(train_path)
        print(f"Saved train split ({split_type}) to {train_path}")
    else:
        print(f"Train data already exists at {train_path}")

    if not val_path.exists():
        val_split.to_pickle(val_path)
        print(f"Saved val split ({split_type}) to {val_path}")
    else:
        print(f"Validation data already exists at {val_path}")

    if not test_path.exists():
        test_split.to_pickle(test_path)
        print(f"Saved test split ({split_type}) to {test_path}")
    else:
        print(f"Test data already exists at {test_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate DrugComb train/val/test splits")
    p.add_argument(
        "--split_type",
        type=str,
        default="all",
        choices=["random", "cold_drug", "all"],
        help="Which split type to generate (or 'all' for both)",
    )
    args = p.parse_args()
    if args.split_type == "all":
        for st in SPLIT_TYPES:
            split_train_val_test(st)
    else:
        split_train_val_test(args.split_type)