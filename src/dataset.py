"""
Description: Split synergy dataset into train, val and test and save splits as pkl (CSV is not recommended because of the heterogenous schema of the data)
Outcome: Train-val-test data as .pkl
Author: Kusal Debnath
"""

import torch
from torch.utils.data import Dataset
from tdc.multi_pred import DrugSyn
from pathlib import Path

from .utils import standardize

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"

RAW_DATA_PATH = DATA_PATH / "raw"
TRAIN_DATA_PATH = RAW_DATA_PATH / "train_split.pkl"
VAL_DATA_PATH = RAW_DATA_PATH / "val_split.pkl"
TEST_DATA_PATH = RAW_DATA_PATH / "test_split.pkl"

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


def split_train_val_test():

    # Load drug-synergy data
    data = DrugSyn(name="DrugComb")
    split = data.get_split()
    # print(type(split['train']))

    train_split, val_split, test_split = split['train'], split['valid'], split['test']
    # print(f"Train shape: {train_split.shape}")
    # print(f"Validation shape: {val_split.shape}")
    # print(f"Test shape: {test_split.shape}")

    if not TRAIN_DATA_PATH.exists():
        train_split.to_pickle(TRAIN_DATA_PATH)
    else:
        print(f'Train data already exists in {TRAIN_DATA_PATH}')

    if not VAL_DATA_PATH.exists():
        val_split.to_pickle(VAL_DATA_PATH)
    else:
        print(f'Validation data already exists in {VAL_DATA_PATH}')

    if not TEST_DATA_PATH.exists():
        test_split.to_pickle(TEST_DATA_PATH)
    else:
        print(f'Test data already exists in {TEST_DATA_PATH}')

if __name__ == "__main__":
    split_train_val_test()