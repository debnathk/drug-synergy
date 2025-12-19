import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

from src.utils import smiles2onehot

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
GUACAMOL_DATA_DIR = DATA_DIR / "guacamol"

class ChEMBLDataset(Dataset):
    def __init__(self, file_path, tokenizer_func=smiles2onehot):
        with open(file_path, 'r') as f:
             self.smiles_list = [line for line in f]
        self.tokenizer = tokenizer_func

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer(smiles)

        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.float32)

        return tokens_tensor, tokens_tensor # same input and target

    
# Load datasets
train_dataset = ChEMBLDataset(file_path=str(GUACAMOL_DATA_DIR / "train_smiles.txt"))
val_dataset = ChEMBLDataset(file_path=str(GUACAMOL_DATA_DIR / "val_smiles.txt"))

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

def main():
    # Sanity check of the loaders
    data_batch, target_batch = next(iter(train_loader))

    # Inspect the shapes
    print(f"Batch Data Shape:   {data_batch.shape}")
    print(f"Batch Target Shape: {target_batch.shape}")
    print(f"Data Type:          {data_batch.dtype}")

    # Check single sample
    sample_molecule = data_batch[0]
    print(f"Single Sample Shape: {sample_molecule.shape}")

if __name__ == "__main__":
    main()