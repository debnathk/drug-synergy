import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data/raw"
EMBEDDINGS_PATH = ROOT / "data/embeddings"

# Dataset
class DrugSynergyDataset(Dataset):
    def __init__(self, df, drug_emb_path, omics_emb_path, target_cols):
        self.df = df.reset_index(drop=True)
        self.drug_emb = torch.load(drug_emb_path)
        self.omics_emb = torch.load(omics_emb_path)
        self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        d1 = self.drug_emb["embeddings"][row['Drug1']]
        d2 = self.drug_emb["embeddings"][row['Drug2']]

        sample_id = f"{row['Drug1']}__{row['Drug2']}__{row['Cell_Line_ID']}"
        omics = self.omics_emb[sample_id]

        x = torch.cat([d1, d2, omics], dim=-1)   # [2304]
        y = torch.tensor(row[self.target_cols].values.astype(float), dtype=torch.float32)

        return x, y

# Load your dataset
with open(RAW_DATA_PATH / "train_split.pkl", 'rb') as file:
    train_df = pickle.load(file)

with open(RAW_DATA_PATH / "val_split.pkl", 'rb') as file:
    val_df = pickle.load(file)

# Define the target columns
target_cols = ['Synergy_ZIP', 'Synergy_Bliss', 'Synergy_Loewe', 'Synergy_HSA']

# Paths to embeddings (replace with actual paths)
drug_emb_path = EMBEDDINGS_PATH / 'drug_embeddings.pt'
train_omics_emb_path = EMBEDDINGS_PATH / 'train_omics.pt'
val_omics_emb_path = EMBEDDINGS_PATH / 'val_omics.pt'

# Create datasets
train_dataset = DrugSynergyDataset(train_df, drug_emb_path, train_omics_emb_path, target_cols)
val_dataset = DrugSynergyDataset(val_df, drug_emb_path, val_omics_emb_path, target_cols)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False) 

# MLP
class SynergyMLP(nn.Module):
    def __init__(self, input_dim=2304, output_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SynergyMLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()   # multi-target regression

def train_epoch(loader):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

# Run training
for epoch in range(20):
    train_loss = train_epoch(train_loader)
    val_loss = eval_epoch(val_loader)

    print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
    
def sanity_check():
    x, y = train_dataset[0]
    print(x, x.shape)   # torch.Size([2304])
    print(y, y.shape)   # torch.Size([4])
    
# if __name__ == "__main__":
#     sanity_check()