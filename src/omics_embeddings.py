"""
Description: Create encoder for omics, save unique omics embeddings as .pt
Outcome: Embeddings of unique omics as .pt
Author: Kusal Debnath
"""

import torch
import torch.nn as nn
from tdc.multi_pred import DrugSyn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Define paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
LOG_PATH = ROOT / "logs"

TRAIN_OMICS_PATH = DATA_PATH / "embeddings/train_omics.pt"
VAL_OMICS_PATH = DATA_PATH / "embeddings/val_omics.pt"
TEST_OMICS_PATH = DATA_PATH / "embeddings/test_omics.pt"

def setup_logging():
    """
    Creates a timestamped log directory and configures logging.
    Returns the run directory path.
    """
    # Create timestamp-based run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_PATH / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = run_dir / "omics_embedding.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    
    return run_dir, logger

data = DrugSyn(name="DrugComb")
split = data.get_split()
# print(type(split))

train_split, val_split, test_split = split['train'], split['valid'], split['test']

class OmicsEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        ) 

    def forward(self, x):
        return self.net(x)
    
class OmicsAttentionFusion(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, embeddings):
        x = embeddings.unsqueeze(0) # [1, M, 256]
        attn_out, _ = self.attn(x, x, x)
        fused = attn_out.mean(dim=1) # [1, 256]
        return fused.squeeze(0)
    
class ProjectionAttention(nn.Module): # Preserves expressivness, unlike a plain linear layer
    def __init__(self, in_dim=256, out_dim=768):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, out_dim))
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, x):
        """
        x: Tensor [256]
        """
        k = self.key(x).unsqueeze(0).unsqueeze(0)    # [1,1,768]
        v = self.value(x).unsqueeze(0).unsqueeze(0)
        q = self.query.unsqueeze(0)                  # [1,1,768]

        out, _ = self.attn(q, k, v)
        return out.squeeze(0).squeeze(0)             # [768]
    
class OmicsFusionModel(nn.Module):
    def __init__(self, mrna_dim, mirna_dim, prot_dim):
        super().__init__()

        self.mrna_encoder = OmicsEncoder(mrna_dim)
        self.mirna_encoder = OmicsEncoder(mirna_dim)
        self.prot_encoder = OmicsEncoder(prot_dim)

        self.fusion = OmicsAttentionFusion(emb_dim=256)
        self.project = ProjectionAttention(256, 768)

    def forward(self, mrna, mirna, prot):
        e1 = self.mrna_encoder(mrna)
        e2 = self.mirna_encoder(mirna)
        e3 = self.prot_encoder(prot)

        stacked = torch.stack([e1, e2, e3], dim=0)
        fused_256 = self.fusion(stacked)
        fused_768 = self.project(fused_256)

        return fused_768
    
def get_splitwise_omics_embeddings(
    split: pd.DataFrame,
    fusion_model: torch.nn.Module,
    save_path: Path,
    device: torch.device
):
    fusion_model.eval().to(device)
    omics_embeddings = {}

    with torch.no_grad():
        for _, row in tqdm(split.iterrows(), total=len(split), desc=f"Saving → {save_path.name}"):

            sample_id = f"{row['Drug1']}__{row['Drug2']}__{row['Cell_Line_ID']}"

            mrna = torch.tensor(row["CellLine"][0], dtype=torch.float32, device=device)
            mirna = torch.tensor(row["CellLine"][2], dtype=torch.float32, device=device)
            prot = torch.tensor(row["CellLine"][1], dtype=torch.float32, device=device)

            emb = fusion_model(mrna, mirna, prot)   # [768]
            omics_embeddings[sample_id] = emb.cpu()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(omics_embeddings, save_path)

    return omics_embeddings

    
def main():
    '''
    Saves embeddings of unique omics as .pt
    '''
    mrna_dim = len(train_split['CellLine'][0][0])
    proteomics_dim = len(train_split['CellLine'][0][1])
    mirna_dim = len(train_split['CellLine'][0][2])

    fusion_model = OmicsFusionModel(
        mrna_dim=mrna_dim,
        prot_dim=proteomics_dim,
        mirna_dim=mirna_dim
    )

    if not TRAIN_OMICS_PATH.exists():
        train_omics = get_splitwise_omics_embeddings(
        train_split,
        fusion_model,
        DATA_PATH / "embeddings/train_omics.pt",
        torch.device('cuda'))
    else: print(f"Omics embeddings for train split already exists in {TRAIN_OMICS_PATH}")

    if not VAL_OMICS_PATH.exists():
        val_omics = get_splitwise_omics_embeddings(
        val_split,
        fusion_model,
        DATA_PATH / "embeddings/val_omics.pt",
        torch.device('cuda'))
    else: print(f"Omics embeddings for validation split already exists in {VAL_OMICS_PATH}")

    if not TEST_OMICS_PATH.exists():
        test_omics = get_splitwise_omics_embeddings(
        test_split,
        fusion_model,
        DATA_PATH / "embeddings/test_omics.pt",
        torch.device('cuda'))
    else: print(f"Omics embeddings for test split already exists in {TEST_OMICS_PATH}")
    
def sanity_check():
    # import math
    import torch

    val_omics = torch.load(DATA_PATH / "embeddings/val_omics.pt")

    print("Type:", type(val_omics))
    print("Num samples:", len(val_omics))

    sample_keys = list(val_omics.keys())[:5]
    for k in sample_keys:
        print(k)

    emb = next(iter(val_omics.values()))

    print("Embedding shape:", emb.shape)
    print("Dtype:", emb.dtype)
    print("Device:", emb.device)


    bad = [
        k for k, v in val_omics.items()
        if torch.isnan(v).any() or torch.isinf(v).any()
    ]

    print("Bad embeddings:", len(bad))

    stack = torch.stack(list(val_omics.values()))

    print("Mean:", stack.mean().item())
    print("Std:", stack.std().item())
    print("Min:", stack.min().item())
    print("Max:", stack.max().item())

    key = sample_keys[0]
    e1 = val_omics[key]
    e2 = val_omics[key]

    print("Deterministic:", torch.allclose(e1, e2))

    keys = list(val_omics.keys())[:3]

    for i in range(len(keys) - 1):
        diff = torch.norm(val_omics[keys[i]] - val_omics[keys[i+1]])
        print(f"Δ({i}):", diff.item())

    df_sample = val_split.iloc[0]
    sample_id = f"{df_sample['Drug1']}__{df_sample['Drug2']}__{df_sample['Cell_Line_ID']}"

    print("Exists:", sample_id in val_omics)

if __name__ == "__main__":
    main()
    # sanity_check()
        