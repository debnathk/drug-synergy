"""
Description: Get embedding for all the unique drugs in DrugComb dataset
Outcome: Embeddings of unique drugs as .pt (per split_type: drug_embeddings_<split_type>.pt)
Author: Kusal Debnath
"""

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle
import argparse

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
EMBEDDINGS_PATH = DATA_PATH / "embeddings"

# Use src.models for in-repo; fallback to models for legacy
try:
    from src.models.language_model import MolFormerMLM
except ImportError:
    from models.language_model import MolFormerMLM


def main(split_type="random"):
    # Load train/val/test from pre-generated splits for this split_type
    raw_dir = RAW_DATA_PATH / split_type
    train_path = raw_dir / "train_split.pkl"
    val_path = raw_dir / "val_split.pkl"
    test_path = raw_dir / "test_split.pkl"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}. Run: python -m src.dataset --split_type {split_type}"
            )
    with open(train_path, "rb") as f:
        train_split = pickle.load(f)
    with open(val_path, "rb") as f:
        val_split = pickle.load(f)
    with open(test_path, "rb") as f:
        test_split = pickle.load(f)
    # print(f"Train shape: {train_split.shape}")
    # print(f"Validation shape: {val_split.shape}")
    # print(f"Test shape: {test_split.shape}")

    # print(train_split.columns)
    # data_columns = ['Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'CSS', 'Synergy_ZIP',
    #     'Synergy_Bliss', 'Synergy_Loewe', 'Synergy_HSA', 'Drug1', 'Drug2',
    #     'CellLine']

    # print(f"Unique drugs in Drug1 and Drug2 columns - Train split: {len(set(train_split['Drug1'].tolist()))}, {len(set(train_split['Drug2'].tolist()))}")

    drug1_train = train_split['Drug1'].tolist() 
    drug1_val = val_split['Drug1'].tolist() 
    drug1_test = test_split['Drug1'].tolist() 

    drug2_train = train_split['Drug2'].tolist() 
    drug2_val = val_split['Drug2'].tolist() 
    drug2_test = test_split['Drug2'].tolist() 

    drug1_all = drug1_train + drug1_val + drug1_test
    drug2_all = drug2_train + drug2_val + drug2_test

    unique_drugs = list(set(drug1_all + drug2_all)) # Generate embeddings for these only

    def batched(iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i+batch_size]

    batch_size = 32
    drug_embeddings_dict = {}
    language_model = MolFormerMLM()

    for batch in tqdm(batched(unique_drugs, batch_size), desc="Embedding", total=len(unique_drugs) // batch_size + 1):
        emb = language_model.get_batch_embeddings(batch)

        # Convert to torch tensor
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)

        emb = emb.float().cpu()

        for smi, vec in zip(batch, emb):
            drug_embeddings_dict[smi] = vec # vec shape: (768,)

    print(f"Drug embeddings type and shape: {type(vec)} ,{vec.shape}")

    # Save embeddings
    embeddings_info = {
        "embeddings": drug_embeddings_dict,
        "model": "MolFormer-XL",
        "dim": next(iter(drug_embeddings_dict.values())).shape[0]
    }

    out_path = EMBEDDINGS_PATH / f"drug_embeddings_{split_type}.pt"
    EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings_info, str(out_path))
    print(f"Embeddings generated and saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drug embeddings for DrugComb splits")
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", "cold_drug"],
        help="Split type (must match pre-generated splits in data/raw/<split_type>/)",
    )
    args = parser.parse_args()
    main(split_type=args.split_type)
