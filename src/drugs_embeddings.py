"""
Description: Get embedding for all the unique drugs in DrugComb dataset
Outcome: Embeddings of unique drugs as .pt
Autor: Kusal Debnath
"""

import torch
from tdc.multi_pred import DrugSyn
from tqdm import tqdm
import numpy as np
from pathlib import Path
from src.models.language_model import MolFormerMLM

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"

def main():

    # Load drug-synergy data
    data = DrugSyn(name="DrugComb")
    split = data.get_split()
    # print(type(split['train']))

    train_split, val_split, test_split = split['train'], split['valid'], split['test']
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
    embeddings = []
    language_model = MolFormerMLM()

    for batch in tqdm(batched(unique_drugs, batch_size), desc="Embedding", total=len(unique_drugs) // batch_size + 1):
        emb = language_model.get_batch_embeddings(batch)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"Drug embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_info = {
        "embeddings": embeddings,
        "smiles": unique_drugs,
        "model": "MolFormer-XL",
        "dim": embeddings.shape[1]
    }

    torch.save(embeddings_info, str(DATA_PATH / "embeddings/drug_embeddings.pt"))
    print(f'Embeddings generated and saved in {str(DATA_PATH / "embeddings/drug_embeddings.pt")}')

if __name__ == "__main__":
    if not Path(DATA_PATH / "embeddings/drug_embeddings.pt").exists():
        main()
    else:
        print(f'Embeddings already generated and can be found in {str(DATA_PATH / "embeddings/embeddings.pt")}')
