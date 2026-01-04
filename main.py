import torch
import pandas as pd
from pathlib import Path
from tdc.multi_pred import DrugSyn
from omics_embeddings import OmicsEncoder

data = DrugSyn(name="DrugComb")
split = data.get_split()

train_split, val_split, test_split = split['train'], split['valid'], split['test']

# print(len(train_split['CellLine'][0][0]), len(train_split['CellLine'][0][1]), len(train_split['CellLine'][0][2]))
# print(len(val_split['CellLine'][0][0]), len(val_split['CellLine'][0][1]), len(val_split['CellLine'][0][2]))
# print(len(test_split['CellLine'][0][0]), len(test_split['CellLine'][0][1]), len(test_split['CellLine'][0][2]))

mrna_dim = len(train_split['CellLine'][0][0])
proteomics_dim = len(train_split['CellLine'][0][1])
mirna_dim = len(train_split['CellLine'][0][2])

mrna_encoder = OmicsEncoder(mrna_dim)
proteomics_encoder = OmicsEncoder(proteomics_dim)
mirna_encoder = OmicsEncoder(mirna_dim)

z_mrna = mrna_encoder(torch.tensor(train_split['CellLine'][0][0], dtype=torch.float32))
z_proteomics = proteomics_encoder(torch.tensor(train_split['CellLine'][0][1], dtype=torch.float32))
z_mirna = mirna_encoder(torch.tensor(train_split['CellLine'][0][2], dtype=torch.float32))

print(z_mrna.shape, z_proteomics.shape, z_mirna.shape)

# Attention-based omics fusion
Z = torch.stack([z_mrna, z_proteomics, z_mirna], dim=1) # (B, 3, 256)
attn_layer = torch.nn.Linear(Z.size(-1), 1)  # Define an attention layer
cell_proj_layer = torch.nn.Sequential(
    torch.nn.Linear(256, 768),
    torch.nn.ReLU()
)
weights = torch.softmax(attn_layer(Z), dim=1) # Importance if each modality embedding
cell_emb = (weights * Z).sum(dim=1)
cell_emb_proj = cell_proj_layer(cell_emb) # projects dim 256 -> 768 (to match drug embedding dim)
print(cell_emb.shape)
print(cell_emb_proj.shape)

# create dataset object, dataloader, MLP, train