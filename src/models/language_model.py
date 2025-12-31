"""
Description: Language model class to generate drug embeddings from molecule SMILES

Author: Kusal Debnath
"""

import torch
from transformers import AutoModel, AutoTokenizer 
import numpy as np

class MolFormerMLM:
    """
    MolFormer Language Model class for getting embeddings for molecules smiles
    """
    def __init__(self, model_name="ibm-research/MoLFormer-XL-both-10pct"):
        self.model = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def get_molecule_embedding(self, smiles):
        "Get embeddings for a single molecule smiles"
        inputs = self.tokenizer(smiles, padding=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs)
        
        # Use the CLS token embeddings as the molecule represenatation
        # return output.pooler_output
        return output.last_hidden_state[:, 0, :].numpy()


    def get_batch_embeddings(self, smiles_list):
        "Get embeddings for a list of molecule smiles"
        embeddings = []
        
        for smiles in smiles_list:
            embedding = self.get_molecule_embedding(smiles)
            embeddings.append(embedding)

        return np.vstack(embeddings)
    
if __name__ == "__main__":
    known_smiles = {
    'Levodopa': 'C1=CC(=C(C=C1CC(C(=O)O)N)O)O',
    'Fluoxetine': 'CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F',
    'Donepezil': 'COC1=C(C=C2C(=C1)CC(=O)C(=C2)CC3CCN(CC3)CC4=CC=CC=C4)OC',
    'Memantine': 'CC12CC3CC(C1)(CC(C3)(C2)N)C',
    'Haloperidol': 'C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F',
    'Risperidone': 'CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=C(C=CC(=C4)F)C5=CC=CC=C5',
    'Carbamazepine': 'C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N',
    'Valproic Acid': 'CCCC(CCC)C(=O)O'
    }

    language_model = MolFormerMLM()

    sample_embedding = language_model.get_molecule_embedding(known_smiles['Carbamazepine'])
    print(f"Sample embedding shape: {sample_embedding.shape}")

    smiles_list = list(known_smiles.values())

    sample_embeddings = language_model.get_batch_embeddings(smiles_list)
    print(f"Batch sample embeddings shape: {sample_embeddings.shape}")