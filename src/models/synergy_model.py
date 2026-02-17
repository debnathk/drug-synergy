"""
Description: End-to-end synergy prediction model with trainable omics encoder
             and attention-based fusion of drug and omics embeddings.
             Supports both MLP and KAN prediction heads.
Author: Kusal Debnath
"""

import torch
import torch.nn as nn

from .mlp import SynergyMLP
from .kan import SynergyKAN


class OmicsEncoder(nn.Module):
    """Encoder for a single omics modality (mRNA, miRNA, or proteomics)."""
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),      # Normalize input features
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class OmicsAttentionFusion(nn.Module):
    """Fuses multiple omics modality embeddings using self-attention."""
    def __init__(self, emb_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            batch_first=True
        )

        # Learnable CLS token: shape [1, 1, emb_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02) # ViT-style init

    def forward(self, embeddings, return_attention=False):
        """
        Args:
            embeddings: Tensor of shape [batch, num_modalities, emb_dim]
            return_attention: If True, also return attention weights
        Returns:
            Fused CLS embedding of shape [batch, emb_dim]
            If return_attention: also returns attention weights [batch, num_heads, num_modalities+1, num_modalities+1]
        """
        B = embeddings.size(0)

        # Expand CLS toekn across the batch: [B, 1, emb_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Prepend CLS token -> [B, num_modalities + 1, emb_dim]
        x = torch.cat([cls_tokens, embeddings], dim=1)


        attn_out, attn_weights = self.attn(x, x, x, average_attn_weights=False)

        # Extract only CLS token output -> [B, emb_dim]
        fused = attn_out[:, 0, :]
        
        if return_attention:
            return fused, attn_weights
        return fused


# class ProjectionAttention(nn.Module): # **Attention is overkill here
#     """Projects embedding to higher dimension using attention mechanism."""
#     def __init__(self, in_dim=256, out_dim=768):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(1, 1, out_dim))
#         self.key = nn.Linear(in_dim, out_dim)
#         self.value = nn.Linear(in_dim, out_dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=out_dim,
#             num_heads=8,
#             batch_first=True
#         )

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape [batch, in_dim]
#         Returns:
#             Projected embedding of shape [batch, out_dim]
#         """
#         batch_size = x.size(0)
#         k = self.key(x).unsqueeze(1)    # [batch, 1, out_dim]
#         v = self.value(x).unsqueeze(1)  # [batch, 1, out_dim]
#         q = self.query.expand(batch_size, -1, -1)  # [batch, 1, out_dim]

#         out, _ = self.attn(q, k, v)
#         return out.squeeze(1)  # [batch, out_dim]


class OmicsFusionModel(nn.Module):
    """
    Fuses mRNA, miRNA, and proteomics data into a single 768-dim embedding.
    This is the trainable omics encoder.
    """
    def __init__(self, mrna_dim, mirna_dim, prot_dim, out_dim=768):
        super().__init__()
        self.mrna_encoder = OmicsEncoder(mrna_dim)
        self.mirna_encoder = OmicsEncoder(mirna_dim)
        self.prot_encoder = OmicsEncoder(prot_dim)

        self.fusion = OmicsAttentionFusion(emb_dim=256)
        # self.project = ProjectionAttention(256, out_dim)
        self.project = nn.Linear(256, out_dim)

    def forward(self, mrna, mirna, prot, return_attention=False):
        """
        Args:
            mrna: Tensor of shape [batch, mrna_dim]
            mirna: Tensor of shape [batch, mirna_dim]
            prot: Tensor of shape [batch, prot_dim]
            return_attention: If True, also return omics fusion attention weights
        Returns:
            Fused omics embedding of shape [batch, 768]
            If return_attention: also returns attention weights [batch, num_heads, 3, 3]
                Attention matrix indices: 0=mRNA, 1=miRNA, 2=Proteomics
        """
        e1 = self.mrna_encoder(mrna)   # [batch, 256]
        e2 = self.mirna_encoder(mirna)  # [batch, 256]
        e3 = self.prot_encoder(prot)    # [batch, 256]

        stacked = torch.stack([e1, e2, e3], dim=1)  # [batch, 3, 256]
        
        if return_attention:
            fused_256, omics_attn = self.fusion(stacked, return_attention=True)
        else:
            fused_256 = self.fusion(stacked)
            
        fused_768 = self.project(fused_256)  # [batch, 768]

        if return_attention:
            return fused_768, omics_attn
        return fused_768




class SynergyModel(nn.Module):
    """
    End-to-end model for drug synergy prediction.
    Combines frozen drug embeddings with trainable omics encoder,
    fuses them via concatenation, and predicts synergy score.
    
    Supports both MLP and KAN prediction heads using SynergyMLP and SynergyKAN.
    """
    def __init__(self, mrna_dim, mirna_dim, prot_dim, embed_dim=768, head_type="mlp", grid_size=5):
        """
        Args:
            mrna_dim: Dimension of mRNA features
            mirna_dim: Dimension of miRNA features
            prot_dim: Dimension of proteomics features
            embed_dim: Embedding dimension (default 768)
            head_type: Type of prediction head - "mlp" or "kan" (default "mlp")
            grid_size: Grid size for KAN layers (only used if head_type="kan")
        """
        super().__init__()
        
        self.head_type = head_type

        # Trainable omics encoder: raw omics -> 768-dim embedding
        self.omics_encoder = OmicsFusionModel(mrna_dim, mirna_dim, prot_dim, out_dim=embed_dim)

        # Prediction head - takes concatenated [drug1, drug2, omics] = 3 * embed_dim
        input_dim = embed_dim * 3  # 2304
        
        if head_type == "mlp":
            # Use ready-made SynergyMLP: 2304 -> 1024 -> 256 -> 1
            self.head = SynergyMLP(input_dim=input_dim, output_dim=1)
        elif head_type == "kan":
            # Use ready-made SynergyKAN: 2304 -> 128 -> 32 -> 1
            self.head = SynergyKAN(input_dim=input_dim, output_dim=1, grid_size=grid_size)
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'mlp' or 'kan'")

    def forward(self, drug1_emb, drug2_emb, mrna, mirna, prot, return_attention=False):
        """
        Args:
            drug1_emb: Tensor of shape [batch, 768] - frozen drug embedding
            drug2_emb: Tensor of shape [batch, 768] - frozen drug embedding
            mrna: Tensor of shape [batch, mrna_dim] - raw mRNA features
            mirna: Tensor of shape [batch, mirna_dim] - raw miRNA features
            prot: Tensor of shape [batch, prot_dim] - raw proteomics features
            return_attention: If True, also return attention weights (only omics fusion)
        Returns:
            Synergy prediction of shape [batch, 1]
            If return_attention: also returns dict with attention weights:
                - 'omics_fusion': [batch, 4, 3, 3] - Omics modality attention
                    (indices: 0=mRNA, 1=miRNA, 2=Proteomics)
        """
        # Encode raw omics to embedding
        if return_attention:
            omics_emb, omics_attn = self.omics_encoder(mrna, mirna, prot, return_attention=True)
        else:
            omics_emb = self.omics_encoder(mrna, mirna, prot)

        # Concatenate drug and omics embeddings
        fused = torch.cat([drug1_emb, drug2_emb, omics_emb], dim=1)  # [batch, 2304]

        # Predict synergy score using the selected head (MLP or KAN)
        output = self.head(fused)
        
        if return_attention:
            attention_weights = {
                'omics_fusion': omics_attn  # [batch, 4, 3, 3] - mRNA, miRNA, Proteomics
            }
            return output, attention_weights
        
        return output
