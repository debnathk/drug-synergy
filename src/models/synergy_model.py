"""
Description: End-to-end synergy prediction model with trainable omics encoder
             and attention-based fusion of drug and omics embeddings.
Author: Kusal Debnath
"""

import torch
import torch.nn as nn


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

    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor of shape [batch, num_modalities, emb_dim]
        Returns:
            Fused embedding of shape [batch, emb_dim]
        """
        attn_out, _ = self.attn(embeddings, embeddings, embeddings)
        fused = attn_out.mean(dim=1)  # [batch, emb_dim]
        return fused


class ProjectionAttention(nn.Module):
    """Projects embedding to higher dimension using attention mechanism."""
    def __init__(self, in_dim=256, out_dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, out_dim))
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, in_dim]
        Returns:
            Projected embedding of shape [batch, out_dim]
        """
        batch_size = x.size(0)
        k = self.key(x).unsqueeze(1)    # [batch, 1, out_dim]
        v = self.value(x).unsqueeze(1)  # [batch, 1, out_dim]
        q = self.query.expand(batch_size, -1, -1)  # [batch, 1, out_dim]

        out, _ = self.attn(q, k, v)
        return out.squeeze(1)  # [batch, out_dim]


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
        self.project = ProjectionAttention(256, out_dim)

    def forward(self, mrna, mirna, prot):
        """
        Args:
            mrna: Tensor of shape [batch, mrna_dim]
            mirna: Tensor of shape [batch, mirna_dim]
            prot: Tensor of shape [batch, prot_dim]
        Returns:
            Fused omics embedding of shape [batch, 768]
        """
        e1 = self.mrna_encoder(mrna)   # [batch, 256]
        e2 = self.mirna_encoder(mirna)  # [batch, 256]
        e3 = self.prot_encoder(prot)    # [batch, 256]

        stacked = torch.stack([e1, e2, e3], dim=1)  # [batch, 3, 256]
        fused_256 = self.fusion(stacked)  # [batch, 256]
        fused_768 = self.project(fused_256)  # [batch, 768]

        return fused_768


class DrugOmicsCrossAttention(nn.Module):
    """
    Cross-attention module to fuse drug embeddings with omics embedding.
    Learns how drugs and cell line omics interact.
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug1_emb, drug2_emb, omics_emb):
        """
        Args:
            drug1_emb: Tensor of shape [batch, 768]
            drug2_emb: Tensor of shape [batch, 768]
            omics_emb: Tensor of shape [batch, 768]
        Returns:
            Fused embedding of shape [batch, 768]
        """
        # Stack as sequence: [drug1, drug2, omics] -> [batch, 3, 768]
        seq = torch.stack([drug1_emb, drug2_emb, omics_emb], dim=1)

        # Self-attention to learn interactions
        attn_out, _ = self.self_attn(seq, seq, seq)
        attn_out = self.dropout(attn_out)
        seq = self.norm(seq + attn_out)

        # Mean pooling to get single representation
        fused = seq.mean(dim=1)  # [batch, 768]
        return fused


class SynergyModel(nn.Module):
    """
    End-to-end model for drug synergy prediction.
    Combines frozen drug embeddings with trainable omics encoder,
    fuses them via cross-attention, and predicts synergy score.
    """
    def __init__(self, mrna_dim, mirna_dim, prot_dim, embed_dim=768):
        super().__init__()

        # Trainable omics encoder: raw omics -> 768-dim embedding
        self.omics_encoder = OmicsFusionModel(mrna_dim, mirna_dim, prot_dim, out_dim=embed_dim)

        # Cross-attention to fuse [drug1, drug2, omics]
        self.cross_attn = DrugOmicsCrossAttention(embed_dim=embed_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, drug1_emb, drug2_emb, mrna, mirna, prot):
        """
        Args:
            drug1_emb: Tensor of shape [batch, 768] - frozen drug embedding
            drug2_emb: Tensor of shape [batch, 768] - frozen drug embedding
            mrna: Tensor of shape [batch, mrna_dim] - raw mRNA features
            mirna: Tensor of shape [batch, mirna_dim] - raw miRNA features
            prot: Tensor of shape [batch, prot_dim] - raw proteomics features
        Returns:
            Synergy prediction of shape [batch, 1]
        """
        # Encode raw omics to embedding
        omics_emb = self.omics_encoder(mrna, mirna, prot)  # [batch, 768]

        # Fuse drug and omics embeddings via cross-attention
        fused = self.cross_attn(drug1_emb, drug2_emb, omics_emb)  # [batch, 768]

        # Predict synergy score
        return self.head(fused)
