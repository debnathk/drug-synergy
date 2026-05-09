"""
Description: End-to-end synergy prediction models for drug synergy prediction.

SynergyModel    -- original model with attention-based fusion of mRNA, miRNA,
                   and proteomics alongside frozen drug embeddings.
SynergyModelL1000 -- L1000-augmented variant that uses mRNA + per-drug LINCS
                     L1000 perturbation profiles (drops miRNA & proteomics).
"""

import torch
import torch.nn as nn

from .mlp import SynergyMLP
from .kan import SynergyKAN


class OmicsEncoder(nn.Module):
    """Encoder for a single omics modality (mRNA, miRNA, or proteomics)."""
    def __init__(self, in_dim, out_dim=768): # Initially tried with 256
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),      # Normalize input features
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class OmicsAttentionFusion(nn.Module):
    """Fuses multiple omics modality embeddings using self-attention."""
    def __init__(self, emb_dim=768):
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


class OmicsConcatFusion(nn.Module):
    """Concatenates modality embeddings along the feature dimension.
    
    Output dimension = num_modalities * emb_dim (e.g., 3 * 768 = 2304).
    Simple baseline that preserves all information from each modality.
    """
    def __init__(self, emb_dim=768, num_modalities=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_modalities = num_modalities
    
    def forward(self, embeddings, return_attention=False):
        """
        Args:
            embeddings: Tensor of shape [batch, num_modalities, emb_dim]
            return_attention: Ignored (no attention weights for concat)
        Returns:
            Concatenated embedding of shape [batch, num_modalities * emb_dim]
        """
        fused = embeddings.view(embeddings.size(0), -1)
        if return_attention:
            return fused, None
        return fused


class OmicsProductFusion(nn.Module):
    """Element-wise product (Hadamard product) across modalities.
    
    Output dimension = emb_dim. Captures multiplicative interactions
    between modalities - a feature must be active in all modalities
    to contribute strongly to the output.
    """
    def __init__(self, emb_dim=768):
        super().__init__()
        self.emb_dim = emb_dim
    
    def forward(self, embeddings, return_attention=False):
        """
        Args:
            embeddings: Tensor of shape [batch, num_modalities, emb_dim]
            return_attention: Ignored (no attention weights for product)
        Returns:
            Element-wise product of shape [batch, emb_dim]
        """
        fused = embeddings.prod(dim=1)
        if return_attention:
            return fused, None
        return fused


class OmicsPoolingFusion(nn.Module):
    """Mean or max pooling across modalities.
    
    Output dimension = emb_dim. Simple aggregation that treats all
    modalities equally (mean) or selects the strongest signal (max).
    """
    def __init__(self, emb_dim=768, pool_type="mean"):
        super().__init__()
        self.emb_dim = emb_dim
        if pool_type not in ("mean", "max"):
            raise ValueError(f"pool_type must be 'mean' or 'max', got {pool_type!r}")
        self.pool_type = pool_type
    
    def forward(self, embeddings, return_attention=False):
        """
        Args:
            embeddings: Tensor of shape [batch, num_modalities, emb_dim]
            return_attention: Ignored (no attention weights for pooling)
        Returns:
            Pooled embedding of shape [batch, emb_dim]
        """
        if self.pool_type == "mean":
            fused = embeddings.mean(dim=1)
        else:
            fused = embeddings.max(dim=1)[0]
        
        if return_attention:
            return fused, None
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


VALID_FUSION_TYPES = ("attention", "concat", "product", "mean_pool", "max_pool")


class OmicsFusionModel(nn.Module):
    """
    Fuses mRNA, miRNA, and proteomics data into a single embedding.
    This is the trainable omics encoder.
    
    Args:
        mrna_dim: Input dimension for mRNA features
        mirna_dim: Input dimension for miRNA features
        prot_dim: Input dimension for proteomics features
        out_dim: Output embedding dimension (must match drug embedding dim)
        fusion_type: Fusion strategy to use. One of:
            - "attention": Self-attention with learnable CLS token (default)
            - "concat": Concatenate modalities (output dim = 3 * out_dim)
            - "product": Element-wise product across modalities
            - "mean_pool": Mean pooling across modalities
            - "max_pool": Max pooling across modalities
    """
    def __init__(self, mrna_dim, mirna_dim, prot_dim, out_dim=768, fusion_type="attention"):
        super().__init__()
        
        if fusion_type not in VALID_FUSION_TYPES:
            raise ValueError(
                f"fusion_type must be one of {VALID_FUSION_TYPES}, got {fusion_type!r}"
            )
        
        self.fusion_type = fusion_type
        self.out_dim = out_dim
        
        self.mrna_encoder = OmicsEncoder(mrna_dim, out_dim=out_dim)
        self.mirna_encoder = OmicsEncoder(mirna_dim, out_dim=out_dim)
        self.prot_encoder = OmicsEncoder(prot_dim, out_dim=out_dim)

        # Initialize fusion module and set output dimension
        num_modalities = 3
        if fusion_type == "attention":
            self.fusion = OmicsAttentionFusion(emb_dim=out_dim)
            self.fused_dim = out_dim
        elif fusion_type == "concat":
            self.fusion = OmicsConcatFusion(emb_dim=out_dim, num_modalities=num_modalities)
            self.fused_dim = out_dim * num_modalities
        elif fusion_type == "product":
            self.fusion = OmicsProductFusion(emb_dim=out_dim)
            self.fused_dim = out_dim
        elif fusion_type == "mean_pool":
            self.fusion = OmicsPoolingFusion(emb_dim=out_dim, pool_type="mean")
            self.fused_dim = out_dim
        elif fusion_type == "max_pool":
            self.fusion = OmicsPoolingFusion(emb_dim=out_dim, pool_type="max")
            self.fused_dim = out_dim

    def forward(self, mrna, mirna, prot, return_attention=False):
        """
        Args:
            mrna: Tensor of shape [batch, mrna_dim]
            mirna: Tensor of shape [batch, mirna_dim]
            prot: Tensor of shape [batch, prot_dim]
            return_attention: If True, also return omics fusion attention weights
                (only meaningful for fusion_type="attention")
        Returns:
            Fused omics embedding of shape [batch, fused_dim]
            If return_attention: also returns attention weights or None
                For attention fusion: [batch, num_heads, 4, 4]
                    Attention matrix indices: 0=CLS, 1=mRNA, 2=miRNA, 3=Proteomics
                For other fusion types: None
        """
        e1 = self.mrna_encoder(mrna)   # [batch, out_dim]
        e2 = self.mirna_encoder(mirna)  # [batch, out_dim]
        e3 = self.prot_encoder(prot)    # [batch, out_dim]

        stacked = torch.stack([e1, e2, e3], dim=1)  # [batch, 3, out_dim]
        
        if return_attention:
            fused, omics_attn = self.fusion(stacked, return_attention=True)
            return fused, omics_attn
        
        return self.fusion(stacked)




class SynergyModel(nn.Module):
    """
    End-to-end model for drug synergy prediction.
    Combines frozen drug embeddings with trainable omics encoder,
    fuses them via concatenation, and predicts synergy score.
    
    Supports both MLP and KAN prediction heads using SynergyMLP and SynergyKAN.
    Supports multiple omics fusion strategies via fusion_type parameter.
    """
    def __init__(self, mrna_dim, mirna_dim, prot_dim, embed_dim=768, head_type="mlp", 
                 grid_size=5, fusion_type="attention"):
        """
        Args:
            mrna_dim: Dimension of mRNA features
            mirna_dim: Dimension of miRNA features
            prot_dim: Dimension of proteomics features
            embed_dim: Embedding dimension (default 768)
            head_type: Type of prediction head - "mlp" or "kan" (default "mlp")
            grid_size: Grid size for KAN layers (only used if head_type="kan")
            fusion_type: Omics fusion strategy. One of:
                - "attention": Self-attention with learnable CLS token (default)
                - "concat": Concatenate modalities
                - "product": Element-wise product across modalities
                - "mean_pool": Mean pooling across modalities
                - "max_pool": Max pooling across modalities
        """
        super().__init__()
        
        self.head_type = head_type
        self.fusion_type = fusion_type

        # Trainable omics encoder: raw omics -> fused_dim embedding
        self.omics_encoder = OmicsFusionModel(
            mrna_dim, mirna_dim, prot_dim, 
            out_dim=embed_dim, 
            fusion_type=fusion_type
        )

        # Prediction head input: [drug1 | drug2 | fused_omics]
        # Drug embeddings are embed_dim each, fused_omics depends on fusion strategy
        input_dim = embed_dim * 2 + self.omics_encoder.fused_dim
        
        if head_type == "mlp":
            self.head = SynergyMLP(input_dim=input_dim, output_dim=1)
        elif head_type == "kan":
            self.head = SynergyKAN(input_dim=input_dim, output_dim=1, grid_size=grid_size)
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'mlp' or 'kan'")

    def forward(self, drug1_emb, drug2_emb, mrna, mirna, prot, return_attention=False):
        """
        Args:
            drug1_emb: Tensor of shape [batch, embed_dim] - frozen drug embedding
            drug2_emb: Tensor of shape [batch, embed_dim] - frozen drug embedding
            mrna: Tensor of shape [batch, mrna_dim] - raw mRNA features
            mirna: Tensor of shape [batch, mirna_dim] - raw miRNA features
            prot: Tensor of shape [batch, prot_dim] - raw proteomics features
            return_attention: If True, also return attention weights 
                (only meaningful for fusion_type="attention")
        Returns:
            Synergy prediction of shape [batch, 1]
            If return_attention: also returns dict with attention weights:
                - 'omics_fusion': [batch, num_heads, 4, 4] for attention fusion
                    (indices: 0=CLS, 1=mRNA, 2=miRNA, 3=Proteomics)
                - 'omics_fusion': None for other fusion types
        """
        # Encode raw omics to embedding
        if return_attention:
            omics_emb, omics_attn = self.omics_encoder(mrna, mirna, prot, return_attention=True)
        else:
            omics_emb = self.omics_encoder(mrna, mirna, prot)

        # Concatenate drug and omics embeddings
        fused = torch.cat([drug1_emb, drug2_emb, omics_emb], dim=1)

        # Predict synergy score using the selected head (MLP or KAN)
        output = self.head(fused)
        
        if return_attention:
            attention_weights = {
                'omics_fusion': omics_attn
            }
            return output, attention_weights
        
        return output


class SynergyModelL1000(nn.Module):
    """
    L1000-augmented synergy prediction model.

    Replaces the multi-omics attention fusion with a simpler architecture:
      - mRNA features encoded via a trainable OmicsEncoder
      - Per-drug L1000 perturbation profiles encoded via a shared OmicsEncoder
      - Frozen drug LM embeddings (MolFormer / ChemBERTa)

    Forward concatenation:
        [drug1_emb, drug2_emb, mRNA_enc, L1000_d1_enc, L1000_d2_enc]
        = 5 * embed_dim  ->  MLP or KAN head  ->  synergy score
    """

    def __init__(self, mrna_dim, l1000_dim, embed_dim=768,
                 head_type="mlp", grid_size=5):
        super().__init__()
        self.head_type = head_type

        self.mrna_encoder = OmicsEncoder(mrna_dim, out_dim=embed_dim)
        self.l1000_encoder = OmicsEncoder(l1000_dim, out_dim=embed_dim)

        input_dim = embed_dim * 5

        if head_type == "mlp":
            self.head = SynergyMLP(input_dim=input_dim, output_dim=1)
        elif head_type == "kan":
            self.head = SynergyKAN(input_dim=input_dim, output_dim=1,
                                   grid_size=grid_size)
        else:
            raise ValueError(
                f"Unknown head_type: {head_type}. Must be 'mlp' or 'kan'")

    def forward(self, drug1_emb, drug2_emb, mrna, l1000_d1, l1000_d2):
        """
        Args:
            drug1_emb: [batch, embed_dim] frozen drug embedding
            drug2_emb: [batch, embed_dim] frozen drug embedding
            mrna:      [batch, mrna_dim]  raw mRNA expression
            l1000_d1:  [batch, l1000_dim] L1000 profile for drug 1
            l1000_d2:  [batch, l1000_dim] L1000 profile for drug 2
        Returns:
            Synergy prediction of shape [batch, 1]
        """
        mrna_enc = self.mrna_encoder(mrna)          # [batch, embed_dim]
        l1000_d1_enc = self.l1000_encoder(l1000_d1) # [batch, embed_dim]
        l1000_d2_enc = self.l1000_encoder(l1000_d2) # [batch, embed_dim]

        fused = torch.cat(
            [drug1_emb, drug2_emb, mrna_enc, l1000_d1_enc, l1000_d2_enc],
            dim=1,
        )  # [batch, 5*embed_dim]

        return self.head(fused)
