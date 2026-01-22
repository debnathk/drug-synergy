# Drug Synergy Prediction - Implementation Plan

**Project**: Attention-Based Multi-Modal Fusion for Drug Synergy Prediction  
**Start Date**: January 2026  
**Duration**: 6 Weeks

---

## Phase 1: Evaluation & Baseline (Week 1)

| Task | Status | Notes |
|------|--------|-------|
| Implement end-to-end SynergyModel with attention fusion | Done | `src/models/synergy_model.py` |
| Train on DrugComb dataset | Done | Val MSE: 13.40 |
| Evaluate on test set with comprehensive metrics | Done | `evaluate.py` - MAE, R², Pearson, Spearman |
| Compare against simple MLP baseline (concatenation) | Done | MLP best val MSE: 21.91 (severe overfitting) |
| Analyze attention weights for interpretability | Done | `analyze_attention.py` - extracts from cross-attention |

**Training Results Summary**:

| Model | Best Val MSE | Final Train MSE | Notes |
|-------|-------------|-----------------|-------|
| SynergyModel (Attention) | 13.40 | 11.66 | Good generalization |
| SynergyMLP (Baseline) | 21.91 | 13.07 | Severe overfitting |

**Deliverables**:
- `evaluate.py` - Test set evaluation with comprehensive metrics (MSE, RMSE, MAE, R², Pearson r, Spearman ρ)
- `analyze_attention.py` - Attention weight extraction and visualization
- `src/models/synergy_model.py` - Updated with `return_attention` parameter for interpretability

**To Run Evaluation**:
```bash
# Evaluate attention model on test set
python evaluate.py --model attention --split test

# Compare both models
python evaluate.py --model both --split test

# Analyze attention patterns
python analyze_attention.py --split val --num_samples 1000
```

---

## Phase 2: Model Improvements (Week 2-3)

| Task | Status | Priority |
|------|--------|----------|
| Implement target standardization | Done | High |
| Add learning rate scheduling (CosineAnnealing) | Pending | High |
| Multi-task learning for all synergy metrics | Pending | High |
| Hyperparameter tuning (hidden dims, attention heads, dropout) | Pending | Medium |
| Cross-validation for robust performance estimates | Pending | Medium |
| Add early stopping | Pending | Medium |
| Implement gradient clipping | Pending | Low |

**Target Standardization** (Completed):
```bash
# Train with standardization (recommended for better convergence)
python main.py --standardize

# Evaluation automatically handles inverse transform
python evaluate.py --model attention --split test
```

The `TargetScaler` class normalizes targets to mean=0, std=1:
- Scaler parameters saved in checkpoint for evaluation
- `evaluate.py` auto-detects and inverse transforms predictions
- Benefits: Faster convergence, more stable gradients

**Hyperparameters to Tune**:
```
- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Batch size: [32, 64, 128]
- Hidden dimension: [256, 512, 1024]
- Attention heads: [4, 8, 16]
- Dropout: [0.1, 0.2, 0.3]
- Weight decay: [0, 1e-5, 1e-4]
```

**Deliverables**:
- Hyperparameter sweep results
- Multi-task model checkpoint
- Updated training curves

---

## Phase 3: Advanced Features (Week 4-5)

| Task | Status | Priority |
|------|--------|----------|
| Implement drug pair symmetry augmentation | Pending | High |
| Add GNN-based drug encoder option (e.g., MPNN) | Pending | Medium |
| Stacked transformer layers for deeper fusion | Pending | Medium |
| Implement Mixup data augmentation | Pending | Low |
| Cell line embedding via graph (if data available) | Pending | Low |
| Ensemble multiple models | Pending | Low |

**Architecture Experiments**:
```
1. Deeper omics encoder (3-4 layers with residual)
2. Separate drug-drug interaction module
3. Multiple cross-attention layers (2-3 stacked)
4. Replace mean pooling with [CLS] token
```

**Deliverables**:
- Ablation study on architecture variants
- Best performing model checkpoint
- Comparison table of all experiments

---

## Phase 4: Analysis & Documentation (Week 6)

| Task | Status | Priority |
|------|--------|----------|
| Comprehensive ablation studies | Pending | High |
| Interpretability analysis (attention visualization) | Pending | High |
| **SHAP explainability analysis** | Pending | High |
| Error analysis on prediction failures | Pending | High |
| Case studies on known synergistic pairs | Pending | Medium |
| Final paper preparation | Pending | High |
| Code cleanup and documentation | Pending | Medium |

**Analysis Questions**:
1. Which omics modality contributes most to predictions?
2. Do attention patterns differ for synergistic vs antagonistic pairs?
3. What drug/cell line combinations have highest prediction error?
4. How does performance vary across different cancer types?
5. **Which input features (genes, proteins, drug substructures) drive predictions?**

### SHAP Explainability Analysis

SHAP (SHapley Additive exPlanations) provides feature-level importance scores for individual predictions.

**Implementation Approach**:
```python
import shap

# Wrap model for SHAP
def model_predict(X):
    # X: concatenated [drug1_emb, drug2_emb, mrna, mirna, prot]
    d1, d2, mrna, mirna, prot = split_features(X)
    return model(d1, d2, mrna, mirna, prot).detach().numpy()

# Use DeepExplainer or KernelExplainer
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_samples)
```

**SHAP Analysis Tasks**:

| Task | Description |
|------|-------------|
| Global feature importance | Rank omics features by mean |SHAP| across all predictions |
| Modality importance | Compare total SHAP contribution of mRNA vs miRNA vs proteomics |
| Drug embedding importance | Identify which drug embedding dimensions matter most |
| Local explanations | Explain individual high/low synergy predictions |
| Interaction effects | SHAP interaction values for drug-omics pairs |

**Visualizations to Generate**:
1. **Summary plot**: Feature importance ranking (beeswarm plot)
2. **Bar plot**: Mean absolute SHAP values per feature/modality
3. **Dependence plots**: Feature value vs SHAP value for top features
4. **Force plots**: Individual prediction explanations
5. **Waterfall plots**: Contribution breakdown for specific samples

**Key Questions SHAP Can Answer**:
- Which genes/proteins are most predictive of synergy?
- Do Drug1 and Drug2 embeddings contribute equally?
- Are there specific omics markers that indicate antagonism?
- How do feature contributions differ across cancer types?

**Deliverables**:
- `shap_analysis.py` - SHAP computation and visualization script
- SHAP summary plots for each modality
- Top-50 most important features list
- Case study explanations for synergistic/antagonistic pairs

---

**General Deliverables**:
- Final technical report/paper
- Supplementary materials (figures, tables)
- Clean, documented codebase
- Model weights and reproducibility scripts

---

## Phase 5: Zero-Shot Prediction Capability (Future/Optional)

### Current Limitations

| Scenario | Currently Supported? | Limitation |
|----------|---------------------|------------|
| New drugs (unseen SMILES) | **NO** | Drug embeddings pre-computed only for DrugComb drugs |
| New cell lines (unseen omics) | **PARTIAL** | Works if omics data has matching dimensions |
| New combinations (seen drugs + cells) | **YES** | Model learns drug-omics interactions |

### Tasks for Zero-Shot Support

| Task | Status | Priority |
|------|--------|----------|
| Create prediction API for raw SMILES input | Pending | High |
| On-the-fly MolFormer embedding inference | Pending | High |
| Cold-start data splits (leave-drug-out) | Pending | High |
| Cold-start data splits (leave-cell-line-out) | Pending | High |
| Benchmark zero-shot generalization | Pending | Medium |
| External dataset validation (e.g., ALMANAC, O'Neil) | Pending | Medium |

### Implementation Details

**1. Prediction Interface for New Drugs**
```python
# Desired API
from src.predict import SynergyPredictor

predictor = SynergyPredictor.from_checkpoint("checkpoint_best.pt")

# Predict for any SMILES + omics
synergy = predictor.predict(
    drug1_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    drug2_smiles="CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    mrna=mrna_array,      # [mrna_dim]
    mirna=mirna_array,    # [mirna_dim]
    proteomics=prot_array # [prot_dim]
)
```

**2. Cold-Start Evaluation Splits**
```
- Leave-Drug-Out (LDO): Test drugs never seen in training
- Leave-Cell-Line-Out (LCO): Test cell lines never seen in training
- Leave-Pair-Out (LPO): Current default (drug pairs unseen, but individual drugs seen)
```

**3. Required Changes**
- `src/predict.py` - New prediction interface with on-the-fly MolFormer
- `src/dataset.py` - Add cold-start split generation functions
- `evaluate.py` - Support for cold-start evaluation metrics
- `src/models/language_model.py` - Expose inference API (currently only batch embeddings)

### Expected Challenges
1. **Generalization gap**: Model may overfit to training drug chemical space
2. **Omics dimensionality**: New cell lines must have same omics features
3. **Computational cost**: On-the-fly MolFormer inference slower than lookup
4. **Domain shift**: External datasets may have different synergy score distributions

**Deliverables**:
- `src/predict.py` - Zero-shot prediction interface
- Cold-start data splits for DrugComb
- Zero-shot benchmark results (LDO, LCO metrics)
- External dataset evaluation report

---

## Timeline Summary

```
Week 1  [■■■■■■■■■■] Phase 1: Evaluation & Baseline
Week 2  [■■■■■□□□□□] Phase 2: Model Improvements (Part 1)
Week 3  [□□□□□■■■■■] Phase 2: Model Improvements (Part 2)
Week 4  [■■■■■■■□□□] Phase 3: Advanced Features (Part 1)
Week 5  [□□□■■■■■■■] Phase 3: Advanced Features (Part 2)
Week 6  [■■■■■■■■■■] Phase 4: Analysis & Documentation
Future  [□□□□□□□□□□] Phase 5: Zero-Shot Prediction (Optional)
```

---

## Current Progress

### Completed
- [x] Dataset preparation (DrugComb splits)
- [x] Drug embeddings (MolFormer-XL)
- [x] SynergyModel architecture
- [x] End-to-end training pipeline
- [x] Initial training run (100 epochs)
- [x] Evaluation script with comprehensive metrics
- [x] MLP baseline training and comparison
- [x] Attention analysis and visualization tools
- [x] Model updated to return attention weights
- [x] Target standardization with `--standardize` flag

### Current Results
| Metric | Attention Model | MLP Baseline |
|--------|-----------------|--------------|
| Best Val MSE | 13.40 | 21.91 |
| Final Train MSE | 11.66 | 13.07 |
| Total Parameters | 20.4M | ~2.5M |
| Training Time | ~1.7 hours | ~0.5 hours |
| Generalization | Good | Poor (overfitting) |

### Next Immediate Actions
1. ~~Run test set evaluation~~ → Use `python evaluate.py --split test`
2. ~~Implement target standardization~~ → Use `python main.py --standardize`
3. Add learning rate scheduler (Phase 2)
4. Set up multi-task learning (Phase 2)

---

## Files Reference

| File | Description |
|------|-------------|
| `main.py` | Training script for SynergyModel |
| `evaluate.py` | Comprehensive evaluation (MSE, MAE, R², Pearson, Spearman) |
| `analyze_attention.py` | Attention weight extraction and visualization |
| `src/models/synergy_model.py` | Model architecture (with attention export) |
| `src/models/language_model.py` | MolFormer wrapper for drug embeddings |
| `src/dataset.py` | Dataset classes |
| `src/models/mlp.py` | Baseline MLP |
| `src/drugs_embeddings.py` | Pre-compute drug embeddings for DrugComb |
| `logs/synergy_attn_*/` | Attention model training logs and checkpoints |
| `logs/synergy_mlp_*/` | MLP baseline training logs and checkpoints |
| `logs/evaluation_*/` | Evaluation results |
| `logs/attention_analysis_*/` | Attention analysis outputs |

### Planned Files (Phase 4 - Analysis)

| File | Description |
|------|-------------|
| `shap_analysis.py` | SHAP explainability analysis (planned) |

### Planned Files (Phase 5 - Zero-Shot)

| File | Description |
|------|-------------|
| `src/predict.py` | Zero-shot prediction interface (planned) |
| `src/cold_start_splits.py` | Generate LDO/LCO evaluation splits (planned) |

---

## Notes

- Target column: `Synergy_ZIP` (can extend to Bliss, Loewe, HSA)
- Drug embeddings are frozen (from MolFormer)
- Omics encoder is trainable
- BatchNorm applied to omics inputs for stability
- **Target Standardization**: Use `--standardize` flag for training (recommended)

### Zero-Shot Prediction Status
⚠️ **Current limitation**: Model cannot predict for drugs outside DrugComb dataset.
- Drug embeddings are pre-computed only for ~4,000 DrugComb drugs
- New SMILES strings will cause KeyError during inference
- See **Phase 5** for planned zero-shot capability

### Explainability Methods

**Two complementary approaches for model interpretability:**

| Method | Level | What It Explains |
|--------|-------|------------------|
| Attention Weights | Modality/Component | How Drug1, Drug2, and Omics interact |
| SHAP Values | Feature | Which specific genes/proteins drive predictions |

#### Attention Analysis (Implemented)
The SynergyModel now supports `return_attention=True` to extract:
- **Omics Fusion Attention**: [batch, 4 heads, 3, 3] - Shows how mRNA, miRNA, and Proteomics attend to each other
- **Cross-Attention**: [batch, 8 heads, 3, 3] - Shows Drug1, Drug2, and Omics interactions

Usage:
```python
preds, attention_weights = model(d1, d2, mrna, mirna, prot, return_attention=True)
# attention_weights['omics_fusion'] - omics modality attention
# attention_weights['cross_attention'] - drug-omics cross attention
```

#### SHAP Analysis (Planned - Phase 4)
SHAP provides fine-grained feature importance:
- Identifies which specific genes/proteins contribute to each prediction
- Enables discovery of biomarkers for drug synergy
- Complements attention by explaining *what* features matter, not just *how* modalities interact
