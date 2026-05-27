# Research Log

## Project: Drug Synergy Prediction with L1000 Perturbation Profiles

---

### 1. Unique Drug Extraction from DrugComb

**Script:** `meta-analysis/unique_drugs.py`

- Loaded the DrugComb dataset via TDC (`tdc.multi_pred.DrugSyn`).
- Extracted Drug1 and Drug2 identifiers (name + SMILES) from train/valid/test splits.
- Merged and deduplicated to produce a single list of unique drugs.
- **Result:** 129 unique drugs saved to `meta-analysis/unique_drugs.csv` (columns: `Drug_ID`, `Drug`).

---

### 2. LINCS L1000 Level 5 Profile Extraction

**Script:** `meta-analysis/l1000/main.py`  
**Runner:** `meta-analysis/l1000/run_main.sh` (SLURM, 128 GB RAM, GPU partition, conda env `aidd`)

- Parsed the Level 5 GCTX file (`level5_beta_trt_cp_n720216x12328.gctx`) in chunks of 10,000 signatures.
- Filtered signatures to `pert_type == trt_cp` and `qc_pass == 1` using `siginfo_beta.txt`.
- Extracted raw moderated z-score matrices (all 12,328 genes and 978 landmark genes).
- Computed per-compound consensus profiles (median modZ across signatures grouped by `pert_id`).
- Enriched consensus profiles with compound metadata (target, MoA, SMILES, InChI key) from `compoundinfo_beta.txt`.

**Outputs** (in `meta-analysis/l1000/output/`):
| File | Description |
|------|-------------|
| `raw_signatures_all_genes.csv` | Raw signature matrix, all genes |
| `raw_signatures_landmark.csv` | Raw signature matrix, landmark genes |
| `consensus_profiles_all_genes.csv` | Consensus profiles (35,391 compounds x 12,328 genes) |
| `consensus_profiles_landmark.csv` | Consensus profiles (35,391 compounds x 978 genes) |
| `signature_metadata.csv` | Per-signature QC and experimental metadata |
| `gene_metadata.csv` | Gene annotations (12,329 genes) |

---

### 3. Drug Matching: DrugComb to L1000

**Script:** `meta-analysis/l1000/extract_unique_drug_perturbations.py`

Matched all 129 DrugComb drugs against L1000 consensus profiles using a 5-pass matching pipeline that guarantees 100% coverage:

| Pass | Strategy | match_type | Description |
|------|----------|-----------|-------------|
| 0 | Manual overrides | `manual` | Hand-curated Drug_ID -> cmap_name mappings for known synonyms (`meta-analysis/l1000/manual_drug_mappings.csv`) |
| 1 | Name match | `name` | Case-insensitive exact match of Drug_ID to L1000 `cmap_name` |
| 2 | Tanimoto >= 0.85 | `tanimoto` | Morgan fingerprint (radius=2, 2048 bits) similarity on original SMILES |
| 2.5 | Salt-stripped Tanimoto >= 0.85 | `salt_tanimoto` | Same as pass 2 but strips multi-component SMILES to the largest fragment (by heavy-atom count) before fingerprinting, recovering salt forms like TAMOXIFEN CITRATE |
| 3 | Nearest-neighbor proxy | `proxy` | Assigns the rank-1 most similar L1000 compound by Tanimoto as a surrogate for drugs not profiled in L1000 |

**Implementation details:**
- **Pass 0:** 3 verified manual mappings (CISPLATINO -> cisplatin, NSC 85998 -> streptozotocin, URACIL MUSTARD -> uracil-mustard). These drugs exist in L1000 under different names confirmed by searching the consensus profiles.
- **Passes 2 and 2.5:** RDKit canonicalization, pre-computed reference fingerprints over ~25K unique SMILES, vectorised `BulkTanimotoSimilarity`. Pass 2.5 uses `Chem.GetMolFrags` to isolate the parent compound from salt counterions.
- **Pass 3:** For drugs genuinely absent from L1000 (platinum complexes, large natural products, experimental compounds), the most structurally similar profiled compound is used as a proxy. The `tanimoto_score` column records the similarity so downstream analysis can assess confidence.

**Coverage evolution:**

| Stage | Matched | Unmatched | New recoveries |
|-------|---------|-----------|----------------|
| After pass 0 (manual overrides) | 3 | 126 | 3 |
| After pass 1 (name match) | ~69 | ~60 | ~66 |
| After pass 2 (Tanimoto >= 0.85) | ~114 | ~15 | ~45 |
| After pass 2.5 (salt-stripped Tanimoto) | ~116 | ~13 | ~2 |
| After pass 3 (proxy) | **129 (100%)** | **0** | ~13 |

**Final coverage: 129/129 drugs matched (100%).**

**Outputs** (in `meta-analysis/l1000/output/drug_query_results/`):
| File | Description |
|------|-------------|
| `drug_match_map.csv` | Full mapping table (Drug_ID, Drug, pert_id, cmap_name, SMILES, InChI key, match_type, tanimoto_score) |
| `unmatched_drugs.csv` | Drugs with no direct L1000 match (matched via proxy in pass 3) |
| `unmatched_nearest_neighbors.csv` | Top-5 most similar L1000 compounds per unmatched drug (for diagnostic review) |
| `matched_consensus_landmark.csv` | Consensus landmark gene rows for all matched pert_ids |
| `matched_consensus_all_genes.csv` | Consensus all-genes rows for all matched pert_ids |
| `matched_signature_metadata.csv` | Signature metadata for all matched pert_ids |

---

### 4. Representative Profiles (Single Profile Per Drug)

**Script:** `meta-analysis/l1000/extract_unique_drug_perturbations.py` (aggregation step)

Collapsed multi-pert_id drugs into a single representative profile per `Drug_ID` by computing the **median** modZ across all matched `pert_id` rows for each gene. Median was chosen for robustness to outlier stereoisomers, consistent with the consensus computation in the L1000 extraction pipeline.

**Outputs** (in `meta-analysis/l1000/output/drug_query_results/`):
| File | Shape | Description |
|------|-------|-------------|
| `representative_profiles_landmark.csv` | 129 x 980 | Drug_ID + Drug + 978 landmark gene columns |
| `representative_profiles_all_genes.csv` | 129 x 12,330 | Drug_ID + Drug + 12,328 gene columns |

These files provide complete perturbation-based drug features for all 129 DrugComb drugs and are ready for integration into the synergy prediction model.

---

### 5. L1000-Augmented Synergy Model

**Model:** `src/models/synergy_model.py` → `SynergyModelL1000`  
**Dataset:** `src/dataset.py` → `DrugSynergyL1000Dataset`  
**Training:** `main.py` (pass `--l1000_path` to activate)

Integrated L1000 representative perturbation profiles as per-drug features alongside frozen chemical language model embeddings. The architecture replaces the multi-omics attention fusion (mRNA + miRNA + proteomics) with a simpler concatenation-based design using only mRNA and L1000 profiles:

**Architecture:**

| Component | Source | Dimension |
|-----------|--------|-----------|
| Drug 1 embedding | Frozen MolFormer CLS | 768 |
| Drug 2 embedding | Frozen MolFormer CLS | 768 |
| mRNA encoding | Trainable `OmicsEncoder` | 768 |
| L1000 drug 1 encoding | Trainable `OmicsEncoder` (shared) | 768 |
| L1000 drug 2 encoding | Trainable `OmicsEncoder` (shared) | 768 |
| **Concatenated input** | | **3840** |

The concatenated 3840-d vector is fed to the MLP or KAN prediction head (same head classes as before, with updated input dimension).

The L1000 encoder is shared between both drugs in a pair (same weights). Each `OmicsEncoder` applies BatchNorm → Linear(in, 1024) → BatchNorm → ReLU → Linear(1024, 768).

**Dataset (`DrugSynergyL1000Dataset`):**
- Loads the representative profiles CSV and builds a SMILES → tensor lookup at init
- Per sample: looks up `row['Drug1']` and `row['Drug2']` SMILES against the L1000 lookup
- Returns 6-tuple: `(drug1_emb, drug2_emb, mrna, l1000_d1, l1000_d2, y)`

**CLI usage:**

```bash
# L1000 mode (landmark genes, 978-d per drug)
python main.py \
    --l1000_path meta-analysis/l1000/output/drug_query_results/representative_profiles_landmark.csv \
    --l1000_genes landmark \
    --head_type kan --standardize

# L1000 mode (all genes, 12328-d per drug)
python main.py \
    --l1000_path meta-analysis/l1000/output/drug_query_results/representative_profiles_all_genes.csv \
    --l1000_genes all \
    --head_type kan --standardize

# Original mode (no L1000, mRNA+miRNA+prot attention fusion — unchanged)
python main.py --head_type kan --standardize
```

**Backward compatibility:** Without `--l1000_path`, the original `SynergyModel` + `DrugSynergyRawOmicsDataset` pipeline is used unchanged. Existing checkpoints and logs are unaffected.

---

### 6. Configurable Multi-Omics Fusion Strategies

**Files modified:** `src/models/synergy_model.py`, `main.py`, `ablation.py`

Added support for multiple fusion strategies in the original `SynergyModel` (mRNA + miRNA + proteomics fusion). Previously, only self-attention with a learnable CLS token was available. Now five strategies can be selected via the `--fusion_type` CLI argument:

| Strategy | Operation | Output Dim | Description |
|----------|-----------|------------|-------------|
| `attention` | CLS + MultiheadAttention | 768 | Learnable cross-modal attention weights (default, original behavior) |
| `concat` | Concatenate along feature dim | 2304 | Simple baseline preserving all modality information |
| `product` | Element-wise product | 768 | Captures multiplicative interactions between modalities |
| `mean_pool` | Mean across modalities | 768 | Simple average with equal weighting |
| `max_pool` | Max across modalities | 768 | Selects strongest features per dimension |

**Implementation:**

Added three new fusion modules (`OmicsConcatFusion`, `OmicsProductFusion`, `OmicsPoolingFusion`) and updated `OmicsFusionModel` with a `fusion_type` parameter that selects the fusion module and sets `fused_dim`. The `SynergyModel` now computes the prediction head input dimension dynamically based on `fused_dim`.

**CLI usage:**

```bash
# Default attention fusion
python main.py --head_type kan --standardize

# Concatenation fusion (larger head input: 3840)
python main.py --fusion_type concat --head_type kan --standardize

# Element-wise product fusion
python main.py --fusion_type product --head_type kan --standardize

# Mean pooling fusion
python main.py --fusion_type mean_pool --head_type kan --standardize

# Ablation study with specific fusion
python ablation.py --fusion_type concat --epochs 100
```

**Backward compatibility:** Default `fusion_type="attention"` preserves original model behavior. The L1000-augmented model (`SynergyModelL1000`) is unaffected as it uses direct concatenation without multi-omics fusion.

---

### 7. Inference Script for External Drug Combinations

**Script:** `inference.py`

Implemented an inference pipeline to predict synergy scores for external/experimental drug combinations using the trained `SynergyModelL1000` model. The script enables evaluation of novel drug combinations not present in the training data.

**Features:**

- Loads trained checkpoint and config (including target scaler parameters)
- Generates MolFormer embeddings for novel drugs from SMILES files
- Handles L1000 profile matching via Tanimoto similarity proxy when drugs are not directly profiled in L1000
- Extracts cell line mRNA data for target cancer types
- Supports all four synergy metrics: `Synergy_ZIP`, `Synergy_Bliss`, `Synergy_Loewe`, `Synergy_HSA`

**CLI Arguments:**

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to model checkpoint |
| `--target` | Synergy metric type |
| `--approved_dir` | Directory with approved drug SMILES |
| `--proprietary_dir` | Directory with proprietary drug SMILES |
| `--l1000_path` | L1000 representative profiles CSV |
| `--output` | Output CSV path |

**Initial Results (Sebti Group Combinations):**

Using `SynergyModelL1000` cold-drug checkpoint (Synergy_Bliss):

| Drug1 | Drug2 | Cancer Type | Avg Synergy |
|-------|-------|-------------|-------------|
| FGTI-2734 | Sotorasib | Lung | +2.39 |
| GGTI-2418 | Belinostat | T-Cell Lymphoma | -0.26 |
| GGTI-2418 | Romidepsin | T-Cell Lymphoma | -3.18 |

FGTI-2734 + Sotorasib shows positive synergy across lung cancer cell lines (highest in NCI-H322M: +4.79), consistent with published literature on KRAS inhibitor combinations.

**Outputs:** `external/sebti-group/synergy_predictions.csv`

---

### 8. Extended Inference Script to Support Both Model Types

**Date:** 2026-05-13

**Implementation:**
- Modified `inference.py` to support both the original `SynergyModel` (mRNA + miRNA + proteomics) and `SynergyModelL1000` (mRNA + L1000 profiles)
- Added model type auto-detection based on `config.json` keys (presence of `l1000_dim` indicates L1000 model)
- Added `embed_dim` inference from checkpoint state_dict for backward compatibility with older checkpoints
- Updated cell line data extraction to return all three omics modalities for `SynergyModel`
- L1000 profile loading is now skipped when using `SynergyModel`
- Added inference commands for original `SynergyModel` to `run_main.sh`

**Rationale:**
- Enable comparison between multi-omics fusion (mRNA + miRNA + proteomics) and L1000-augmented approaches for external drug predictions
- Allow inference with different checkpoint architectures without code modifications

**Files modified:**
- `inference.py` - Model type auto-detection, embed_dim inference, conditional omics extraction
- `run_main.sh` - Added inference commands for original SynergyModel checkpoints

**Observations:**
- Original `SynergyModel` (multi-omics attention fusion) predictions for Sebti group combinations:

| Drug1 | Drug2 | Cancer Type | ZIP Avg | Bliss Avg | HSA Avg | Loewe Avg |
|-------|-------|-------------|---------|-----------|---------|-----------|
| FGTI-2734 | Sotorasib | Lung | +1.20 | +1.14 | -1.42 | -4.56 |
| GGTI-2418 | Belinostat | T-Cell Lymphoma | +1.19 | +1.65 | -2.87 | -10.59 |
| GGTI-2418 | Romidepsin | T-Cell Lymphoma | +0.03 | -1.14 | -3.31 | -12.30 |

- Different synergy metrics show varying predictions; ZIP and Bliss show positive synergy for FGTI-2734 combinations while HSA and Loewe show antagonism
- GGTI-2418 + Belinostat shows strongest Bliss synergy (+4.38 in MOLT-4 cells)

**Outputs:** 
- `external/sebti-group/results/synergylm_zip_predictions.csv`
- `external/sebti-group/results/synergylm_hsa_predictions.csv`
- `external/sebti-group/results/synergylm_bliss_predictions.csv`
- `external/sebti-group/results/synergylm_loewe_predictions.csv`

---

### 9. Comprehensive Inference Analysis: SynergyLM vs SynergyLM+L1000

**Date:** 2026-05-19

**Implementation:**
- Created comprehensive analysis comparing predictions from both model architectures for Sebti group drug combinations
- Analyzed synergy predictions across multiple metrics (ZIP, Bliss, HSA, Loewe) and cell lines
- Evaluated L1000 proxy matching quality and its impact on predictions

**Rationale:**
- Understand model-specific prediction patterns for experimental prioritization
- Identify reliable predictions versus those sensitive to modeling assumptions

**Files modified:**
- `external/sebti-group/results/analysis.md` - Comprehensive write-up

**Key Findings:**

1. **FGTI-2734 + Sotorasib**: Both models predict positive synergy (SynergyLM: +1.14, L1000: +2.39 Bliss avg). Strongest in NCI-H322M cells. Recommended for experimental validation.

2. **GGTI-2418 + Belinostat**: Model divergence observed. SynergyLM predicts synergy (+1.65, strongest in MOLT-4: +4.38), while L1000 predicts slight antagonism (-0.26). Low L1000 proxy similarity (0.27) may explain discrepancy.

3. **GGTI-2418 + Romidepsin**: Consistent antagonism across both models (-1.14 to -3.18). De-prioritized despite direct L1000 match for Romidepsin.

4. **Metric-dependent predictions**: ZIP/Bliss generally positive, HSA/Loewe generally negative for same combinations.

5. **L1000 proxy limitation**: Low Tanimoto similarity (<0.3) for most drugs suggests prediction uncertainty. Direct L1000 profiling recommended.

**Outputs:** 
- `external/sebti-group/results/analysis.md`

---

### 10. FGTI-2734 + Sotorasib Detailed Report

**Date:** 2026-05-19

**Implementation:**
- Created detailed report for FGTI-2734 + Sotorasib combination with cell-line specific predictions
- Included KRAS mutation status for each cell line to highlight biological relevance
- Compared predictions with published experimental synergy data

**Rationale:**
- Focus analysis on the biologically relevant KRAS G12C cell line (NCIH23) since Sotorasib targets this specific mutation
- Provide actionable insights for experimental validation

**Files modified:**
- `external/sebti-group/results/FGTI-2734+Sotorasib_report.md`

**Key Findings:**

1. **NCIH23 (KRAS G12C)** - the only biologically relevant cell line:
   - ZIP: +1.44 (SynergyLM), +0.91 (L1000) - modest positive synergy
   - Bliss: +0.38 (SynergyLM), +1.65 (L1000) - weak to moderate synergy
   - HSA: -1.39 (SynergyLM), -1.37 (L1000) - slight antagonism
   - Loewe: -6.28 (SynergyLM), -12.12 (L1000) - strong antagonism

2. **Discrepancy with literature**: Published studies report synergy values of 5-9 (Loewe/HSA scale 0-14), but our models predict antagonism for these metrics in NCIH23

3. **KRAS wild-type cell lines** (NCI-H322M, NCI-H226) show high predicted synergy, but these are biologically questionable since Sotorasib requires KRAS G12C for activity

**Outputs:** 
- `external/sebti-group/results/FGTI-2734+Sotorasib_report.md`

---

### Next Steps

- Investigate model prediction discrepancy for NCIH23 vs. published experimental data
- Validate FGTI-2734 + Sotorasib predictions experimentally in NCIH23 (KRAS G12C)
- Consider retraining with KRAS mutation-aware features
- Generate direct L1000 profiles for FGTI-2734 and Sotorasib to improve L1000 model predictions
- Compare fusion strategies (attention vs. concat vs. product vs. pooling) on synergy prediction metrics
