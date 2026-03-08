#!/bin/bash
#SBATCH --job-name=synergy_exp
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=64G

echo "Job started on $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd

module load cuda/13.0

python -c "import torch; print('Device:', torch.cuda.get_device_name(0))"

# ==============================================================================
# TRAINING: KAN head, standardized target, 10 epochs
# Outputs: logs/<exp_name>_<timestamp>/ with RMSE, SCC, PCC (std) in PrettyTable.
# Split types: random (default) or cold_drug. If using cold_drug, run first:
# python -m src.dataset --split_type all
# python -m src.drugs_embeddings --split_type random
# python -m src.drugs_embeddings --split_type cold_drug
# ==============================================================================

epochs=50
omics_emb_dim=768

# KAN head, standardize target, 10 epochs, random split
# python main.py --exp_name kan_std_random --standardize --head_type kan --epochs 100 --split_type random

# Run all experiments after simplifying omics projection
# for target in Synergy_ZIP Synergy_HSA Synergy_Bliss Synergy_Loewe; do
#   for head in kan mlp; do
#     for split in random cold_drug; do
#       exp_name="${head}_std_CLS_${split}_${target}_e100"
#       echo "====== TRAIN: $exp_name ======"
#       python main.py --exp_name $exp_name --standardize --head_type $head --epochs 100 --split_type $split --target $target
#       echo "====== EVAL: $exp_name ======"
#       python evaluate.py --target $target --split_type $split --split test
#       echo ""
#     done
#   done
# done


# for target in Synergy_ZIP, Synergy_BLISS; do
#   for head in mlp; do
#     for split in random; do
#       exp_name="${head}_std_CLS_${split}_${target}_e${epochs}_omics_emb_dim${omics_emb_dim}"
#       echo "====== TRAIN: $exp_name ======"
#       python main.py --exp_name $exp_name --standardize --head_type $head --epochs $epochs --split_type $split --target $target
#       echo "====== EVAL: $exp_name ======"
#       python evaluate.py --target $target --split_type $split --split test
#       echo ""
#   done
# done


# ZIP
# KAN
# python main.py --exp_name kan_std_random_ZIP_e100 --standardize --head_type kan --epochs 100 --split_type random --target Synergy_ZIP # First run after simplifying omics projection
# python evaluate.py --target Synergy_ZIP --split_type random --split test

# python main.py --exp_name kan_std_cold_drug_ZIP_e100 --standardize --head_type kan --epochs 100 --split_type cold_drug --target Synergy_ZIP
# python evaluate.py --target Synergy_ZIP --split_type cold_drug --split test

# MLP
# python main.py --exp_name mlp_std_random_ZIP_e100 --standardize --head_type mlp --epochs 100 --split_type random --target Synergy_ZIP
# python evaluate.py --target Synergy_ZIP --split_type random --split test

# python main.py --exp_name mlp_std_cold_drug_ZIP_e100 --standardize --head_type mlp --epochs 100 --split_type cold_drug --target Synergy_ZIP
# python evaluate.py --target Synergy_ZIP --split_type cold_drug --split test

# Loewe
# KAN
# python main.py --exp_name kan_std_random_Loewe_e100 --standardize --head_type kan --epochs 100 --split_type random --target Synergy_Loewe
# python evaluate.py --target Synergy_Loewe --split_type random --split test

# python main.py --exp_name kan_std_cold_drug_Loewe_e100 --standardize --head_type kan --epochs 100 --split_type cold_drug --target Synergy_Loewe
# python evaluate.py --target Synergy_Loewe --split_type cold_drug --split test

# MlP
# python main.py --exp_name mlp_std_random_Loewe_e100 --standardize --head_type mlp --epochs 100 --split_type random --target Synergy_Loewe
# python evaluate.py --target Synergy_Loewe --split_type random --split test

# python main.py --exp_name mlp_std_cold_drug_Loewe_e100 --standardize --head_type mlp --epochs 100 --split_type cold_drug --target Synergy_Loewe
# python evaluate.py --target Synergy_Loewe --split_type cold_drug --split test

# python main.py --exp_name mlp_std_random --standardize --head_type mlp --epochs 100 --split_type random


# Same settings, cold_drug split (uncomment after generating cold_drug splits/embeddings)
# python main.py --exp_name kan_std_cold_drug --standardize --head_type kan --epochs 10 --split_type cold_drug

# Evaluate (after training)
# Default (Synergy_ZIP)
# python evaluate.py --target Synergy_ZIP --split_type cold_drug --split test

# Other synergy types (must match training)
# python evaluate.py --target Synergy_Bliss --split_type random --split test
# python evaluate.py --target Synergy_Loewe --split test
# python evaluate.py --target Synergy_HSA --split_type cold_drug --split test

# ==============================================================================
# ABLATION STUDY: Omics leave-one-out (drug embeddings always active)
# Runs three conditions sequentially: no_mrna, no_mirna, no_prot.
# Outputs per condition: logs/ablation_<condition>_<timestamp>/
# Summary PrettyTable (RMSE, SCC, PCC with std) is printed after all conditions.
# Summary JSON: logs/ablation_summary_<timestamp>.json
#

# for target in Synergy_HSA Synergy_Bliss Synergy_Loewe; do
#   for head in mlp; do
#     for split in random cold_drug; do
#       exp_name="${head}_std_${split}_${target}_e${epochs}"
#       python ablation.py --exp_name $exp_name --standardize --head_type $head --epochs $epochs --split_type $split --target $target
#     done
#   done
# done

# Analyze attention weights for ZIP (original) - cold_drug split
# python main.py --exp_name mlp_std_CLS_cold_drug_Synergy_ZIP_e50 --standardize --head_type mlp --epochs 50 --split_type cold_drug --target Synergy_ZIP

# python main.py --exp_name mlp_std_CLS_cold_drug_Synergy_HSA_e50 --standardize --head_type mlp --epochs 50 --split_type cold_drug --target Synergy_HSA

# python main.py --exp_name mlp_std_CLS_cold_drug_Synergy_Loewe_e50 --standardize --head_type mlp --epochs 50 --split_type cold_drug --target Synergy_Loewe

# python main.py --exp_name mlp_std_CLS_cold_drug_Synergy_Bliss_e50 --standardize --head_type mlp --epochs 50 --split_type cold_drug --target Synergy_Bliss


# Visualize attention weights
# python analyze_attention.py \
#     --checkpoint logs/mlp_std_CLS_cold_drug_Synergy_ZIP_e50_20260305_153002/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_ZIP

# python analyze_attention.py \
#     --checkpoint logs/mlp_std_CLS_cold_drug_Synergy_HSA_e50_20260305_161932/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_HSA

# python analyze_attention.py \
#     --checkpoint logs/mlp_std_CLS_cold_drug_Synergy_Loewe_e50_20260305_171353/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_Loewe

# python analyze_attention.py \
#     --checkpoint logs/mlp_std_CLS_cold_drug_Synergy_Bliss_e50_20260306_121254/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_Bliss


# Analyze attention weights for ZIP (Ablation study) - no mRNA, no miRNA, no protein, no omics (cold_drug split)
# python analyze_attention.py \
#     --checkpoint logs/mlp_std_cold_drug_Synergy_ZIP_e100_no_mrna_20260227_202128/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_ZIP

# python analyze_attention.py \
#     --checkpoint logs/mlp_std_cold_drug_Synergy_ZIP_e100_no_mirna_20260227_220747/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_ZIP

# python analyze_attention.py \
#     --checkpoint logs/mlp_std_cold_drug_Synergy_ZIP_e100_no_prot_20260227_235004/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_ZIP

# python analyze_attention.py \
#     --checkpoint logs/mlp_std_cold_drug_Synergy_ZIP_e100_no_omics_20260228_012943/checkpoint_best.pt \
#     --split test --split_type cold_drug --target Synergy_ZIP


# Run all three conditions (default):
# python ablation.py --standardize --head_type kan --epochs 100 --split_type random --target Synergy_ZIP
# python ablation.py --standardize --head_type mlp --epochs 100 --split_type random --target Synergy_ZIP

#
# Run a single condition:
# python ablation.py --ablation no_mrna --standardize --head_type kan --epochs 100 --split_type random
#
# With cold_drug split:
# python ablation.py --standardize --head_type kan --epochs 100 --split_type cold_drug --target Synergy_ZIP
# python ablation.py --standardize --head_type mlp --epochs 100 --split_type cold_drug --target Synergy_ZIP
# ==============================================================================

# ==============================================================================
# DATA SCALING STUDY: Train on 25%, 50%, 75%, 100% of training data
# Subsamples only training data; validation and test sets remain fixed.
# Uses random split only. Target: Synergy_ZIP (default).
# Outputs per fraction: logs/data_scaling_frac_<frac>_<timestamp>/
# Summary plot: logs/data_scaling_study_<timestamp>/scaling_curves.png
# Summary JSON: logs/data_scaling_study_<timestamp>/scaling_summary.json
#
# Run all fractions (default: 25%, 50%, 75%, 100%):
# python data_scaling.py --fractions 0.25 0.50 0.75 --standardize --head_type mlp --drug_model molformer --epochs 50 --seed 42 --exp_name mlp_std_CLS_random_Synergy_ZIP_e50_scaling_study
#
# Generate embeddings with ChemBERTa (384-dim)
# python -m src.drugs_embeddings --split_type random --model chemberta

# Train with ChemBERTa embeddings
# python main.py --drug_emb_dim 384 --drug_model chemberta --standardize --head_type mlp --epochs 50 --split_type random --target Synergy_ZIP --exp_name mlp_std_CLS_random_Synergy_ZIP_e50_chemberta_384

# python evaluate.py --target Synergy_ZIP --split_type random --split test
python evaluate.py --checkpoint logs/mlp_std_CLS_random_Synergy_ZIP_e50_chemberta_384_20260306_204007/checkpoint_best.pt --split test

#
# Custom fractions (e.g., 10%, 30%, 50%, 70%, 90%, 100%):
# python data_scaling.py --fractions 0.10 0.30 0.50 0.70 0.90 1.0 --standardize --head_type kan --epochs 100
#
# Different seed for subsampling:
# python data_scaling.py --standardize --head_type kan --epochs 100 --seed 123
# ==============================================================================

echo "Job ended on $(date)"