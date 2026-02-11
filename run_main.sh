#!/bin/bash
#SBATCH --job-name=kan_exp
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

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

# KAN head, standardize target, 10 epochs, random split
# python main.py --exp_name kan_std_random --standardize --head_type kan --epochs 100 --split_type random

# ZIP
# KAN
python main.py --exp_name kan_std_random_ZIP_e100 --standardize --head_type kan --epochs 100 --split_type random --target Synergy_ZIP
python evaluate.py --target Synergy_ZIP --split_type random --split test

python main.py --exp_name kan_std_cold_drug_ZIP_e100 --standardize --head_type kan --epochs 100 --split_type cold_drug --target Synergy_ZIP
python evaluate.py --target Synergy_ZIP --split_type cold_drug --split test

# MLP
python main.py --exp_name mlp_std_random_ZIP_e100 --standardize --head_type mlp --epochs 100 --split_type random --target Synergy_ZIP
python evaluate.py --target Synergy_ZIP --split_type random --split test

python main.py --exp_name mlp_std_cold_drug_ZIP_e100 --standardize --head_type mlp --epochs 100 --split_type cold_drug --target Synergy_ZIP
python evaluate.py --target Synergy_ZIP --split_type cold_drug --split test

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

echo "Job ended on $(date)"