#!/bin/bash
#SBATCH --job-name=main
#SBATCH --output=./logs/main.out
#SBATCH --error=./logs/main.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

echo "Job started on $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd

module load cuda/13.0

python -c "import torch; print('Device:', torch.cuda.get_device_name(0))"

# Training options:
# python main.py                          # Train without standardization
python main.py --standardize            # Train with target standardization
# python main.py --standardize --epochs 150 --lr 5e-5  # Custom hyperparameters

# Train with target standardization (recommended)
# python main.py --standardize --epochs 100

# Test set evaluation (auto-handles standardized models)
# python evaluate.py --model attention --split test

# Compare both models
# python evaluate.py --model both --split test

# Analyze attention patterns
# python analyze_attention.py --split val --num_samples 1000

echo "Job ended on $(date)"