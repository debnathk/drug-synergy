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

module load cuda/12.8

python src/omics_embeddings.py

echo "Job ended on $(date)"