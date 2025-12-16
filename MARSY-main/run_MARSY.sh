#!/bin/bash
#SBATCH --job-name=marsy
#SBATCH --output=./logs/run_marsy.out
#SBATCH --error=./logs/run_marsy.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=100-00:00:00

echo "Experiment started on $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd-tensorflow

module load cuda/12.3

# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python MARSY.py

echo "Experiment ended on $(date)"