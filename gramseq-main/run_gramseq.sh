#!/bin/bash
#SBATCH --job-name=gramseq
#SBATCH --output=./logs/gramseq.out
#SBATCH --error=./logs/gramseq.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=150G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00

echo "Experiment starts at: $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd-tensorflow
# conda activate py10-tf210
echo "Using python from $(which python)"

module load cuda/12.3

# Print cuda-core info
# python -c "import tensorflow as tf; print(f'Device(s): {tf.config.list_physical_devices()}')"

# python -m src.test.test_gvae

# python -m src/download_data.py

python -m src.training.dataloader

# python src/train.py --dataset bindingdb\
#                     --protenc CNN\
#                     --epochs 500\
#                     --folds 5

echo "Experiment ends at: $(date)"
