#!/bin/bash
#SBATCH --job-name=gramseq
#SBATCH --output=./logs/gramseq.out
#SBATCH --error=./logs/gramseq.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=150G
#SBATCH --time=30-00:00:00

echo "Experiment starts at: $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd
# conda activate py10-tf210
echo "Using python from $(which python)"

module load cuda/12.8

# Print cuda-core info
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# python -m src.test.test_gvae

# python -m src/download_data.py

# python -m src.training.dataloader
python -m src.training.pretrain_moleculevae

# python src/train.py --dataset bindingdb\
#                     --protenc CNN\
#                     --epochs 500\
#                     --folds 5

echo "Experiment ends at: $(date)"
