#!/bin/bash
#SBATCH --job-name=gramseq
#SBATCH --output=./logs/gramseq.out
#SBATCH --error=./logs/gramseq.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=30-00:00:00

echo "Experiment starts at: $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd-tensorflow
echo "Using python from $(which python)"

module loade cuda/12.3

python src/train.py --dataset bindingdb\
                    --protenc CNN\
                    --epochs 500\
                    --folds 5

echo "Experiment ends at: $(date)"
