#!/bin/bash
#SBATCH --job-name=gramseq
#SBATCH --output=./logs/gramseq.out
#SBATCH --error=./logs/gramseq.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --partition=cpu
#SBATCH --time=30-00:00:00

echo "Experiment starts at: $(date)"

eval "$(conda shell.bash hook)"
conda activate aidd-tensorflow
echo "Using python from $(which python)"

module load cuda/12.3

# Print cuda-core info
# python -c "import torch; print(f"Using device: {torch.cuda.get_device_name(0)}")"

python -m src.test.test_gvae

# python src/train.py --dataset bindingdb\
#                     --protenc CNN\
#                     --epochs 500\
#                     --folds 5

echo "Experiment ends at: $(date)"
