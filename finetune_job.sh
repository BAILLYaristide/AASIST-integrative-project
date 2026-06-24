#!/bin/bash
#SBATCH --job-name=aasist_finetune
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

CHECKPOINT=${CHECKPOINT:-models/weights/AASIST.pth}

module load miniconda3/25.5.1
cd ~/aasist

echo "=== Fine-tuning AASIST depuis : $CHECKPOINT ==="
python3 main.py \
    --config ./config/AASIST_finetune_noise.conf \
    --resume "$CHECKPOINT"
