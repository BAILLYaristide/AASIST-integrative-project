#!/bin/bash
#SBATCH --job-name=aasist_resume
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

module load miniconda3/25.5.1
cd ~/aasist

CKPT=./exp_result/LA_RawGATST_baseline_ep100_bs24/weights/best.pth

echo "=== Reprise entraînement RawGAT-ST depuis epoch 76 ==="
python3 main.py \
    --config ./config/RawGATST_baseline.conf \
    --resume $CKPT \
    --start_epoch 76
