#!/bin/bash
#SBATCH --job-name=aasist_distill
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

module load miniconda3/25.5.1
cd ~/aasist

echo "=== Distillation AASIST (teacher) → AASIST-L (student) ==="
python3 distillation.py --config ./config/AASIST_distill.conf
