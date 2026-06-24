#!/bin/bash
#SBATCH --job-name=aasist_eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

MODEL=${MODEL:-AASIST-L}

module load miniconda3/25.5.1
cd ~/aasist

python3 main.py --eval --config ./config/${MODEL}.conf
