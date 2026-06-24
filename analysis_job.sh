#!/bin/bash
#SBATCH --job-name=aasist_analysis
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

module load miniconda3/25.5.1
cd ~/aasist

SCORES=./exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt

echo "=== Breakdown par attaque ==="
python3 breakdown_analysis.py --scores $SCORES

echo "=== Robustesse au bruit ==="
python3 -u noise_robustness.py --config ./config/AASIST.conf --snr_levels 30 20 10 5 0
