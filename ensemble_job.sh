#!/bin/bash
#SBATCH --job-name=ensemble_eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

module load miniconda3/25.5.1
cd ~/aasist

SCORES=(
    exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt
    exp_result/LA_AASIST-L_ep100_bs24/eval_scores_using_best_dev_model.txt
    exp_result/LA_RawGATST_baseline_ep100_bs24/eval_scores_using_best_dev_model.txt
    exp_result/LA_RawNet2_baseline_ep100_bs32/eval_scores_using_best_dev_model.txt
)

LABELS="AASIST AASIST-L RawGAT-ST RawNet2"

echo "=== Fusion uniforme ==="
python3 ensemble_eval.py \
    --scores "${SCORES[@]}" \
    --labels $LABELS \
    --output exp_result/ensemble/uniform_scores.txt

echo ""
echo "=== Fusion pondérée (1/EER_eval) ==="
python3 ensemble_eval.py \
    --scores "${SCORES[@]}" \
    --labels $LABELS \
    --weights 1.2056 1.0084 0.3185 0.2364 \
    --output exp_result/ensemble/weighted_scores.txt
