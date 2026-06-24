"""
Courbe DET (Detection Error Tradeoff) multi-modèles.
Superpose les 4 modèles + l'ensemble sur un même graphique.
Standard dans la littérature ASVspoof.

Usage :
  python3 det_curve.py \
    --scores exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt \
             exp_result/LA_AASIST-L_ep100_bs24/eval_scores_using_best_dev_model.txt \
             exp_result/LA_RawGATST_baseline_ep100_bs24/eval_scores_using_best_dev_model.txt \
             exp_result/LA_RawNet2_baseline_ep100_bs32/eval_scores_using_best_dev_model.txt \
    --labels AASIST AASIST-L RawGAT-ST RawNet2 \
    --output exp_result/det_curve.png

Ajouter un fichier d'ensemble avec --ensemble_score pour le superposer.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

from evaluation import compute_det_curve, compute_eer


def load_scores(path):
    data = np.genfromtxt(path, dtype=str)
    keys = data[:, 2]
    scores = data[:, 3].astype(float)
    bona = scores[keys == "bonafide"]
    spoof = scores[keys == "spoof"]
    return bona, spoof


def to_normal_deviate(p):
    """Transforme une probabilité en dévié normal (axe DET standard)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return norm.ppf(p)


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
STYLES = ["-", "--", "-.", ":", "-", "--"]


def plot_det(ax, frr, far, label, color, style, eer):
    x = to_normal_deviate(far)
    y = to_normal_deviate(frr)
    ax.plot(x, y, linestyle=style, color=color,
            label=f"{label} (EER {eer:.2f}%)", linewidth=1.8)


def main(args):
    fig, ax = plt.subplots(figsize=(8, 7))

    all_paths = list(args.scores)
    all_labels = list(args.labels) if args.labels else [f"model_{i}" for i in range(len(all_paths))]
    if args.ensemble_score:
        all_paths.append(args.ensemble_score)
        all_labels.append("Ensemble")

    assert len(all_labels) == len(all_paths)

    for i, (path, label) in enumerate(zip(all_paths, all_labels)):
        bona, spoof = load_scores(path)
        frr, far, _ = compute_det_curve(bona, spoof)
        eer, _ = compute_eer(bona, spoof)
        color = COLORS[i % len(COLORS)]
        style = STYLES[i % len(STYLES)]
        plot_det(ax, frr, far, label, color, style, eer * 100)

    # Axe en pourcentages (ticks en dévié normal, labels en %)
    ticks_pct = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    ticks_nd = [to_normal_deviate(p / 100) for p in ticks_pct]
    ax.set_xticks(ticks_nd)
    ax.set_xticklabels([str(p) for p in ticks_pct])
    ax.set_yticks(ticks_nd)
    ax.set_yticklabels([str(p) for p in ticks_pct])

    # Diagonale EER
    lim = to_normal_deviate(np.array([0.001, 0.4]))
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.4, label="EER line")

    ax.set_xlim(to_normal_deviate(0.001), to_normal_deviate(0.4))
    ax.set_ylim(to_normal_deviate(0.001), to_normal_deviate(0.4))
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title("DET Curve — ASVspoof 2019 LA eval")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"DET curve sauvegardée : {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True,
                        help="Fichiers de scores des modèles")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Noms des modèles")
    parser.add_argument("--ensemble_score", type=str, default=None,
                        help="Fichier de scores de l'ensemble (optionnel)")
    parser.add_argument("--output", type=str,
                        default="./exp_result/det_curve.png")
    main(parser.parse_args())
