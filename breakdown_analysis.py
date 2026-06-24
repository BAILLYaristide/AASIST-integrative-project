"""
Breakdown de l'EER par type d'attaque (A07-A19) à partir du fichier de scores.
Usage : python3 breakdown_analysis.py --scores <chemin_vers_scores.txt>
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from evaluation import compute_eer


def load_scores(path):
    data = np.genfromtxt(path, dtype=str)
    sources = data[:, 1]
    keys = data[:, 2]
    scores = data[:, 3].astype(float)
    return sources, keys, scores


def main(args):
    sources, keys, scores = load_scores(args.scores)

    bona_scores = scores[keys == "bonafide"]
    attack_types = np.unique(sources[keys == "spoof"])

    results = {}
    for attack in attack_types:
        spoof_scores = scores[sources == attack]
        if len(spoof_scores) == 0:
            continue
        eer, _ = compute_eer(bona_scores, spoof_scores)
        results[attack] = eer * 100

    print(f"\n{'Attaque':<10} {'EER (%)':>10} {'Difficulté':>12}")
    print("-" * 35)
    for attack, eer in sorted(results.items(), key=lambda x: x[1]):
        bar = "█" * int(eer / 0.5)
        print(f"{attack:<10} {eer:>10.4f}  {bar}")

    overall_eer, _ = compute_eer(bona_scores, scores[keys == "spoof"])
    print(f"\nEER global : {overall_eer * 100:.4f} %")

    # Graphique
    attacks = list(results.keys())
    eers = [results[a] for a in attacks]
    colors = ["#d62728" if e < 1.0 else "#ff7f0e" if e < 5.0 else "#1f77b4" for e in eers]

    import os
    model_name = args.model_name if args.model_name else os.path.basename(os.path.dirname(args.scores))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(attacks, eers, color=colors)
    ax.axhline(overall_eer * 100, color="black", linestyle="--", label=f"EER global ({overall_eer*100:.2f}%)")
    ax.set_xlabel("Type d'attaque")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"EER par type d'attaque — {model_name}")
    ax.legend()
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{eer:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = args.scores.replace(".txt", "_breakdown.png")
    plt.savefig(out_path, dpi=150)
    print(f"Graphique sauvegardé : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True,
                        help="Chemin vers le fichier de scores (eval_scores_using_best_dev_model.txt)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Nom du modèle à afficher dans le titre du graphique")
    main(parser.parse_args())
