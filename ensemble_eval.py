"""
Fusion de scores (ensemble) de plusieurs modèles anti-spoofing.

Normalise les scores de chaque modèle (zero-mean / unit-variance) avant
la fusion, car les sorties AASIST et RawNet2 ne sont pas sur la même échelle.
Calcule ensuite EER et min-tDCF sur les scores fusionnés.

Usage — moyenne uniforme :
  python3 ensemble_eval.py \
    --scores exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt \
             exp_result/LA_AASIST-L_ep100_bs24/eval_scores_using_best_dev_model.txt \
             exp_result/LA_RawGATST_baseline_ep100_bs24/eval_scores_using_best_dev_model.txt \
             exp_result/LA_RawNet2_baseline_ep100_bs32/eval_scores_using_best_dev_model.txt \
    --labels AASIST AASIST-L RawGAT-ST RawNet2

Usage — pondéré par 1/EER_eval (valeurs de resultats.md) :
  python3 ensemble_eval.py ... --weights 1.2056 1.0084 0.3185 0.2364
"""
import argparse
import json
import numpy as np
from pathlib import Path

from evaluation import calculate_tDCF_EER


def load_scores(path):
    """Retourne un dict {utt_id: (src, key, score)} pour un fichier de scores."""
    data = np.genfromtxt(path, dtype=str)
    result = {}
    for row in data:
        utt_id, src, key, score = row[0], row[1], row[2], float(row[3])
        result[utt_id] = (src, key, score)
    return result


def normalize(scores):
    mu, sigma = scores.mean(), scores.std()
    return (scores - mu) / (sigma + 1e-8)


def main(args):
    assert len(args.scores) >= 2, "Au moins 2 fichiers de scores requis"

    labels = args.labels or [f"model_{i}" for i in range(len(args.scores))]
    assert len(labels) == len(args.scores)

    weights = np.array(args.weights if args.weights else [1.0] * len(args.scores))
    weights /= weights.sum()

    all_dicts = [load_scores(p) for p in args.scores]
    utt_ids = list(all_dicts[0].keys())
    for i, d in enumerate(all_dicts[1:], 1):
        assert set(d.keys()) == set(utt_ids), \
            f"Les utterances de {args.scores[i]} ne correspondent pas au premier fichier"

    print(f"\n{'Modèle':<16} {'mean':>8} {'std':>8} {'poids':>8}")
    print("-" * 44)

    fused = np.zeros(len(utt_ids))
    ref_srcs = np.array([all_dicts[0][u][0] for u in utt_ids])
    ref_keys = np.array([all_dicts[0][u][1] for u in utt_ids])

    for i, (d, label, w) in enumerate(zip(all_dicts, labels, weights)):
        raw = np.array([d[u][2] for u in utt_ids])
        normed = normalize(raw) if args.normalize else raw
        print(f"{label:<16} {normed.mean():>8.3f} {normed.std():>8.3f} {w:>8.3f}")
        fused += w * normed

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for utt_id, src, key, score in zip(utt_ids, ref_srcs, ref_keys, fused):
            f.write(f"{utt_id} {src} {key} {score:.6f}\n")
    print(f"\nScores fusionnés : {output_path}")

    with open(args.config) as f:
        config = json.load(f)
    db = Path(config["database_path"])
    asv_score_file = db / config["asv_score_path"]
    output_txt = output_path.parent / "ensemble_t-DCF_EER.txt"

    eer, tdcf = calculate_tDCF_EER(
        cm_scores_file=output_path,
        asv_score_file=asv_score_file,
        output_file=output_txt,
    )
    print(f"\nEnsemble — EER : {eer:.4f}%   min-tDCF : {tdcf:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True,
                        help="Fichiers de scores à fusionner")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Noms des modèles (pour affichage)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Poids de chaque modèle avant normalisation (défaut : uniforme). "
                             "Suggestion pondérée par 1/EER_eval : 1.2056 1.0084 0.3185 0.2364")
    parser.add_argument("--config", type=str, default="./config/AASIST.conf",
                        help="Config pour database_path et asv_score_path")
    parser.add_argument("--output", type=str,
                        default="./exp_result/ensemble/ensemble_scores.txt")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Désactiver la normalisation avant fusion (déconseillé)")
    parser.set_defaults(normalize=True)
    main(parser.parse_args())
