"""
Stacking ensemble — régression logistique sur les scores dev des 4 modèles.
Compare l'EER du stacking à celui de l'ensemble pondéré 1/EER.

Protocole :
  - Features = [score_AASIST, score_AASIST-L, score_RawGAT-ST, score_RawNet2] normalisés
  - Labels    = 1 (bonafide) / 0 (spoof) sur le jeu de développement
  - Le modèle entraîné est ensuite appliqué sur l'eval pour calculer EER + tDCF

Usage :
  python3 stacking_ensemble.py \
    --dev_scores  exp_result/LA_AASIST_ep100_bs24/dev_scores_using_best_dev_model.txt \
                  exp_result/LA_AASIST-L_ep100_bs24/dev_scores_using_best_dev_model.txt \
                  exp_result/LA_RawGATST_baseline_ep100_bs24/dev_scores_using_best_dev_model.txt \
                  exp_result/LA_RawNet2_baseline_ep100_bs32/dev_scores_using_best_dev_model.txt \
    --eval_scores exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt \
                  exp_result/LA_AASIST-L_ep100_bs24/eval_scores_using_best_dev_model.txt \
                  exp_result/LA_RawGATST_baseline_ep100_bs24/eval_scores_using_best_dev_model.txt \
                  exp_result/LA_RawNet2_baseline_ep100_bs32/eval_scores_using_best_dev_model.txt \
    --labels AASIST AASIST-L RawGAT-ST RawNet2 \
    --config ./config/AASIST.conf \
    --output exp_result/ensemble/stacking_scores.txt
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from evaluation import calculate_tDCF_EER, compute_eer


def load_scores(path):
    """Retourne (utt_ids, srcs, keys, scores) triés par utt_id."""
    data = np.genfromtxt(path, dtype=str)
    utt_ids = data[:, 0]
    srcs    = data[:, 1]
    keys    = data[:, 2]
    scores  = data[:, 3].astype(float)
    order = np.argsort(utt_ids)
    return utt_ids[order], srcs[order], keys[order], scores[order]


def build_matrix(score_files):
    """Construit la matrice features (N x M) + labels/metadata alignés."""
    all_data = [load_scores(p) for p in score_files]
    utt_ids_ref = all_data[0][0]
    for i, (uids, _, _, _) in enumerate(all_data[1:], 1):
        assert np.array_equal(uids, utt_ids_ref), \
            f"Utterances du fichier {i} ne correspondent pas au fichier 0"

    X = np.column_stack([d[3] for d in all_data])
    srcs = all_data[0][1]
    keys = all_data[0][2]
    y = (keys == "bonafide").astype(int)
    return utt_ids_ref, srcs, keys, X, y


def write_scores(path, utt_ids, srcs, keys, scores):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for uid, src, key, s in zip(utt_ids, srcs, keys, scores):
            f.write(f"{uid} {src} {key} {s:.6f}\n")


def main(args):
    labels = args.labels or [f"model_{i}" for i in range(len(args.dev_scores))]
    assert len(labels) == len(args.dev_scores) == len(args.eval_scores)

    print("Chargement des scores dev...")
    utt_dev, srcs_dev, keys_dev, X_dev, y_dev = build_matrix(args.dev_scores)

    print("Chargement des scores eval...")
    utt_eval, srcs_eval, keys_eval, X_eval, _ = build_matrix(args.eval_scores)

    print(f"Dev  : {len(y_dev)} utterances ({y_dev.sum()} bonafide, {(~y_dev.astype(bool)).sum()} spoof)")

    # EER individuel sur le dev pour référence
    print(f"\n{'Modèle':<16} {'EER dev (%)':>12}")
    print("-" * 30)
    for i, label in enumerate(labels):
        bona = X_dev[y_dev == 1, i]
        spoof = X_dev[y_dev == 0, i]
        eer, _ = compute_eer(bona, spoof)
        print(f"{label:<16} {eer*100:>12.4f}")

    # Entraînement du stacking (LogisticRegression avec normalisation)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=args.C, max_iter=1000, random_state=42)),
    ])
    clf.fit(X_dev, y_dev)

    coefs = clf.named_steps["lr"].coef_[0]
    print(f"\nCoefficients LogReg (après StandardScaler) :")
    for label, c in zip(labels, coefs):
        print(f"  {label:<16} {c:>+.4f}")

    # Application sur l'eval
    # score = P(bonafide) — plus élevé = plus probablement bonafide
    stacking_scores_eval = clf.predict_proba(X_eval)[:, 1]

    output_path = Path(args.output)
    write_scores(output_path, utt_eval, srcs_eval, keys_eval, stacking_scores_eval)
    print(f"\nScores stacking eval sauvegardés : {output_path}")

    # EER + tDCF sur l'eval
    with open(args.config) as f:
        config = json.load(f)
    db = Path(config["database_path"])
    asv_score_file = db / config["asv_score_path"]
    output_txt = output_path.parent / "stacking_t-DCF_EER.txt"

    eer, tdcf = calculate_tDCF_EER(
        cm_scores_file=output_path,
        asv_score_file=asv_score_file,
        output_file=output_txt,
    )
    print(f"\nStacking ensemble — EER : {eer:.4f}%   min-tDCF : {tdcf:.5f}")
    print("\n(Référence : ensemble pondéré 1/EER → EER 0.7342% / tDCF 0.02163)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_scores", nargs="+", required=True,
                        help="Fichiers de scores sur le jeu de développement")
    parser.add_argument("--eval_scores", nargs="+", required=True,
                        help="Fichiers de scores sur le jeu d'évaluation")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Noms des modèles (pour affichage)")
    parser.add_argument("--config", type=str, default="./config/AASIST.conf",
                        help="Config pour database_path et asv_score_path")
    parser.add_argument("--output", type=str,
                        default="./exp_result/ensemble/stacking_scores.txt")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Régularisation LogisticRegression (défaut 1.0)")
    main(parser.parse_args())
