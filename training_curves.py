"""
Plot dev EER (all epochs via TensorBoard) and eval EER (best-dev epochs) for each model.

Usage:
    python3 training_curves.py
    python3 training_curves.py --output exp_result/training_curves.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

EXP_MODELS = [
    ("AASIST",             "LA_AASIST_ep100_bs24"),
    ("AASIST-L",           "LA_AASIST-L_ep100_bs24"),
    ("RawGAT-ST",          "LA_RawGATST_baseline_ep100_bs24"),
    ("RawNet2",            "LA_RawNet2_baseline_ep100_bs32"),
    ("AASIST+freq_aug",    "LA_AASIST_freq_aug_ep100_bs24"),
    ("AASIST-L+freq_aug",  "LA_AASIST-L_freq_aug_ep100_bs24"),
    ("AASIST+noise_aug",   "LA_AASIST_noise_aug_ep100_bs24"),
    ("AASIST+label_smooth","LA_AASIST_label_smooth_ep100_bs24"),
    ("AASIST+scheduler",   "LA_AASIST_scheduler_plateau_ep100_bs24"),
]

EER_RE = re.compile(r"EER\s*=\s*([\d.]+)")


def parse_eer_file(path: Path):
    try:
        text = path.read_text()
        m = EER_RE.search(text)
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def load_tb_dev(exp_dir: Path):
    """Lit dev_eer depuis les fichiers TensorBoard events.*"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None

    event_files = sorted(exp_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None

    epochs, eers = [], []
    for ef in event_files:
        ea = EventAccumulator(str(ef))
        ea.Reload()
        if "dev_eer" not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars("dev_eer"):
            epochs.append(int(e.step))
            eers.append(e.value * 100)  # stocké en fraction, converti en %

    if not epochs:
        return None, None

    # trie et déduplique par epoch
    pairs = sorted(set(zip(epochs, eers)))
    epochs, eers = zip(*pairs)
    return np.array(epochs), np.array(eers)


def load_eval_curve(metric_dir: Path):
    """Lit eval EER aux epochs où le dev s'est amélioré."""
    eval_epochs, eval_eers = [], []
    for f in sorted(metric_dir.glob("t-DCF_EER_*epo.txt")):
        m = re.search(r"(\d+)epo", f.name)
        if m:
            eer = parse_eer_file(f)
            if eer is not None:
                eval_epochs.append(int(m.group(1)))
                eval_eers.append(eer)
    if not eval_epochs:
        return None, None
    return np.array(eval_epochs), np.array(eval_eers)


def load_curves(exp_dir: Path):
    metric_dir = exp_dir / "metrics"
    dev_ep, dev_eer = load_tb_dev(exp_dir)
    eval_ep, eval_eer = load_eval_curve(metric_dir) if metric_dir.exists() else (None, None)
    return dev_ep, dev_eer, eval_ep, eval_eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="./exp_result")
    parser.add_argument("--output",   default="exp_result/training_curves.png")
    parser.add_argument("--models",   nargs="*")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)

    models = [(name, tag) for name, tag in EXP_MODELS
              if args.models is None or name in args.models]
    available = [(name, tag) for name, tag in models
                 if (exp_root / tag).exists()]

    if not available:
        print("Aucun dossier trouvé dans", exp_root)
        return

    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, (name, tag) in zip(axes, available):
        dev_ep, dev_eer, eval_ep, eval_eer = load_curves(exp_root / tag)

        has_dev  = dev_ep  is not None and len(dev_ep)  > 0
        has_eval = eval_ep is not None and len(eval_ep) > 0

        if not has_dev and not has_eval:
            ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name)
            continue

        if has_dev:
            ax.plot(dev_ep, dev_eer, color="steelblue", linewidth=1.2,
                    label="Dev EER (toutes epochs)")

        if has_eval:
            ax.scatter(eval_ep, eval_eer, color="tomato", s=30, zorder=5,
                       label="Eval EER (best dev epochs)")
            ax.plot(eval_ep, eval_eer, color="tomato", linewidth=0.8,
                    linestyle="--", alpha=0.6)

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("EER (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle("Courbes d'entraînement — Dev EER et Eval EER par epoch",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
