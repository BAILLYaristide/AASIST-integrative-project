"""
Test de robustesse au bruit gaussien — AASIST.
Évalue l'EER pour différents niveaux de SNR.
Usage : python3 noise_robustness.py --config ./config/AASIST.conf --snr_levels 0 5 10 20 30
"""
import argparse
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from models.AASIST import Model
from evaluation import compute_eer
from data_utils import pad


class NoisyEvalDataset(Dataset):
    def __init__(self, protocol_path, audio_dir, snr_db=None, max_len=64600):
        self.samples = []
        self.snr_db = snr_db
        self.max_len = max_len
        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                _, utt_id, _, src, key = parts
                self.samples.append((utt_id, src, key))
        self.audio_dir = Path(audio_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, src, key = self.samples[idx]
        path = self.audio_dir / "flac" / f"{utt_id}.flac"
        x, _ = sf.read(str(path), dtype="float32")
        x = pad(x, self.max_len)
        if self.snr_db is not None:
            signal_power = np.mean(x ** 2) + 1e-10
            noise_power = signal_power / (10 ** (self.snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), x.shape).astype(np.float32)
            x = x + noise
        return torch.tensor(x), src, key


def evaluate(model, loader, device):
    model.eval()
    all_scores, all_keys, all_srcs = [], [], []
    with torch.no_grad():
        for batch_x, srcs, keys in loader:
            batch_x = batch_x.to(device)
            _, out = model(batch_x)
            scores = out[:, 1].cpu().numpy()
            all_scores.extend(scores)
            all_keys.extend(keys)
            all_srcs.extend(srcs)
    scores = np.array(all_scores)
    keys = np.array(all_keys)
    bona = scores[keys == "bonafide"]
    spoof = scores[keys == "spoof"]
    eer, _ = compute_eer(bona, spoof)
    return eer * 100


def main(args):
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(config["model_config"]).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    print(f"Modèle chargé : {config['model_path']}")

    db = Path(config["database_path"])
    protocol = db / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    audio_dir = db / "ASVspoof2019_LA_eval"

    snr_levels = [None] + args.snr_levels  # None = pas de bruit

    print(f"\n{'SNR (dB)':<12} {'EER (%)':>10}")
    print("-" * 25)
    for snr in snr_levels:
        dataset = NoisyEvalDataset(protocol, audio_dir, snr_db=snr)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        eer = evaluate(model, loader, device)
        label = "Propre" if snr is None else f"{snr} dB"
        print(f"{label:<12} {eer:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/AASIST.conf")
    parser.add_argument("--snr_levels", type=int, nargs="+", default=[30, 20, 10, 5, 0])
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
