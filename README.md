# Détection de deepfakes vocaux — Projet Intégrateur 2A

Validation et extension du modèle [AASIST](https://arxiv.org/abs/2110.01200) (Jung et al., ICASSP 2022) sur le dataset ASVspoof 2019 LA.

**Aristide Bailly — Télécom Paris, 2025–2026**

---

## Résultats principaux

| Modèle | EER (%) | min-tDCF |
|--------|---------|----------|
| AASIST (poids officiels) | 0,83 | 0,0275 |
| AASIST-L (poids officiels) | 0,99 | 0,0309 |
| AASIST from scratch | 1,34 | 0,0387 |
| RawGAT-ST from scratch | 1,67 | 0,0546 |
| RawNet2 from scratch | 4,23 | 0,1130 |
| **Fusion pondérée (×4 modèles)** | **0,73** | **0,0216** |

Robustesse au bruit gaussien (AASIST officiel) : 1,02% à 30 dB → 63,1% à 0 dB.

---

## Structure du repo

```
├── main.py / evaluation.py / data_utils.py / utils.py   # code principal
├── models/          # architectures (AASIST, AASIST-L, RawGAT-ST, RawNet2)
├── config/          # configs des 11 expériences
├── exp_result/      # résultats : t-DCF_EER.txt, metric_log.txt, figures
│   └── ensemble/    # scores de fusion (uniforme et pondérée)
├── rapport/         # rapport final (.tex + .pdf)
│
├── breakdown_analysis.py   # EER par type d'attaque
├── det_curve.py            # courbes DET
├── training_curves.py      # courbes d'entraînement
├── ensemble_eval.py        # fusion de scores
├── distillation.py         # distillation AASIST → AASIST-L
├── noise_robustness.py     # robustesse au bruit gaussien
└── *.sh                    # jobs SLURM (cluster GPU Télécom Paris)
```

## Installation

```bash
pip install -r requirements.txt
```

Dataset : [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) — à placer dans `./LA/`.

## Utilisation

```bash
# Entraînement
python main.py --config ./config/AASIST.conf

# Évaluation (poids officiels)
python main.py --eval --config ./config/AASIST.conf

# Fusion de scores
python ensemble_eval.py

# Robustesse au bruit
python noise_robustness.py --model_path ./models/weights/AASIST.pth --config ./config/AASIST.conf
```

## Expériences

| Expérience | EER (%) | ΔEER vs baseline |
|-----------|---------|-----------------|
| AASIST from scratch (référence) | 1,34 | — |
| + frequency augmentation | 1,14 | −0,19 |
| + label smoothing | 1,36 | +0,02 |
| + scheduler adaptatif | 1,43 | +0,09 |
| + noise augmentation | 1,62 | +0,28 |
| Fine-tuning + noise aug | 1,22 | +0,39 vs officiel |
| Distillation AASIST→AASIST-L | 1,39 | +0,40 vs AASIST-L scratch |

## Référence

> Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, ICASSP 2022.

Code officiel original : [clovaai/aasist](https://github.com/clovaai/aasist)
