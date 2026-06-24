# Résultats — Projet Intégrateur AASIST

Dataset : ASVspoof 2019 LA (eval set)
Métriques : EER (Equal Error Rate) et min-tDCF (tandem Detection Cost Function)

---

## 1. Reproduction des poids officiels

| Modèle | EER (%) | min-tDCF | Résultat papier | Verdict |
|--------|---------|----------|-----------------|---------|
| AASIST | 0.8295 | 0.02753 | 0.83% / 0.0275 | ✅ Reproduit |
| AASIST-L | 0.9917 | 0.03090 | 0.99% / 0.0309 | ✅ Reproduit |

---

## 2. Entraînements from scratch

Valeurs rapportées = best checkpoint (best.pth), pas le modèle SWA final.

| Modèle | EER (%) | min-tDCF | Best eval epoch | Notes |
|--------|---------|----------|-----------------|-------|
| AASIST | **1.335** | **0.03871** | ep98 | ✅ job 858484 (22/06/2026) |
| AASIST-L | **0.9917** | **0.0309** | ep76 | ✅ Correspond au papier |
| RawGAT-ST | **1.67** | **0.0546** | ep85 | ✅ SWA : 1.672% / 0.0551 (légère amélioration) |
| RawNet2 | 4.23 | 0.113 | ep54 | ✅ SWA : 4.46% / 0.1288 (dégradation — checkpoints ep70-88 sur-apprennent le dev) |

---

## 3. Breakdown par attaque

### 3a. From scratch (RawNet2, RawGAT-ST, AASIST-L)

| Attaque | RawNet2 | RawGAT-ST | AASIST-L |
|---------|---------|-----------|----------|
| A09 | 0.14% | 0.02% | 0.00% |
| A13 | 0.24% | 0.30% | 0.20% |
| A14 | 0.63% | 0.59% | 0.06% |
| A11 | 1.08% | 0.55% | 0.14% |
| A16 | 0.86% | 1.26% | 0.63% |
| A19 | 1.75% | 0.94% | 0.71% |
| A08 | 3.13% | 0.42% | 0.20% |
| A07 | 2.76% | 1.53% | 0.55% |
| A10 | 2.67% | 1.96% | 1.00% |
| A12 | 2.77% | 1.67% | 0.69% |
| A15 | 2.50% | 1.10% | 0.57% |
| A17 | 8.08% | 2.18% | 1.92% |
| A18 | **14.26%** | 4.70% | 3.05% |

**Observations :**
- A09 / A13 : attaques les plus faciles pour tous les modèles
- A17 / A18 : attaques les plus difficiles (vocoders neuronaux / TTS sophistiqués)
- RawNet2 s'effondre sur A17 (8%) et A18 (14%)
- RawGAT-ST (EER global 1.67%) — homogène sauf sur A18 (4.70%)
- AASIST-L est le plus homogène et robuste sur toutes les attaques

### 3b. AASIST poids officiels

| Attaque | AASIST (officiel) |
|---------|-------------------|
| A09 | 0.00% |
| A13 | 0.15% |
| A14 | 0.16% |
| A11 | 0.18% |
| A08 | 0.42% |
| A07 | 0.53% |
| A15 | 0.55% |
| A16 | 0.65% |
| A19 | 0.65% |
| A12 | 0.71% |
| A10 | 0.86% |
| A17 | 1.26% |
| A18 | **2.61%** |

EER global : 0.8295%

---

## 4. Fusion de scores (ensemble)

| Fusion | EER (%) | min-tDCF |
|--------|---------|----------|
| Uniforme (poids égaux) | 0.8703 | 0.02710 |
| **Pondérée (1/EER_eval)** | **0.7342** | **0.02163** |

Poids pondérés : AASIST 0.395 / AASIST-L 0.331 / RawGAT-ST 0.196 / RawNet2 0.078
Gain vs meilleur modèle seul (AASIST 0.8295%) : **+0.10% EER, -21% tDCF**

### Breakdown ensemble pondéré

| Attaque | Ensemble pondéré |
|---------|-----------------|
| A09 | 0.00% |
| A13 | 0.06% |
| A14 | 0.08% |
| A11 | 0.10% |
| A15 | 0.30% |
| A08 | 0.24% |
| A07 | 0.35% |
| A16 | 0.47% |
| A19 | 0.49% |
| A12 | 0.42% |
| A10 | 0.67% |
| A17 | **1.18%** |
| A18 | **2.46%** |

Gain ensemble vs AASIST-L from scratch sur les attaques difficiles : A17 1.92%→1.18%, A18 3.05%→2.46%

---

## 5. Robustesse au bruit (AASIST)

✅ **job 858449 COMPLETED (21/06/2026, 53 min)** — modèle : poids officiels AASIST.pth, snr_levels 30 20 10 5 0, seed 1234, GPU.

| SNR (dB) | EER (%) |
|----------|---------|
| Propre | 0.8295 |
| 30 dB | 1.0170 |
| 20 dB | 4.2014 |
| 10 dB | 9.3644 |
| 5 dB | 25.8977 |
| 0 dB | 63.1003 |

Dégradation brutale : ×5 entre propre et 20 dB, quasi-aléatoire (63%) à 0 dB. Expliqué par l'absence totale de bruit dans ASVspoof 2019 LA — le modèle apprend des signatures spectrales fines masquées par le bruit additif.

---

## 6. Freq augmentation

Masquage aléatoire de bandes fréquentielles pendant l'entraînement (flag `freq_aug: True`).
Entraînements from scratch sur 100 epochs, mêmes hyperparamètres que les baselines.

| Modèle | EER (%) | min-tDCF | Δ EER vs baseline |
|--------|---------|----------|-------------------|
| AASIST (officiel, référence) | 0.8295 | 0.02753 | — |
| AASIST + freq_aug (from scratch) | **1.142** | 0.03646 | +0.31% |
| AASIST-L (from scratch, référence) | 0.9917 | 0.0309 | — |
| AASIST-L + freq_aug (from scratch) | **1.102** | 0.03583 | +0.11% |

**Conclusion** : freq_aug dégrade légèrement l'EER global mais améliore la robustesse sur A18 (AASIST-L no-aug 3.05% → freq_aug 2.64%).

### Breakdown freq_aug

| Attaque | AASIST+aug | AASIST-L+aug | AASIST-L no-aug |
|---------|------------|--------------|-----------------|
| A09 | 0.00% | 0.04% | 0.00% |
| A13 | 0.15% | 0.15% | 0.20% |
| A14 | 0.29% | 0.34% | 0.06% |
| A11 | 0.34% | 0.30% | 0.14% |
| A08 | 0.39% | 0.31% | 0.20% |
| A07 | 1.06% | 0.96% | 0.55% |
| A15 | 0.83% | 1.04% | 0.57% |
| A16 | 0.73% | 0.63% | 0.63% |
| A19 | 0.61% | 0.65% | 0.71% |
| A12 | 0.75% | 0.94% | 0.69% |
| A10 | 1.24% | 1.10% | 1.00% |
| A17 | 1.81% | 1.91% | 1.92% |
| A18 | 2.99% | **2.64%** | 3.05% |

### Historique best checkpoints freq_aug

**AASIST + freq_aug :** ep18 (1.70%), ep19, ep22, ep28 (1.59%), ep34 (1.50%), ep51, ep62
**AASIST-L + freq_aug :** ep18 (1.63%), ep30 (1.29%), ep31, ep55, ep61, ep62 (1.09%), ep73 (0.95%)

## 7. Historique entraînement RawNet2 (best checkpoints)

| Epoch (best dev) | dev EER (%) | eval EER (%) | Commentaire |
|------------------|-------------|--------------|-------------|
| 28 | 0.900 | 5.83 | premier best |
| 40 | 0.825 | 5.06 | |
| 44 | 0.825 | 4.69 | |
| 46 | 0.785 | 4.62 | |
| 54 | 0.708 | **4.23** | ← meilleur eval |
| 70 | 0.708 | 5.11 | ← dev stable, eval se dégrade |
| 73 | 0.628 | 4.53 | |
| 82 | 0.551 | 4.65 | |
| 88 | 0.551 | 4.46 | best dev, pire eval |
| Final SWA | — | **4.231** | SWA ≈ niveau ep54 |

**Observation clé** : après ep54, dev EER continue de baisser (0.708→0.551%) mais eval EER se dégrade (4.23→5.11%). Divergence dev/eval claire = sur-ajustement du dev set. Le SWA lisse les poids et retrouve le niveau ep54.

## 8. Label smoothing, noise augmentation & scheduler plateau

Entraînements from scratch sur 100 epochs, AASIST baseline, mêmes hyperparamètres sauf la modification testée.

*⚠️ Valeurs corrigées le 16/06 — les chiffres précédents de label_smooth et scheduler_plateau étaient erronés (relevés à partir de `t-DCF_EER.txt` sur le cluster).*

| Expérience | EER (%) | min-tDCF | Δ EER vs baseline AASIST (0.83%) | Job SLURM |
|-----------|---------|----------|----------------------------------|-----------|
| AASIST (référence officielle) | 0.8295 | 0.02753 | — | — |
| `label_smooth` (from scratch) | **1.360** | 0.04516 | +0.53 pt ❌ | 849825 |
| `noise_aug` (from scratch) | **1.618** | 0.05401 | +0.79 pt ❌ | 849826 |
| `scheduler_plateau` (from scratch) | **1.425** | 0.04647 | +0.60 pt ❌ | 849827 |

**Conclusion** : les trois modifications dégradent les performances. La baseline cosine scheduler sans augmentation reste optimale sur ASVspoof 2019 LA.

### Breakdown par attaque

| Attaque | label_smooth | noise_aug | scheduler_plateau | AASIST (baseline) |
|---------|-------------|-----------|--------------------|--------------------|
| A09 | 0.105% | 0.261% | 0.098% | 0.00% |
| A13 | 0.326% | 0.587% | 0.302% | 0.15% |
| A14 | 0.180% | 0.489% | 0.343% | 0.16% |
| A11 | 0.448% | 0.954% | 0.465% | 0.18% |
| A08 | 0.750% | 1.083% | 0.570% | 0.42% |
| A07 | 1.124% | 1.002% | 1.199% | 0.53% |
| A19 | 0.693% | 0.937% | 0.652% | 0.65% |
| A16 | 1.362% | 0.913% | 0.832% | 0.65% |
| A12 | 1.368% | 1.321% | 1.222% | 0.71% |
| A10 | 1.891% | 1.752% | 1.606% | 0.86% |
| A15 | 1.083% | 1.222% | 1.019% | 0.55% |
| A17 | 1.548% | 1.623% | 1.467% | 1.26% |
| A18 | **6.512%** | **4.703%** | **3.602%** | **2.61%** |

**Observation clé** : les trois modifications dégradent *fortement* la robustesse sur A18, l'attaque la plus difficile — l'effet est le plus marqué pour `label_smooth` (×2.5 par rapport à la baseline). Cela contredit l'hypothèse initiale que le label smoothing améliorerait la généralisation sur attaques inconnues : ici il semble plutôt diluer le signal d'apprentissage sur les cas difficiles.

---

## 9. Fine-tuning et distillation

### 9.1 Fine-tuning AASIST + noise_aug (depuis best.pth)

| Modèle | EER (%) | min-tDCF | Job SLURM | Notes |
|--------|---------|----------|-----------|-------|
| AASIST fine-tuné (30 ep, LR 1e-5, noise_aug) | **1.155** | 0.0358 | 849829 | ❌ Dégrade vs baseline (0.83%) — SWA final ; 3 tentatives ont échoué avant (849810, 849816, 849828) |

Le fine-tuning depuis le best checkpoint avec noise augmentation n'apporte pas de gain — cohérent avec le résultat `noise_aug` from scratch (1.618%), qui dégrade également la baseline.

### 9.2 Distillation AASIST → AASIST-L

✅ **Job 856672 — COMPLETED (21/06/2026, 16h15)** — teacher = poids officiels AASIST, student = AASIST-L from scratch, α=0.5, T=4, 100 epochs.

| Modèle | EER (%) | min-tDCF |
|--------|---------|----------|
| AASIST-L from scratch (référence) | 0.9917 | 0.03090 |
| **AASIST-L distillé** | **1.390** | **0.03826** |

La distillation dégrade la performance vs from scratch (+0.40 pt EER). Résultats dans `exp_result/LA_AASIST_distill_ep100_bs24/t-DCF_EER.txt`.

#### Breakdown distillation vs AASIST-L from scratch

| Attaque | Distillé | From scratch |
|---------|----------|--------------|
| A09 | 0.017 % | 0.00 % |
| A13 | 0.227 % | 0.20 % |
| A14 | 0.081 % | 0.06 % |
| A11 | 0.105 % | 0.14 % |
| A08 | 0.791 % | 0.20 % |
| A07 | 0.424 % | 0.55 % |
| A15 | 0.390 % | 0.57 % |
| A16 | 1.100 % | 0.63 % |
| A19 | 1.362 % | 0.71 % |
| A12 | 0.587 % | 0.69 % |
| A10 | 0.611 % | 1.00 % |
| A17 | 2.136 % | 1.92 % |
| A18 | **3.562 %** | **3.05 %** |

---

## Fichiers utiles sur le cluster

```
~/aasist/exp_result/
├── LA_AASIST_ep100_bs24/
├── LA_AASIST-L_ep100_bs24/
│   ├── eval_scores_using_best_dev_model.txt
│   ├── eval_scores_using_best_dev_model_breakdown.png
│   └── weights/best.pth
├── LA_RawGATST_baseline_ep100_bs24/
│   ├── eval_scores_using_best_dev_model.txt
│   └── eval_scores_using_best_dev_model_breakdown.png
└── LA_RawNet2_baseline_ep100_bs32/
    ├── eval_scores_using_best_dev_model.txt
    └── eval_scores_using_best_dev_model_breakdown.png
```
