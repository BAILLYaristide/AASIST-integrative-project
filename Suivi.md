# Suivi — Projet Intégrateur 2A : AASIST Audio Anti-Spoofing

## Contexte
Projet intégrateur 2A à Télécom Paris.
Sujet : reproduction et exploration du modèle AASIST (détection de deepfakes vocaux).
Papier : Jung et al., ICASSP 2022 — [arXiv 2110.01200](https://arxiv.org/abs/2110.01200)
Dépôt personnel : https://github.com/BAILLYaristide/AASIST-integrative-project
Cluster : `abailly-24@gpu-gw.enst.fr` (partition 3090, nœuds nodemm01/node40)

---

## Fait ✅

### Mise en place
- Clonage du dépôt officiel `clovaai/aasist`, remote changé vers dépôt personnel
- Code copié sur le cluster : `~/aasist/`, dataset ASVspoof 2019 LA dans `~/aasist/LA/`
- Dépendances installées via `module load miniconda3/25.5.1` + `pip` (dont matplotlib)

### Corrections de bugs
- `evaluation.py` : `np.float` → `float` (supprimé dans NumPy 2.0)
- `data_utils.py:53` — `pad_random` : `np.random.randint(0)` levait `ValueError` si fichier exactement 64600 samples → `stt = 0`
- `main.py:168` — `log_text` : loggait chaque epoch au lieu des seules epochs avec amélioration → `log_text = ""`
- `noise_robustness.py` — seed manquant : `np.random.seed(args.seed)` + argument `--seed` ajouté
- `breakdown_analysis.py:23` — types d'attaques hard-codés → `np.unique(sources[keys == "spoof"])`
- `config/RawGATST_baseline.conf` et `config/RawNet2_baseline.conf` — `model_path` pointait vers serveur interne NAVER (inaccessible) → mis à jour vers `./exp_result/.../weights/best.pth` (10/06/2026)

### Améliorations du code d'entraînement
- `main.py` : gradient clipping ajouté (`nn.utils.clip_grad_norm_`, max_norm=1.0)
- Nouveaux configs : `config/AASIST_freq_aug.conf` et `config/AASIST-L_freq_aug.conf`
- `main.py` : label smoothing ajouté (ε = 0.1) — activé via clé `"label_smoothing"` dans le config (rétrocompatible : absent = comportement original)
- Nouveau config : `config/AASIST_label_smooth.conf` — AASIST + label smoothing ε=0.1, 100 epochs → **job 849802** (lancé 12/06/2026, PD)
- `data_utils.py` : noise augmentation ajoutée dans `Dataset_ASVspoof2019_train` — bruit gaussien aléatoire SNR ∈ [10, 40] dB, activé via clé `"noise_aug": true` dans le config (rétrocompatible)
- `main.py` : passage de `noise_aug` depuis le config vers le dataset (ligne `get_loader`)
- Nouveau config : `config/AASIST_noise_aug.conf` — AASIST + noise augmentation, 100 epochs
- `utils.py` : scheduler `reduce_on_plateau` ajouté (`ReduceLROnPlateau`, patience=5, factor=0.5)
- `main.py` : appel `scheduler.step(dev_eer)` au niveau epoch pour `reduce_on_plateau`
- Nouveau config : `config/AASIST_scheduler_plateau.conf` — AASIST + ReduceLROnPlateau, 100 epochs
- `config/AASIST_finetune_noise.conf` — fine-tuning depuis best.pth AASIST, 30 epochs, LR 1e-5, noise_aug=true
- `finetune_job.sh` — job SLURM dédié au fine-tuning (passe `--resume` vers best.pth)
- `distillation.py` — distillation AASIST (teacher gelé) → AASIST-L (student) : loss = α×CE + (1-α)×KL, T=4, α=0.5
- `config/AASIST_distill.conf` — teacher AASIST best.pth, student AASIST-L from scratch, 100 epochs
- `distill_job.sh` — job SLURM dédié distillation

### Nouveaux scripts produits (ajout 24/06/2026)
- `training_curves.py` — courbes dev EER (toutes epochs) + eval EER (epochs best dev) pour chaque modèle. Usage : `python3 training_curves.py` → `exp_result/training_curves.png`

### Nouveaux scripts produits
- `breakdown_analysis.py` — EER par type d'attaque + graphique
- `noise_robustness.py` — EER vs SNR (bruit gaussien)
- `ensemble_eval.py` + `ensemble_job.sh` — fusion des scores des 4 modèles
- `det_curve.py` — courbe DET multi-modèles
- Job scripts Slurm : `eval_job.sh`, `train_job.sh`, `resume_job.sh`, `analysis_job.sh`

### Évaluations — poids officiels
| Modèle | EER (%) | min-tDCF | Résultat papier | Statut |
|--------|---------|----------|-----------------|--------|
| AASIST | 0.8295 | 0.02753 | 0.83% / 0.0275 | ✅ Reproduit |
| AASIST-L | 0.9917 | 0.03090 | 0.99% / 0.0309 | ✅ Reproduit |

### Entraînements from scratch
| Modèle | EER (%) | min-tDCF | Notes |
|--------|---------|----------|-------|
| AASIST | **1.335** | **0.03871** | ✅ best ep98, job 858484 (22/06/2026) |
| AASIST-L | **0.9917** | **0.0309** | ✅ best ep76, égale le papier |
| RawNet2 | 4.231 | 0.1134 | ✅ 100 epochs, best eval ep54 (best dev ep88, sans gain eval après ep54) |
| RawGAT-ST | **1.67** | **0.0546** | ✅ Complet — best.pth ep85, EER corrigé (était noté 3.14% par erreur) |

### Analyse et exploration
- Breakdown EER par attaque A07–A19 pour les 3 modèles ✅
  - A09/A13 faciles — A17/A18 difficiles (vocoders neuronaux)
  - RawNet2 s'effondre sur A18 (14.26%) — AASIST-L le plus homogène
  - Graphiques PNG dans chaque dossier `exp_result/`
- Courbe DET 4 modèles générée ✅ → `exp_result/det_curve_no_ensemble.png`
- Courbe DET avec ensemble générée ✅ → `exp_result/det_curve.png`
- Breakdown AASIST poids officiels récupéré ✅ (A18: 2.61%)
- Breakdown ensemble pondéré récupéré ✅ (A17: 1.18%, A18: 2.46%)
- Historique training RawNet2 documenté ✅ — overfitting dev/eval après ep54

### Freq augmentation
| Modèle | EER (%) | min-tDCF | Δ vs baseline |
|--------|---------|----------|---------------|
| AASIST + freq_aug | 1.142 | 0.03646 | +0.31% |
| AASIST-L + freq_aug | 1.102 | 0.03583 | +0.11% |
- Légèrement dégradé globalement, mais A18 amélioré : 3.05% → 2.64% pour AASIST-L ✅
- **Interprétation ajoutée au rapport (24/06/2026)** : A17/A18 sont les attaques les plus récentes (vocoders neuronaux). Dans un contexte où les menaces réelles sont de plus en plus dominées par ce type de générateurs, la freq aug pourrait représenter un meilleur équilibre en conditions réelles — légère dégradation sur les attaques simples, gain ciblé sur les plus difficiles. À confirmer sur plusieurs seeds et datasets plus récents (ASVspoof 5, In-the-Wild).

### Rapport
- V1 rédigée ✅ → `rapport_v1.md` (en cours d'édition manuelle par Aristide, ne pas écraser)
- **Restructuration complète en `rapport_v2.md`** ✅ (16/06/2026) — nouveau plan type article scientifique : résumé, introduction (contexte/gap/contributions), travaux connexes, méthode, protocole expérimental (+ remarques d'implémentation), résultats et discussion (avec étude d'ablation explicite sur les 5 variantes d'entraînement), limitations (section dédiée), conclusion et perspectives, annexes (breakdowns détaillés, historique RawNet2)
  - Intègre toutes les données à jour de `resultats.md` (label_smooth/noise_aug/plateau corrigés, fine-tuning, statut distillation)
  - Reformule l'intro pour distinguer "validation des poids officiels d'AASIST" (jamais ré-entraîné from scratch) de "reproduction de l'entraînement" (AASIST-L/RawGAT-ST/RawNet2)
  - Sections 5.6 (bruit) et 5.7 (distillation) encore incomplètes — jobs 853480/853490 en cours
  - **Reste à faire** : décider si `rapport_v2.md` remplace `rapport_v1.md` ou si les deux sont fusionnés à la main

---

## Jobs terminés depuis le 12/06/2026 (vérifié 16/06/2026)

| Job | Nom | Statut | Résultat |
|-----|-----|--------|----------|
| 849825 | aasist_train | ✅ terminé | label_smooth — EER 1.360% (corrigé, voir resultats.md §8) |
| 849826 | aasist_train | ✅ terminé | noise_aug — EER 1.618% |
| 849827 | aasist_train | ✅ terminé | scheduler_plateau — EER 1.425% (corrigé) |
| 849810 / 849816 / 849828 | aasist_finetune | ❌ échecs successifs | logs minimalistes — relancés |
| 849829 | aasist_finetune | ✅ terminé | EER 1.224%, tDCF 0.0383 |
| 849817 | aasist_distill | ❌ échec | `FileNotFoundError` sur teacher_path — training jamais lancé |
| 850485 | aasist_analysis | ⚠️ partiel | breakdown AASIST OK ; robustesse au bruit cancelled TIME LIMIT (4h) |

---

## 🐛 Bugs trouvés et corrigés le 16/06/2026

- **`analysis_job.sh` ne demandait pas de GPU** (`--gres=gpu:1` absent) → `noise_robustness.py` tournait sur CPU → bien trop lent pour tenir en 4h sur ~71k échantillons × 6 passages SNR. Explique les 2 échecs successifs (845307, 850485). **Corrigé** : relancé en sbatch direct avec `--gres=gpu:1` → **job 853480** (16/06).
- **`exp_result/LA_AASIST_ep100_bs24/weights/` est vide** — ce dossier n'a en réalité jamais contenu d'entraînement from scratch d'AASIST (juste une évaluation des poids officiels rangée sous ce nom). C'est pour ça que `teacher_path = exp_result/.../best.pth` était introuvable lors du job 849817.
- **Fix distillation** : `teacher_path` du config cluster pointe maintenant vers `./models/weights/AASIST.pth` (poids officiels, vérifiés présents : `models/weights/AASIST.pth`, 1.28 Mo, 03/06). Relancé → **job 853490** (16/06).

### Jobs en cours (16/06/2026)
| Job | Nom | Description |
|-----|-----|-------------|
| 853480 | noise_robust | Robustesse au bruit AASIST, snr_levels 30 20 10 5 0, avec GPU |
| 853490 | aasist_distill | Distillation AASIST (teacher=poids officiels) → AASIST-L (student), α=0.5, T=4 |

---

## Reste à faire 🔲

### Résultats — tout récupéré ✅
- [x] Robustesse au bruit — job 858449 COMPLETED (21/06/2026) : propre 0.83%, 30dB 1.02%, 20dB 4.20%, 10dB 9.36%, 5dB 25.90%, 0dB 63.10%
- [x] Distillation — job 856672 COMPLETED (21/06/2026) : EER 1.390%, tDCF 0.03826
- [x] AASIST from scratch — job 858484 COMPLETED (22/06/2026) : EER 1.335%, tDCF 0.03871, best ep98

### Rapport
- [x] Section 4.6 (robustesse au bruit) — complétée ✅
- [x] Section 4.7 (distillation) — complétée ✅
- [x] Section 4.4 (variantes) — complétée ✅ ; tableaux AASIST + AASIST-L fusionnés en un seul (24/06/2026)
- [x] Section 4.2 (from scratch) — AASIST from scratch ajouté ✅
- [x] Résumé — mis à jour (plus de mention "en cours") ✅
- [x] Conclusion — mise à jour ✅
- [ ] Insérer figures : `det_curve.png` + breakdown PNG (marquer chemins dans le rapport)
- [x] Section 2.3 noise aug — précision ajoutée : bruit gaussien blanc, densité spectrale plate ✅ (24/06/2026)
- [x] Section 4.4 noise aug — commentaire résultats ajouté : dataset propre, écart train/test, bruit blanc masque les artefacts ✅ (24/06/2026)
- [x] Section 2.3 + 4.4 — label smoothing et scheduler réduits à une note courte, retirés des tableaux ✅ (24/06/2026)
- [x] Résumé, intro, liste contributions, limitations, conclusion — label smoothing et scheduler retirés de toutes ces sections (gardés uniquement comme note en section 4.4) ✅ (24/06/2026)
- [ ] Relecture et mise en forme finale

### Scripts
- [x] `stacking_ensemble.py` — créé ✅
- [x] `distillation.py` — créé ✅

### Présentation
- [ ] Préparer slides (15 min)

### Divers
- [ ] Envoyer mail au prof (format de rendu, deadline, fork ou diff pour le code)
- [ ] Nettoyer git (historiques divergents cluster/GitHub)

---

## Commandes utiles sur le cluster

```bash
# Connexion
ssh abailly-24@gpu-gw.enst.fr

# Voir ses jobs
squeue -u abailly-24

# Logs d'un job
cat ~/aasist/aasist_analysis_JOBID.log
cat ~/aasist/aasist_analysis_JOBID.err

# Annuler un job
scancel JOBID

# Lancer analysis (breakdown + robustesse)
sbatch ~/aasist/analysis_job.sh

# Lancer ensemble
sbatch ~/aasist/ensemble_job.sh

# Lancer DET curve (pas de GPU, instantané)
cd ~/aasist && python3 det_curve.py \
  --scores exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt \
           exp_result/LA_AASIST-L_ep100_bs24/eval_scores_using_best_dev_model.txt \
           exp_result/LA_RawGATST_baseline_ep100_bs24/eval_scores_using_best_dev_model.txt \
           exp_result/LA_RawNet2_baseline_ep100_bs32/eval_scores_using_best_dev_model.txt \
  --labels AASIST AASIST-L RawGAT-ST RawNet2 \
  --ensemble_score exp_result/ensemble/weighted_scores.txt \
  --output exp_result/det_curve.png
```
