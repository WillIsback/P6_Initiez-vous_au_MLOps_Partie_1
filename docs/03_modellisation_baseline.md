# 03 — Modélisation : baselines (LogReg & LightGBM)

Notebook : `p6_ml.ipynb`, sections 8 et 9.

## La métrique métier (cœur du projet)

**Règle FN = 10 × FP** : un mauvais client accepté (FN) coûte **10 unités**, un bon client
refusé (FP) coûte **1 unité**. La métrique pilotante est le **coût par client** en validation.

### Optimisation du seuil

- Plutôt que le défaut 0.5, le seuil est optimisé **sur le train** par balayage exact
  (tri des probas + comptage cumulé FN/FP en O(n log n)) puis **appliqué à la validation** —
  aucune fuite : le seuil de décision ne voit jamais la validation.
- Le seuil théorique bayésien avec ces coûts vaut `FP/(FN+FP) = 1/11 ≈ 0,091` **sur une
  probabilité calibrée**. Nos deux modèles sont entraînés avec `class_weight='balanced'` (poids
  des défauts ≈ 0,92/0,08 ≈ 11,4) : leurs scores sont décalés vers le haut, et le seuil théorique
  devient ≈ 0,535 sur ce score pondéré. Mesuré : **0,518** (LogReg) et **0,506** (LightGBM) —
  cohérent avec la théorie. *(Version précédente, LightGBM non pondéré : seuil 0,0888 ≈ 1/11.)*

### Traitement du déséquilibre : `class_weight='balanced'`

- Appliqué aux **deux** baselines (et à tous les LightGBM du pipeline : CV, OpenFE, Boruta, tuning).
- Il compense le 92/8 pendant l'apprentissage ; le seuil optimisé sur le coût métier absorbe
  ensuite le décalage des scores. Conséquence assumée : le score servi **n'est pas une
  probabilité calibrée** (c'est un score de risque pondéré, comparé à un seuil métier figé).
- ⚠️ **Piège rencontré** : avec la pondération, l'early stopping doit surveiller la logloss
  **pondérée** de validation (`eval_class_weight=['balanced']`). Sans cela, la logloss non
  pondérée continue de baisser à mesure que le modèle se recale, l'arrêt ne se déclenche jamais
  (3 000 itérations, AUC train 0,99) et le coût de validation se dégrade (0,583). Corrigé : arrêt
  à 318 itérations, coût 0,487.

### Références (holdout validation)

| Politique | Coût/client |
|---|---|
| Tout refuser (prédire 1 partout) | 0,8073 |
| Tout accorder (prédire 0 partout) | **0,8073 → 0,9193** |

Ces bornes ancrent l'évaluation : tout modèle doit battre ~0,81.

## Choix des baselines

| Modèle | Encodage | Pourquoi |
|---|---|---|
| **Logistic Regression** (C=1, lbfgs, max_iter=1000, `class_weight='balanced'`) | OHE + scaling (221 features) | Référence linéaire, interprétable, rapide — sert de plancher |
| **LightGBM** (3000 arbres, lr=0.05, leaves=31, min_child=20, subsample 0.8, colsample 0.8, `class_weight='balanced'`) | **Ordinal natif** (pas d'OHE) + `early_stopping(50)` sur logloss pondérée | Modèle cible : gère le non-linéaire, le NA natif, et l'ordinal évite la dimensionnalité OHE |

- Seuils optimisés entraînés sur le **train complet** pour l'artefact final ; la CV sert à la
  **stabilité** seulement.
- **Registry MLflow** : `home-credit-logreg` v7, `home-credit-lightgbm` v8 — avec **signature**
  + **input example** + `skops_trusted_types` explicites (MLflow 3.x refuse sinon le chargement skops).

## Résultats (221 features : application + tables secondaires)

### Holdout (validation 61 503 lignes)

| Modèle | ROC-AUC | PR-AUC | Seuil optimal | Coût/client |
|---|---|---|---|---|
| LogReg | 0,7761 | 0,2616 | 0,5181 | 0,5033 |
| **LightGBM** | **0,7880** | 0,2876 | **0,5059** | **0,4873** |

→ LightGBM gagne partout ; rapporté aux bornes : le coût chute de 0,92 (tout accorder) à 0,49,
soit **−47 % de coût vs la politique nulle**. *(Version précédente, application seule :
LogReg 0,7485 / 0,5414 ; LightGBM 0,7585 / 0,5269.)*

### CV 5-fold (stabilité)

| Modèle | ROC-AUC (moy ± écart) | Coût/client (moy ± écart) |
|---|---|---|
| LogReg | 0,7741 ± 0,0019 | 0,5066 ± 0,0033 |
| LightGBM | 0,7844 ± 0,0013 | 0,4946 ± 0,0032 |

→ Écarts-types faibles : la validation aléatoire ne masque pas d'instabilité. Le coût par fold
ré-optimise son seuil **sur le fold d'entraînement** puis mesure sur le fold de validation.

### Artefacts logués (runs MLflow)

`baseline_logreg`, `baseline_lightgbm`, `cv_logreg`, `cv_lightgbm`, `comparaison_baselines` :
courbes **ROC** (`roc_curve_val.png`), **coût vs seuil** (`cost_vs_threshold_val.png`),
performance par fold (`cv_folds_performance.png`), comparaison des deux modèles.

## Limites

- Features « manuelles » uniquement (221, dont 111 agrégats) : les ratios et interactions ne
  sont pas encore explorés systématiquement — cf. section 4 (OpenFE).
- Score pondéré non calibré : pour afficher une vraie probabilité de défaut, il faudrait une
  calibration (isotonic / Platt) en aval.
- Le coût est mesuré en holdout aléatoire ; la dérive temporelle réelle n'est pas testée (pas de
  date, cf. section 6).

## Coût de calcul

LogReg ~40 s ; LightGBM ~30 s ; CV 5-fold LogReg ~2 min, LightGBM ~1 min. Pic RAM ~4 Go.
