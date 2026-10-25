# 03 — Modélisation : baselines (LogReg & LightGBM)

Notebook : `p6_ml.ipynb`, sections 8 et 9.

## La métrique métier (cœur du projet)

**Règle FN = 10 × FP** : un mauvais client accepté (FN) coûte **10 unités**, un bon client
refusé (FP) coûte **1 unité**. La métrique pilotante est le **coût par client** en validation.

### Optimisation du seuil

- Plutôt que le défaut 0.5, le seuil est optimisé **sur le train** par balayage exact
  (tri des probas + comptage cumulé FN/FP en O(n log n)) puis **appliqué à la validation** —
  aucune fuite : le seuil de décision ne voit jamais la validation.
- Le seuil théorique bayésien avec ces coûts vaut `FP/(FN+FP) = 1/11 ≈ 0,091` : en pratique le
  seuil optimal LightGBM mesuré est **0,0888**, en accord remarquable avec la théorie — les
  scores du modèle sont bien calibrés en région basse.

### Références (holdout validation)

| Politique | Coût/client |
|---|---|
| Tout refuser (prédire 1 partout) | 0,8073 |
| Tout accorder (prédire 0 partout) | **0,8073 → 0,9193** |

Ces bornes ancrent l'évaluation : tout modèle doit battre ~0,81.

## Choix des baselines

| Modèle | Encodage | Pourquoi |
|---|---|---|
| **Logistic Regression** (C=1, lbfgs, max_iter=1000) | OHE (110 → ~240 features), scaling | Référence linéaire, interprétable, rapide — sert de plancher |
| **LightGBM** (3000 arbres, lr=0.05, leaves=31, min_child=20, subsample 0.8, colsample 0.8) | **Ordinal natif** (pas d'OHE) + `early_stopping(50)` | Modèle cible : gère le non-linéaire, le NA natif, et l'ordinal évite la dimensionnalité OHE |

- Seuils optimisés entraînés sur le **train complet** pour l'artefact final ; la CV sert à la
  **stabilité** seulement.
- **Registry MLflow** : `home-credit-logreg` v2, `home-credit-lightgbm` v2 — avec **signature**
  + **input example** + `skops_trusted_types` explicites (MLflow 3.x refuse sinon le chargement skops).

## Résuitats

### Holdout (validation 61 503 lignes)

| Modèle | ROC-AUC | PR-AUC | Seuil optimal | Coût/client |
|---|---|---|---|---|
| LogReg | 0,7485 | — | 0,5223 | 0,5414 |
| **LightGBM** | **0,7585** | 0,2490 | **0,0888** | **0,5269** |

→ LightGBM gagne partout ; rapporté aux bornes : le coût chute de 0,92 (tout accorder) à 0,53,
soit **−43 % de coût vs la politique nulle**.

### CV 5-fold (stabilité)

| Modèle | ROC-AUC (moy ± écart) | Coût/client (moy ± écart) |
|---|---|---|
| LogReg | 0,7454 ± 0,0031 | 0,5462 ± 0,0074 |
| LightGBM | 0,7555 ± 0,0019 | 0,5340 ± 0,0078 |

→ Écarts-types faibles : la validation aléatoire ne masque pas d'instabilité. Le coût par fold
ré-optimise son seuil **sur le fold d'entraînement** puis mesure sur le fold de validation.

### Artefacts logués (runs MLflow)

`baseline_logreg`, `baseline_lightgbm`, `cv_logreg`, `cv_lightgbm`, `comparaison_baselines` :
courbes **ROC** (`roc_curve_val.png`), **coût vs seuil** (`cost_vs_threshold_val.png`),
performance par fold (`cv_folds_performance.png`), comparaison des deux modèles.

## Limites

- Features brutes uniquement (110) : le potentiel des ratios métier (credit/income…) n'est pas
  exploité — cf. section 4 (OpenFE).
- Le coût est mesuré en holdout aléatoire ; la dérive temporelle réelle n'est pas testée (pas de
  date, cf. section 6).

## Coût de calcul

LogReg ~20 s ; LightGBM ~30-60 s ; les deux CV 5-fold ~3 min au total. Pic RAM ~2,5 Go.
