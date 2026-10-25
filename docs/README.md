# Documentation — Pipeline MLOps Home Credit Scoring

Documentation du projet de scoring crédit (Home Credit), structurée par étape du pipeline.
Chaque document explicite **les choix méthodologiques**, **leur application concrète**
(paramètres réels du projet), **les résultats obtenus**, **les limites** et **le coût de calcul**
(voir [00 — Environnement & dimensionnement](00_environnement_et_dimensionnement.md)).

## Pipeline

| # | Étape | Document | Statut |
|---|-------|----------|--------|
| 0 | Environnement de dev & dimensionnement (DAT) | [00](00_environnement_et_dimensionnement.md) | ✅ |
| 1 | Exploration des données (EDA) | [01](01_exploration_des_donnees.md) | ✅ |
| 2 | Préparation des données | [02](02_preparation_des_donnees.md) | ✅ |
| 3 | Modélisation — baselines (LogReg & LightGBM) | [03](03_modellisation_baseline.md) | ✅ |
| 4 | Feature engineering automatisé (OpenFE) | [04](04_feature_engineering.md) | ✅ |
| 5 | Feature selection (filtre variance + BorutaShap) | [05](05_feature_selection.md) | ✅ |
| 6 | Tuning modèle & hyperparamètres | [06](06_tuning_hyperparametres.md) | ✅ |
| 7 | Serving du modèle (MLflow pyfunc + Docker) | [07](07_serving.md) | ✅ |
| 8 | Application Gradio (inférence interactive + traçage) | [08](08_gradio_inference.md) | ✅ |

## Artefacts principaux

- **Notebook** : [`notebooks/p6_ml.ipynb`](../notebooks/p6_ml.ipynb) — pipeline complet, sections 1 à 11.
- **Données traitées** : `data/processed/` — snapshots `*_prepared.csv` (base), `*_fe.csv` (features OpenFE),
  `*_fs.csv` (features sélectionnées), `preprocessing_bundle.joblib`.
- **Tracking** : MLflow (experiment `home-credit-scoring`) — runs, métriques métier, artifacts (ROC, coût/seuil),
  Model Registry (`home-credit-logreg`, `home-credit-lightgbm`).
- **Feature selection** : `data/processed/feature_selection_boruta.csv` — décision par feature
  (`confirmed` / `rejected` / `tentative_kept` / `removed_variance`).

## Règle métier centrale

Le coût métier est **FN = 10 × FP** (un mauvais client accepté coûte 10× un bon client refusé).
Toutes les décisions de pipeline (seuil, sélection, tuning) sont guidées par le
**coût par client en validation**, jamais par l'accuracy.

## Données

- `application_train.csv` : 307 511 demandes, 122 colonnes, cible `TARGET` (8,07 % de défauts).
- `application_test.csv` : 48 744 demandes — **jamais utilisé** avant l'évaluation finale (set inviolé).
- Source : Home Credit Default Risk (Kaggle), chargée via `data/home-credit-default-risk.zip`.
