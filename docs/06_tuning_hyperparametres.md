# 06 — Tuning modèle & hyperparamètres

Notebook : `p6_ml.ipynb`, section 12 *(exécuté le 02/11/2026 — version avec tables secondaires
et pondération LightGBM)*.

## Plan appliqué

0. **Jeu de features : `train_fs.csv` / `val_fs.csv` (122 features Boruta)** — et non les 603
   features OpenFE. AUC proche (0,7866 vs 0,7895), modèle 5× plus compact : entretien,
   interprétabilité (contexte réglementaire crédit), latence d'inférence, budget d'entraînement
   et **surface de dérive** (moins de features à surveiller). Prix de ce choix sur le holdout :
   +0,010/client avant tuning (cf. doc 05).
1. **Optuna** (TPE sampler, `seed=42`) sur les hyperparamètres LightGBM : `learning_rate`,
   `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`,
   `reg_alpha`, `reg_lambda` (log-espaces où pertinent) ; `n_estimators` = 3 000 + early stopping.
2. **Objectif** : **minimiser le coût métier moyen en CV 5-fold** (chaque fold ré-optimise son
   seuil sur le fold d'entraînement, mesure sur le fold de validation — aucune fuite).
3. **Pondération** : `class_weight='balanced'` dans chaque essai ; early stopping sur la logloss
   **pondérée** (`eval_class_weight=['balanced']`, cf. doc 03).
4. **Budget CPU** : 25 essais × 5 fits, `n_jobs=8` par fit, early stopping 50 par fold.
5. **Artefacts** : run MLflow `tuning_optuna` (params, coût CV, historique d'optimisation,
   importances des hyperparamètres), modèle final au **Model Registry**.
6. **Évaluation finale** : une seule fois sur `application_test.csv` (toujours inviolé).

## Résultats (mesurés, rejeu du 02/11/2026)

**Étude Optuna** : 25 essais × 5-fold CV, **27,3 min** (1 641 s). Run MLflow `tuning_optuna` :
`63e372ca58df4f09b38758c21f9dcf9e`.

| | Coût CV (objectif) | AUC CV | AUC validation | PR-AUC | Coût validation | Seuil optimal |
|---|---|---|---|---|---|---|
| **Tuned** | **0,4899** | 0,7861 ± 0,0024 | **0,7893** | 0,2887 | **0,4814** | **0,5348** |
| Non tuné (mêmes 122 features) | — | — | 0,7866 | 0,2844 | 0,4962 | 0,4881 |
| *Version précédente (application seule, 61 features, non pondéré)* | *0,5211* | *0,7638* | *0,7662* | *0,2579* | *0,5220* | *0,0856* |

Gain : **−0,0148/client** vs le modèle non tuné, et **−0,0406/client (−7,8 %)** vs le modèle
final de la version précédente. C'est le meilleur coût de tout le pipeline (il bat aussi les 603
features OpenFE non tunées, 0,4861).

**Meilleurs hyperparamètres** :

```
learning_rate     = 0.0169      subsample        = 0.777
num_leaves        = 28          colsample_bytree = 0.625
max_depth         = 3           reg_alpha        = 9.21
min_child_samples = 34          reg_lambda       = 0.035
class_weight      = balanced
best iteration (fit final)      = 2 983 / 3 000
```

Lecture : arbres **très peu profonds (3)**, `learning_rate` faible et **forte régularisation L1**
(`reg_alpha` ≈ 9,2) — le modèle préfère beaucoup d'arbres faibles et parcimonieux, ce qui limite
le surapprentissage sur les combos OpenFE redondants. **Point d'attention** : 2 983 itérations
sur 3 000 — la borne `n_estimators` est presque atteinte ; la relever (5 000) est une piste
d'amélioration simple.

**Seuil métier** : 0,5348 — très proche du seuil théorique ≈ 0,535 attendu pour un score
pondéré `balanced` (odds × 11,4) avec FN = 10 × FP. La pondération décale le score, le seuil
optimisé l'absorbe : la **décision** reste celle qui minimise le coût métier.

**Registry** : `home-credit-lightgbm` **v9** (signature + input example + description). Le
script `scripts/build_serving_model.py` résout dynamiquement la dernière version et produit
`home-credit-serving` **v2** (seuil figé 0,5348).

## Captures MLflow

- Liste des runs du pipeline (tag `data_scope`) : [`img/mlflow_01_runs_pipeline.png`](img/mlflow_01_runs_pipeline.png)
- Comparaison des runs (coût, AUC, seuil) : [`img/mlflow_02_comparaison_runs.png`](img/mlflow_02_comparaison_runs.png)
- Détail du run `tuning_optuna` : [`img/mlflow_03_run_tuning_optuna.png`](img/mlflow_03_run_tuning_optuna.png)
- Model Registry : [`img/mlflow_05_model_registry.png`](img/mlflow_05_model_registry.png)

## Étape suivante

Évaluation finale sur `application_test.csv` (toujours inviolé à ce stade), puis serving
(`docs/07`) et app Gradio (`docs/08`).
