# 06 — Tuning modèle & hyperparamètres

Notebook : `p6_ml.ipynb`, section 12 *(exécuté le 24/10/2026)*.

## Plan arrêté (appliqué tel quel)

0. **✅ Décision tranchée : tuning sur le jeu sélectionné `train_fs.csv` / `val_fs.csv` (61 features)**
   — et non sur les 400 features d'OpenFE. Justification (principe de simplicité) : à performance
   quasi équivalente (AUC 0,7647 vs 0,7656 ; coût 0,5229 vs 0,5187), le modèle plus simple est
   supérieur sur **tous** les coûts d'exploitation : entretien, interprétabilité (contexte
   réglementaire crédit), latence d'inférence, budget d'entraînement (~5× plus rapide par fold →
   budget Optuna plus large) et **surface de dérive** (moins de features à surveiller en
   production). Le léger écart de coût (+0,0042/client) est le prix assumé de cette simplicité.
1. **Optuna** (TPE sampler) pour optimiser les hyperparamètres LightGBM :
   `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`,
   `max_depth`, `n_estimators` (borne haute + early stopping).
2. **Objectif** : **minimiser le coût métier moyen en CV 5-fold** (chaque fold ré-optimise son
   seuil sur le fold d'entraînement, mesure sur le fold de validation — aucune fuite).
3. **Jeu de features** : `train_fs.csv` / `val_fs.csv` (**61 features Boruta détendu** — cf. point 0).
4. **Budget CPU** : ~25 essais Optuna bornés, `n_jobs=8` par fit, pruners
   (`MedianPruner`) sur les fits intermédiaires.
5. **Artefacts** : study Optuna + run MLflow `tuning_optuna` (meilleurs params, coût CV),
   puis ré-entraînement final du meilleur modèle enregistré au **Model Registry**
   (`home-credit-lightgbm` v3) avec signature + seuil optimal figé.
6. **Évaluation finale** : une seule fois sur `application_test.csv` (toujours inviolé à ce stade).

## Résultats (mesurés, rejeu du 24/10/2026)

**Étude Optuna** : TPE sampler, 25 essais × 5-fold CV, **43,5 min** (2 611 s),
`n_jobs=8` par fit, early stopping 50 par fold, seuil ré-optimisé par fold
(coût métier moyen = objectif). Run MLflow `tuning_optuna` :
`2bab0b5ba06f435fa50be2de9c12ca72`.

| | Coût CV (objectif) | AUC CV | AUC validation | PR-AUC | Coût/validation | Seuil optimal |
|---|---|---|---|---|---|---|
| **Tuned (essai 18)** | **0.5211** | 0.7638 ± 0.0015 | **0.7662** | 0.2579 | **0.5220** | **0.0856** |
| Untuned (réf. 61 feat.) | 0.5229 | 0.7647 | 0.7647 | — | 0.5229 | 0.0888 |

Gain : **−0.0009 / client** sur la validation (et AUC +0.0015) malgré un espace de
recherche volontairement étroit — le modèle 61 features était déjà proche de son
plafond, ce qui **confirme le choix de simplicité** (l'écart résiduel avec le
400 features OpenFE non tuné, 0.5187, tombe à +0.0033/client).

**Meilleurs hyperparamètres** (essai 18) :

```
learning_rate     = 0.0107      subsample        = 0.783
num_leaves        = 54          colsample_bytree = 0.957
max_depth         = 4           reg_alpha        = 0.0157
min_child_samples = 166         reg_lambda       = 0.948
best iteration (fit final)      = 2 525
```

Lecture : arbres **peu profonds (4)** et très régularisés (`min_child_samples=166`,
lr faible 0,0107, 2 525 itérations) — cohérent avec un jeu réduit de 61 features
déjà fortes : le modèle préfère l'average de nombreux arbres faibles à des arbres
complexes, limitant le surapprentissage sur les combos redondantes.

**Convergence** : la courbe de l'étude stagne après l'essai 18 (essais 19-24 sans
gain) → le budget de 25 essais était suffisant ; TPE a épuisé l'espace utile.

**Registry** : le modèle final est enregistré dans `home-credit-lightgbm` —
**version 6** (les versions 3-5 ont été consommées par les re-exécutions des
sections 9-11 ; l'impression « v3 » affichée par le notebook est un libellé codé
en dur). Signature + input example logués ; le script
`scripts/build_serving_model.py` résout dynamiquement la dernière version.

**Seuil métier** : 0.0856 — à nouveau très proche du seuil bayésien théorique
1/(1+10) ≈ 0,091 (régime FN = 10 × FP), ce qui valide la cohérence de
l'optimisation.

## Étape suivante

Évaluation finale sur `application_test.csv` (toujours inviolé à ce stade), puis
serving (`docs/07`) et app Gradio (`docs/08`).
