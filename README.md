# P6 — Initiez-vous au MLOps (Partie 1) : scoring crédit Home Credit

Pipeline MLOps complet de bout en bout dans un seul notebook exécutable —
EDA → préparation → baselines (LogReg / LightGBM) → feature engineering automatisé
(OpenFE) → feature selection (BorutaShap) → tuning Optuna (coût métier FN = 10 × FP)
→ MLflow Model Registry → serving pyfunc → app Gradio d'inférence :

- Notebook : [`notebooks/p6_ml.ipynb`](notebooks/p6_ml.ipynb)
- Documentation détaillée (choix + application, en français) : [`docs/`](docs/README.md)

## Installation

```bash
git clone <ce-dépôt> && cd P6_Initiez-vous_au_MLOps_Partie_1
uv sync                 # Python 3.11 + toutes les dépendances (uv.lock figé)
```

Prérequis : [uv](https://docs.astral.sh/uv/), ~5 Go de RAM libres, aucun GPU.

## Données

Les données (~3 Go) ne sont **pas versionnées** : le notebook télécharge et
extrait lui-même le jeu Home Credit au premier run (voir
[`data/README.md`](data/README.md) pour la reconstitution complète depuis un
clone vierge).

## Temps d'exécution du notebook

**~80 min au total (1 h 20)** — mesuré sur un rejeu complet
(`jupyter nbconvert --execute`, du 24/10/2026), horodatages extraits des cellules.

Configuration de mesure (hôte de dev) : **AMD Ryzen 7 5700G, 8 cœurs / 16 threads,
30 Go RAM, Ubuntu 24.04 — aucun GPU**. Toute la parallélisation est bornée à
`n_jobs=8` (moitié des cœurs), conformément à la contrainte CPU du projet.

| Section | Contenu | Temps | Part |
|---|---|---:|---:|
| 1 → 4 | Setup, imports, chargement des CSV | ~1 min | 1 % |
| 5 | EDA exploratoire (pandas) | 0,4 min | 1 % |
| 6 | Préparation des données (sentinelles, clip, imputation) | 1,6 min | 2 % |
| 8 | Entraînement & Model Registry (baselines) | 2,6 min | 3 % |
| 9 | Cross-validation 5-fold & comparaison | 5,2 min | 7 % |
| 10 | Feature engineering automatisé (OpenFE) | 11,7 min | 15 % |
| 11 | Feature selection — filtre variance + BorutaShap | 7,9 min | 10 % |
| 12 | Tuning Optuna (25 essais × 5-fold CV) | **49,7 min** | **62 %** |
| **Total** | | **~80 min** | **100 %** |

À savoir avant de lancer :
- **Le tuning Optuna domine le runtime (62 %)** — 25 essais × 5 fits avec
  early stopping sur les 61 features sélectionnées. OpenFE (~12 min) et
  BorutaShap (~8 min) sont les deux autres étapes gourmandes.
- C'est un pipeline **CPU-bound** : prévoir ~5 Go de RAM libres, ne pas
  lancer un autre travail lourd en parallèle, et compter ~1 h 30 pour un
  premier rejeu (marge d'IO et de premier chargement des CSV).
- Le serveur MLflow doit être actif avant de démarrer (sinon les cellules
  d'enregistrement échouent) :

```bash
./scripts/serve_mlflow.sh                 # 1) serveur MLflow sur :5000
.venv/bin/jupyter lab notebooks/p6_ml.ipynb  # 2) Run All (~80 min)
```

Chronique indicative du coût de calcul par étape et dimensionnement de la VM de
déploiement : [`docs/00_environnement_et_dimensionnement.md`](docs/00_environnement_et_dimensionnement.md)
(règle DAT du projet).

## Phase inférence (après le notebook)

```bash
.venv/bin/python scripts/build_serving_model.py   # wrapper pyfunc -> Registry home-credit-serving
# le PATH doit privilégier uvicorn du venv (voir docs/07 — piège documenté) :
PATH="$PWD/.venv/bin:$PATH" .venv/bin/mlflow models serve \
    -m 'models:/home-credit-serving/1' -p 5001 --env-manager local
.venv/bin/python scripts/build_inference_db.py    # base SQLite des 48 744 clients de test (~5 min)
.venv/bin/python apps/gradio_app.py               # app Gradio sur :7860
```

- Serving du modèle (wrapper pyfunc encodeur + LightGBM + seuil) : [`docs/07_serving.md`](docs/07_serving.md)
- App Gradio + base SQLite d'inférence + traçage « Prod Inference » : [`docs/08_gradio_inference.md`](docs/08_gradio_inference.md)
