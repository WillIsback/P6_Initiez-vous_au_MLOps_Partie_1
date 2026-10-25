# Données (ignorées par git — ~3 Go)

Ce dossier **n'est pas versionné** (3 Go de CSV). Il se reconstitue intégralement
avec le pipeline du projet :

| Chemin | Origine | Taille |
|---|---|---|
| `raw/home-credit-default-risk.zip` | téléchargé automatiquement par la **cellule 2 du notebook** (URL OpenClassrooms S3) | ~690 MB |
| `raw/application_train.csv`, `raw/application_test.csv`, `raw/HomeCredit_columns_description.csv` | extraction sélective (cellule 3 du notebook) | ~200 Mo |
| `processed/*_prepared.csv`, `*_fe.csv`, `*_fs.csv`, `preprocessing_bundle.joblib`, `openfe_bank*.joblib/json`, `feature_selection_boruta.csv` | sections 6, 10 et 11 du notebook | ~2,1 Go (dont 1,4 Go pour `train_fe.csv`) |
| `inference/clients.db`, `inference/features.json` | `scripts/build_inference_db.py` | 26 Mo |

## Reconstitution depuis un clone vierge

1. `uv sync` (Python 3.11)
2. Ouvrir `notebooks/p6_ml.ipynb` et faire un **Run All** (~1 h 20, voir la
   section « Temps d'exécution » du README racine) — le zip est téléchargé et
   extrait automatiquement, tous les fichiers `processed/` sont régénérés
3. Pour la phase inférence (optionnel) :
   `scripts/build_serving_model.py` puis `scripts/build_inference_db.py`

> Si l'URL du cours n'est plus valide, récupérer le jeu
> [Home Credit Default Risk sur Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
> et placer les CSV dans `raw/` (le reste de la chaîne est inchangé).
