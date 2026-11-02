# 00 — Environnement de dev & dimensionnement (DAT)

Fiche destinée à un administrateur système : elle décrit **la machine de développement**,
**la consommation réelle mesurée** par étape du pipeline, et **le dimensionnement recommandé (DAT)**
d'une VM capable de reproduire ou déployer l'ensemble.

## 1. Machine de développement

| Composant | Valeur |
|---|---|
| CPU | **AMD Ryzen 7 5700G** (Cezanne, Zen 3) |
| Cœurs / threads | 8 cœurs physiques / 16 threads (SMT) |
| Fréquence | 4,0 GHz base — 4,67 GHz boost (scaling mesuré ≈ 93 %) |
| RAM | **30 GiB** DDR4 (double canal) |
| GPU | Aucun utilisé (Radeon intégré ignoré — tout le pipeline est CPU) |
| OS | Ubuntu 24.04.4 LTS, noyau 6.8 |
| Python | 3.11, environnement `uv` (`.venv/`) |
| Serveur de tracking | MLflow 3.x local (`scripts/serve_mlflow.sh`, port 5000, SQLite + artifacts locaux) |

Toutes les charges du projet sont **parallélisées sur CPU** avec `n_jobs=8` (la moitié des
threads logiques) pour laisser de la marge à la machine — contrainte exprimée lors du développement.

## 2. Consommation réelle mesurée (par étape)

Durées observées sur la machine ci-dessus, données réelles du projet (307 511 demandes) :

| Étape | Durée | Parallélisme | Pic RAM estimé | Commentaire |
|---|---|---|---|---|
| Agrégation des 6 tables secondaires (section 5bis) | ~50 s (exploration 31 s + agrégats 21 s) | 1 thread | ~3,3 Go | 57 M lignes d'historique lues |
| EDA (section 5) | < 30 s | 1 thread | ~1,5 Go | agrégations pandas |
| Préparation (section 6) | ~1 min | 1 thread | ~4 Go | imputation, winsorisation, snapshots (221 features) |
| Baseline LogReg (OHE) | ~40 s | multi | ~3 Go | 221 features, 246k lignes |
| Baseline LightGBM + early stopping | ~30-60 s | 8 threads | ~2 Go | ordinal natif |
| CV 5-fold (×2 modèles) | ~3 min | séquentiel (5 fits) | ~3 Go | coût métier par fold |
| **OpenFE** — recherche AutoFE | **262 s** | 8 process | ~3 Go | 5k lignes, 5 676 candidats, CPU-borné volontairement |
| OpenFE — transform + snapshots CSV | 427 s | 8 process | ~6 Go | 382 nouvelles features sur 307k lignes (603 colonnes) |
| Fit LightGBM + OpenFE (603 feat.) | 53 s | 16 threads | ~6 Go | 262 itérations (early stopping pondéré) |
| **BorutaShap** — sélection | **~5 min** (287 s au rejeu du 02/11) | 8 threads | ~4 Go | 30k lignes × 603 feat., SHAP par trial ; variable selon la charge machine |
| **Tuning Optuna** (section 12) — 25 essais × 5-fold CV | **~28 min** (27,3 min d'étude + fit final 49 s) | 8 threads/fit | ~3 Go | **étape la plus longue** ; seuil ré-optimisé par fold |
| Exécution notebook complète (**sections 1→12**) | **~53 min mesuré** (rejeu du 02/11/2026) | variable | pic ~6 Go | exécution cellule par cellule (`nbclient`) |

Chiffres clés du dimensionnement :
- **Le goulot d'étranglement est le tuning Optuna** (~28 min, 52 % du runtime) devant l'AutoFE OpenFE (~12 min) et BorutaShap (~5 min).
- Le pic RAM global observé est **≈ 6 Go** (section 10) — la machine de dev exploite ~20 % de sa RAM.
- Aucune étape n'a besoin de GPU : le pipeline est 100 % CPU, LightGBM/OpenFE/skops.

## 3. DAT — dimensionnement VM cible

### 3.1 VM de reproducibilité (rejeu du pipeline)

| Ressource | Minimum | Recommandé | Justification |
|---|---|---|---|
| vCPU | 6 | **8** (équivalent 5700G) | `n_jobs=8` borné par conception ; < 4 vCPU ferait ×2 le temps d'OpenFE/Boruta |
| RAM | 12 Go | **16 Go** | pic mesuré 5 Go + marge OS/cache ; 8 Go suffirait mais sans marge |
| Stockage | 10 Go (SSD) | **20 Go SSD** | brut ~3,3 Go (zip + 9 CSV) + `processed/` ~4 Go + `mlflow-artifacts/` ~1 Go + venv |
| Durée totale mesurée / dimensionnée | — | **~53 min mesuré** (rejeu du 02/11/2026, sections 1→12) ; **1,5 h** retenu en marge | Étapes lourdes : tuning Optuna + OpenFE |
| Réseau | aucun requis | — | tout est local ; Kaggle only pour le téléchargement initial |

### 3.2 Composants de service (si mise en production du tracking)

| Composant | Recommandation |
|---|---|
| MLflow server | Conteneur dédié, 2 vCPU / 4 Go, base PostgreSQL (au lieu de SQLite), backend artifacts S3/MinIO |
| Notebook | Conteneur Jupyter ou job `nbconvert` planifié (cron CI) plutôt qu'un poste ouvert |
| Secrets / modèle | Registry MLflow + chargement `skops` (types de confiance explicitement listés) |

### 3.3 Notes d'exploitation

- Les jobs longs (OpenFE, BorutaShap) sont **bornés par conception** : sous-échantillons
  (5k / 30k lignes), `n_jobs` limité, fichiers temporaires dirigés vers `/tmp`.
- La reproductibilité est assurée par `pyproject.toml` + `uv.lock` (lightgbm 4.7.0, openfe 0.0.12,
  borutashap 1.0.17, shap) et les seeds fixés (`random_state=42` partout).
- ⚠️ BorutaShap 1.0.17 nécessite deux shims de compatibilité (scipy ≥ 1.12 : `binom_test` ;
  numpy ≥ 2.0 : `np.NaN`) — voir `notebooks/p6_ml.ipynb` section 11.1.
