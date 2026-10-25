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
| Chargement `train_fe.csv` (1,4 Go) | ~72 s | 1 thread (I/O) | ~3,2 Go | CSV large (400 colonnes) |
| EDA (section 5) | < 30 s | 1 thread | ~1,5 Go | agrégations pandas |
| Préparation (section 6) | ~1-2 min | 1 thread | ~2,5 Go | imputation, winsorisation, snapshots |
| Baseline LogReg (OHE) | ~20 s | multi | ~2 Go | 110 features, 246k lignes |
| Baseline LightGBM + early stopping | ~30-60 s | 8 threads | ~2 Go | ordinal natif |
| CV 5-fold (×2 modèles) | ~3 min | séquentiel (5 fits) | ~2,5 Go | coût métier par fold |
| **OpenFE** — recherche AutoFE | **220 s** | 8 process | ~3 Go | 5k lignes, ~3k candidats, CPU-borné volontairement |
| OpenFE — transform complet | 147 s | 8 process | ~4 Go | 290 nouvelles features sur 307k lignes |
| Fit LightGBM + OpenFE (400 feat.) | 136 s | 16 threads | ~5 Go | 252 itérations (early stopping) |
| **BorutaShap** — sélection (2 passes : strict diagnostic + détendu) | **~8-20 min** (7,9 min au rejeu du 24/10) | 8 threads | **~3-4 Go** (mesuré 3,2 Go en phase données) | 30k lignes × 400 feat., SHAP par trial ; variable selon la charge machine |
| **Tuning Optuna** (section 12) — 25 essais × 5-fold CV | **~50 min** (43,5 min d'étude + fit final) | 8 threads/fit | ~2,5 Go | **étape la plus longue** ; seuil ré-optimisé par fold |
| Exécution notebook complète (**sections 1→12**) | **~80 min mesuré** (rejeu du 24/10/2026) | variable | pic ~5 Go | `nbconvert --execute` |

Chiffres clés du dimensionnement :
- **Le gouot d'étranglement est le tuning Optuna** (~50 min, 62 % du runtime) devant l'AutoFE OpenFE (~12 min) et BorutaShap (~8-20 min).
- Le pic RAM global observé est **≈ 5 Go** (section 10) — la machine de dev exploite ~17 % de sa RAM.
- Aucune étape n'a besoin de GPU : le pipeline est 100 % CPU, LightGBM/OpenFE/skops.

## 3. DAT — dimensionnement VM cible

### 3.1 VM de reproducibilité (rejeu du pipeline)

| Ressource | Minimum | Recommandé | Justification |
|---|---|---|---|
| vCPU | 6 | **8** (équivalent 5700G) | `n_jobs=8` borné par conception ; < 4 vCPU ferait ×2 le temps d'OpenFE/Boruta |
| RAM | 12 Go | **16 Go** | pic mesuré 5 Go + marge OS/cache ; 8 Go suffirait mais sans marge |
| Stockage | 8 Go (SSD) | **15 Go SSD** | brut ~1 Go + `processed/` ~2,5 Go + `mlflow-artifacts/` ~1 Go + zip + venv |
| Durée totale mesurée / dimensionnée | — | **~1 h 20 mesuré** (rejeu du 24/10/2026, sections 1→12) ; **1,5 h** retenu en marge | Étapes lourdes : tuning Optuna + OpenFE |
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
