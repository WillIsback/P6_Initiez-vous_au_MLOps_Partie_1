# 08 — Application Gradio (inférence interactive)

Notebook : `p6_ml.ipynb` · Scripts : `scripts/build_inference_db.py`, `apps/gradio_app.py`.

## Objectif

Interface de démonstration « inférence production » :
1. **Parcourir** les 48 744 clients du fichier test (jamais observés à l'entraînement),
2. **Prédire** sur le client sélectionné via le **serving MLflow natif** (endpoint
   `/invocations` du wrapper pyfunc `home-credit-serving`),
3. **Décrypter** la décision (SHAP waterfall + commentaires),
4. **Tracer** chaque prédiction dans MLflow côté **inférence** (tag `stage=prod_inference`),
   distinct des runs d'entraînement du notebook.

## Évolution de spec : pas de bouton « ground truth »

Vérification faite à la construction de la base : **`application_test.csv` ne contient
pas de colonne `TARGET`** (121 colonnes, identifiant seul — protocole Kaggle : les
labels du set de test ne sont jamais fournis). Le bouton de révélation « ground
truth » est donc **retiré du cahier des charges** — impossible à honnêtement
implementer sur ce fichier. (Réintroduction possible plus tard en mode « audit » sur
`val_fs.csv`, qui, lui, porte une cible.)

## Architecture

```
┌───────────────────────┐        ┌──────────────────────────────┐
│  apps/gradio_app.py   │        │  mlflow models serve :5001   │
│  navigation + UI      │───────▶│  pyfunc {encodeur ordinal +  │
│  (:7860)              │ requête│   LightGBM tuned + seuil}    │
└──────────┬────────────┘  JSON  └──────────────┬───────────────┘
           │ lecture paginée                    │ charge
           ▼                                    ▼
┌───────────────────────┐        ┌──────────────────────────────┐
│  data/inference/      │        │  MLflow Registry             │
│  clients.db (SQLite)  │        │  home-credit-serving v1      │
│  48 744 × 61 features │        │  (build : scripts/)          │
└───────────────────────┘        └──────────────┬───────────────┘
           ▲                                    │ 1 run par prédiction
           │ construit par                      ▼
┌───────────────────────┐        ┌──────────────────────────────┐
│ build_inference_db.py │        │  MLflow :5000                │
│ bundle → OpenFE →     │        │  runs tag stage=prod_inference│
│ Boruta (rejeu figé)   │        └──────────────────────────────┘
└───────────────────────┘
```

## Pourquoi une base SQLite préparée en offline (et pas du « à la volée »)

Le cahier des charges demandait « le meilleur et le plus élégant par défaut » :
**préparation offline + base légère** l'emporte sur le calcul à la volée :

| Critère | Offline → SQLite | À la volée par requête |
|---|---|---|
| Latence par client | ~1 ms (SELECT) | 2-5 s (OpenFE transform complet) |
| Cohérence | Features identiques à l'entraînement (mêmes bundles figés) | Recalcul risqué (dérive de versions) |
| Mémoire app | ~30 Mo (SQLite mappé) | charge complète du train comme référence OpenFE |
| Pattern industrie | « feature store » léger, versionné | — |

Le script `build_inference_db.py` rejoue **exactement** la chaîne du notebook :

1. **Bundle de préparation** (section 6) : sentinelle `DAYS_EMPLOYED` → NaN, `DAYS_*`
   → années, colonnes supprimées, winsorisation aux bornes train, imputation
   médiane/mode — le tout figé par `preprocessing_bundle.joblib` ;
2. **Banc OpenFE** (section 10) : la recherche est **régénérée avec les mêmes
   graines** (5 000 lignes, `seed=42`, candidats restreints) puis **mise en cache**
   (`openfe_bank.joblib` + formules JSON lisibles) — pas de rejeu inutile au prochain
   build ;
3. **Vérifications intégrées** (échec = arrêt, jamais de base empoisonnée) :
   - préparation répliquée comparée à `train/val_prepared.csv` (≥ 99,9 % de cellules
     conformes, tolérance round-trip CSV) ;
   - banc OpenFE comparé aux snapshots `train/val_fe.csv` (les 290 colonnes autoFE
     doivent se reproduire à l'identique) ;
4. **Sélection Boruta** : les 61 features confirmées (`feature_selection_boruta.csv`)
   sont extraites ; toute absence = arrêt explicite ;
5. **Écriture SQLite** : `clients(SK_ID_CURR INTEGER PRIMARY KEY, …61 features…)`,
   catégorielles brutes en TEXT, numériques en REAL, NULL = valeur absente
   (LightGBM gère nativement les NaN résiduels des features autoFE).

## Traçage « Prod Inference » dans MLflow

Chaque clic sur « 🔮 Prédire » crée un **run MLflow** :

- nom `prod_inference_gradio` ;
- tags : `stage=prod_inference`, `source=gradio_app`, `serving_uri`, `model_uri` ;
- params : `sk_id_curr` ;
- metrics : `prob_default`, `decision`, `threshold`, `latency_ms`.

Dans l'UI MLflow (experiment `home-credit-scoring`) : filtrer par tag
`stage=prod_inference` — ces runs sont **nettement séparés des runs d'entraînement**
(tagués `stage=training` côté notebook). Le serveur de scoring lui-même est le
serving natif MLflow (`mlflow models serve`) : l'inférence Gradio ne passe **que**
par l'endpoint REST, jamais par un chargement local du modèle.

## Décryptage des résultats

Le volet SHAP reconstruit la décision du modèle, feature par feature :
- **waterfall** des 8 principales contributions (log-odds) ;
- les features `autoFE_f_i` opaques sont **retraduites par leur formule lisible**
  (fichiers `openfe_bank_formulas.json`, ex. `autoFE_f_91 = (AMT_CREDIT/AMT_ANNUITY)`) ;
- lecture métier : coût attendu selon la règle **FN = 10 × FP**, comparaison au seuil
  bayésien théorique 1/11 ≈ 9,1 %.

## Lancement (3 process)

```bash
# 1) serveur MLflow (si absent)
./scripts/serve_mlflow.sh
# 2) serving du modèle (wrapper pyfunc)
mlflow models serve -m 'models:/home-credit-serving/1' -p 5001 --env-manager local
# 3) app Gradio
.venv/bin/python apps/gradio_app.py          # http://<host>:7860
# test headless de la chaîne complète :
.venv/bin/python apps/gradio_app.py --selftest
```

La construction de la base (une fois, ~5 min au premier build dont 240 s de
recherche OpenFE, ~1 min ensuite grâce au cache du banc) :
`.venv/bin/python scripts/build_inference_db.py`

## DAT — dimensionnement (règle permanente)

Hôte de dev (fiche `00_environnement_et_dimensionnement.md`) : AMD Ryzen 7 5700G,
8 C/16 T, 30 Go RAM, Ubuntu 24.04.

| Étape | CPU | RAM pic | Durée mesurée |
|---|---|---|---|
| `build_inference_db.py` (1er build, recherche OpenFE incluse) | 8 threads (OpenFE) | ~7 Go | **~5 min** (recherche 240 s + 44 s) |
| `build_inference_db.py` (rejeu, banc en cache) | 8 threads | **6,8 Go mesuré** (`/usr/bin/time -v` : 7 138 836 Ko) | **44-66 s mesuré** |
| App Gradio — navigation (SELECT SQLite) | 1 thread | ~0,3 Go (lecture seule) | **~0,0 ms/ sélection** (mesuré) |
| App Gradio — prédiction (HTTP serving) | 1 thread | ~0,5 Go (modèle chargé) | **16-25 ms** (1er appel 102 ms, selftest 3 clients) |
| App Gradio — décryptage SHAP (waterfall inclus) | 1 thread | + ~0,3 Go (TreeExplainer) | **120 ms/client** (mesuré sur 20 clients) |
| Transformation OpenFE (290 features, 307k lignes) | 8 threads | — | **~4 s** en mémoire (le coût du notebook venait de l'écriture CSV) |

VM de déploiement (inférence seule, sans build) : **2 vCPU / 2-4 Go / 3 Go SSD**
(3 process : serveur de scoring + app Gradio + SQLite ; ports 5001 et 7860 à
exposer derrière un reverse proxy ; le serveur MLflow n'est requis qu'au chargement
initial du modèle). Le build de la base peut rester en CI/batch : il ne dépend pas
de la VM de serving.

## Limites connues & évolutions

- **Un client réellement nouveau** (hors fichier test préparé) exigerait la chaîne
  complète embarquée (bundle → banc OpenFE en cache → sélection → encodeur) — le
  fichier de formules JSON est là pour ça, mais le mode « à la volée » n'est pas
  implémenté (choix : le batch offline est le mode de production de référence) ;
- **Pas d'authentification** sur Gradio ni sur `mlflow models serve` : en prod,
  passer derrière un reverse proxy avec clé API (cf. doc 07) ;
- Ground truth : réintroduire le bouton exigerait un jeu étiqueté — option « audit »
  possible sur `val_fs.csv` (61 503 clients, cible connue, jamais vus par le
  gradient).
