# 07 — Serving du modèle (MLflow)

Notebook : `p6_ml.ipynb` (Registry v6 — dernière version du modèle tuné) + `scripts/build_serving_model.py`.

## Objectif

Exposer le **scoreur crédit** en API REST prête pour la production, via l'écosystème MLflow :
`home-credit-lightgbm` **v6** (LightGBM tuné sur 61 features) devient un service **`home-credit-serving` v1**.

## Architecture retenue : wrapper pyfunc {encodeur + modèle + seuil}

Le modèle du Registry v6 attend des **features déjà encodées** (ordinal-encodage des 4
catégorielles) : le servir tel quel repousserait la logique de prétraitement sur le client —
non soutenable (fuite de contrat, dérive potentielle). Le service embarque donc la chaîne
complète dans un **`mlflow.pyfunc.PythonModel`** :

```
requête JSON (61 features brutes)
        │
        ▼
┌─────────────────────────────────────────────┐
│  home-credit-serving (pyfunc)               │
│  1. OrdinalEncoder (fit train, figé)        │
│  2. LightGBM tuned (Registry v6)            │
│  3. Seuil métier figé (val_best_threshold)  │
└─────────────────────────────────────────────┘
        │
        ▼
réponse : { prob_default, decision (1=refuser), threshold }
```

Points clés :
- **Le seuil est figé côté serveur** (optimisé sur le coût métier FN=10×FP) : le client ne peut
  pas le contourner, la décision business est cohérente pour tous les appelsants.
- **`handle_unknown='use_encoded_value'`** : une catégorie inédite en production encode en −1
  au lieu d'échouer (robustesse à la dérive).
- Le **préprocessing ne dérive jamais silencieusement** : l'encodeur est un artefact versionné
  du run MLflow, immuable.

## Lancement

### Développement (à partir du venv du projet)

```bash
mlflow models serve -m 'models:/home-credit-serving/1' -p 5001 --env-manager local
```

> **Piège d'environnement (mesuré)** : `--env-manager local` exécute la commande
> `uvicorn` trouvée sur le PATH. Sans préfixe, un uvicorn système (ex. linuxbrew,
> Python sans mlflow) peut être utilisé → `ModuleNotFoundError: No module named
> 'mlflow'`. Lancer avec :
> `PATH="$PWD/.venv/bin:$PATH" .venv/bin/mlflow models serve ...`

### Production (image autonome pour le sysadmin)

```bash
mlflow models build-docker -m 'models:/home-credit-serving/1' -n home-credit-serving
docker run -p 5001:8080 home-credit-serving
```

L'image embarque Python + lightgbm + scikit-learn + le modèle : **aucune dépendance au serveur
MLflow au runtime** (le Registry n'est requis qu'au *build*).

## Contrat d'API

Consommateurs du service : pipeline batch (notebook), et l'**application Gradio**
(doc 08) qui envoie les 61 features des clients de test depuis sa base SQLite.
Requête (features brutes, mêmes noms que `train_fs.csv` sans `TARGET`) :

```bash
curl http://localhost:5001/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
    "columns": ["<61 colonnes features brutes>"],
    "data": [["< valeurs ... >"]]
  }
}'
```

Réponse :

```json
{"predictions": [{"prob_default": 0.073, "decision": 0, "threshold": 0.098}]}
```

- `decision = 1` ⟺ `prob_default >= threshold` ⟺ **refuser la demande** (règle FN = 10 × FP).
- Test de non-régression : appel curls sur 3 lignes de `val_fs.csv` — comparer `prob_default`
  aux prédictions du notebook (tolérance exacte, déterminisme du modèle).

## DAT — dimensionnement du serving

| Ressource | Développement (dev laptop) | Production (VM dédiée) |
|---|---|---|
| vCPU | 2 | **2 vCPU** (inférence LightGBM mono-thread < 10 ms/req) |
| RAM | ~1 Go | **2-4 Go** (modèle ~10 Mo + overhead serveur) |
| Stockage | venv ~1,5 Go | **image Docker ~2 Go** |
| Latence cible | — | < 10 ms/req (p95) — 61 features, arbre seul |
| Débit | — | > 100 req/s sur 2 vCPU sans batching |
| MLflow server requis ? | oui (local) | **non** au runtime (image autonome) |
| HA | — | redondance ≥ 2 instances derrière un load balancer ; stateless |

Monitoring à prévoir côté infra :
- journalisation des requêtes/réponses (audit crédit : conservation réglementaire),
- alerte sur la **dérive** des distributions des 61 features (PSI) — justification de la
  sélection Boruta : moins de features = moins de points de dérive à surveiller,
- re-fit périodique : relancer sections 11→12 avec les données récentes, pousser une nouvelle
  version du Registry (le seuil est ré-optimisé à chaque version).

## Sécurité

- `mlflow models serve` n'a **aucune authentification** : en prod, placer derrière reverse
  proxy/gateway (mTLS ou clé API) — le service lui-même reste stateless et non exposé directement.
- Le modèle ne retourne que probabilité + décision : **pas de détail discriminant par individu**
  (conformité RGPD/explainability crédit : le SHAP du modèle reste côté batch, dans le notebook).

## Résultats mesurés (rejeu du 24/10/2026)

- **Amorçage** : ~15 s (téléchargement des artefacts depuis le tracking server +
  chargement du pyfunc + démarrage uvicorn).
- **Premier appel** : **HTTP 200 en 283 ms** (setup de connexion inclus) — réponse
  conforme : `{"prob_default": 0.0642, "decision": 0, "threshold": 0.0856}`
  (client `train_fs` correctement accepté : 6,42 % < seuil 8,56 %).
- **Modèle servi** : `home-credit-serving` **v1** (run `build_serving_model`
  `aa877b28`), base `home-credit-lightgbm` **v6**, seuil figé **0.0856**.
- **Build Docker** : chemin documenté ci-dessus, **non exécuté** dans cette session
  (priorité au test local) — prévoir ~5-10 min et ~2 Go supplémentaires si requis.
- Latences steady-state et app Gradio : mesurées dans le doc 08 après lancement
  complet de la chaîne.
