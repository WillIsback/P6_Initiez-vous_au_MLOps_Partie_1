# 04 — Feature engineering automatisé (OpenFE)

Notebook : `p6_ml.ipynb`, section 10.

## Objectif

Les baselines n'utilisent que les **110 features brutes**. Or en scoring crédit, l'information
est souvent dans les **ratios** (`AMT_CREDIT / AMT_INCOME_TOTAL` = charge de crédit,
`DAYS_EMPLOYED / DAYS_BIRTH` = ancienneté relative). Plutôt que d'écrire ces ratios à la main,
on fait **découvrir** les features pertinentes par une librairie SOTA d'AutoFE.

## Choix de la librairie : OpenFE

**OpenFE** (ICML 2023) : référence de l'automatisation du feature engineering sur données
tabulaires (hors approches LLM, qui exigeraient une clé API et du réseau — incompatibles avec
notre environnement local). Méthode :

1. **Génération de candidats** : opérateurs unaires (abs, log…), combinaisons 2-à-2
   (`+ − × ÷`), opérateurs d'agrégation par groupe (`GroupByThenMean`, `GroupByThenRank`…).
2. **Stage I** — *pruning prédictif* : évaluation rapide de chaque candidat (petit LightGBM,
   early stopping agressif) → élimination du bruit.
3. **Stage II** — sélection itérative : ajout glouton des candidats restants guidé par
   l'importance LightGBM (`gain_importance`).

La métrique de recherche est **l'AUC** (seule métrique native du moteur de sélection — le coût
métier, nécessitant un seuil optimisé, n'est pas exprimable comme metric LightGBM). C'est
acceptable : AUC et coût métier sont monotones l'un par l'autre dans notre régime ; le seuil est
ré-optimisé en aval sur le coût réel.

## Contrainte CPU — stratégie de bornage

La machine de dev (cf. [00](00_environnement_et_dimensionnement.md)) impose une empreinte
raisonnable ; le pipeline OpenFE est donc borné **par conception** :

| Levier | Valeur | Effet |
|---|---|---|
| Sous-échantillon de recherche | **5 000 lignes** stratifiées (vs 246 008) | ~50× moins de calcul par candidat |
| Espace de candidats | `candidate_features_list` limité aux **top-30 features par importance LightGBM** + 16 catégorielles → **≈ 3 000 candidats au lieu de 71 729** | −96 % de candidats (mécanisme prévu par OpenFE pour les tables larges) |
| Parallélisme | `n_jobs=8` (½ des 16 threads) | laisse la machine disponible |
| Fichiers temporaires | `/tmp/opencode/*.feather` | pas de pollution du dépôt |

## Application

- Recherche : `task='classification'`, `metric='auc'`, `stage1_metric='predictive'`,
  `min_candidate_features=200`, `seed=42` → **220 s**, **290 features découvertes**.
- `transform` appliqué aux 246 008 + 61 503 lignes : 147 s → **400 features** (110 + 290).
- Sécurités aval : `inf → NaN` (divisions par zéro des ratios — LightGBM gère le NA, pas l'inf),
  catégories autoFE → codes.
- Snapshots : `data/processed/train_fe.csv` (1,4 Go) / `val_fe.csv` (351 Mo).

## Résuitats

### Top features découvertes (score de sélection OpenFE)

| Feature | Score | Interprétation métier |
|---|---|---|
| `EXT_SOURCE_3 + EXT_SOURCE_2` | 35 038 | Combinaison des deux meilleurs scores externes |
| `AMT_CREDIT / AMT_ANNUITY` | 5 540 | Durée implicite de remboursement |
| `EXT_SOURCE_3 * EXT_SOURCE_2` | 2 642 | Interaction des scores |
| `AMT_ANNUITY / AMT_GOODS_PRICE` | 1 926 | Charge annuelle vs valeur du bien |
| `max(EXT_SOURCE_3, EXT_SOURCE_1)` | 1 613 | Meilleur score externe disponible |
| `EXT_SOURCE_1 * DAYS_EMPLOYED` | 1 092 | Interaction score/ancienneté |
| `GroupByThenRank(AMT_GOODS_PRICE, NAME_CONTRACT_TYPE)` | 811 | Rang du bien dans son type de contrat |

Cohérence remarquable : OpenFE redécouvre **les ratios canoniques du scoring** sans les
connaître.

### Gain validé (LightGBM, run `lightgbm_autofe`)

| | ROC-AUC | PR-AUC | Seuil optimal | **Coût/client** |
|---|---|---|---|---|
| 110 features (baseline) | 0,7585 | 0,2490 | 0,0888 | 0,5269 |
| **400 features (+ OpenFE)** | **0,7656** | 0,2597 | 0,0899 | **0,5187** |

→ **+0,007 ROC-AUC ; −0,0082 de coût/client (−1,6 %)**. Gain modeste mais réel, à coût de
calcul maîtrisé (early stopping à 252 itérations seulement).

## Limites

- 290 features ajoutées sur 5 000 lignes : une partie est du bruit redondant → motivé la
  section 5 (feature selection).
- Recherche bornée aux top-30 features de base : des interactions impliquant des features
  « faibles » seules mais utiles en combinaison ont pu être écartées (compromis CPU assumé).

## Coût de calcul

Recherche 220 s (8 process) ; transform 147 s ; fit de validation 136 s (~5 Go pic).
Total section ≈ 8 min.
