# 04 — Feature engineering automatisé (OpenFE)

Notebook : `p6_ml.ipynb`, section 10.

## Objectif

Les baselines utilisent **221 features** (application + 111 agrégats des tables secondaires). Or en scoring crédit, l'information
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
| Espace de candidats | `candidate_features_list` limité aux **top-30 features par importance LightGBM** + 16 catégorielles → **5 676 candidats** (vs plusieurs centaines de milliers sans restriction) | mécanisme prévu par OpenFE pour les tables larges |
| Parallélisme | `n_jobs=8` (½ des 16 threads) | laisse la machine disponible |
| Fichiers temporaires | `/tmp/opencode/*.feather` | pas de pollution du dépôt |

## Application

- Top-30 de base : `EXT_SOURCE_1/2/3`, `AMT_ANNUITY`, `DAYS_BIRTH`, mais aussi des agrégats
  secondaires (`PREV_DAYS_LAST_DUE_1ST_VERSION_MAX`, `INST_AMT_PAYMENT_SUM`,
  `BUR_DEBT_CREDIT_RATIO_MAX`…).
- Recherche : `task='classification'`, `metric='auc'`, `stage1_metric='predictive'`,
  `min_candidate_features=200`, `seed=42` → **262 s**, **382 features découvertes**.
- `transform` appliqué aux 246 008 + 61 503 lignes → **603 features** (221 + 382).
- **Banc figé** : `openfe_bank.joblib` + `openfe_bank_formulas.json` sauvegardés par le notebook
  et rejoués à l'identique par l'inférence (vérification 100 % des cellules).
- Sécurités aval : `inf → NaN` (divisions par zéro des ratios — LightGBM gère le NA, pas l'inf),
  catégories autoFE → codes.
- Snapshots : `data/processed/train_fe.csv` (1,4 Go) / `val_fe.csv` (351 Mo).

## Résuitats

### Top features découvertes (score de sélection OpenFE)

| Feature | Score | Interprétation métier |
|---|---|---|
| `EXT_SOURCE_2 + EXT_SOURCE_3` | 33 815 | Combinaison des deux meilleurs scores externes |
| `AMT_ANNUITY / AMT_CREDIT` | 2 910 | Taux d'annuité (inverse de la durée de remboursement) |
| `EXT_SOURCE_1 + EXT_SOURCE_3` | 2 251 | Combinaison de scores externes |
| `max(EXT_SOURCE_1, EXT_SOURCE_3)` | 1 852 | Meilleur score externe disponible |
| `BUR_DEBT_CREDIT_RATIO_MAX + INST_LAST1Y_DPD_MEAN` | 1 818 | Endettement bureau × retards récents |
| `INST_AMT_PAYMENT_SUM / CC_CNT_DRAWINGS_CURRENT_MEAN` | 1 777 | Volume remboursé vs usage de la carte |
| `EXT_SOURCE_1 − INST_LAST1Y_DPD_MEAN` | 1 695 | Score externe corrigé des retards récents |
| `AMT_ANNUITY / AMT_GOODS_PRICE` | 1 653 | Charge annuelle vs valeur du bien |

OpenFE redécouvre **les ratios canoniques du scoring** et les **croise avec l'historique**
(endettement, retards de paiement) apporté par les tables secondaires.

### Gain validé (LightGBM, run `lightgbm_autofe`)

| | ROC-AUC | PR-AUC | Seuil optimal | **Coût/client** |
|---|---|---|---|---|
| 221 features (baseline) | 0,7880 | 0,2876 | 0,5059 | 0,4873 |
| **603 features (+ OpenFE)** | **0,7895** | 0,2886 | 0,4943 | **0,4861** |

→ **+0,0015 ROC-AUC ; −0,0012 de coût/client**. Gain faible : une grande partie du signal
« ratios / historique » est déjà apportée par les agrégats des tables secondaires.
*(Version précédente, sans tables secondaires : +0,007 AUC, −0,0082 de coût.)*

## Limites

- 382 features ajoutées sur 5 000 lignes : une partie est du bruit redondant → motive la
  section 5 (feature selection).
- Recherche bornée aux top-30 features de base : des interactions impliquant des features
  « faibles » seules mais utiles en combinaison ont pu être écartées (compromis CPU assumé).

## Coût de calcul

Recherche 262 s (8 process) ; transform + écriture des snapshots 427 s ; fit de validation 53 s.
Total section ≈ 12 min.
