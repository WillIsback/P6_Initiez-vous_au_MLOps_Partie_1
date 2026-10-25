# 01 — Exploration des données (EDA)

Notebook : `p6_ml.ipynb`, section 5.

## Contexte

Jeu **Home Credit Default Risk** (Kaggle) : prédire la probabilité de défaut de remboursement
d'une demande de crédit. L'EDA pose les fondations du prétraitement et des choix de features.

## Application

- **Shape** : `application_train.csv` — **307 511 lignes × 122 colonnes**, dont `SK_ID_CURR`
  (identifiant) et `TARGET` (binaire, défaut = 1).
- **Déséquilibre** : **8,07 % de défauts** (24 825 positifs) — classe minoritaire marquée, ce qui
  condamne d'emblée l'accuracy comme métrique et motive la métrique métier (coût, section 3).
- **Valeurs manquantes** : **67 colonnes contiennent des NaN**, dont **41 avec > 50 % de
  valeurs manquantes** (`COMMONAREA_*`, `LANDAREA_*`, `FLOORSMIN_*`… toutes issues du bureau de
  crédit agrégé). Les features `EXT_SOURCE_*` (scores externes) ont 19-20 % de NaN mais sont les
  plus corrélées à la cible.
- **Sentinelle cachée** : `DAYS_EMPLOYED` vaut **365 243** pour **18 %** des lignes — ce n'est pas
  une vraie valeur mais le marqueur « sans emploi » positionné à la date d'agrégation du fichier.
  Le laisser tel quel polluerait les trees (les 365 243 formeraient un nœud massif) et les
  imputations futures.
- **Corrélations linéaires faibles** avec la cible : max ≈ |−0,18| (`EXT_SOURCE_3`), |−0,16|
  (`EXT_SOURCE_2`), `DAYS_BIRTH` ≈ |0,08|. Rien n'est linéairement séparable → les modèles
  d'arbres sont justifiés.
- **Variables catégorielles** : 16 colonnes `str` (dont ordinales naturelles
  `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, `NAME_INCOME_TYPE`…).

## Choix méthodologiques (décisions issues de l'EDA)

| Observation | Décision |
|---|---|
| 41 colonnes > 50 % NaN | **Conserver** — LightGBM gère nativement le manquant ; l'imputation balaierait l'information |
| Sentinelle 365 243 | **Convertir en NaN** avant imputation (traitement section 6) |
| Corrélations faibles | Baseline linéaire (LogReg) + non-linéaire (LightGBM) pour mesurer l'écart |
| Déséquilibre 8 % | Stratification systématique des splits + métrique coût métier |
| `SK_ID_CURR` | Identifiant pur → **supprimé** (aucune valeur prédictive, risque de fuite) |

## Résultats

Les statistiques EDA (plots de distribution, top NaN, corrélations) sont dans le notebook —
3 figures loguées : distributions des features clés, proportions de NaN par colonne,
matrice de corrélation ciblée.

## Limites

- Pas de colonne temporelle dans `application_train` (le split devra être aléatoire stratifié —
  risque de recouvrement temporel documenté en section 6).
- Les 9 tables secondaires (bureau, previous applications, installments…) ne sont **pas** intégrées
  dans ce pipeline : l'EDA se concentre sur `application_train/test`.

## Coût de calcul

< 30 s, 1 thread, ~1,5 Go RAM (agrégations pandas sur 122 colonnes).
