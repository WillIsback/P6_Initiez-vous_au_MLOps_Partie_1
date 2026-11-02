# 01b — Tables secondaires : agrégats & jointures

Notebook : `p6_ml.ipynb`, section 5bis · Code : [`scripts/secondary_tables.py`](../scripts/secondary_tables.py)
(module partagé par le notebook et par `scripts/build_inference_db.py`).

## Pourquoi

`application_train/test` ne décrit que la **demande en cours**. L'historique de crédit du client
(crédits chez d'autres organismes, demandes passées, retards de paiement, utilisation des
cartes) est réparti dans **6 tables secondaires**. C'est l'information la plus naturelle en
scoring crédit : ignorer ces tables revenait à scorer un client sans regarder son passé.

## Schéma relationnel

```
application_{train|test}   (1 ligne / client, clé SK_ID_CURR)
  ├── bureau                 crédits déclarés par d'autres organismes   (clé SK_ID_BUREAU)
  │     └── bureau_balance   historique mensuel de chaque crédit bureau
  └── previous_application   demandes passées chez Home Credit           (clé SK_ID_PREV)
        ├── installments_payments   échéances et paiements
        ├── POS_CASH_balance        historique mensuel des prêts POS / cash
        └── credit_card_balance     historique mensuel des cartes de crédit
```

## Exploration (formats, clés, doublons, couverture)

| Table | Lignes | Colonnes (num / cat) | Clé | Lignes / clé | Doublons | % NaN | % clients train | % clients test |
|---|---|---|---|---|---|---|---|---|
| bureau | 1 716 428 | 17 (14 / 3) | `SK_ID_CURR` | 5,6 | 0 | 13,5 | 85,7 | 86,8 |
| bureau_balance | 27 299 925 | 3 (2 / 1) | `SK_ID_BUREAU` | 33,4 | 0 | 0,0 | **30,0** | **86,8** |
| previous_application | 1 670 214 | 37 (21 / 16) | `SK_ID_CURR` | 4,9 | 0 | 18,0 | 94,6 | 98,1 |
| installments_payments | 13 605 401 | 8 (8 / 0) | `SK_ID_CURR` | 40,1 | 0 | 0,0 | 94,8 | 98,4 |
| POS_CASH_balance | 10 001 358 | 8 (7 / 1) | `SK_ID_CURR` | 29,7 | 0 | 0,1 | 94,1 | 98,1 |
| credit_card_balance | 3 840 312 | 23 (22 / 1) | `SK_ID_CURR` | 37,1 | 0 | 6,7 | 28,3 | 34,2 |

## Choix méthodologiques

| Choix | Justification |
|---|---|
| **Agréger au grain client** avant de joindre | 5 à 40 lignes d'historique par client : une jointure directe multiplierait les lignes |
| **Jointure gauche** sur `SK_ID_CURR`, `validate="one_to_one"` | Aucune demande créée ni perdue (307 511 lignes avant / après) |
| Agrégats **sans `TARGET`** | Pas de fuite : calculables une fois pour tous les clients (train et test) |
| `*_COUNT` = 0 sans historique | L'absence d'historique est une information (pas une valeur manquante à imputer) |
| Autres agrégats laissés NaN | Imputés en section 6 par la médiane **fit sur le train** |
| Sentinelle `365243` des `DAYS_*` de `previous_application` → NaN | Même anomalie que `DAYS_EMPLOYED` |
| Ratios métier | `DEBT_CREDIT_RATIO` (endettement), `APP_CREDIT_RATIO` (montant demandé / accordé), `PAYMENT_RATIO`, `UTILIZATION` (solde / plafond carte), `REFUSED_RATIO` |
| Retards de paiement | `DPD` / `DBD` (jours de retard / d'avance par échéance), `LATE_RATIO`, versions « 12 derniers mois » |
| Module Python partagé | Mêmes features à l'entraînement et en inférence (pas de divergence train/serve) |

## Résultats

- **111 features** : bureau + bureau_balance **37**, previous_application **29**,
  installments_payments **21**, POS_CASH_balance **11**, credit_card_balance **13**.
- `application_train` : **122 → 233 colonnes** ; après préparation (section 6) **221 features**.
- Calcul : **21 s** pour les 6 tables (opérations vectorisées), ~3,3 Go RAM au pic.
- **Signal** : 24 agrégats sur 111 ont |corr| > 0,05 avec `TARGET`. Les plus forts sont tous
  lisibles métier :

| Agrégat | Corrélation avec TARGET | Lecture |
|---|---|---|
| `CC_UTILIZATION_MEAN` | +0,136 | Carte de crédit utilisée près du plafond |
| `BUR_DAYS_CREDIT_MEAN` | +0,090 | Crédits bureau récents (historique court) |
| `INST_LAST1Y_LATE_MEAN` | +0,082 | Retards de paiement sur les 12 derniers mois |
| `PREV_REFUSED_RATIO` | +0,078 | Part de demandes passées refusées |
| `BUR_ACTIVE_RATIO` | +0,077 | Part de crédits encore actifs |

- **Impact sur les modèles** (validation, seuil optimisé sur le train) :

| Modèle | Avant (application seule) | Après (+ tables secondaires) |
|---|---|---|
| Baseline LogReg | AUC 0,7485 · coût 0,5414 | AUC **0,7761** · coût **0,5033** |
| Baseline LightGBM | AUC 0,7585 · coût 0,5269 | AUC **0,7880** · coût **0,4873** |
| Modèle final (tuné) | AUC 0,7662 · coût 0,5220 | AUC **0,7893** · coût **0,4814** |

C'est **le levier le plus important du projet** : il apporte plus que l'AutoFE et le tuning réunis.

## Visualisation

Figure `secondary_features_signal.png` (run MLflow `secondary_tables_join`) : taux de défaut
avec / sans historique par table, top 15 des corrélations, taux de défaut par décile du
meilleur agrégat.

## Limites & points de vigilance

- ⚠️ **`bureau_balance` : couverture 30 % (train) vs 87 % (test)**. Les features dérivées
  (`BUR_MONTHS_MEAN`, `BUR_LATE_RATIO_*`, `BUR_DPD_LEVEL_MAX`) ont donc une distribution
  différente en production → à surveiller en priorité (dérive de données).
- Agrégats « manuels » (moyennes, max, comptages) : pas de pondération temporelle fine au-delà
  de la fenêtre « 12 derniers mois » des paiements.
- Tables agrégées indépendamment : pas d'agrégats croisés `previous_application ↔ installments`
  par crédit (ex. retard moyen des seuls crédits refusés).

## Coût de calcul

Lecture des 6 CSV (~2,5 Go) + agrégats : ~30 s pour l'exploration, 21 s pour l'agrégation,
pic ~3,3 Go RAM, 1 thread.
