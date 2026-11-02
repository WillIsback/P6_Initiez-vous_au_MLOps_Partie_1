# 05 — Feature selection (filtre variance + BorutaShap)

Notebook : `p6_ml.ipynb`, section 11.

## Objectif

Le jeu enrichi par OpenFE compte **603 features** (221 + 382 OpenFE) dont une partie est :
- **sans variance** (constantes ou quasi-constantes issues des opérateurs) ;
- **à impact nul** (pas meilleures que du bruit pour prédire le défaut) ;
- **redondantes** (doublons de ratios déjà couverts).

La sélection vise à ne garder que les features **réellement pertinentes**, en contrôlant
statistiquement la notion d'« impact nul ».

## Méthode : BorutaShap (sélection « all-relevant »)

**BorutaShap** = protocole de Boruta (Kursa & Rudnicki) avec importance par **valeurs SHAP**
(Lundberg) au lieu de l'importance MDI/gini, biaisée en faveur des features à haute
cardinalité (nos colonnes catégorielles ordinal-encodées).

### Principe du test

1. À chaque trial, chaque feature réelle est **comparée à ses shadow features** — des copies
   dont les valeurs sont mélangées (permutation) : *n* features réelles + *n* shadows de même
   distribution/cardinalité.
2. Un LightGBM (200 arbres) est entraîné sur réelles + shadows (split 70/30 interne,
   `train_or_test='test'` : l'importance est évaluée sur des lignes non vues par le fit).
3. Importance = **|SHAP| moyen** (TreeExplainer, `tree_path_dependent`, `approximate=True`),
   normalisé en z-score.
4. Une feature marque un « hit » si son importance **dépasse le meilleur shadow**. Le taux de
   hits est testé par un **test binomial exact bilatéral** (`p = 0.05`, i.e. hypothèse : « au
   hasard, elle battrait le max shadow dans 5 % des cas ») avec **correction FDR
   Benjamini-Hochberg** sur les tests multiples.
5. Après `n_trials` trials : **confirmed** (impact réel prouvé), **rejected** (impact nul —
   jamais meilleur que le shadow), **tentative** (indécis) → tranché par `TentativeRoughFix()`
   (comparaison de la médiane d'importance à la médiane du max shadow).

C'est précisément la réponse formelle à la question « cette feature a-t-elle un impact nul ? » :
une feature sans variance ou sans signal est statistiquement indistinguable de son shadow
permuté → rejected.

### Filtre préalable (gratuit)

Avant Boruta, retrait **mécanique** (section 11.1) :
- features **constantes** (`nunique ≤ 1`) ;
- features **quasi-constantes** : une même valeur pour **> 99,9 %** des lignes.

Ces colonnes n'ont aucune variance exploitable et pollueraient inutilement le budget de trials.

## Contrainte CPU — bornage

| Levier | Valeur |
|---|---|
| Échantillon de recherche | **30 000 lignes** stratifiées (vs 246 008) |
| Trials | **25** (au-delà, le taux de hits se stabilise) |
| Modèle par trial | LightGBM 200 arbres, `n_jobs=8` (½ des threads) |
| Mesure d'importance | **shap** (mesuré : ~11 s/trial à 30k lignes → ~5 min) |
| Compatibilité | shims `scipy≥1.12` (`binom_test`) et `numpy≥2.0` (`np.NaN`) — BorutaShap 1.0.17 est antérieur à ces ruptures |

## Application

- Encodage ordinal des 16 catégorielles **avant** Boruta (SHAP requiert du numérique) ;
  les noms de colonnes d'origine sont conservés pour l'interprétabilité de la sélection.
- `BorutaShap(model=LGBM(200), importance_measure='shap', percentile=50, pvalue=0.05)`,
  `fit(n_trials=25, random_state=42)`.
- Artefacts : `data/processed/feature_selection_boruta.csv` (décision par feature) ;
  run MLflow `feature_selection_boruta`.
- Jeu réduit sauvegardé : `train_fs.csv` / `val_fs.csv` (features `confirmed` uniquement).

### Application retenu (protocole détendu)

La barre du max-shadow est abaissée via **`percentile=50`** (shadows à la moitié des réelles). Ce réglage est documenté et justifié par le diagnostic du protocole strict
(ci-dessous). Les autres paramètres sont inchangés (`pvalue=0.05`, 25 trials, 30k lignes, 200 arbres).

### Résultats

> **Protocole strict (`percentile=100`) — diagnostic établi sur la version précédente du pipeline
> (table principale seule, 400 features)** :
> - Filtre variance : **0 constante / 0 quasi-constante** — OpenFE ne produit que des combinaisons
>   de features vivantes ; le cas « sans variance » se traitait mais était vide sur ce jeu.
> - BorutaShap strict : **10 confirmées / 390 rejetées**, dont `EXT_SOURCE_1/2/3`, `DAYS_BIRTH`,
>   `AMT_INCOME_TOTAL`, `CODE_GENDER` — les features les plus fortes du jeu.
> - Modèle LightGBM sur les 10 confirmées : ROC-AUC **0,7562**, coût/client **0,5307**
>   → **pire que la baseline 110 features (0,5269)** et loin des 400 features (**0,5187**).
>
> **Diagnostic** : deux effets combinés sur-prunent.
> 1. *Barre du max-shadow* : il faut battre le **maximum** de 400 bruits permutés à chaque trial
>    (extrême-value de 400 tirages) — barre très haute.
> 2. *Dilution SHAP par redondance* : face aux 290 combos autoFE (`EXT_SOURCE_3+EXT_SOURCE_2`…),
>    LightGBM concentre le signal dans le **combo représentant** ; l'individu (`EXT_SOURCE_3` seul)
>    devient presque inutilisé dans le modèle ajusté → SHAP ≈ 0 → rejeté comme *irrelevant
>    conditionnellement au modèle*. Boruta conserve les représentants et rejette les individus
>    redondants : comportement attendu du protocole, mais excessif ici.

**Résultats du protocole détendu (`percentile=50`, mesuré — version actuelle)** :
- Filtre variance : 0 constante / 0 quasi-constante sur 603 features.
- **122 features confirmées** / 481 rejetées. Les agrégats secondaires y sont bien représentés
  (`INST_DBD_MEAN`, `INST_COUNT`, `BUR_ACTIVE_COUNT`, `BUR_DAYS_CREDIT_MEAN`,
  `CC_UTILIZATION_MEAN/MAX`, `PREV_APR_DAYS_DECISION_MAX`…).
- Modèle LightGBM pondéré sur les 122 features (run `feature_selection_boruta`) :
  ROC-AUC **0,7866**, PR-AUC 0,2844, seuil 0,4881, **coût/client 0,4962**

### Mise en perspective (toutes étapes, version actuelle)

| Modèle | Features | ROC-AUC | Coût/client |
|---|---|---|---|
| Baseline LogReg | 221 (OHE) | 0,7761 | 0,5033 |
| Baseline LightGBM | 221 | 0,7880 | 0,4873 |
| + OpenFE | 603 | 0,7895 | **0,4861** |
| **+ Boruta détendu** | **122** | **0,7866** | 0,4962 |
| + tuning Optuna (doc 06) | 122 | 0,7893 | **0,4814** |

→ La sélection conserve **99,6 % de l'AUC** avec **80 % de features en moins**, mais coûte
**+0,010/client** sur ce holdout (contre +0,004 dans la version précédente). Le choix des 122
features est maintenu pour l'exploitabilité (inférence, SHAP dans l'app, surface de dérive) ;
l'écart est rattrapé par le tuning (0,4814, meilleur résultat du pipeline). Tuner directement
les 603 features reste une piste.

## Limites

- Sélection calibrée sur **30k lignes** : les features à signal faible peuvent basculer
  rejected/confirmed selon l'échantillon (le `tentative_kept` documente cette zone grise).
- Le protocole teste la pertinence **individuelle** (all-relevant), pas la contribution
  marginale conditionnée à un sous-ensemble donné (une feature redondante mais prédictive peut
  être confirmée).
- Importance évaluée par SHAP sur le LightGBM de trial (200 arbres) : une métrique plus stable
  (plus d'arbres) donnerait des hit-rates légèrement différents — compromis CPU assumé.

## Coût de calcul

Filtre variance : ~10 s. BorutaShap : **~5 min** (287 s), 8 threads, pic ~4 Go RAM (mesuré).
Validation finale du sous-ensemble : ~30 s.
