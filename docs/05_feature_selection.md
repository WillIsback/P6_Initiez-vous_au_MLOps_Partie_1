# 05 — Feature selection (filtre variance + BorutaShap)

Notebook : `p6_ml.ipynb`, section 11.

## Objectif

Le jeu enrichi par OpenFE compte **400 features** dont une partie est :
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
   dont les valeurs sont mélangées (permutation) : 400 features réelles + 400 shadows de même
   distribution/cardinalité.
2. Un LightGBM (200 arbres) est entraîné sur les 800 colonnes (split 70/30 interne,
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
| Mesure d'importance | **shap** (calibré : ~41 s/trial à 30k lignes → ~17 min) |
| Compatibilité | shims `scipy≥1.12` (`binom_test`) et `numpy≥2.0` (`np.NaN`) — BorutaShap 1.0.17 est antérieur à ces ruptures |

## Application

- Encodage ordinal des 16 catégorielles **avant** Boruta (SHAP requiert du numérique) ;
  les noms de colonnes d'origine sont conservés pour l'interprétabilité de la sélection.
- `BorutaShap(model=LGBM(200), importance_measure='shap', percentile=100, pvalue=0.05)`,
  `fit(n_trials=25, random_state=42)`.
- Artefacts : `data/processed/feature_selection_boruta.csv` (décision par feature) ;
  run MLflow `feature_selection_boruta`.
- Jeu réduit sauvegardé : `train_fs.csv` / `val_fs.csv` (features `confirmed` uniquement).

### Application retenu (protocole détendu)

La barre du max-shadow est abaissée via **`percentile=50`** : 200 shadow features pour 400 réelles
(au lieu de 400). Ce réglage est documenté et justifié par le diagnostic du protocole strict
(ci-dessous). Les autres paramètres sont inchangés (`pvalue=0.05`, 25 trials, 30k lignes, 200 arbres).

### Résultats

> **Protocole strict (`percentile=100`, mesuré)** :
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

**Résultats du protocole détendu (`percentile=50`, mesuré)** :
- **61 features confirmées** / 339 rejetées (dont 73 « tentative » tranchées par `TentativeRoughFix`)
- Top confirmées (ordre de sélection Boruta) : `autoFE_f_91`, `autoFE_f_83`, `CODE_GENDER`,
  `autoFE_f_55`, `AMT_CREDIT`, `autoFE_f_4`…
- Modèle LightGBM sur les 61 features (run `feature_selection_boruta`) :
  ROC-AUC **0,7647**, PR-AUC 0,2568, seuil 0,0978, **coût/client 0,5229**

### Mise en perspective (toutes étapes)

| Modèle | Features | ROC-AUC | Coût/client |
|---|---|---|---|
| Baseline LogReg | 110 (OHE) | 0,7485 | 0,5414 |
| Baseline LightGBM | 110 | 0,7585 | 0,5269 |
| + OpenFE | 400 | 0,7656 | **0,5187** |
| + Boruta strict (10 feat.) | 10 | 0,7562 | 0,5307 ❌ |
| **+ Boruta détendu** | **61** | **0,7647** | **0,5229** |

→ La sélection détendue **conserve 99,9 % de l'AUC** des 400 features avec **85 % de features en
moins**, et **battent toujours la baseline 110 features** (coût 0,5229 < 0,5269). Compromis
compactcité/performance nettement favorable — arbitrage final (400 vs 61 pour le tuning) tranché
en section 6 du doc tuning.

## Limites

- Sélection calibrée sur **30k lignes** : les features à signal faible peuvent basculer
  rejected/confirmed selon l'échantillon (le `tentative_kept` documente cette zone grise).
- Le protocole teste la pertinence **individuelle** (all-relevant), pas la contribution
  marginale conditionnée à un sous-ensemble donné (une feature redondante mais prédictive peut
  être confirmée).
- Importance évaluée par SHAP sur le LightGBM de trial (200 arbres) : une métrique plus stable
  (plus d'arbres) donnerait des hit-rates légèrement différents — compromis CPU assumé.

## Coût de calcul

Filtre variance : ~30 s. BorutaShap : **~17-20 min**, 8 threads, pic ~3,7 Go RAM (mesuré).
Validation finale du sous-ensemble : ~30 s.
