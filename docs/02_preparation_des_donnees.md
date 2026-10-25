# 02 — Préparation des données

Notebook : `p6_ml.ipynb`, section 6.

## Objectif

Produire un jeu de features **propre, sans fuite, réutilisable** (snapshots sur disque) servant
de base aux baselines, au feature engineering et à la feature selection.

## Choix méthodologiques & application

### 1. Split train/validation — stratifié 80/20

- **246 008 lignes train** / **61 503 lignes validation**, `stratify=TARGET`, `random_state=42`.
- **Pourquoi stratifié** : conserver les 8,07 % de défauts dans les deux ensembles (sinon la
  variance du coût métier explose en validation).
- **⚠️ Piège identifié** : idéalement le split devrait être **temporel** (simuler le déploiement
  sur des demandes postérieures), mais aucune colonne date exploitable n'existe dans
  `application_train` (`DAYS_BIRTH` est relatif au jour de demande, non absolu). Le split
  aléatoire stratifié est conservé **avec cette limitation documentée**.
- **`application_test.csv` reste inviolé** : aucun stat, aucune imputation, aucun fit ne le touche
  avant l'évaluation finale.

### 2. Traitements par colonne (fit sur le train uniquement)

| Traitement | Règle | Justification |
|---|---|---|
| Sentinelle | `DAYS_EMPLOYED == 365243` → NaN | Valeur non informative (section 1) |
| Conversion temporelle | `DAYS_*` (négatifs) → **années positives** | Lisibilité + cohérence de signe pour les ratios |
| Imputation num | **médiane**, fit train | Robuste aux distributions heavy-tail (features de type montant) |
| Imputation cat | **mode**, fit train | Reprise de la valeur la plus fréquente |
| Winsorisation | quantiles **0,5 % / 99,5 %** si \|skew\| > 10 | Coupe les queue extrêmes des montants (`AMT_*`) sans toucher aux features normales |
| Quasi-constantes | suppression si **> 99,9 %** de la même valeur | Aucune variance exploitable, bruit pour les splits |
| Identifiant | `SK_ID_CURR` supprimé | Fuite/pas de sens prédictif |

### 3. Encodage — délégué aux modèles (pas de OHE ici)

- La préparation garde les **16 colonnes catégorielles en `str`** (pandas 3.x) dans les snapshots :
  l'encodage est décidé par modèle (OHE pour la LogReg, **encodage ordinal natif** pour LightGBM),
  ce qui évite de figer une représentation inadaptée à l'aval.

## Résultats

- **110 features** finales (122 → 111 après identifiant → 110 après quasi-constantes),
  **0 NaN**, 246 008 / 61 503 lignes.
- Artefacts écrits : `data/processed/train_prepared.csv`, `val_prepared.csv`,
  `preprocessing_bundle.joblib` (transformers réutilisables, refit impossibles — pas de fuite).

## Limites

- Split aléatoire (pas temporel) — documenté ci-dessus.
- Imputation globale (pas de missing-indicator) : acceptable pour LightGBM natif-NA, mais les
  colonnes à > 50 % NaN gardent une imputation « médiane » qui lisse — compensé par le fait que
  LightGBM apprend ses propres règles de gestion du NA à l'entraînement (les NaN imputés restent
  minoritaires après winsorisation ciblée).

## Coût de calcul

~1-2 min, 1 thread, pic ~2,5 Go RAM (écriture des snapshots ~230 Mo).
