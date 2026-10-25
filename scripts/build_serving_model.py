"""Construit et enregistre le modele de SERVING (pyfunc) : encodeur + LightGBM tuned + seuil metier.
A executer APRES la section 12 (Registry v3 disponible).
Sortie : registry `home-credit-serving` v1 — servi via `mlflow models serve` ou image Docker.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import mlflow
import mlflow.pyfunc

PROCESSED = Path("data/processed")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("home-credit-scoring")
client = mlflow.MlflowClient()

# --- 1. chargement du jeu servi (61 features Boruta, RAW — catégorielles non encodées) ---
train = pd.read_csv(PROCESSED / "train_fs.csv")
y_tr = train["TARGET"]
X_tr = train.drop(columns=["TARGET"])
feature_names = list(X_tr.columns)
cat_cols = [c for c in feature_names if not pd.api.types.is_numeric_dtype(X_tr[c])]
print(f"features servies : {len(feature_names)} (dont {len(cat_cols)} catégorielles brutes)")

# --- 2. encodeur ordinal, identique section 12.1 (fit train : pas de fuite) ---
enc = ColumnTransformer(
    [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
    remainder="passthrough",
)
enc.fit(X_tr)

# --- 3. modele tuned depuis le Registry (derniere version creee par la section 12) ---
versions = client.search_model_versions("name='home-credit-lightgbm'")
latest = max(versions, key=lambda v: int(v.version))
model_uri = f"models:/home-credit-lightgbm/{latest.version}"
lgb_tuned = mlflow.lightgbm.load_model(model_uri)
print(f"modèle chargé : {model_uri}")

# --- 4. seuil optimal fige : lu dans la run tuning_optuna (metric val_best_threshold) ---
runs = client.search_runs(
    "1", filter_string="attributes.run_name = 'tuning_optuna'",
    order_by=["attributes.start_time DESC"], max_results=1,
)
thr = runs[0].data.metrics["val_best_threshold"]
print(f"seuil métier figé : {thr:.4f}")


# --- 5. wrapper pyfunc : RAW -> encode -> proba -> decision ---
class HomeCreditScorer(mlflow.pyfunc.PythonModel):
    """API de scoring : entrées = 61 features brutes (telles que train_fs.csv sans TARGET).
    Sorties : prob_default (P(défaut)), decision (1 = refuser, seuil métier figé)."""

    def __init__(self, encoder, model, feature_names, threshold):
        self.encoder = encoder
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            df = model_input.reindex(columns=self.feature_names)  # alignement par nom
        else:
            df = pd.DataFrame(np.asarray(model_input), columns=self.feature_names)
        X_enc = self.encoder.transform(df)
        proba = self.model.predict_proba(X_enc)[:, 1]
        return pd.DataFrame({
            "prob_default": proba,
            "decision": (proba >= self.threshold).astype(int),
            "threshold": np.full(len(proba), self.threshold),
        })


# toutes les colonnes numeriques declarees en double : le serving accepte floats/nulls
X_example = X_tr[:5].copy()
for c in X_example.columns:
    if c not in cat_cols:
        X_example[c] = X_example[c].astype("float64")

signature = mlflow.models.infer_signature(
    X_example,
    pd.DataFrame({
        "prob_default": np.zeros(5),
        "decision": np.zeros(5, dtype=int),
        "threshold": np.full(5, thr),
    }),
)

with mlflow.start_run(run_name="build_serving_model") as run:
    mlflow.pyfunc.log_model(
        name="home_credit_scorer",
        python_model=HomeCreditScorer(enc, lgb_tuned, feature_names, thr),
        signature=signature,
        input_example=X_example,
        registered_model_name="home-credit-serving",
        pip_requirements=["lightgbm", "scikit-learn", "pandas", "numpy", "joblib"],
    )
    mlflow.log_params({
        "base_model": f"home-credit-lightgbm v{latest.version} (tuned)",
        "n_features_raw": len(feature_names),
        "threshold_frozen": thr,
        "fn_cost": 10.0, "fp_cost": 1.0,
    })
    mlflow.set_tags({"stage": "serving", "api": "pyfunc_scoring"})
    print(f"Run build_serving_model : {run.info.run_id}")

print("Registry : home-credit-serving v1 prêt")
print("Lancer : mlflow models serve -m 'models:/home-credit-serving/1' -p 5001 --env-manager local")
