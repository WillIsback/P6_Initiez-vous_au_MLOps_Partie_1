"""Construction de la base d'inférence SQLite pour l'app Gradio.

Rejoue la chaîne exacte du notebook sur application_test.csv (48 744 clients
jamais observés à l'entraînement) :
  1. bundle de préparation (section 6) : sentinelle, unités, clip, imputation
  2. banc OpenFE (section 10) : régénéré avec les mêmes graines puis mis en cache
  3. sélection Boruta (section 11) : les 61 features confirmées

Vérifications intégrées (échec => abort, jamais de base empoisonnée) :
  a) la réplication de la préparation est comparée à train/val_prepared.csv
  b) le banc OpenFE régénéré est comparé à train_fe.csv / val_fe.csv

Sortie : data/inference/clients.db (table `clients` : SK_ID_CURR + 61 features).

Usage :
  .venv/bin/python scripts/build_inference_db.py               # build complet
  .venv/bin/python scripts/build_inference_db.py --skip-verify # sans comparaison
  .venv/bin/python scripts/build_inference_db.py --refresh-bank
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from openfe import OpenFE, get_candidate_features
from openfe.utils import tree_to_formula

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
INFERENCE = ROOT / "data" / "inference"
BANK_PATH = PROCESSED / "openfe_bank.joblib"
BANK_FORMULAS_PATH = PROCESSED / "openfe_bank_formulas.json"
DB_PATH = INFERENCE / "clients.db"

# graines / hyperparamètres — identiques au notebook (sections 10.1 / 10.2)
SEED = 42
SEARCH_ROWS = 5000
N_JOBS = 8  # moitié des cœurs (contrainte DAT)


def apply_bundle(df_raw: pd.DataFrame, bundle: dict, base_cols: list[str]) -> pd.DataFrame:
    """Chaîne de préparation section 6, figée par preprocessing_bundle.joblib."""
    df = df_raw.copy()

    # 1) sentinelle DAYS_EMPLOYED -> NaN
    df.loc[df["DAYS_EMPLOYED"] == bundle["sentinel_days_employed"], "DAYS_EMPLOYED"] = np.nan

    # 2) jours negatifs -> annees positives
    for col in bundle["days_cols"]:
        df[col] = (-df[col] / 365.25).round(2)

    # 3) identifiant + quasi-constantes (listes figées fit sur le train)
    df = df.drop(columns=[c for c in bundle["dropped_cols"] if c in df.columns])

    # 4) winsorisation aux bornes fit sur le train
    for col, (lo, hi) in bundle["clip_bounds"].items():
        df[col] = df[col].clip(lo, hi)

    # 5) imputation mediane / mode (imputeurs fit sur le train)
    num_cols = [c for c in bundle["num_cols"] if c in df.columns]
    cat_cols = [c for c in bundle["cat_cols"] if c in df.columns]
    df[num_cols] = bundle["num_imputer"].transform(df[num_cols])
    df[cat_cols] = bundle["cat_imputer"].transform(df[cat_cols])

    return df.reindex(columns=base_cols)


def regen_openfe_bank(X_tr: pd.DataFrame, y_tr: pd.Series, cat_cols: list[str]) -> list:
    """Régénère le banc de features OpenFE (section 10.1 + 10.2, mêmes graines)."""
    _, X_search, _, y_search = train_test_split(
        X_tr, y_tr, train_size=SEARCH_ROWS, stratify=y_tr, random_state=SEED,
    )

    # espace de candidats restreint : top-30 par importance LightGBM rapide
    enc_imp = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
        remainder="passthrough",
    )
    X_search_enc = enc_imp.fit_transform(X_search)
    imp_names = [n.split("__", 1)[-1] for n in enc_imp.get_feature_names_out()]
    gbm_imp = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        random_state=SEED, n_jobs=N_JOBS, verbosity=-1,
    )
    gbm_imp.fit(X_search_enc, y_search)
    imp = pd.Series(gbm_imp.feature_importances_, index=imp_names).sort_values(ascending=False)
    top_num = [n for n in imp.index[:30] if n not in cat_cols]
    candidates = get_candidate_features(
        numerical_features=top_num, categorical_features=cat_cols, order=1,
    )

    ofe = OpenFE()
    new_features = ofe.fit(
        data=X_search, label=pd.DataFrame(y_search),
        task="classification", metric="auc",
        candidate_features_list=candidates,
        stage1_metric="predictive", min_candidate_features=200,
        n_jobs=N_JOBS, seed=SEED, verbose=True,
        tmp_save_path="/tmp/opencode/openfe_tmp.feather",
    )
    return list(new_features)


def fresh_ofe() -> OpenFE:
    """OpenFE pret pour transform() sans fit() — fit() pose tmp_save_path + verbose."""
    ofe = OpenFE()
    ofe.tmp_save_path = "/tmp/opencode/openfe_tmp.feather"
    ofe.verbose = True
    return ofe


def compare_frames(got: pd.DataFrame, ref: pd.DataFrame, cols: list[str], label: str) -> float:
    """Comparaison tolérante (round-trip CSV) : renvoie la fraction de cellules conformes."""
    a, b = got[cols].reset_index(drop=True), ref[cols].reset_index(drop=True)
    a_num, b_num = a.apply(pd.to_numeric, errors="coerce"), b.apply(pd.to_numeric, errors="coerce")
    close = np.isclose(a_num.to_numpy(), b_num.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)
    frac = float(close.mean())
    print(f"  [{label}] {frac:.4%} cellules conformes ({len(cols)} colonnes, {len(a)} lignes)")
    return frac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-verify", action="store_true", help="sans comparaison aux snapshots")
    parser.add_argument("--refresh-bank", action="store_true", help="regénère le banc OpenFE (cache)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    INFERENCE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ entrees
    bundle = joblib.load(PROCESSED / "preprocessing_bundle.joblib")
    prep = pd.read_csv(PROCESSED / "train_prepared.csv")
    base_cols = [c for c in prep.columns if c != "TARGET"]
    fs = pd.read_csv(PROCESSED / "feature_selection_boruta.csv")
    fs_cols = fs.loc[fs["decision"] == "confirmed", "feature"].tolist()
    autofe_needed = [c for c in fs_cols if c.startswith("autoFE_f_")]
    print(f"features Boruta : {len(fs_cols)} (dont {len(autofe_needed)} autoFE)")

    app_train = pd.read_csv(RAW / "application_train.csv")
    y_full = app_train["TARGET"]
    app_test = pd.read_csv(RAW / "application_test.csv")
    test_ids = app_test["SK_ID_CURR"].copy()

    # ------------------------------------- 1) replication de la preparation (6)
    print("\n=== 1) preparation (bundle section 6) ===")
    train_base = apply_bundle(app_train.drop(columns=["TARGET"]), bundle, base_cols)
    test_base = apply_bundle(app_test, bundle, base_cols)
    X_tr, X_va, y_tr, y_va = train_test_split(
        train_base, y_full, test_size=0.2, stratify=y_full, random_state=SEED,
    )
    print(f"train_base : {train_base.shape} | X_tr : {X_tr.shape} | test_base : {test_base.shape}")

    if not args.skip_verify:
        prep_va = pd.read_csv(PROCESSED / "val_prepared.csv").drop(columns=["TARGET"])
        # la reproduction du split stratifie doit etre identique au notebook
        frac_tr = compare_frames(X_tr, pd.read_csv(PROCESSED / "train_prepared.csv").drop(columns=["TARGET"]), base_cols, "prep train")
        frac_va = compare_frames(X_va, prep_va, base_cols, "prep validation")
        if min(frac_tr, frac_va) < 0.999:
            raise SystemExit("ARRET : la reproduction de la preparation diverge du notebook")

    # ------------------------------------------------- 2) banc OpenFE (cache)
    print("\n=== 2) banc OpenFE ===")
    if BANK_PATH.exists() and not args.refresh_bank:
        new_features = joblib.load(BANK_PATH)
        print(f"  cache charge : {BANK_PATH.name} ({len(new_features)} features)")
    else:
        t0 = time.perf_counter()
        new_features = regen_openfe_bank(X_tr, y_tr, bundle["cat_cols"])
        joblib.dump(new_features, BANK_PATH)
        BANK_FORMULAS_PATH.write_text(json.dumps(
            [tree_to_formula(n) for n in new_features], indent=1, ensure_ascii=False))
        print(f"  banc regenere en {time.perf_counter() - t0:.0f} s -> cache + formules")

    # ------------------------------------------------ 3) verification du banc
    if not args.skip_verify:
        print("\n=== 3) verification du banc OpenFE (snapshots *_fe.csv) ===")
        ofe = fresh_ofe()
        X_tr_chk, X_va_chk = ofe.transform(X_tr.copy(), X_va.copy(), new_features, n_jobs=N_JOBS)
        fe_cols = [c for c in pd.read_csv(PROCESSED / "train_fe.csv", nrows=1).columns
                   if c.startswith("autoFE_f_")]
        missing = [c for c in fe_cols if c not in X_tr_chk.columns]
        if missing:
            raise SystemExit(f"ARRET : {len(missing)} features autoFE du notebook absentes du banc regenere")
        frac_tr = compare_frames(X_tr_chk, pd.read_csv(PROCESSED / "train_fe.csv"), fe_cols, "autoFE train")
        frac_va = compare_frames(X_va_chk, pd.read_csv(PROCESSED / "val_fe.csv"), fe_cols, "autoFE validation")
        if min(frac_tr, frac_va) < 0.999:
            raise SystemExit("ARRET : le banc OpenFE regenere ne reproduit pas les snapshots")
        print("  banc OpenFE reproduit a l'identique -> base fiable")

    # ------------------------------------- 4) features des clients de test
    print("\n=== 4) features des 48 744 clients de test ===")
    t0 = time.perf_counter()
    ofe = fresh_ofe()
    _, X_test_fe = ofe.transform(X_tr.copy(), test_base.copy(), new_features, n_jobs=N_JOBS)
    X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan)
    for c in X_test_fe.columns:
        if str(X_test_fe[c].dtype) == "category":
            X_test_fe[c] = X_test_fe[c].cat.codes
    missing = [c for c in autofe_needed if c not in X_test_fe.columns]
    if missing:
        raise SystemExit(f"ARRET : features Boruta absentes de la transformation test : {missing[:5]}")
    test_fs = X_test_fe[fs_cols].copy()
    test_fs.insert(0, "SK_ID_CURR", test_ids.to_numpy())
    print(f"  {time.perf_counter() - t0:.0f} s | frame finale : {test_fs.shape}")

    # ------------------------------------------------------- 5) ecriture SQLite
    print("\n=== 5) ecriture SQLite ===")
    import sqlite3
    if DB_PATH.exists():
        DB_PATH.unlink()
    cat_in_fs = [c for c in fs_cols if c in bundle["cat_cols"]]
    for c in cat_in_fs:
        test_fs[c] = test_fs[c].astype(object).where(test_fs[c].notna(), None)
    con = sqlite3.connect(DB_PATH)
    col_types = {c: "TEXT" for c in cat_in_fs}
    col_types.update({c: "REAL" for c in fs_cols if c not in cat_in_fs})
    col_types["SK_ID_CURR"] = "INTEGER"
    test_fs.to_sql("clients", con, index=False, if_exists="replace", dtype=col_types)
    con.execute("CREATE INDEX IF NOT EXISTS idx_clients_sk ON clients(SK_ID_CURR)")
    n_rows = con.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
    con.commit()
    con.close()
    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"  {DB_PATH.relative_to(ROOT)} : {n_rows:,} clients, {size_mb:.1f} Mo, {len(fs_cols)} features")

    # --------------------------------------------------- 6) tracé MLflow (prep)
    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("home-credit-scoring")
        with mlflow.start_run(run_name="prepare_inference_db") as run:
            mlflow.set_tags({"stage": "inference-prep", "output": "sqlite_clients_db"})
            mlflow.log_params({
                "source": "application_test.csv", "n_clients": n_rows,
                "n_features": len(fs_cols), "n_autofeatures": len(autofe_needed),
                "openfe_bank_size": len(new_features), "n_jobs": N_JOBS,
            })
            mlflow.log_metrics({"prep_seconds": time.perf_counter() - t_start,
                                "db_size_mb": size_mb})
            (INFERENCE / "features.json").write_text(json.dumps({
                "features": fs_cols, "cat_features": cat_in_fs,
                "build_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=1, ensure_ascii=False))
            mlflow.log_artifact(str(INFERENCE / "features.json"))
            print(f"  run MLflow : {run.info.run_id}")
    except Exception as exc:  # serveur MLflow absent : la base reste valide
        print(f"  [WARN] tracé MLflow impossible : {exc}")

    print(f"\nTerminé en {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
