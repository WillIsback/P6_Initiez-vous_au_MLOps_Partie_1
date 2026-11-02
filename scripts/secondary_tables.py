"""Agrégats des 6 tables secondaires Home Credit, au grain client (SK_ID_CURR).

Module partagé par le notebook (section 5bis) et `scripts/build_inference_db.py` :
l'entraînement et l'inférence calculent exactement les mêmes features.

Schéma relationnel :
    application_{train|test} (1 ligne / client, clé SK_ID_CURR)
      ├── bureau (crédits déclarés par d'autres organismes, clé SK_ID_BUREAU)
      │     └── bureau_balance (historique mensuel de chaque crédit bureau)
      └── previous_application (demandes passées chez Home Credit, clé SK_ID_PREV)
            ├── installments_payments (échéances et paiements)
            ├── POS_CASH_balance (historique mensuel des prêts POS / cash)
            └── credit_card_balance (historique mensuel des cartes de crédit)

Pas de fuite de cible : aucun agrégat n'utilise TARGET ; ils ne dépendent que de
l'historique du client, on peut donc les calculer une fois pour tous les clients.

Convention : les colonnes `*_COUNT` valent 0 pour un client sans historique dans
la table (absence d'historique = information) ; les autres agrégats restent NaN
et sont imputés plus loin par la préparation (section 6), fit sur le train.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SECONDARY_FILES = [
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]
DAYS_SENTINEL = 365243  # même sentinelle « 1000 ans » que DAYS_EMPLOYED


def _flat(agg: pd.DataFrame, prefix: str) -> pd.DataFrame:
    agg.columns = [f"{prefix}_{c}_{s}".upper() for c, s in agg.columns]
    return agg


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _count_where(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Nombre de lignes vérifiant `mask`, par client (vectorisé)."""
    return mask.astype("int32").groupby(df["SK_ID_CURR"]).sum()


def agg_bureau(raw_dir: Path) -> pd.DataFrame:
    bur = pd.read_csv(raw_dir / "bureau.csv")
    bb = pd.read_csv(raw_dir / "bureau_balance.csv")

    # bureau_balance -> grain crédit bureau : retards (STATUS 1..5 = mois de retard)
    bb["LATE"] = bb["STATUS"].isin(list("12345")).astype("int8")
    bb["DPD_LEVEL"] = pd.to_numeric(bb["STATUS"], errors="coerce").fillna(0)
    bb_per_credit = bb.groupby("SK_ID_BUREAU").agg(
        MONTHS=("MONTHS_BALANCE", "size"),
        LATE_RATIO=("LATE", "mean"),
        DPD_LEVEL_MAX=("DPD_LEVEL", "max"),
    )
    bur = bur.join(bb_per_credit, on="SK_ID_BUREAU")

    bur["DEBT_CREDIT_RATIO"] = _ratio(bur["AMT_CREDIT_SUM_DEBT"], bur["AMT_CREDIT_SUM"])
    g = bur.groupby("SK_ID_CURR")
    out = _flat(g.agg({
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum"],
        "AMT_CREDIT_SUM_LIMIT": ["sum"],
        "AMT_ANNUITY": ["mean"],
        "DEBT_CREDIT_RATIO": ["mean", "max"],
        "MONTHS": ["mean"],
        "LATE_RATIO": ["mean", "max"],
        "DPD_LEVEL_MAX": ["max"],
    }), "BUR")
    out["BUR_COUNT"] = g.size()
    out["BUR_ACTIVE_COUNT"] = _count_where(bur, bur["CREDIT_ACTIVE"] == "Active")
    out["BUR_ACTIVE_RATIO"] = out["BUR_ACTIVE_COUNT"] / out["BUR_COUNT"]
    out["BUR_TOTAL_DEBT_CREDIT_RATIO"] = _ratio(
        out["BUR_AMT_CREDIT_SUM_DEBT_SUM"], out["BUR_AMT_CREDIT_SUM_SUM"])
    for label, ctype in [("CONSUMER", "Consumer credit"), ("CARD", "Credit card"),
                         ("CAR", "Car loan"), ("MORTGAGE", "Mortgage"),
                         ("MICROLOAN", "Microloan")]:
        out[f"BUR_TYPE_{label}_COUNT"] = _count_where(bur, bur["CREDIT_TYPE"] == ctype)

    act = bur[bur["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR")
    out["BUR_ACT_AMT_CREDIT_SUM_SUM"] = act["AMT_CREDIT_SUM"].sum()
    out["BUR_ACT_AMT_CREDIT_SUM_DEBT_SUM"] = act["AMT_CREDIT_SUM_DEBT"].sum()
    out["BUR_ACT_DAYS_CREDIT_MEAN"] = act["DAYS_CREDIT"].mean()
    return out


def agg_previous(raw_dir: Path) -> pd.DataFrame:
    prev = pd.read_csv(raw_dir / "previous_application.csv")
    for c in [c for c in prev.columns if c.startswith("DAYS_")]:
        prev.loc[prev[c] == DAYS_SENTINEL, c] = np.nan
    prev["APP_CREDIT_RATIO"] = _ratio(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])

    g = prev.groupby("SK_ID_CURR")
    out = _flat(g.agg({
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean"],
        "AMT_CREDIT": ["mean", "max"],
        "AMT_GOODS_PRICE": ["mean"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "RATE_DOWN_PAYMENT": ["mean"],
        "APP_CREDIT_RATIO": ["mean", "min"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "DAYS_LAST_DUE_1ST_VERSION": ["max"],
        "DAYS_TERMINATION": ["max"],
    }), "PREV")
    out["PREV_COUNT"] = g.size()
    for label, status in [("APPROVED", "Approved"), ("REFUSED", "Refused"),
                          ("CANCELED", "Canceled")]:
        out[f"PREV_{label}_COUNT"] = _count_where(prev, prev["NAME_CONTRACT_STATUS"] == status)
    out["PREV_REFUSED_RATIO"] = out["PREV_REFUSED_COUNT"] / out["PREV_COUNT"]
    for label, ctype in [("CASH", "Cash loans"), ("CONSUMER", "Consumer loans"),
                         ("REVOLVING", "Revolving loans")]:
        out[f"PREV_TYPE_{label}_COUNT"] = _count_where(prev, prev["NAME_CONTRACT_TYPE"] == ctype)

    apr = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR")
    out["PREV_APR_AMT_CREDIT_MEAN"] = apr["AMT_CREDIT"].mean()
    out["PREV_APR_DAYS_DECISION_MAX"] = apr["DAYS_DECISION"].max()
    ref = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR")
    out["PREV_REF_AMT_APPLICATION_MEAN"] = ref["AMT_APPLICATION"].mean()
    out["PREV_REF_DAYS_DECISION_MAX"] = ref["DAYS_DECISION"].max()
    return out


def agg_installments(raw_dir: Path) -> pd.DataFrame:
    ins = pd.read_csv(raw_dir / "installments_payments.csv")
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAYMENT_RATIO"] = _ratio(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["LATE"] = (ins["DPD"] > 0).astype("int8")
    ins["UNDERPAID"] = (ins["PAYMENT_DIFF"] > 0).astype("int8")

    g = ins.groupby("SK_ID_CURR")
    out = _flat(g.agg({
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean"],
        "LATE": ["mean"],
        "UNDERPAID": ["mean"],
        "PAYMENT_DIFF": ["mean", "max", "sum"],
        "PAYMENT_RATIO": ["mean", "min"],
        "AMT_PAYMENT": ["sum", "mean"],
        "AMT_INSTALMENT": ["sum"],
        "DAYS_ENTRY_PAYMENT": ["max"],
        "NUM_INSTALMENT_VERSION": ["nunique"],
    }), "INST")
    out["INST_COUNT"] = g.size()
    out["INST_PREV_COUNT"] = g["SK_ID_PREV"].nunique()

    recent = ins[ins["DAYS_INSTALMENT"] >= -365].groupby("SK_ID_CURR")
    out["INST_LAST1Y_DPD_MEAN"] = recent["DPD"].mean()
    out["INST_LAST1Y_LATE_MEAN"] = recent["LATE"].mean()
    return out


def agg_pos_cash(raw_dir: Path) -> pd.DataFrame:
    pos = pd.read_csv(raw_dir / "POS_CASH_balance.csv")
    pos["LATE"] = (pos["SK_DPD"] > 0).astype("int8")
    pos["COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype("int8")
    g = pos.groupby("SK_ID_CURR")
    out = _flat(g.agg({
        "MONTHS_BALANCE": ["min", "max"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
        "LATE": ["mean"],
        "COMPLETED": ["mean"],
    }), "POS")
    out["POS_COUNT"] = g.size()
    out["POS_PREV_COUNT"] = g["SK_ID_PREV"].nunique()
    return out


def agg_credit_card(raw_dir: Path) -> pd.DataFrame:
    cc = pd.read_csv(raw_dir / "credit_card_balance.csv")
    cc["UTILIZATION"] = _ratio(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["PAYMENT_MIN_RATIO"] = _ratio(cc["AMT_PAYMENT_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    cc["LATE"] = (cc["SK_DPD"] > 0).astype("int8")
    g = cc.groupby("SK_ID_CURR")
    out = _flat(g.agg({
        "AMT_BALANCE": ["mean", "max"],
        "UTILIZATION": ["mean", "max"],
        "AMT_DRAWINGS_CURRENT": ["mean"],
        "AMT_PAYMENT_CURRENT": ["mean"],
        "PAYMENT_MIN_RATIO": ["mean"],
        "CNT_DRAWINGS_CURRENT": ["mean"],
        "SK_DPD": ["max", "mean"],
        "LATE": ["mean"],
    }), "CC")
    out["CC_COUNT"] = g.size()
    out["CC_PREV_COUNT"] = g["SK_ID_PREV"].nunique()
    return out


AGGREGATORS = {
    "bureau + bureau_balance": agg_bureau,
    "previous_application": agg_previous,
    "installments_payments": agg_installments,
    "POS_CASH_balance": agg_pos_cash,
    "credit_card_balance": agg_credit_card,
}


def build_aggregates(raw_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Calcule tous les agrégats (index = SK_ID_CURR, une ligne par client connu)."""
    import time
    parts = []
    for name, fn in AGGREGATORS.items():
        t0 = time.perf_counter()
        part = fn(Path(raw_dir)).astype("float64")
        parts.append(part)
        if verbose:
            print(f"  {name:26s} {part.shape[1]:3d} features | {len(part):>7,} clients "
                  f"| {time.perf_counter() - t0:5.1f} s")
    agg = pd.concat(parts, axis=1)
    agg.index.name = "SK_ID_CURR"
    return agg


def count_columns(agg: pd.DataFrame) -> list[str]:
    return [c for c in agg.columns if c.endswith("_COUNT")]


def join_aggregates(app: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Jointure gauche sur SK_ID_CURR ; `*_COUNT` absents -> 0 (pas d'historique)."""
    out = app.merge(agg, how="left", left_on="SK_ID_CURR", right_index=True, validate="one_to_one")
    assert len(out) == len(app), "la jointure a dupliqué des lignes"
    cnt = count_columns(agg)
    out[cnt] = out[cnt].fillna(0)
    return out


def load_or_build(raw_dir: Path, cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Agrégats depuis le cache parquet, ou recalcul (puis mise en cache)."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    agg = build_aggregates(raw_dir)
    agg.to_parquet(cache_path)
    return agg
