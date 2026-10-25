"""App Gradio — scoring crédit Home Credit en inférence production.

Volet 1 : navigation paginée dans la base SQLite des 48 744 clients du fichier
          test (jamais observés à l'entraînement).
Volet 2 : prédiction sur le client sélectionné, via le **serving MLflow natif**
          (endpoint /invocations du wrapper pyfunc `home-credit-serving`) :
          probabilité de défaut + décision au seuil métier figé (FN = 10 × FP).
Volet 3 : décryptage SHAP (waterfall + commentaires) des facteurs de décision.

Spec : le fichier test ne contient pas de colonne TARGET (vérifié à la construction
de la base) -> le bouton « ground truth » est retiré du cahier des charges.

Traçage : chaque prédiction crée un run MLflow nommé `prod_inference_gradio`
(tag `stage=prod_inference`) — visible côté inférence, distinct des runs d'entraînement.

Lancement :
  .venv/bin/python apps/gradio_app.py             # interface sur :7860
  .venv/bin/python apps/gradio_app.py --selftest  # test headless de la chaîne
Prérequis : serveur MLflow (:5000) + `mlflow models serve` (:5001) actifs.
"""
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import gradio as gr
import mlflow
import mlflow.pyfunc
import shap

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "inference" / "clients.db"
FEATURES_JSON = ROOT / "data" / "inference" / "features.json"
FORMULAS_PATH = ROOT / "data" / "processed" / "openfe_bank_formulas.json"

TRACKING_URI = "http://127.0.0.1:5000"
SERVING_URI = "http://127.0.0.1:5001/invocations"
SERVING_MODEL_URI = "models:/home-credit-serving/1"
PAGE_SIZE = 12

# --------------------------------------------------------------- chargements
if not DB_PATH.exists():
    raise SystemExit(
        f"Base absente : {DB_PATH}\n"
        "Lancer d'abord : .venv/bin/python scripts/build_inference_db.py"
    )
con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)

_meta = json.loads(FEATURES_JSON.read_text())
FEATURES: list[str] = _meta["features"]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("home-credit-scoring")
try:
    scorer = mlflow.pyfunc.load_model(SERVING_MODEL_URI)
except Exception as exc:
    raise SystemExit(
        f"Modèle de serving indisponible ({SERVING_MODEL_URI}) : {exc}\n"
        "Lancer d'abord : mlflow models serve -m 'models:/home-credit-serving/1' "
        "-p 5001 --env-manager local"
    )
def _unwrap_python_model(scorer):
    """Acces au PythonModel du wrapper pyfunc (API publique MLflow 3.x + replis)."""
    for getter in (
        lambda m: m.unwrap_python_model(),       # API publique MLflow 3.x
        lambda m: m._model_impl.python_model,    # wrapper interne
        lambda m: m.python_model,                # style MLflow 2.x
    ):
        try:
            return getter(scorer)
        except Exception:
            continue
    raise RuntimeError("impossible d'extraire le PythonModel du scorer pyfunc")


_pm = _unwrap_python_model(scorer)
_encoder = _pm.encoder
_model = _pm.model
_THRESHOLD = float(_pm.threshold)
_TREE_EXPLAINER = shap.TreeExplainer(_model)
_ENC_NAMES = [n.split("__", 1)[-1] for n in _encoder.get_feature_names_out()]

try:
    _FORMULAS: list[str] = json.loads(FORMULAS_PATH.read_text())
except Exception:
    _FORMULAS = []

DISPLAY_COLS = ["SK_ID_CURR", "CODE_GENDER", "AMT_CREDIT",
                "OCCUPATION_TYPE", "HOUR_APPR_PROCESS_START"]
_page_cache: list[int] = []  # SK_ID_CURR de la page courante


# ------------------------------------------------------------------ helpers
def _feature_label(name: str) -> str:
    """autoFE_f_i -> nom + formule lisible (banc OpenFE)."""
    if name.startswith("autoFE_f_") and _FORMULAS:
        try:
            return f"{name} = {_FORMULAS[int(name.split('_')[-1])]}"
        except (IndexError, ValueError):
            return name
    return name


def load_client(sk_id: int) -> dict | None:
    cur = con.execute("SELECT * FROM clients WHERE SK_ID_CURR = ?", (sk_id,))
    row = cur.fetchone()
    if row is None:
        return None
    values = dict(zip([d[0] for d in cur.description], row))
    return {"sk_id": int(values.pop("SK_ID_CURR")),
            "features": {k: values[k] for k in FEATURES}}


# ------------------------------------------------------- volet 1 : navigation
def browse(search: str, page: int):
    global _page_cache
    pattern = f"%{search.strip()}%" if search.strip() else "%"
    total = con.execute(
        "SELECT COUNT(*) FROM clients WHERE CAST(SK_ID_CURR AS TEXT) LIKE ?",
        (pattern,)).fetchone()[0]
    max_page = max(0, (total - 1) // PAGE_SIZE)
    page = min(max(page, 0), max_page)
    cur = con.execute(
        f"SELECT {', '.join(DISPLAY_COLS)} FROM clients "
        "WHERE CAST(SK_ID_CURR AS TEXT) LIKE ? ORDER BY SK_ID_CURR LIMIT ? OFFSET ?",
        (pattern, PAGE_SIZE, page * PAGE_SIZE))
    rows = cur.fetchall()
    _page_cache = [r[0] for r in rows]
    info = f"**{total:,} clients** — page {page + 1}/{max_page + 1}"
    return gr.update(value=pd.DataFrame(rows, columns=DISPLAY_COLS)), page, info


def goto_page(delta: int, page: int):
    return browse("", page + delta)


def select_row(evt: gr.SelectData, page: int):
    if not _page_cache or evt.index[0] >= len(_page_cache):
        return None, "*Aucun client sélectionné*"
    client = load_client(_page_cache[evt.index[0]])
    raw = client["features"]
    card_lines = [f"### Client **{client['sk_id']}** (fichier test, jamais vu à l'entraînement)"]
    card_lines += [f"- `{k}` : {raw[k]}" for k in FEATURES
                   if not k.startswith("autoFE_f_")]
    card_lines.append(f"\n*+ {sum(k.startswith('autoFE_f_') for k in FEATURES)} "
                      "features autoFE calculées — détail dans le décryptage SHAP*")
    return client, "\n".join(card_lines)


# ------------------------------------------------- volet 2 : prédiction + tracé
def predict(client: dict | None):
    if client is None:
        return "—", "—", "—", "Sélectionnez d'abord un client."
    values = [None if (isinstance(v, float) and np.isnan(v)) else v
              for v in (client["features"][f] for f in FEATURES)]
    payload = {"dataframe_split": {"columns": FEATURES, "data": [values]}}

    t0 = time.perf_counter()
    resp = requests.post(SERVING_URI, json=payload, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    pred = resp.json()["predictions"][0]
    prob, decision = float(pred["prob_default"]), int(pred["decision"])

    if decision == 1:
        decision_txt = "REFUSER la demande (probabilité ≥ seuil)"
        cost_txt = ("Si ce client remboursait : coût **1** (FP — marge perdue). "
                    "Sans refus, le défaut attendu coûterait 10 : le refus est l'option la moins chère.")
    else:
        decision_txt = "ACCEPTER la demande (probabilité < seuil)"
        cost_txt = ("Si ce client faisait défaut : coût **10** (FN). "
                    f"La probabilité estimée ({prob:.1%}) reste sous le seuil {_THRESHOLD:.1%} : "
                    "risque jugé inférieur au gain.")

    try:
        with mlflow.start_run(run_name="prod_inference_gradio") as run:
            mlflow.set_tags({
                "stage": "prod_inference", "source": "gradio_app",
                "serving_uri": SERVING_URI, "model_uri": SERVING_MODEL_URI,
            })
            mlflow.log_params({"sk_id_curr": client["sk_id"]})
            mlflow.log_metrics({"prob_default": prob, "decision": decision,
                                "threshold": _THRESHOLD, "latency_ms": latency_ms})
        trace_txt = (f"latence {latency_ms:.0f} ms · tracé MLflow : run `{run.info.run_id}` "
                     "(filtre UI : tag `stage=prod_inference`)")
    except Exception as exc:
        trace_txt = f"latence {latency_ms:.0f} ms · [WARN] tracé MLflow impossible : {exc}"

    return f"{prob:.2%}", decision_txt, cost_txt, trace_txt


# ------------------------------------------------ volet 3 : décryptage SHAP
def explain(client: dict | None):
    if client is None:
        return None, "Sélectionnez d'abord un client."
    df1 = pd.DataFrame([[client["features"][f] for f in FEATURES]], columns=FEATURES)
    X_enc = _encoder.transform(df1)
    expl = _TREE_EXPLAINER(X_enc)

    plt.figure(figsize=(8, 4.5))
    shap.plots.waterfall(expl[0], max_display=8, show=False)
    fig = plt.gcf()
    plt.tight_layout()

    values = np.asarray(expl[0].values)
    order = np.argsort(-values)
    top_up = [i for i in order[:3] if values[i] > 0]
    top_dn = [i for i in order[::-1][:3] if values[i] < 0]

    def fmt(i: int) -> str:
        return f"- **{_feature_label(_ENC_NAMES[i])}** : {values[i]:+.3f}"

    lines = [f"### Décryptage — client {client['sk_id']}",
             f"Base (taux de défaut moyen) : **{expl[0].base_values:.3f}** (log-odds) → "
             f"probabilité de départ **{1 / (1 + np.exp(-expl[0].base_values)):.2%}**.",
             "\n**Facteurs qui augmentent le risque de défaut :**"]
    lines += [fmt(i) for i in top_up] or ["- aucun"]
    lines += ["\n**Facteurs qui réduisent le risque :**"]
    lines += [fmt(i) for i in top_dn] or ["- aucun"]
    proba_final = 1 / (1 + np.exp(-(expl[0].base_values + values.sum())))
    lines.append(f"\nDécision servie : **{'REFUSER' if proba_final >= _THRESHOLD else 'ACCEPTER'}** "
                 f"(probabilité reconstituée {proba_final:.2%}) au seuil figé {_THRESHOLD:.2%} "
                 "(seuil bayésien théorique FN=10×FP : 1/11 ≈ 9,1 %).")
    return fig, "\n".join(lines)


# ---------------------------------------------------------------------- UI
with gr.Blocks(title="Home Credit — Scoring crédit (inférence MLflow") as demo:
    gr.Markdown(
        "# Home Credit — Scoring crédit (inférence production)\n"
        "Navigateur sur les **48 744 clients du fichier test** (jamais observés à l'entraînement). "
        "Le fichier test ne contient pas de cible : le bouton « ground truth » est retiré du cahier "
        "des charges. Chaque prédiction passe par le **serving MLflow natif** et est tracée dans "
        "MLflow (tag `stage=prod_inference`).")
    client_state = gr.State(None)
    page_state = gr.State(0)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1 — Parcourir les clients")
            search_box = gr.Textbox(label="Rechercher un SK_ID_CURR (vide = tous)")
            with gr.Row():
                prev_btn = gr.Button("◀ Précédent")
                next_btn = gr.Button("Suivant ▶")
            page_info = gr.Markdown()
            clients_df = gr.Dataframe(headers=DISPLAY_COLS, interactive=False,
                                      wrap=True, label="Cliquez une ligne pour sélectionner")
        with gr.Column(scale=1):
            gr.Markdown("## 2 — Prédiction & décryptage")
            client_card = gr.Markdown("*Aucun client sélectionné*")
            predict_btn = gr.Button("🔮 Prédire (via serving MLflow)", variant="primary")
            with gr.Row():
                prob_out = gr.Textbox(label="P(défaut)", scale=1)
                decision_out = gr.Textbox(label="Décision au seuil métier", scale=2)
            cost_out = gr.Textbox(label="Lecture métier (coût FN = 10 × FP)")
            trace_out = gr.Textbox(label="Inférence tracée (MLflow Prod Inference)")
            explain_btn = gr.Button("🧩 Décrypter (SHAP)")
            shap_plot = gr.Plot(label="Waterfall SHAP")
            decode_out = gr.Markdown()

    demo.load(browse, inputs=[search_box, page_state], outputs=[clients_df, page_state, page_info])
    search_box.submit(browse, inputs=[search_box, page_state], outputs=[clients_df, page_state, page_info])
    prev_btn.click(lambda p: goto_page(-1, p), inputs=[page_state], outputs=[clients_df, page_state, page_info])
    next_btn.click(lambda p: goto_page(+1, p), inputs=[page_state], outputs=[clients_df, page_state, page_info])
    clients_df.select(select_row, inputs=[page_state], outputs=[client_state, client_card])
    predict_btn.click(predict, inputs=[client_state],
                      outputs=[prob_out, decision_out, cost_out, trace_out])
    explain_btn.click(explain, inputs=[client_state], outputs=[shap_plot, decode_out])


def selftest(n: int = 3) -> None:
    """Chaîne complète headless : navigation -> prediction -> SHAP -> tracé MLflow."""
    ids = [r[0] for r in con.execute("SELECT SK_ID_CURR FROM clients ORDER BY RANDOM() LIMIT ?", (n,))]
    for sk in ids:
        client = load_client(sk)
        assert client is not None
        prob, decision, cost, trace = predict(client)
        fig, decode = explain(client)
        print(f"\n=== client {sk} ===\np {prob} | {decision}\n{trace}\n{decode.splitlines()[0]}")
        assert fig is not None and "Décryptage" in decode
    print(f"\nselftest OK — {n} clients servis, tracés et décryptés")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
