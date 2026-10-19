"""Governed, read-only tools exposed to the agent (via MCP). Only SELECT on the transactions
table; row cap enforced; every call is audit-logged."""
import json
import re
import sqlite3
import time

from .data import DATA, DB

AUDIT = DATA / "audit.log"
MAX_ROWS = 50
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace)\b", re.I)
# Ground-truth labels must never reach the agent: block queries that mention them.
LABELS = re.compile(r"isfraud|isflaggedfraud|select\s+\*", re.I)
HIDDEN_COLS = ("isFraud", "isFlaggedFraud")


def _audit(tool, args, ok):
    DATA.mkdir(exist_ok=True)
    with open(AUDIT, "a") as f:
        f.write(json.dumps({"t": time.time(), "tool": tool, "args": args, "ok": ok}) + "\n")


def run_select(sql: str):
    ok = not LABELS.search(sql) and sql.strip().lower().startswith("select") and not FORBIDDEN.search(sql) and ";" not in sql.strip().rstrip(";")
    _audit("run_select", {"sql": sql}, ok)
    if not ok:
        raise ValueError("only a single read-only SELECT is allowed")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchmany(MAX_ROWS)]
    finally:
        con.close()


def get_transaction(txn_id: int):
    _audit("get_transaction", {"txn_id": txn_id}, True)
    cols = "txn_id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest"
    rows = run_select(f"select {cols} from transactions where txn_id = {int(txn_id)}")
    return rows[0] if rows else None


def label_for_eval(txn_id: int) -> int:
    """Ground truth for evals only; deliberately not exposed through the agent's tools."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute("select isFraud from transactions where txn_id = ?", (int(txn_id),)).fetchone()[0]
    finally:
        con.close()


_model = None


def score_transaction(txn_id: int):
    """Fraud-model score for one transaction (pre-transaction features only; no label access)."""
    global _model
    import pandas as pd

    from .model import featurize, load_model
    _audit("score_transaction", {"txn_id": txn_id}, True)
    _model = _model or load_model()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        df = pd.read_sql("select txn_id, step, type, amount, oldbalanceOrg, oldbalanceDest from transactions where txn_id = ?",
                         con, params=(int(txn_id),))
    finally:
        con.close()
    if df.empty:
        return None
    s = float(_model["clf"].predict_proba(featurize(df, _model["feature_set"]))[:, 1][0])
    return {"txn_id": int(txn_id), "model_score": round(s, 4), "flag_threshold": round(_model["threshold"], 4),
            "model_flags_fraud": s >= _model["threshold"], "features_used": list(_model["clf"].feature_names_in_)
            if hasattr(_model["clf"], "feature_names_in_") else _model["feature_set"]}
