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


def _connect():
    if not DB.exists():
        raise FileNotFoundError(f"database not found at {DB}; run `python -m finguard init` first")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def _select_authorizer(action, arg1, arg2, _database, _trigger):
    """Restrict reads to non-label columns on the transactions table."""
    if action == sqlite3.SQLITE_READ:
        if arg1 != "transactions" or arg2.lower() in {c.lower() for c in HIDDEN_COLS}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def run_select(sql: str):
    ok = not LABELS.search(sql) and sql.strip().lower().startswith("select") and not FORBIDDEN.search(sql) and ";" not in sql.strip().rstrip(";")
    if not ok:
        _audit("run_select", {"sql": sql}, False)
        raise ValueError("only a single read-only SELECT is allowed")
    con = _connect()
    try:
        con.set_authorizer(_select_authorizer)
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(MAX_ROWS)]
        _audit("run_select", {"sql": sql, "rows": len(rows)}, True)
        return rows
    except sqlite3.Error:
        _audit("run_select", {"sql": sql}, False)
        raise
    finally:
        con.close()


def get_transaction(txn_id: int):
    _audit("get_transaction", {"txn_id": txn_id}, True)
    cols = "txn_id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest"
    rows = run_select(f"select {cols} from transactions where txn_id = {int(txn_id)}")
    return rows[0] if rows else None


def label_for_eval(txn_id: int) -> int:
    """Ground truth for evals only; deliberately not exposed through the agent's tools."""
    con = _connect()
    try:
        row = con.execute("select isFraud from transactions where txn_id = ?", (int(txn_id),)).fetchone()
        if row is None:
            raise LookupError(f"transaction {txn_id} not found")
        return row[0]
    finally:
        con.close()


def _scoring_row(txn_id):
    import pandas as pd
    con = _connect()
    try:
        return pd.read_sql("select txn_id, step, type, amount, oldbalanceOrg, oldbalanceDest "
                           "from transactions where txn_id = ?", con, params=(int(txn_id),))
    finally:
        con.close()


def _score_row(frame, model_id, artifact):
    from .model import MODEL_NAMES, featurize
    features = featurize(frame, artifact["feature_set"])
    score = float(artifact["clf"].predict_proba(features)[0, 1])
    threshold = artifact["threshold"]
    display_score = round(score, 8)
    if display_score in (0.0, 1.0):
        display_score = score  # Keep real endpoints distinct from tiny nonzero tails.
    return {"txn_id": int(frame.iloc[0]["txn_id"]), "model_id": model_id, "name": MODEL_NAMES[model_id],
            "model_score": display_score, "flag_threshold": round(threshold, 6),
            "model_flags_fraud": score >= threshold, "features_used": features.columns.tolist()}


class ModelUnavailable(ValueError):
    pass


def score_transaction(txn_id: int, model_id="gradient_boosting"):
    """Score one transaction with a trained model using only pre-transaction data."""
    from .model import load_model, model_catalog
    catalog = model_catalog(load_model())
    if model_id not in catalog:
        raise ModelUnavailable(f"Model unavailable: {model_id}. Available models: {', '.join(catalog)}")
    _audit("score_transaction", {"txn_id": txn_id, "model_id": model_id}, True)
    frame = _scoring_row(txn_id)
    return None if frame.empty else _score_row(frame, model_id, catalog[model_id])


def compare_scores(txn_id: int):
    from .model import load_model, model_catalog
    _audit("compare_scores", {"txn_id": txn_id}, True)
    frame = _scoring_row(txn_id)
    if frame.empty:
        return None
    return [_score_row(frame, key, item) for key, item in model_catalog(load_model()).items()]
