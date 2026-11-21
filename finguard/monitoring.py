"""Operational metrics and feature-drift monitoring."""
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import tools
from .model import DRIFT_FEATURES, load_model

_lock = threading.Lock()
_requests = defaultdict(lambda: {"count": 0, "latency": 0.0})


def record_request(method, path, status, elapsed):
    with _lock:
        row = _requests[(method, path, status)]
        row["count"] += 1
        row["latency"] += elapsed


def prometheus_metrics():
    lines = [
        "# HELP finguard_http_requests_total HTTP requests handled.",
        "# TYPE finguard_http_requests_total counter",
        "# HELP finguard_http_request_duration_seconds HTTP request duration.",
        "# TYPE finguard_http_request_duration_seconds summary",
    ]
    with _lock:
        snapshot = list(_requests.items())
    for (method, path, status), values in snapshot:
        labels = f'method="{method}",path="{path}",status="{status}"'
        lines.append(f"finguard_http_requests_total{{{labels}}} {values['count']}")
        lines.append(f"finguard_http_request_duration_seconds_count{{{labels}}} {values['count']}")
        lines.append(f"finguard_http_request_duration_seconds_sum{{{labels}}} {values['latency']:.6f}")
    return "\n".join(lines) + "\n"


def _psi(expected, actual, epsilon=1e-6):
    expected = np.maximum(np.asarray(expected, dtype=float), epsilon)
    actual = np.maximum(np.asarray(actual, dtype=float), epsilon)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(sample_size=10000):
    """Compare the latest transactions with distributions captured during training."""
    artifact = load_model()
    reference = artifact.get("drift_reference")
    if not reference:
        raise RuntimeError("model has no drift reference; run `python -m finguard init` to retrain it")
    sample_size = max(100, min(int(sample_size), 100_000))
    columns = ", ".join(DRIFT_FEATURES)
    con = tools._connect()
    try:
        recent = pd.read_sql(f"select {columns} from transactions order by step desc, txn_id desc limit ?",
                             con, params=(sample_size,))
    finally:
        con.close()
    if recent.empty:
        raise RuntimeError("no transactions available for drift monitoring")

    features = {}
    for col, ref in reference.items():
        if ref["kind"] == "numeric":
            actual = pd.cut(recent[col], bins=ref["edges"], include_lowest=True).value_counts(
                sort=False, normalize=True).tolist()
            score = _psi(ref["proportions"], actual)
        else:
            categories = sorted(set(ref["proportions"]) | set(recent[col].astype(str)))
            actual_map = recent[col].astype(str).value_counts(normalize=True).to_dict()
            score = _psi([ref["proportions"].get(c, 0) for c in categories],
                         [actual_map.get(c, 0) for c in categories])
        features[col] = {"psi": round(score, 4), "status": "stable" if score < 0.1 else "warning" if score < 0.2 else "drift"}
    worst = max(v["psi"] for v in features.values())
    return {"status": "stable" if worst < 0.1 else "warning" if worst < 0.2 else "drift",
            "sample_size": len(recent), "generated_at": time.time(), "thresholds": {"warning": 0.1, "drift": 0.2},
            "features": features}
