"""Baseline fraud scorer with a temporal split (train < val < test by `step`), so evaluation
mimics scoring future transactions. Two feature sets:
  full     - includes post-transaction balances, which leak the label in PaySim (fraud often zeroes
             the origin account); reported only to show the size of the leak
  pre_txn  - only what is known before the transaction settles (the honest setting)
The threshold is chosen on the validation period for a target precision, then applied to test."""
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

from .data import DATA, load_frame

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
FEATURE_SETS = {
    "pre_txn": ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "type_code"],
    "full": ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "type_code"],
}
DRIFT_FEATURES = ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "type"]
DEFAULT_MODEL = "gradient_boosting"
MODEL_NAMES = {DEFAULT_MODEL: "Gradient boosting", "random_forest": "Random forest",
               "logistic_regression": "Logistic regression"}


def make_drift_reference(df, bins=10):
    """Capture training-distribution buckets used by the drift monitor."""
    reference = {}
    for col in DRIFT_FEATURES:
        if col == "type":
            reference[col] = {"kind": "categorical", "proportions": df[col].value_counts(normalize=True).to_dict()}
            continue
        edges = np.unique(df[col].quantile(np.linspace(0, 1, bins + 1)).to_numpy(dtype=float))
        if len(edges) < 2:
            edges = np.array([float(df[col].iloc[0]) - 1, float(df[col].iloc[0]) + 1])
        edges[0], edges[-1] = -np.inf, np.inf
        proportions = pd.cut(df[col], bins=edges, include_lowest=True).value_counts(sort=False, normalize=True)
        reference[col] = {"kind": "numeric", "edges": edges.tolist(), "proportions": proportions.tolist()}
    return reference


def featurize(df, feature_set="pre_txn"):
    out = df.copy()
    out["type_code"] = out["type"].map({t: i for i, t in enumerate(TYPES)}).fillna(-1).astype(int)
    return out[FEATURE_SETS[feature_set]]


def temporal_split(df, train_frac=0.7, val_frac=0.15):
    """Split by `step` (time), not at random. Returns (train, val, test) frames."""
    lo, hi = df["step"].min(), df["step"].max()
    t1 = lo + (hi - lo) * train_frac
    t2 = lo + (hi - lo) * (train_frac + val_frac)
    return df[df["step"] <= t1], df[(df["step"] > t1) & (df["step"] <= t2)], df[df["step"] > t2]


def pick_threshold(y, scores, target_precision=0.8):
    prec, _, thr = precision_recall_curve(y, scores)
    ok = np.where(prec[:-1] >= target_precision)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5


def _classifier(model_id, feature_set, seed):
    if model_id == DEFAULT_MODEL:
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(n_estimators=40, max_depth=18, min_samples_leaf=4,
                                      max_samples=0.12, n_jobs=2, random_state=seed)
    if model_id == "logistic_regression":
        numeric = [col for col in FEATURE_SETS[feature_set] if col != "type_code"]
        preprocess = ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type_code"]),
        ])
        return make_pipeline(preprocess, LogisticRegression(solver="newton-cholesky", max_iter=150))
    raise ValueError(f"unknown model: {model_id}")


def _train_split(split, feature_set, target_precision, seed, model_id):
    tr, va, te = split
    clf = _classifier(model_id, feature_set, seed)
    clf.fit(featurize(tr, feature_set), tr["isFraud"])
    val_scores = clf.predict_proba(featurize(va, feature_set))[:, 1]
    thr = pick_threshold(va["isFraud"], val_scores, target_precision)
    val_precision = precision_score(va["isFraud"], val_scores >= thr, zero_division=0)
    s = clf.predict_proba(featurize(te, feature_set))[:, 1]
    pred = (s >= thr).astype(int)
    y = te["isFraud"]
    return clf, {
        "feature_set": feature_set,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test_fraud_rate": float(y.mean()),
        "test_pr_auc": float(average_precision_score(y, s)),
        "threshold": thr,
        "validation_precision": float(val_precision),
        "threshold_method": "validation_target" if val_precision >= target_precision else "fallback_0.5",
        "test_precision": float(precision_score(y, pred, zero_division=0)),
        "test_recall": float(recall_score(y, pred)),
    }


def train_eval(feature_set="pre_txn", target_precision=0.8, seed=0, df=None, model_id=DEFAULT_MODEL):
    df = load_frame() if df is None else df
    return _train_split(temporal_split(df), feature_set, target_precision, seed, model_id)


if __name__ == "__main__":
    for fs in FEATURE_SETS:
        print(train_eval(fs)[1])


MODEL_PATH = DATA / "model.joblib"


def train_models(feature_set="pre_txn"):
    columns = ["type", "isFraud", *(c for c in FEATURE_SETS[feature_set] if c != "type_code")]
    split = temporal_split(load_frame()[columns])
    trained = {}
    for model_id in MODEL_NAMES:
        clf, info = _train_split(split, feature_set, 0.8, 0, model_id)
        trained[model_id] = {"clf": clf, "threshold": info["threshold"], "feature_set": feature_set, "info": info}
    # Keep the original top-level schema for existing scoring and monitoring clients.
    primary = trained[DEFAULT_MODEL].copy()
    primary["alternatives"] = {key: value for key, value in trained.items() if key != DEFAULT_MODEL}
    primary["drift_reference"] = make_drift_reference(split[0])
    with NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        joblib.dump(primary, temp_path)
        temp_path.replace(MODEL_PATH)
    finally:
        temp_path.unlink(missing_ok=True)
    return {key: item["info"] for key, item in trained.items()}


def save_model(feature_set="pre_txn"):
    return train_models(feature_set)[DEFAULT_MODEL]


def model_catalog(artifact):
    return {DEFAULT_MODEL: artifact, **artifact.get("alternatives", {})}


@lru_cache(maxsize=1)
def _load_version(path, inode, modification_ns):
    return joblib.load(path)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model not found at {MODEL_PATH}; run `python -m finguard init` first"
        )
    stat = MODEL_PATH.stat()
    return _load_version(MODEL_PATH, stat.st_ino, stat.st_mtime_ns)
