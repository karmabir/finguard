"""Baseline fraud scorer with a temporal split (train < val < test by `step`), so evaluation
mimics scoring future transactions. Two feature sets:
  full     - includes post-transaction balances, which leak the label in PaySim (fraud often zeroes
             the origin account); reported only to show the size of the leak
  pre_txn  - only what is known before the transaction settles (the honest setting)
The threshold is chosen on the validation period for a target precision, then applied to test."""
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

from .data import DATA, load_frame

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
FEATURE_SETS = {
    "pre_txn": ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "type_code"],
    "full": ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "type_code"],
}


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


def train_eval(feature_set="pre_txn", target_precision=0.8, seed=0, df=None):
    df = load_frame() if df is None else df
    tr, va, te = temporal_split(df)
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    clf.fit(featurize(tr, feature_set), tr["isFraud"])
    thr = pick_threshold(va["isFraud"], clf.predict_proba(featurize(va, feature_set))[:, 1], target_precision)
    s = clf.predict_proba(featurize(te, feature_set))[:, 1]
    pred = (s >= thr).astype(int)
    y = te["isFraud"]
    return clf, {
        "feature_set": feature_set,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test_fraud_rate": float(y.mean()),
        "test_pr_auc": float(average_precision_score(y, s)),
        "threshold": thr,
        "test_precision": float(precision_score(y, pred, zero_division=0)),
        "test_recall": float(recall_score(y, pred)),
    }


if __name__ == "__main__":
    for fs in FEATURE_SETS:
        print(train_eval(fs)[1])


MODEL_PATH = DATA / "model.joblib"


def save_model(feature_set="pre_txn"):
    clf, info = train_eval(feature_set)
    joblib.dump({"clf": clf, "threshold": info["threshold"], "feature_set": feature_set, "info": info}, MODEL_PATH)
    return info


def load_model():
    return joblib.load(MODEL_PATH)
