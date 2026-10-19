"""Transaction data: loads PaySim (data/paysim.csv) if present, else generates a synthetic stand-in
so the pipeline runs end to end. Swap in the real dataset before reporting any numbers."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
DB = DATA / "finguard.db"
PAYSIM = DATA / "paysim.csv"


def synthetic(n=20000, fraud_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "step": rng.integers(1, 744, n),
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n, p=[.35, .1, .3, .2, .05]),
        "amount": np.round(rng.lognormal(7, 1.5, n), 2),
        "oldbalanceOrg": np.round(rng.lognormal(8, 1.5, n), 2),
        "newbalanceOrig": 0.0,
        "oldbalanceDest": np.round(rng.lognormal(8, 1.5, n), 2),
        "isFraud": 0,
    })
    risky = df["type"].isin(["TRANSFER", "CASH_OUT"])
    p = np.where(risky, fraud_rate * 4, fraud_rate * 0.05) * (1 + (df["amount"] > df["oldbalanceOrg"] * 0.9))
    df["isFraud"] = (rng.random(n) < p).astype(int)
    df["newbalanceOrig"] = np.where(df["isFraud"] == 1, 0.0, np.maximum(df["oldbalanceOrg"] - df["amount"], 0)).round(2)
    df.insert(0, "txn_id", np.arange(n))
    return df


def load_frame():
    if PAYSIM.exists():
        df = pd.read_csv(PAYSIM)
        df.insert(0, "txn_id", np.arange(len(df)))
        return df
    return synthetic()


def build_db():
    DATA.mkdir(exist_ok=True)
    df = load_frame()
    with sqlite3.connect(DB) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)
    return len(df)


if __name__ == "__main__":
    print(f"loaded {build_db()} transactions into {DB}")
