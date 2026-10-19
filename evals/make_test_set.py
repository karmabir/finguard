"""Larger labeled test set from the held-out period, disjoint from the dev set. 50 fraud + 50 legit
(legit restricted to TRANSFER/CASH_OUT). Writes evals/test_cases.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finguard.data import load_frame  # noqa: E402
from finguard.model import temporal_split  # noqa: E402

HERE = Path(__file__).parent


def main(n_per_class=50, seed=1):
    dev_ids = {c["txn_id"] for c in json.loads((HERE / "dev_cases.json").read_text())}
    te = temporal_split(load_frame())[2]
    te = te[~te["txn_id"].isin(dev_ids)]
    cases = []
    for label in (1, 0):
        sub = te[te["isFraud"] == label]
        if label == 0:
            sub = sub[sub["type"].isin(["TRANSFER", "CASH_OUT"])]
        cases += [{"txn_id": int(i), "label": label} for i in sub.sample(n_per_class, random_state=seed)["txn_id"]]
    (HERE / "test_cases.json").write_text(json.dumps(cases, indent=1))
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
