"""Sample a small labeled dev set from the held-out test period (same split as the model).
Balanced fraud/legit, fixed seed. Writes evals/dev_cases.json: [{"txn_id": int, "label": 0/1}]."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finguard.data import load_frame  # noqa: E402
from finguard.model import temporal_split  # noqa: E402


def main(n_per_class=10, seed=0):
    te = temporal_split(load_frame())[2]
    cases = []
    for label in (1, 0):
        sub = te[te["isFraud"] == label]
        # legit sample is restricted to TRANSFER/CASH_OUT so the agent can't win on type alone
        if label == 0:
            sub = sub[sub["type"].isin(["TRANSFER", "CASH_OUT"])]
        cases += [{"txn_id": int(i), "label": label} for i in sub.sample(n_per_class, random_state=seed)["txn_id"]]
    out = Path(__file__).with_name("dev_cases.json")
    out.write_text(json.dumps(cases, indent=1))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
