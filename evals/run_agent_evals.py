"""Run the investigation agent on a labeled set. Reports decision accuracy, agreement with the model flag,
and groundedness (numbers in the agent's summary/evidence that do not appear in any tool output).
Usage: python evals/run_agent_evals.py [--set dev|test] [--n 20]   (costs API credits)"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finguard import tools  # noqa: E402
from finguard.agent import investigate  # noqa: E402

HERE = Path(__file__).parent
NUM = re.compile(r"\d[\d,]*\.?\d*")


def _norm(tok):
    t = tok.replace(",", "").rstrip(".")
    try:
        f = float(t)
    except ValueError:
        return None
    return round(f, 2)


def ungrounded_numbers(result):
    """Numbers cited by the agent that never appear in tool outputs (tiny ints like 1-3 and policy ids ignored)."""
    seen = {_norm(m) for out in result["tool_outputs"] for m in NUM.findall(out)}
    text = result["summary"] + " " + " ".join(f"{e.get('claim','')} {e.get('source','')}" for e in result["evidence"])
    text = re.sub(r"P-\d+", "", text)
    cited = [_norm(m) for m in NUM.findall(text)]
    return [c for c in cited if c is not None and c > 10 and c not in seen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="continue from the saved partial results")
    args = ap.parse_args()
    cases = json.loads((HERE / f"{args.set}_cases.json").read_text())
    fraud = [c for c in cases if c["label"] == 1]
    legit = [c for c in cases if c["label"] == 0]
    cases = [c for pair in zip(fraud, legit) for c in pair][: args.n]
    out_path = HERE / f"agent_results_{args.set}.json"
    rows = json.loads(out_path.read_text())["rows"] if out_path.exists() and args.resume else []
    done = {r["txn_id"] for r in rows}
    cost = sum(r["cost"] for r in rows)
    for c in cases:
        if c["txn_id"] in done:
            continue
        r = None
        for attempt in range(3):  # retry transient API failures instead of losing the whole run
            try:
                r = investigate(c["txn_id"])
                break
            except Exception as e:
                print(f"txn {c['txn_id']} attempt {attempt + 1} failed: {type(e).__name__}", flush=True)
                time.sleep(5)
        if r is None:
            continue
        assert tools.label_for_eval(c["txn_id"]) == c["label"]
        flagged = r["decision"] in ("review", "escalate")
        model_flag = tools.score_transaction(c["txn_id"])["model_flags_fraud"]
        bad = ungrounded_numbers(r)
        rows.append({"txn_id": c["txn_id"], "label": c["label"], "decision": r["decision"], "correct": flagged == bool(c["label"]),
                     "model_flag": model_flag, "model_correct": model_flag == bool(c["label"]),
                     "agrees_with_model": flagged == model_flag, "ungrounded": bad, "tool_calls": len(r["tool_calls"]),
                     "cost": r["cost_usd"], "summary": r["summary"]})
        cost += r["cost_usd"]
        print({k: rows[-1][k] for k in ("txn_id", "label", "decision", "model_flag", "ungrounded")}, flush=True)
        out_path.write_text(json.dumps({"rows": rows}, indent=1))
    n = len(rows)
    summary = {"set": args.set, "n": n,
               "agent_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
               "model_accuracy": round(sum(r["model_correct"] for r in rows) / n, 3),
               "agent_model_agreement": round(sum(r["agrees_with_model"] for r in rows) / n, 3),
               "cases_with_ungrounded_numbers": sum(1 for r in rows if r["ungrounded"]),
               "total_cost_usd": round(cost, 3)}
    print(json.dumps(summary, indent=2))
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
