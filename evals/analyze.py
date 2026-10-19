"""Summarize an agent eval run: confusion matrices for the model flag and the agent decision, bootstrap
95% intervals, where they disagree, groundedness, cost. Usage: python evals/analyze.py [dev|test]"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent


def conf(rows, key):
    tp = sum(1 for r in rows if r["label"] and key(r))
    fn = sum(1 for r in rows if r["label"] and not key(r))
    fp = sum(1 for r in rows if not r["label"] and key(r))
    tn = sum(1 for r in rows if not r["label"] and not key(r))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": round(prec, 3), "recall": round(rec, 3),
            "accuracy": round((tp + tn) / len(rows), 3)}


def boot_ci(rows, key, n=2000, seed=0):
    rnd = random.Random(seed)
    accs = sorted(sum(1 for r in s if key(r) == bool(r["label"])) / len(s)
                  for s in ([rnd.choice(rows) for _ in rows] for _ in range(n)))
    return round(accs[int(0.025 * n)], 3), round(accs[int(0.975 * n)], 3)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    rows = json.loads((HERE / f"agent_results_{which}.json").read_text())["rows"]
    agent = lambda r: r["decision"] != "clear"
    model = lambda r: r["model_flag"]
    out = {"set": which, "n": len(rows),
           "model": {**conf(rows, model), "acc_95ci": boot_ci(rows, model)},
           "agent": {**conf(rows, agent), "acc_95ci": boot_ci(rows, agent)},
           "agent_escalate_only": conf(rows, lambda r: r["decision"] == "escalate"),
           "disagreements": sum(1 for r in rows if agent(r) != model(r)),
           "agent_right_model_wrong": sum(1 for r in rows if agent(r) == bool(r["label"]) and model(r) != bool(r["label"])),
           "model_right_agent_wrong": sum(1 for r in rows if model(r) == bool(r["label"]) and agent(r) != bool(r["label"])),
           "cases_with_ungrounded_numbers": sum(1 for r in rows if r["ungrounded"]),
           "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / len(rows), 2),
           "cost_per_case_usd": round(sum(r["cost"] for r in rows) / len(rows), 4)}
    print(json.dumps(out, indent=2))
    (HERE / f"analysis_{which}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
