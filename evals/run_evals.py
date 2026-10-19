"""Eval harness. Model evals use the held-out test period only (temporal split), for both the
leaky and the honest feature set. Agent-level evals (decision accuracy vs labels, citation
groundedness, cost per case, prompt-injection suite) plug in below once the copilot exists."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finguard.model import FEATURE_SETS, train_eval  # noqa: E402


def main():
    results = [train_eval(fs)[1] for fs in FEATURE_SETS]
    print(json.dumps(results, indent=2))
    # TODO: agent evals - decision accuracy vs labels, citation groundedness, prompt-injection suite


if __name__ == "__main__":
    main()
