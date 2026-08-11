from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.private_schedule_v4_gates import (
    evaluate_private_schedule_v4_gates,
)
from lifetwin.private_artifacts import atomic_write_json, exclusive_run_lock


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply frozen private schedule V4/V4.1 calibration gates."
    )
    parser.add_argument("baseline_scores")
    parser.add_argument("candidate_scores")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    with exclusive_run_lock(output.parent):
        result = evaluate_private_schedule_v4_gates(
            pd.read_csv(args.baseline_scores),
            pd.read_csv(args.candidate_scores),
            _json(args.baseline_summary),
            _json(args.candidate_summary),
            _json(args.preregistration),
        )
        atomic_write_json(result, output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
