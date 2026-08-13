"""Deterministic capability-minimal CLI for the fixed V2.7 profile."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_BOOTSTRAP_MARKER = "LIFETWIN_V027_DETERMINISTIC_BOOTSTRAP"
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _bootstrap_if_needed() -> None:
    expected = {name: "1" for name in _THREAD_VARIABLES}
    expected["PYTHONHASHSEED"] = "0"
    if os.environ.get(_BOOTSTRAP_MARKER) == "1" and all(
        os.environ.get(name) == value for name, value in expected.items()
    ):
        return
    environment = os.environ.copy()
    environment.update(expected)
    environment[_BOOTSTRAP_MARKER] = "1"
    completed = subprocess.run(
        (sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]),
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


_bootstrap_if_needed()

import argparse  # noqa: E402
import json  # noqa: E402


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ATTEMPT_ID = "v027-formal-20260813-a1"
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the one-shot frozen V2.7 lifecycle.",
    )
    parser.add_argument("--internal-stage", choices=("generation", "prediction"))
    parser.add_argument("--attempt-id")
    parser.add_argument("--label-free-root", type=Path, required=True)
    parser.add_argument("--sealed-truth-root", type=Path)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--termination-root", type=Path)
    parser.add_argument("--repo-root", type=Path)
    return parser


def _require_attempt_id(value: str | None) -> str:
    if value != _ATTEMPT_ID:
        raise SystemExit(f"attempt ID must equal {_ATTEMPT_ID}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_stage == "generation":
        if args.sealed_truth_root is None:
            raise SystemExit("generation requires --sealed-truth-root")
        if any(
            value is not None
            for value in (
                args.attempt_id,
                args.score_root,
                args.termination_root,
                args.repo_root,
            )
        ):
            raise SystemExit("generation received an unnecessary capability")
        from lifetwin.experiments.calendar_long_horizon_v022_runner import (  # noqa: PLC0415
            run_isolated_generation_stage_v027,
        )

        run_isolated_generation_stage_v027(
            label_free_root=args.label_free_root,
            sealed_truth_root=args.sealed_truth_root,
        )
        return 0

    if args.internal_stage == "prediction":
        attempt_id = _require_attempt_id(args.attempt_id)
        if any(
            value is not None
            for value in (
                args.sealed_truth_root,
                args.score_root,
                args.termination_root,
            )
        ):
            raise SystemExit("prediction received a forbidden capability")
        from lifetwin.experiments.calendar_long_horizon_v022_runner import (  # noqa: PLC0415
            run_isolated_prediction_process_v027,
        )

        run_isolated_prediction_process_v027(
            label_free_root=args.label_free_root,
            attempt_id=attempt_id,
            repo_root=args.repo_root or _PROJECT_ROOT,
        )
        return 0

    attempt_id = _require_attempt_id(args.attempt_id)
    if any(
        value is None
        for value in (
            args.sealed_truth_root,
            args.score_root,
            args.termination_root,
        )
    ):
        raise SystemExit("full run requires sealed-truth, score, and termination roots")
    from lifetwin.experiments.calendar_long_horizon_v022_runner import (  # noqa: PLC0415
        run_formal_attempt_v027,
    )

    result = run_formal_attempt_v027(
        attempt_id=attempt_id,
        label_free_root=args.label_free_root,
        sealed_truth_root=args.sealed_truth_root,
        score_root=args.score_root,
        termination_root=args.termination_root,
        repo_root=args.repo_root or _PROJECT_ROOT,
    )
    print(
        json.dumps(
            {
                "attempt_id": result.attempt_id,
                "git_commit": result.git_commit,
                "truth_commitment_byte_sha256": (result.truth_commitment_byte_sha256),
                "actual_analysis_hash_ledger_commitment_byte_sha256": (
                    result.actual_analysis_hash_ledger_commitment_byte_sha256
                ),
                "prediction_commitment_byte_sha256": (
                    result.prediction_commitment_byte_sha256
                ),
                "score_status": result.score_status,
                "wall_time_seconds": result.wall_time_seconds,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
