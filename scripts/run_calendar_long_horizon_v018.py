"""Deterministic bootstrap and capability-minimal CLI for V2.3."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_BOOTSTRAP_MARKER = "LIFETWIN_V023_DETERMINISTIC_BOOTSTRAP"
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
    ready = os.environ.get(_BOOTSTRAP_MARKER) == "1" and all(
        os.environ.get(name) == value for name, value in expected.items()
    )
    if ready:
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
_ATTEMPT_ID = "v023-formal-20260810-a1"
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the one-shot frozen V2.3 lifecycle.",
    )
    parser.add_argument(
        "--internal-stage",
        choices=("generation", "prediction"),
    )
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
        if (
            args.attempt_id is not None
            or args.score_root is not None
            or args.termination_root is not None
            or args.repo_root is not None
        ):
            raise SystemExit("generation received an unnecessary capability")
        from lifetwin.experiments.calendar_long_horizon_v018_runner import (  # noqa: PLC0415
            run_isolated_generation_stage,
        )

        run_isolated_generation_stage(
            label_free_root=args.label_free_root,
            sealed_truth_root=args.sealed_truth_root,
        )
        return 0

    if args.internal_stage == "prediction":
        attempt_id = _require_attempt_id(args.attempt_id)
        if (
            args.sealed_truth_root is not None
            or args.score_root is not None
            or args.termination_root is not None
        ):
            raise SystemExit("prediction received a forbidden capability")
        from lifetwin.experiments.calendar_long_horizon_v018_prediction import (  # noqa: PLC0415
            run_isolated_prediction_process_v023,
        )

        run_isolated_prediction_process_v023(
            label_free_root=args.label_free_root,
            attempt_id=attempt_id,
            repo_root=args.repo_root or _PROJECT_ROOT,
        )
        return 0

    attempt_id = _require_attempt_id(args.attempt_id)
    if (
        args.sealed_truth_root is None
        or args.score_root is None
        or args.termination_root is None
    ):
        raise SystemExit(
            "full run requires --sealed-truth-root, --score-root, and "
            "--termination-root"
        )
    from lifetwin.experiments.calendar_long_horizon_v018_runner import (  # noqa: PLC0415
        run_formal_attempt,
    )

    result = run_formal_attempt(
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
