from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import lifetwin.experiments.private_dual_clock_post_outcome_audit as audit_impl
from lifetwin.experiments.nasa_prefix_loco import canonical_json_sha256
from lifetwin.experiments.private_dual_clock_post_outcome_audit import (
    audit_private_dual_clock_v3,
)
from lifetwin.experiments.private_dual_clock_prior_v3 import (
    score_private_dual_clock_prior_v3,
)
from lifetwin.private_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    build_completion_manifest,
    exclusive_run_lock,
    file_sha256,
    verify_completion_manifest,
)


DEFAULT_INPUT_DIRECTORY = Path("artifacts/private-dual-clock-prior-v3")
DEFAULT_TRUTH_DIRECTORY = Path("artifacts/snl-lfp-rpt-loco-v1")
DEFAULT_OUTPUT_DIRECTORY = Path(
    "artifacts/private-dual-clock-prior-v3-post-outcome-audit"
)


def _paths(output_directory: Path) -> dict[str, Path]:
    return {
        "cells": output_directory / "cell_diagnostics.csv",
        "conditions": output_directory / "condition_diagnostics.csv",
        "summary": output_directory / "audit_summary.json",
        "complete": output_directory / "run_complete.json",
    }


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite private V3 audit artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional private replay."
        )


def run(args: argparse.Namespace) -> int:
    input_directory = Path(args.input_directory)
    truth_directory = Path(args.truth_directory)
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    _ensure_available(list(paths.values()), overwrite=args.overwrite)

    upstream_completion = json.loads(
        (input_directory / "run_complete.json").read_text(encoding="utf-8")
    )
    upstream_verification = verify_completion_manifest(
        input_directory,
        upstream_completion,
        required_names=[
            "config",
            "predictions",
            "decisions",
            "decisions_csv",
            "manifest",
            "scores",
            "summary",
            "capsule",
        ],
    )
    config = json.loads(
        (input_directory / "private_config.json").read_text(encoding="utf-8")
    )
    truth = pd.read_parquet(truth_directory / "target_truth.parquet")
    predictions = pd.read_parquet(input_directory / "predictions.parquet")
    decisions = pd.read_parquet(input_directory / "model_decisions.parquet")
    manifest = json.loads(
        (input_directory / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    scores = pd.read_csv(
        input_directory / "scores.csv", float_precision="round_trip"
    )
    score_summary = json.loads(
        (input_directory / "score_summary.json").read_text(encoding="utf-8")
    )
    replay_scores, replay_summary = score_private_dual_clock_prior_v3(
        truth, predictions, decisions, manifest, config
    )
    pd.testing.assert_frame_equal(scores, replay_scores, check_exact=True)
    if score_summary != replay_summary:
        raise ValueError("Private V3 score summary does not replay exactly")

    cells, conditions, summary = audit_private_dual_clock_v3(
        truth, predictions, decisions, scores, config
    )
    summary.pop("summary_content_sha256")
    summary["upstream_completion_verification"] = upstream_verification
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    atomic_write_csv(cells, paths["cells"])
    atomic_write_csv(conditions, paths["conditions"])
    atomic_write_json(summary, paths["summary"])
    completion = build_completion_manifest(
        output_directory,
        {name: path for name, path in paths.items() if name != "complete"},
        metadata={
            "experiment_id": "private_dual_clock_prior_v3_post_outcome_audit",
            "upstream_completion_manifest_sha256": upstream_verification[
                "manifest_content_sha256"
            ],
            "audit_summary_content_sha256": summary["summary_content_sha256"],
            "implementation_file_sha256": file_sha256(Path(audit_impl.__file__)),
            "runner_file_sha256": file_sha256(Path(__file__)),
            "public_release_permitted": False,
        },
    )
    atomic_write_json(completion, paths["complete"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the private outcome-exposed V3 failure-mode audit."
    )
    parser.add_argument("--input-directory", default=str(DEFAULT_INPUT_DIRECTORY))
    parser.add_argument("--truth-directory", default=str(DEFAULT_TRUTH_DIRECTORY))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with exclusive_run_lock(Path(args.output_directory)):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
