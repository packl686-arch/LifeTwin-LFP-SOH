from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.snl import extract_snl_rpt_trajectories, load_snl_metadata
from lifetwin.experiments.nasa_prefix_loco import canonical_json_sha256
from lifetwin.experiments.snl_rpt_loco import load_snl_rpt_loco_config
from lifetwin.experiments.snl_rpt_post_outcome_audit import (
    audit_snl_rpt_loco_result,
    summarize_snl_rpt_extraction_sensitivity,
)


DEFAULT_CONFIG = Path("configs/experiments/snl_lfp_rpt_loco_v1.json")
DEFAULT_INPUT_DIRECTORY = Path("artifacts/snl-lfp-rpt-loco-v1")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/snl-lfp-rpt-post-outcome-audit")
REST_GAP_HOURS_GRID = (0.5, 1.0, 2.0, 4.0)
DUPLICATE_VISIT_EFC_GRID = (5.0, 10.0, 15.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _paths(output_directory: Path) -> dict[str, Path]:
    return {
        "cells": output_directory / "cell_diagnostics.csv",
        "conditions": output_directory / "condition_diagnostics.csv",
        "models": output_directory / "model_metrics.csv",
        "choices": output_directory / "selector_choice_summary.csv",
        "sensitivity": output_directory / "extraction_sensitivity.csv",
        "summary": output_directory / "audit_summary.json",
    }


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite SNL post-outcome audit artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional replay."
        )


def run(args: argparse.Namespace) -> int:
    config = load_snl_rpt_loco_config(args.config)
    raw_zip = Path(args.raw_zip)
    metadata_xlsx = Path(args.metadata_xlsx)
    if _sha256(raw_zip) != config["dataset"]["raw_zip_sha256"]:
        raise ValueError("SNL raw ZIP SHA-256 changed")
    if _sha256(metadata_xlsx) != config["dataset"]["metadata_xlsx_sha256"]:
        raise ValueError("SNL metadata workbook SHA-256 changed")

    input_directory = Path(args.input_directory)
    output_paths = _paths(Path(args.output_directory))
    _ensure_available(list(output_paths.values()), overwrite=args.overwrite)
    truth = pd.read_parquet(input_directory / "target_truth.parquet")
    predictions = pd.read_parquet(input_directory / "predictions.parquet")
    decisions = pd.read_parquet(input_directory / "selector_decisions.parquet")
    scores = pd.read_csv(
        input_directory / "scores.csv", float_precision="round_trip"
    )
    prediction_manifest = json.loads(
        (input_directory / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    score_summary = json.loads(
        (input_directory / "score_summary.json").read_text(encoding="utf-8")
    )
    cells, conditions, models, choices, summary = audit_snl_rpt_loco_result(
        truth,
        predictions,
        decisions,
        prediction_manifest,
        scores,
        score_summary,
        config,
    )

    metadata, _ = load_snl_metadata(
        metadata_xlsx,
        expected_lfp_rows_sha256=config["dataset"][
            "metadata_lfp_rows_canonical_sha256"
        ],
        expected_cell_count=int(config["dataset"]["physical_cell_count"]),
        expected_condition_count=int(config["dataset"]["condition_cluster_count"]),
    )
    primary_trajectories = pd.read_parquet(
        input_directory / "rpt_trajectories.parquet"
    )
    candidate_runs = []
    for rest_gap_hours in REST_GAP_HOURS_GRID:
        for duplicate_visit_efc in DUPLICATE_VISIT_EFC_GRID:
            trajectories, extraction_audit = extract_snl_rpt_trajectories(
                raw_zip,
                metadata,
                rest_gap_hours=rest_gap_hours,
                duplicate_visit_efc=duplicate_visit_efc,
            )
            candidate_runs.append(
                (
                    rest_gap_hours,
                    duplicate_visit_efc,
                    trajectories,
                    extraction_audit,
                )
            )
    sensitivity, sensitivity_summary = summarize_snl_rpt_extraction_sensitivity(
        primary_trajectories,
        candidate_runs,
        primary_rest_gap_hours=float(config["rpt_adapter"]["long_rest_gap_hours"]),
        primary_duplicate_visit_efc=float(
            config["rpt_adapter"]["adjacent_check_collapse_maximum_efc"]
        ),
    )
    summary.pop("summary_content_sha256")
    summary["extraction_sensitivity"] = sensitivity_summary
    summary["summary_content_sha256"] = canonical_json_sha256(summary)

    _write_csv(cells, output_paths["cells"])
    _write_csv(conditions, output_paths["conditions"])
    _write_csv(models, output_paths["models"])
    _write_csv(choices, output_paths["choices"])
    _write_csv(sensitivity, output_paths["sensitivity"])
    _write_json(summary, output_paths["summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen SNL result and run fixed post-outcome model and "
            "RPT-extraction diagnostics"
        )
    )
    parser.add_argument("raw_zip")
    parser.add_argument("metadata_xlsx")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-directory", default=str(DEFAULT_INPUT_DIRECTORY))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
