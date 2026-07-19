from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys

import pandas as pd

from lifetwin.experiments.calendar_v2_uncertainty_development import (
    PRIMARY_PREFIX,
    build_t40_soc12_5_failure_audit,
    run_calendar_v2_uncertainty_development,
)


DEFAULT_INPUT = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_CONFIG = Path(
    "configs/experiments/naumann_calendar_v2_uncertainty_development.json"
)
DEFAULT_PHASE6_PREDICTIONS = Path(
    "artifacts/calendar_v2_development_v1_hardened/label_free_predictions.csv"
)
DEFAULT_PHASE6_DIAGNOSTICS = Path(
    "artifacts/calendar_v2_development_v1_hardened/target_diagnostics.csv"
)
DEFAULT_ARTIFACT_DIR = Path(
    "artifacts/calendar_v2_uncertainty_development_v1_hardened"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)
EXPECTED_PHASE6_PREDICTIONS_SHA256 = (
    "e22b382445f59fcf3b1ef5de9ab6395817d0350669d255f567305d79c1d28296"
)
EXPECTED_PHASE6_DIAGNOSTICS_SHA256 = (
    "8543e9965778d7fa47bd53c05b8ad67bddee0961675dab197e371f7226cd10fd"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, *, label: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {observed}"
        )
    return observed


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _source_provenance() -> dict[str, object]:
    paths = [
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/lifetwin/data/naumann.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2_uncertainty.py",
        PROJECT_ROOT / "src/lifetwin/experiments/calendar_v2_development.py",
        PROJECT_ROOT
        / "src/lifetwin/experiments/calendar_v2_uncertainty_development.py",
    ]
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths
    }
    encoded = json.dumps(
        source_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_sha256": source_hashes,
        "source_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "python": sys.version,
        "packages": {
            package: importlib_metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }


def _require_new_outputs(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "The uncertainty runner never overwrites artifacts; existing="
            f"{existing}"
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return {
        "path": path.as_posix(),
        "row_count": len(frame),
        "sha256": _sha256(path),
    }


def _write_json(payload: dict[str, object], path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run time-honest prefix-only uncertainty diagnostics for Calendar V2."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase6-predictions", type=Path, default=DEFAULT_PHASE6_PREDICTIONS
    )
    parser.add_argument(
        "--phase6-diagnostics", type=Path, default=DEFAULT_PHASE6_DIAGNOSTICS
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    output_paths = {
        "result": args.artifact_dir / "result.json",
        "label_free_intervals": args.artifact_dir / "label_free_intervals.csv",
        "calibration_condition_scores": (
            args.artifact_dir / "calibration_condition_scores.csv"
        ),
        "calibration_quantiles": args.artifact_dir / "calibration_quantiles.csv",
        "interval_point_scores": args.artifact_dir / "interval_point_scores.csv",
        "interval_condition_metrics": (
            args.artifact_dir / "interval_condition_metrics.csv"
        ),
        "interval_summary": args.artifact_dir / "interval_summary.csv",
        "target_uncertainty_diagnostics": (
            args.artifact_dir / "target_uncertainty_diagnostics.csv"
        ),
        "condition_splits": args.artifact_dir / "condition_splits.csv",
        "t40_soc12_5_failure_audit": (
            args.artifact_dir / "t40_soc12_5_failure_audit.json"
        ),
        "t40_soc12_5_residuals": (
            args.artifact_dir / "t40_soc12_5_residuals.csv"
        ),
        "t40_soc12_5_parameter_path": (
            args.artifact_dir / "t40_soc12_5_parameter_path.csv"
        ),
        "t40_low_soc_neighbor_audit": (
            args.artifact_dir / "t40_low_soc_neighbor_audit.csv"
        ),
    }
    _require_new_outputs(list(output_paths.values()))
    input_hash = _require_hash(
        args.input, EXPECTED_INPUT_SHA256, label="Naumann uncertainty input"
    )
    phase6_prediction_hash = _require_hash(
        args.phase6_predictions,
        EXPECTED_PHASE6_PREDICTIONS_SHA256,
        label="Frozen Phase 6 predictions",
    )
    phase6_diagnostic_hash = _require_hash(
        args.phase6_diagnostics,
        EXPECTED_PHASE6_DIAGNOSTICS_SHA256,
        label="Frozen Phase 6 diagnostics",
    )
    config_file_hash = _sha256(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observations = _read_table(args.input)
    (
        result,
        predictions,
        calibration_scores,
        calibration_quantiles,
        point_scores,
        condition_metrics,
        interval_summary,
        target_diagnostics,
        splits,
    ) = run_calendar_v2_uncertainty_development(observations, config=config)
    failure_audit, residuals, parameter_path, neighbor_audit = (
        build_t40_soc12_5_failure_audit(
            observations,
            phase6_predictions=_read_table(args.phase6_predictions),
            phase6_diagnostics=_read_table(args.phase6_diagnostics),
        )
    )
    artifacts = {
        "label_free_intervals": _write_csv(
            predictions, output_paths["label_free_intervals"]
        ),
        "calibration_condition_scores": _write_csv(
            calibration_scores, output_paths["calibration_condition_scores"]
        ),
        "calibration_quantiles": _write_csv(
            calibration_quantiles, output_paths["calibration_quantiles"]
        ),
        "interval_point_scores": _write_csv(
            point_scores, output_paths["interval_point_scores"]
        ),
        "interval_condition_metrics": _write_csv(
            condition_metrics, output_paths["interval_condition_metrics"]
        ),
        "interval_summary": _write_csv(
            interval_summary, output_paths["interval_summary"]
        ),
        "target_uncertainty_diagnostics": _write_csv(
            target_diagnostics, output_paths["target_uncertainty_diagnostics"]
        ),
        "condition_splits": _write_csv(splits, output_paths["condition_splits"]),
        "t40_soc12_5_failure_audit": _write_json(
            failure_audit, output_paths["t40_soc12_5_failure_audit"]
        ),
        "t40_soc12_5_residuals": _write_csv(
            residuals, output_paths["t40_soc12_5_residuals"]
        ),
        "t40_soc12_5_parameter_path": _write_csv(
            parameter_path, output_paths["t40_soc12_5_parameter_path"]
        ),
        "t40_low_soc_neighbor_audit": _write_csv(
            neighbor_audit, output_paths["t40_low_soc_neighbor_audit"]
        ),
    }
    result["t40_soc12_5_failure_audit"] = failure_audit
    result["provenance"] = {
        "input_path": args.input.as_posix(),
        "input_sha256": input_hash,
        "config_path": args.config.as_posix(),
        "config_file_sha256": config_file_hash,
        "frozen_phase6_predictions_path": args.phase6_predictions.as_posix(),
        "frozen_phase6_predictions_sha256": phase6_prediction_hash,
        "frozen_phase6_diagnostics_path": args.phase6_diagnostics.as_posix(),
        "frozen_phase6_diagnostics_sha256": phase6_diagnostic_hash,
        **_source_provenance(),
    }
    result["artifacts"] = artifacts
    _write_json(result, output_paths["result"])
    primary = [
        row
        for row in result["primary_prefix_summary"]
        if int(row["prefix_checkups"]) == PRIMARY_PREFIX
    ]
    print(
        json.dumps(
            {
                "result": output_paths["result"].as_posix(),
                "status": result["status"],
                "development_gate": result["development_gate"],
                "primary_prefix_summary": primary,
                "t40_primary_driver": failure_audit["classification"][
                    "primary_driver"
                ],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
