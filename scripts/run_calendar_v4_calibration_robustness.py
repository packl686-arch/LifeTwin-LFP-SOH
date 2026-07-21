from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import pandas as pd

from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
)
from lifetwin.experiments.calendar_v4_calibration_robustness import (
    EXPECTED_INPUT_FILE_SHA256,
    run_calendar_v4_calibration_robustness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_UPSTREAM_CONFIG = Path(
    "configs/experiments/naumann_calendar_v4_hybrid_development.json"
)
DEFAULT_AUDIT_CONFIG = Path(
    "configs/experiments/naumann_calendar_v4_calibration_robustness.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/calendar_v4_calibration_robustness_v1")
OUTPUT_FILES = {
    "result": "result.json",
    "candidate_label_free_predictions": ("candidate_label_free_predictions.csv"),
    "candidate_condition_scores": "candidate_condition_scores.csv",
    "baseline_route_metrics": "baseline_route_metrics.csv",
    "baseline_condition_metrics": "baseline_condition_metrics.csv",
    "loco_route_metrics": "loco_route_metrics.csv",
    "loco_condition_metrics": "loco_condition_metrics.csv",
    "partition_catalog": "partition_catalog.csv",
    "partition_route_metrics": "partition_route_metrics.csv",
    "partition_condition_metrics": "partition_condition_metrics.csv",
    "sensitivity_summary": "sensitivity_summary.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/lifetwin/data/naumann.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2_uncertainty.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v3_activation.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v4_hybrid.py",
        PROJECT_ROOT / "src/lifetwin/experiments/calendar_v4_hybrid_development.py",
        PROJECT_ROOT / "src/lifetwin/experiments/calendar_v4_calibration_robustness.py",
    )
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths
    }
    encoded = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "source_sha256": source_hashes,
        "source_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "python": sys.version,
        "packages": {
            package: importlib_metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return {"row_count": len(frame), "sha256": _sha256(path)}


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(
    input_path: Path,
    upstream_config_path: Path,
    audit_config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            "The Calendar V4 robustness runner never overwrites an evidence "
            f"directory: {output_dir}"
        )
    input_sha256 = _sha256(input_path)
    if input_sha256 != EXPECTED_INPUT_FILE_SHA256:
        raise ValueError(
            "Naumann robustness input SHA-256 mismatch: expected "
            f"{EXPECTED_INPUT_FILE_SHA256}, found {input_sha256}"
        )
    upstream_config_file_sha256 = _sha256(upstream_config_path)
    audit_config_file_sha256 = _sha256(audit_config_path)
    upstream_config = json.loads(upstream_config_path.read_text(encoding="utf-8"))
    audit_config = json.loads(audit_config_path.read_text(encoding="utf-8"))
    observations = pd.read_csv(input_path)
    canonical_outcome_sha256 = canonical_naumann_outcome_sha256(observations)
    if canonical_outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError("Naumann robustness canonical outcome snapshot mismatch")
    outputs = run_calendar_v4_calibration_robustness(
        observations,
        upstream_config=upstream_config,
        audit_config=audit_config,
    )
    result = outputs[0]
    frames = dict(
        zip(
            (
                "candidate_label_free_predictions",
                "candidate_condition_scores",
                "baseline_route_metrics",
                "baseline_condition_metrics",
                "loco_route_metrics",
                "loco_condition_metrics",
                "partition_catalog",
                "partition_route_metrics",
                "partition_condition_metrics",
                "sensitivity_summary",
            ),
            outputs[1:],
            strict=True,
        )
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        artifacts: dict[str, object] = {}
        for name, frame in frames.items():
            metadata = _write_csv(frame, staging / OUTPUT_FILES[name])
            artifacts[name] = {
                "path": (output_dir / OUTPUT_FILES[name]).as_posix(),
                **metadata,
            }
        result["provenance"] = {
            "input_path": input_path.as_posix(),
            "input_file_sha256": input_sha256,
            "canonical_outcome_sha256": canonical_outcome_sha256,
            "upstream_config_path": upstream_config_path.as_posix(),
            "upstream_config_file_sha256": upstream_config_file_sha256,
            "audit_config_path": audit_config_path.as_posix(),
            "audit_config_file_sha256": audit_config_file_sha256,
            **_source_provenance(),
        }
        result["artifacts"] = artifacts
        _write_json(result, staging / OUTPUT_FILES["result"])
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked retrospective Calendar V4 route-conditional "
            "calibration partition sensitivity audit."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--upstream-config", type=Path, default=DEFAULT_UPSTREAM_CONFIG)
    parser.add_argument("--audit-config", type=Path, default=DEFAULT_AUDIT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        args.input,
        args.upstream_config,
        args.audit_config,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "result": (args.output_dir / OUTPUT_FILES["result"]).as_posix(),
                "status": result["status"],
                "loco": result["leave_one_calibration_condition_out_primary_80pct"],
                "exhaustive": result["exhaustive_partition_primary_80pct"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
