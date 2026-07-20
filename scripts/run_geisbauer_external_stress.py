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

from lifetwin.data.geisbauer_calendar import (
    GEISBAUER_CALENDAR_OBSERVATIONS_SHA256,
    GEISBAUER_CALENDAR_MEMBER_SHA256,
    geisbauer_calendar_observations_sha256,
    load_geisbauer_calendar_observations,
)
from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
)
from lifetwin.experiments.geisbauer_external_stress import (
    run_geisbauer_external_stress,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_TARGET = Path("data/external/geisbauer_2022/LFP_Data.csv")
DEFAULT_PROTOCOL = Path(
    "configs/experiments/geisbauer_lfp_calendar_external_stress.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/geisbauer_external_stress_v1")
EXPECTED_SOURCE_INPUT_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)
EXPECTED_TARGET_INPUT_SHA256 = GEISBAUER_CALENDAR_MEMBER_SHA256
OUTPUT_FILES = {
    "result": "result.json",
    "label_free_predictions": "label_free_predictions.csv",
    "cell_metrics": "cell_metrics.csv",
    "condition_summary": "condition_summary.csv",
    "comparison_summary": "comparison_summary.csv",
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
        PROJECT_ROOT / "src/lifetwin/data/geisbauer_calendar.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v3_activation.py",
        PROJECT_ROOT / "src/lifetwin/experiments/geisbauer_external_stress.py",
    )
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
    source_path: Path,
    target_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            "The Geisbauer runner never overwrites an evidence directory: "
            f"{output_dir}"
        )
    source_file_sha256 = _sha256(source_path)
    if source_file_sha256 != EXPECTED_SOURCE_INPUT_SHA256:
        raise ValueError(
            "Naumann source input SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_INPUT_SHA256}, found {source_file_sha256}"
        )
    target_file_sha256 = _sha256(target_path)
    if target_file_sha256 != EXPECTED_TARGET_INPUT_SHA256:
        raise ValueError(
            "Geisbauer target input SHA-256 mismatch: "
            f"expected {EXPECTED_TARGET_INPUT_SHA256}, found {target_file_sha256}"
        )
    source_observations = pd.read_csv(source_path)
    source_canonical_outcome_sha256 = canonical_naumann_outcome_sha256(
        source_observations
    )
    if source_canonical_outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError("Naumann source canonical outcome snapshot mismatch")
    target_observations, target_audit = load_geisbauer_calendar_observations(
        target_path
    )
    target_audit["source"]["provided_path"] = target_path.as_posix()
    target_fingerprint = geisbauer_calendar_observations_sha256(
        target_observations
    )
    if target_fingerprint != GEISBAUER_CALENDAR_OBSERVATIONS_SHA256:
        raise ValueError("Geisbauer adapter canonical fingerprint mismatch")
    protocol_file_sha256 = _sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result, predictions, cell_metrics, condition_summary, comparison_summary = (
        run_geisbauer_external_stress(
            source_observations,
            target_observations,
            protocol=protocol,
        )
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        frames = {
            "label_free_predictions": predictions,
            "cell_metrics": cell_metrics,
            "condition_summary": condition_summary,
            "comparison_summary": comparison_summary,
        }
        artifacts: dict[str, object] = {}
        for name, frame in frames.items():
            metadata = _write_csv(frame, staging / OUTPUT_FILES[name])
            artifacts[name] = {
                "path": (output_dir / OUTPUT_FILES[name]).as_posix(),
                **metadata,
            }
        result["provenance"] = {
            "source_path": source_path.as_posix(),
            "source_input_file_sha256": source_file_sha256,
            "source_canonical_outcome_sha256": (
                source_canonical_outcome_sha256
            ),
            "target_path": target_path.as_posix(),
            "target_input_file_sha256": target_file_sha256,
            "target_canonical_observations_sha256": target_fingerprint,
            "target_adapter_audit": target_audit,
            "protocol_path": protocol_path.as_posix(),
            "protocol_file_sha256": protocol_file_sha256,
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
            "Run the frozen 120-day, 60 C Geisbauer LFP external transfer "
            "stress screen. This command does not perform long-term validation."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.source, args.target, args.protocol, args.output_dir)
    print(
        json.dumps(
            {
                "result": (args.output_dir / OUTPUT_FILES["result"]).as_posix(),
                "status": result["status"],
                "model_validation_status": result["model_validation_status"],
                "descriptive_signal_status": result["descriptive_signal_status"],
                "primary_comparison": result["primary_comparison"],
                "decision": result["decision"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
