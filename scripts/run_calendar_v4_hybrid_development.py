from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import shutil
import sys
import uuid

import pandas as pd

from lifetwin.atomic_publish import (
    AtomicPublishRetryExhausted,
    publish_directory,
)
from lifetwin.experiments.calendar_v4_hybrid_development import (
    run_calendar_v4_hybrid_development,
)
from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_CONFIG = Path(
    "configs/experiments/naumann_calendar_v4_hybrid_development.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/calendar_v4_hybrid_development_v1")
EXPECTED_INPUT_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)
OUTPUT_FILES = {
    "result": "result.json",
    "label_free_predictions": "label_free_predictions.csv",
    "training_residual_crossfit": "training_residual_crossfit.csv",
    "calibration_condition_scores": "calibration_condition_scores.csv",
    "calibration_quantiles": "calibration_quantiles.csv",
    "condition_metrics": "condition_metrics.csv",
    "condition_splits": "condition_splits.csv",
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
        PROJECT_ROOT / "src/lifetwin/atomic_publish.py",
        PROJECT_ROOT / "src/lifetwin/data/naumann.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2_uncertainty.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v3_activation.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v4_hybrid.py",
        PROJECT_ROOT
        / "src/lifetwin/experiments/calendar_v4_hybrid_development.py",
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


def run(input_path: Path, config_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            "The Calendar V4 runner never overwrites an evidence directory: "
            f"{output_dir}"
        )
    input_sha256 = _sha256(input_path)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Naumann Calendar V4 input SHA-256 mismatch: "
            f"expected {EXPECTED_INPUT_SHA256}, found {input_sha256}"
        )
    config_file_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observations = pd.read_csv(input_path)
    canonical_outcome_sha256 = canonical_naumann_outcome_sha256(observations)
    if canonical_outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError("Naumann Calendar V4 canonical outcome snapshot mismatch")
    (
        result,
        predictions,
        residual_crossfit,
        calibration_scores,
        calibration_quantiles,
        condition_metrics,
        splits,
    ) = run_calendar_v4_hybrid_development(observations, config=config)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        frames = {
            "label_free_predictions": predictions,
            "training_residual_crossfit": residual_crossfit,
            "calibration_condition_scores": calibration_scores,
            "calibration_quantiles": calibration_quantiles,
            "condition_metrics": condition_metrics,
            "condition_splits": splits,
        }
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
            "config_path": config_path.as_posix(),
            "config_file_sha256": config_file_sha256,
            **_source_provenance(),
        }
        result["artifacts"] = artifacts
        _write_json(result, staging / OUTPUT_FILES["result"])
        publish_directory(staging, output_dir)
    except AtomicPublishRetryExhausted:
        raise
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked retrospective Calendar V4 mechanistic hierarchy, "
            "bounded residual, and conservative interval diagnostic."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.input, args.config, args.output_dir)
    print(
        json.dumps(
            {
                "result": (args.output_dir / OUTPUT_FILES["result"]).as_posix(),
                "status": result["status"],
                "confirmation": result["confirmation"],
                "calibration": result["calibration"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
