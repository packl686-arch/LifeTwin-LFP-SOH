from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys

import pandas as pd

from lifetwin.experiments.calendar_v2_development import (
    HIERARCHICAL_POWER_METHOD,
    PRIMARY_PREFIX,
    run_calendar_v2_development,
)


DEFAULT_INPUT = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_CONFIG = Path("configs/experiments/naumann_calendar_v2_development.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts/calendar_v2_development_v1_hardened")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_frozen_input(path: Path) -> str:
    observed = _sha256(path)
    if observed != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Naumann Calendar V2 input SHA-256 mismatch: "
            f"expected {EXPECTED_INPUT_SHA256}, found {observed}"
        )
    return observed


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported observation format: {path.suffix}")


def _source_provenance() -> dict[str, object]:
    paths = [
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/lifetwin/data/naumann.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2.py",
        PROJECT_ROOT / "src/lifetwin/experiments/calendar_v2_development.py",
    ]
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths
    }
    payload = json.dumps(
        source_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "source_sha256": source_hashes,
        "source_tree_sha256": hashlib.sha256(payload).hexdigest(),
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
            "The Calendar V2 runner never overwrites artifacts; existing="
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated, time-honest Naumann Calendar V2 development bakeoff."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    output_paths = {
        "result": args.artifact_dir / "result.json",
        "label_free_predictions": args.artifact_dir / "label_free_predictions.csv",
        "condition_metrics": args.artifact_dir / "condition_metrics.csv",
        "paired_condition_metrics": args.artifact_dir / "paired_condition_metrics.csv",
        "comparison_summary": args.artifact_dir / "comparison_summary.csv",
        "target_diagnostics": args.artifact_dir / "target_diagnostics.csv",
        "fold_parameters": args.artifact_dir / "fold_parameters.csv",
        "condition_splits": args.artifact_dir / "condition_splits.csv",
    }
    _require_new_outputs(list(output_paths.values()))
    input_sha256 = _require_frozen_input(args.input)
    config_file_sha256 = _sha256(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observations = _read_table(args.input)
    (
        result,
        predictions,
        condition_metrics,
        paired_metrics,
        comparisons,
        diagnostics,
        parameters,
        splits,
    ) = run_calendar_v2_development(observations, config=config)
    artifacts = {
        "label_free_predictions": _write_csv(
            predictions, output_paths["label_free_predictions"]
        ),
        "condition_metrics": _write_csv(
            condition_metrics, output_paths["condition_metrics"]
        ),
        "paired_condition_metrics": _write_csv(
            paired_metrics, output_paths["paired_condition_metrics"]
        ),
        "comparison_summary": _write_csv(
            comparisons, output_paths["comparison_summary"]
        ),
        "target_diagnostics": _write_csv(
            diagnostics, output_paths["target_diagnostics"]
        ),
        "fold_parameters": _write_csv(parameters, output_paths["fold_parameters"]),
        "condition_splits": _write_csv(splits, output_paths["condition_splits"]),
    }
    result["provenance"] = {
        "input_path": args.input.as_posix(),
        "input_sha256": input_sha256,
        "config_path": args.config.as_posix(),
        "config_file_sha256": config_file_sha256,
        **_source_provenance(),
    }
    result["artifacts"] = artifacts
    output_paths["result"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["result"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    primary = [
        row
        for row in result["comparison_summary"]
        if row["prefix_checkups"] == PRIMARY_PREFIX
        and row["candidate_method"] == HIERARCHICAL_POWER_METHOD
    ]
    print(
        json.dumps(
            {
                "result": output_paths["result"].as_posix(),
                "status": result["status"],
                "development_gate": result["development_gate"],
                "primary_hierarchical_power_results": primary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
