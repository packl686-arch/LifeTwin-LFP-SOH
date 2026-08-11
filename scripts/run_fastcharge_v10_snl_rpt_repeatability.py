"""Characterize private SNL within-visit RPT repeatability for V10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from lifetwin.data.snl import (
    extract_snl_rpt_repeat_measurements,
    load_snl_metadata,
)
from lifetwin.experiments.fastcharge_v10_snl_rpt_noise import (
    characterize_snl_rpt_repeatability,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/v10_snl_rpt_repeatability_development.json"
DEFAULT_OUTPUT = ROOT / "artifacts/private-v10-snl-rpt-repeatability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    raw_zip = Path(args.raw_zip)
    metadata_xlsx = Path(args.metadata_xlsx)
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite nonempty private V10 output: {output}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    if _sha256(raw_zip) != str(dataset["raw_zip_sha256"]):
        raise ValueError("SNL raw ZIP SHA-256 changed")
    if _sha256(metadata_xlsx) != str(dataset["metadata_xlsx_sha256"]):
        raise ValueError("SNL metadata workbook SHA-256 changed")
    metadata, metadata_audit = load_snl_metadata(
        metadata_xlsx,
        expected_lfp_rows_sha256=str(dataset["metadata_lfp_rows_canonical_sha256"]),
        expected_cell_count=int(dataset["physical_cell_count"]),
        expected_condition_count=int(dataset["condition_cluster_count"]),
    )
    repeats, extraction_audit = extract_snl_rpt_repeat_measurements(
        raw_zip,
        metadata,
        rest_gap_hours=float(config["rpt_adapter"]["long_rest_gap_hours"]),
        duplicate_visit_efc=float(
            config["rpt_adapter"]["adjacent_check_collapse_maximum_efc"]
        ),
    )
    residuals, scores, condition_scales, model, decision = (
        characterize_snl_rpt_repeatability(repeats, config)
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "repeat_measurements": output / "private_repeat_measurements.csv",
        "residuals": output / "private_repeat_residuals.csv",
        "scores": output / "noise_candidate_scores.csv",
        "condition_scales": output / "private_condition_scales.csv",
        "metadata_audit": output / "private_metadata_audit.json",
        "extraction_audit": output / "private_repeat_extraction_audit.json",
        "decision": output / "private_decision.json",
    }
    _write_csv(repeats, paths["repeat_measurements"])
    _write_csv(residuals, paths["residuals"])
    _write_csv(scores, paths["scores"])
    _write_csv(condition_scales, paths["condition_scales"])
    _write_json(metadata_audit, paths["metadata_audit"])
    _write_json(extraction_audit, paths["extraction_audit"])
    _write_json(
        {
            **decision,
            "config_sha256": _sha256(config_path),
            "raw_zip_sha256": _sha256(raw_zip),
            "metadata_xlsx_sha256": _sha256(metadata_xlsx),
        },
        paths["decision"],
    )
    manifest_artifacts = {
        name: {
            "sha256": _sha256(path),
            "byte_count": path.stat().st_size,
            "public_release_permitted": False,
        }
        for name, path in paths.items()
    }
    _write_json(
        {
            "schema_version": "lifetwin.fastcharge_v10.private_artifact_manifest.v1",
            "experiment_id": config["experiment_id"],
            "artifacts": manifest_artifacts,
            "public_release_permitted": False,
        },
        output / "private_manifest.json",
    )
    print(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "selected_distribution": model.distribution,
                "selected_degrees_of_freedom": model.degrees_of_freedom,
                "repeatability_scale_pp": model.scale_pp,
                "repeatability_component_passed": decision[
                    "repeatability_component_gates"
                ]["passed"],
                "full_measurement_model_identified": decision[
                    "full_measurement_model_identified"
                ],
                "eligible_for_full_v9_qualification": decision[
                    "eligible_for_full_v9_qualification"
                ],
                "next_action": decision["next_action"],
                "private_output": str(output),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_zip")
    parser.add_argument("metadata_xlsx")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
