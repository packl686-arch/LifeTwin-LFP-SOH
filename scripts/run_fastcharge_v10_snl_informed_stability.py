"""Propagate a private SNL repeatability component through 1,024 V5 refits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import lifetwin.experiments.fastcharge_v5_pairwise as v5
from lifetwin.experiments import fastcharge_v9_end_to_end_stability as v9
from lifetwin.experiments.fastcharge_safe_prior_v2 import (
    _core_config,
    load_fastcharge_safe_prior_v2_config,
)

try:
    from scripts import run_fastcharge_v9_end_to_end_synthetic_dry_run as run_v9
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    import run_fastcharge_v9_end_to_end_synthetic_dry_run as run_v9


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = ROOT / "configs/experiments/v10_snl_informed_end_to_end_stability.json"
DEFAULT_OUTPUT = ROOT / "artifacts/private-v10-snl-informed-stability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


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


def _load_measurement_component(
    path: Path, config: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    decision = json.loads(path.read_text(encoding="utf-8"))
    source = config["measurement_component_config"]
    source_path = _resolve(str(source["path"]))
    if _sha256(source_path) != str(source["sha256"]):
        raise v5.FastChargeV5PairwiseError("V10 measurement config hash changed")
    required = {
        "schema_version": "lifetwin.fastcharge_v10.snl_rpt_repeatability.v1",
        "experiment_id": "fastcharge_v10_snl_rpt_repeatability_development",
        "future_outcomes_used_for_noise_estimation": False,
        "target_accuracy_evidence_created": False,
        "public_aggregate_release_permitted": False,
        "full_measurement_model_identified": False,
        "eligible_for_full_v9_qualification": False,
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            raise v5.FastChargeV5PairwiseError(
                f"Private V10 measurement decision changed: {key}"
            )
    if decision.get("config_sha256") != str(source["sha256"]):
        raise v5.FastChargeV5PairwiseError(
            "Private V10 decision is not bound to the frozen measurement config"
        )
    selected = decision.get("selected_noise_model")
    if not isinstance(selected, dict):
        raise v5.FastChargeV5PairwiseError("Private V10 noise model is missing")
    distribution = str(selected.get("distribution"))
    scale = float(selected.get("scale_pp", 0.0))
    degrees = selected.get("degrees_of_freedom")
    if distribution not in {"gaussian", "student_t"} or not np.isfinite(scale):
        raise v5.FastChargeV5PairwiseError("Private V10 noise model is invalid")
    if scale <= 0.0:
        raise v5.FastChargeV5PairwiseError("Private V10 noise scale must be positive")
    if distribution == "student_t" and (degrees is None or float(degrees) <= 2.0):
        raise v5.FastChargeV5PairwiseError(
            "Private V10 Student-t model requires df > 2"
        )
    capacity = config["correlated_error_model"]["capacity"]
    capacity["iid_distribution"] = distribution
    capacity["iid_sigma_pp"] = scale
    capacity["iid_degrees_of_freedom"] = float(degrees) if degrees is not None else 0.0
    return decision, config


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    decision_path = Path(args.private_v10_decision)
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty private V10 output: {output}"
        )
    config = deepcopy(json.loads(config_path.read_text(encoding="utf-8")))
    measurement, config = _load_measurement_component(decision_path, config)
    sources: dict[str, Path] = {}
    for name in (
        "parent_protocol",
        "frozen_v7_rule",
        "frozen_v5_selection",
        "frozen_v5_core_config",
    ):
        record = config[name]
        path = _resolve(str(record["path"]))
        if _sha256(path) != str(record["sha256"]):
            raise v5.FastChargeV5PairwiseError(
                f"V10 frozen source hash changed: {path.name}"
            )
        sources[name] = path
    candidate = json.loads(sources["frozen_v7_rule"].read_text(encoding="utf-8"))
    selection = json.loads(sources["frozen_v5_selection"].read_text(encoding="utf-8"))
    v2_config = load_fastcharge_safe_prior_v2_config(sources["frozen_v5_core_config"])
    core = _core_config(v2_config)
    training, target_prefix = run_v9.build_synthetic_inputs(config)
    protocol_sha256 = _sha256(sources["parent_protocol"])
    ledger = run_v9.build_end_to_end_ledger(
        config,
        training,
        target_prefix,
        protocol_sha256=protocol_sha256,
        selection=selection,
        core=core,
    )
    correction, status, draw_metrics = v9.evaluate_end_to_end_stability(
        ledger,
        candidate,
        config,
        protocol_sha256=protocol_sha256,
    )
    repeatability_passed = bool(measurement["repeatability_component_gates"]["passed"])
    full_model_identified = bool(measurement["full_measurement_model_identified"])
    fully_qualified = bool(
        status["quality_activated"] and repeatability_passed and full_model_identified
    )
    effective_correction = correction if fully_qualified else np.zeros_like(correction)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "private_draw_metrics.csv"
    correction_path = output / "private_effective_correction.csv"
    _write_csv(draw_metrics, metrics_path)
    _write_csv(
        pd.DataFrame(
            {
                "forecast_cycle": np.arange(101, 301, dtype=int),
                "effective_correction_pp": effective_correction,
            }
        ),
        correction_path,
    )
    result = {
        "schema_version": "lifetwin.fastcharge_v10.snl_informed_stability.result.v1",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_sha256": _sha256(config_path),
        "private_measurement_decision_sha256": _sha256(decision_path),
        "draw_count": int(config["stability_gate"]["draw_count"]),
        "v9_component_stability": status,
        "repeatability_component_passed": repeatability_passed,
        "full_measurement_model_identified": full_model_identified,
        "fully_qualified": fully_qualified,
        "effective_correction_is_exact_zero": bool(np.all(effective_correction == 0.0)),
        "decision": (
            "retire_v7_dynamic_update_retain_v5"
            if not repeatability_passed
            else "block_v7_pending_reference_and_bridge_records_retain_v5"
        ),
        "missing_measurement_components": measurement["missing_components"],
        "future_target_outcomes_read": False,
        "model_accuracy_evidence_created": False,
        "v5_champion_changed": False,
        "public_release_permitted": False,
        "private_artifacts": {
            "draw_metrics_sha256": _sha256(metrics_path),
            "effective_correction_sha256": _sha256(correction_path),
        },
        "claim_boundaries": config["claim_boundaries"],
    }
    decision_output = output / "private_decision.json"
    _write_json(result, decision_output)
    _write_json(
        {
            "schema_version": "lifetwin.fastcharge_v10.private_stability_manifest.v1",
            "experiment_id": config["experiment_id"],
            "public_release_permitted": False,
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "byte_count": path.stat().st_size,
                }
                for path in (metrics_path, correction_path, decision_output)
            },
        },
        output / "private_manifest.json",
    )
    print(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "draw_count": result["draw_count"],
                "fully_qualified": fully_qualified,
                "decision": result["decision"],
                "effective_correction_is_exact_zero": result[
                    "effective_correction_is_exact_zero"
                ],
                "private_output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_v10_decision")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
