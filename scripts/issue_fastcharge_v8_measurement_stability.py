"""Issue one outcome-free V8 stability-gated forecast correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy

from lifetwin.experiments import fastcharge_v8_measurement_stability as v8
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATH = (
    ROOT / "src/lifetwin/experiments/fastcharge_v8_measurement_stability.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FastChargeV5PairwiseError(f"Cannot read V8 {label}: {path}") from error
    if not isinstance(value, dict):
        raise FastChargeV5PairwiseError(f"V8 {label} must be a JSON object")
    return value


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require_sha_binding(*, observed_path: Path, expected: object, label: str) -> str:
    observed = _sha256(observed_path)
    if observed != str(expected):
        raise FastChargeV5PairwiseError(
            f"V8 {label} SHA-256 does not match its frozen binding"
        )
    return observed


def _validate_quality_decision(
    decision: Mapping[str, object],
    *,
    config_sha256: str,
    ledger_sha256: str,
) -> bool:
    if decision.get("schema_version") != (
        "lifetwin.fastcharge_v8.measurement_quality.result.v1"
    ):
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality decision schema is unsupported"
        )
    if decision.get("config_sha256") != config_sha256:
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality decision belongs to another config"
        )
    if decision.get("noise_ledger_sha256") != ledger_sha256:
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality decision does not bind this noise ledger"
        )
    if (
        decision.get("future_outcome_columns_accepted") is not False
        or decision.get("future_outcome_access_permitted_by_this_result") is not False
    ):
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality decision does not preserve the outcome firewall"
        )
    if decision.get("model_accuracy_evidence_created") is not False:
        raise FastChargeV5PairwiseError(
            "V8 Stage A must not claim model-accuracy evidence"
        )
    implementation = decision.get("implementation")
    if not isinstance(implementation, dict) or implementation.get(
        "module_sha256"
    ) != _sha256(IMPLEMENTATION_PATH):
        raise FastChargeV5PairwiseError(
            "V8 Stage A and Stage B implementation hashes differ"
        )
    quality = decision.get("measurement_quality")
    if not isinstance(quality, dict) or not isinstance(
        quality.get("measurement_quality_passed"), bool
    ):
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality decision lacks a Boolean gate result"
        )
    passed = bool(quality["measurement_quality_passed"])
    expected_decision = (
        "measurement_quality_passed_for_outcome_free_stage_b_issuance"
        if passed
        else "measurement_quality_failed_retain_v5_and_stop_before_outcomes"
    )
    if decision.get("decision") != expected_decision:
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality decision text contradicts its gate result"
        )
    return passed


def _validate_selected_model(
    ledger: pd.DataFrame, decision: Mapping[str, object]
) -> pd.DataFrame:
    validated = v8.validate_noise_ledger(ledger)
    quality = decision["measurement_quality"]
    assert isinstance(quality, dict)
    selected = str(quality.get("selected_noise_model_id", ""))
    if not selected or set(validated["model_id"]) != {selected}:
        raise FastChargeV5PairwiseError(
            "V8 noise ledger does not use the frozen selected model"
        )
    return validated


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FastChargeV5PairwiseError(
            "V8 issuance output directory must be new or empty"
        )
    path.mkdir(parents=True, exist_ok=True)


def _validate_execution_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != (
        "lifetwin.fastcharge_v8_measurement_stability.execution_config.template.v1"
    ):
        raise FastChargeV5PairwiseError(
            "V8 real issuance requires the registered execution config"
        )
    gate = config.get("stability_gate")
    if not isinstance(gate, dict) or not (
        gate.get("draw_count") == 1024
        and gate.get("seed_derivation")
        == "sha256(protocol_hash|cell_id|draw_index)_first_64_bits"
        and gate.get("minimum_measurement_resampled_activation_probability") == 0.95
        and gate.get("minimum_measurement_resampled_correction_sign_probability")
        == 0.95
        and gate.get("maximum_p95_endpoint_effective_correction_deviation_pp") == 0.05
        and gate.get("stable_correction")
        == "pointwise_median_of_measurement_resampled_effective_corrections"
        and gate.get("failed_action") == "exact_zero_update_to_v5_center"
    ):
        raise FastChargeV5PairwiseError(
            "V8 real issuance stability gate differs from the registered rule"
        )
    firewall = config.get("data_firewall")
    if not isinstance(firewall, dict) or any(
        firewall.get(field) is not False
        for field in (
            "same_41_cell_outcomes_permitted",
            "exposed_81_cell_evaluation_permitted",
            "future_outcomes_permitted_before_cohort_commitment",
            "hithium_measurements_in_public_repository_permitted",
        )
    ):
        raise FastChargeV5PairwiseError(
            "V8 execution config does not preserve the registered data firewall"
        )


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    candidate_path = Path(args.candidate).resolve()
    request_path = Path(args.request).resolve()
    quality_path = Path(args.measurement_quality_decision).resolve()
    ledger_path = Path(args.noise_ledger).resolve()
    output = Path(args.output_directory).resolve()

    config = _load_object(config_path, "config")
    candidate = _load_object(candidate_path, "candidate")
    request_raw = _load_object(request_path, "stability request")
    quality_decision = _load_object(quality_path, "measurement-quality decision")
    _validate_execution_config(config)

    frozen_candidate = config.get("frozen_v7_rule")
    if not isinstance(frozen_candidate, dict) or "sha256" not in frozen_candidate:
        raise FastChargeV5PairwiseError("V8 config lacks a frozen V7 candidate binding")
    candidate_sha256 = _require_sha_binding(
        observed_path=candidate_path,
        expected=frozen_candidate["sha256"],
        label="candidate",
    )
    config_sha256 = _sha256(config_path)
    ledger_sha256 = _sha256(ledger_path)
    measurement_quality_passed = _validate_quality_decision(
        quality_decision,
        config_sha256=config_sha256,
        ledger_sha256=ledger_sha256,
    )
    try:
        ledger_raw = pd.read_csv(ledger_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise FastChargeV5PairwiseError(
            f"Cannot read V8 noise ledger: {ledger_path}"
        ) from error
    ledger = _validate_selected_model(ledger_raw, quality_decision)
    request = v8.validate_stability_request(request_raw, candidate)
    noise_model = v8.measurement_noise_model(
        ledger,
        tester_id=request.tester_id,
        temperature_chamber_id=request.temperature_chamber_id,
    )
    history_residuals = (
        request.history_observed_retention_pct - request.history_previous_v5_center_pct
    )
    correction, stability = v8.measurement_stability_update(
        request.history_cycles,
        history_residuals,
        request.future_cycles,
        request.previous_v5_center_pct,
        request.current_v5_center_pct,
        candidate,
        config["stability_gate"],
        noise_model,
        protocol_sha256=config_sha256,
        cell_id=request.cell_id,
        measurement_quality_passed=measurement_quality_passed,
    )
    clip = tuple(
        float(value)
        for value in candidate["frozen_update_rule"]["future_prediction_clip_pct"]
    )
    activated = bool(stability["quality_activated"])
    candidate_center = (
        np.clip(request.current_v5_center_pct + correction, *clip)
        if activated
        else request.current_v5_center_pct.copy()
    )
    if not activated and not np.array_equal(
        candidate_center, request.current_v5_center_pct
    ):
        raise FastChargeV5PairwiseError(
            "V8 failed stability gate did not preserve the exact V5 center"
        )

    _prepare_output(output)
    forecast_path = output / "forecast_correction.csv"
    pd.DataFrame(
        {
            "cycle": request.future_cycles.astype(int),
            "previous_p60_v5_center_pct": request.previous_v5_center_pct,
            "current_p100_v5_center_pct": request.current_v5_center_pct,
            "v8_effective_correction_pp": correction,
            "v8_candidate_center_pct": candidate_center,
        }
    ).to_csv(
        forecast_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    decision_path = output / "decision.json"
    result = {
        "schema_version": (
            "lifetwin.fastcharge_v8.measurement_stability.issuance_result.v1"
        ),
        "issuance_id": request.issuance_id,
        "cell_id": request.cell_id,
        "manufacturing_batch_id": request.manufacturing_batch_id,
        "tester_id": request.tester_id,
        "temperature_chamber_id": request.temperature_chamber_id,
        "config_sha256": config_sha256,
        "candidate_sha256": candidate_sha256,
        "request_sha256": _sha256(request_path),
        "measurement_quality_decision_sha256": _sha256(quality_path),
        "noise_ledger_sha256": ledger_sha256,
        "forecast_correction_sha256": _sha256(forecast_path),
        "measurement_quality_passed": measurement_quality_passed,
        "stability": stability,
        "decision": (
            "v8_stable_correction_issued" if activated else "exact_v5_fallback_issued"
        ),
        "exact_v5_fallback": not activated,
        "future_outcomes_read": False,
        "model_accuracy_evidence_created": False,
        "v5_champion_changed": False,
        "runtime_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "implementation": {
            "module_path": str(IMPLEMENTATION_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "module_sha256": _sha256(IMPLEMENTATION_PATH),
            "runner_path": str(Path(__file__).resolve().relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(result, decision_path)
    commitment_path = output / "prediction_commitment.json"
    _write_json(
        {
            "schema_version": (
                "lifetwin.fastcharge_v8.measurement_stability.commitment.v1"
            ),
            "issuance_id": request.issuance_id,
            "cell_id": request.cell_id,
            "manufacturing_batch_id": request.manufacturing_batch_id,
            "config_sha256": config_sha256,
            "candidate_sha256": candidate_sha256,
            "request_sha256": _sha256(request_path),
            "measurement_quality_decision_sha256": _sha256(quality_path),
            "noise_ledger_sha256": ledger_sha256,
            "forecast_correction_sha256": _sha256(forecast_path),
            "decision_sha256": _sha256(decision_path),
            "future_outcomes_read": False,
            "single_cell_commitment_only": True,
            "stage_c_outcome_opening_authorized": False,
        },
        commitment_path,
    )
    manifest_path = output / "manifest.json"
    artifacts = {
        path.name: {"sha256": _sha256(path), "byte_count": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    _write_json(
        {
            "schema_version": (
                "lifetwin.fastcharge_v8.measurement_stability.manifest.v1"
            ),
            "issuance_id": request.issuance_id,
            "artifacts": artifacts,
        },
        manifest_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--measurement-quality-decision", required=True)
    parser.add_argument("--noise-ledger", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
