"""Compile hash-bound V8 cell issuances into a pre-outcome cohort commitment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from lifetwin.experiments import fastcharge_v8_measurement_stability as v8
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ISSUANCE_FILES = {
    "decision.json",
    "forecast_correction.csv",
    "manifest.json",
    "prediction_commitment.json",
}


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


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FastChargeV5PairwiseError(
            "V8 cohort output directory must be new or empty"
        )
    path.mkdir(parents=True, exist_ok=True)


def _verify_manifest(directory: Path, manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != (
        "lifetwin.fastcharge_v8.measurement_stability.manifest.v1"
    ):
        raise FastChargeV5PairwiseError(
            "V8 cell issuance manifest schema is unsupported"
        )
    artifacts = manifest.get("artifacts")
    expected_names = ISSUANCE_FILES - {"manifest.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise FastChargeV5PairwiseError(
            "V8 cell issuance manifest has an incomplete artifact set"
        )
    for name in sorted(expected_names):
        record = artifacts[name]
        if not isinstance(record, dict):
            raise FastChargeV5PairwiseError(
                "V8 cell issuance manifest artifact is malformed"
            )
        path = directory / name
        if (
            record.get("sha256") != _sha256(path)
            or record.get("byte_count") != path.stat().st_size
        ):
            raise FastChargeV5PairwiseError(
                f"V8 cell issuance artifact changed after manifest: {name}"
            )


def _load_verified_issuance(
    directory: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    observed_names = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_names != ISSUANCE_FILES or any(
        path.is_dir() for path in directory.iterdir()
    ):
        raise FastChargeV5PairwiseError(
            f"V8 cell issuance directory has unexpected contents: {directory}"
        )
    decision_path = directory / "decision.json"
    forecast_path = directory / "forecast_correction.csv"
    commitment_path = directory / "prediction_commitment.json"
    decision = _load_object(decision_path, "cell issuance decision")
    commitment = _load_object(commitment_path, "cell prediction commitment")
    manifest = _load_object(directory / "manifest.json", "cell issuance manifest")
    _verify_manifest(directory, manifest)
    if commitment.get("schema_version") != (
        "lifetwin.fastcharge_v8.measurement_stability.commitment.v1"
    ):
        raise FastChargeV5PairwiseError(
            "V8 cell prediction commitment schema is unsupported"
        )
    if (
        commitment.get("stage_c_outcome_opening_authorized") is not False
        or commitment.get("future_outcomes_read") is not False
    ):
        raise FastChargeV5PairwiseError(
            "V8 cell commitment crossed the cohort outcome firewall"
        )
    exact_bindings = {
        "issuance_id": decision.get("issuance_id"),
        "cell_id": decision.get("cell_id"),
        "manufacturing_batch_id": decision.get("manufacturing_batch_id"),
        "config_sha256": decision.get("config_sha256"),
        "candidate_sha256": decision.get("candidate_sha256"),
        "request_sha256": decision.get("request_sha256"),
        "measurement_quality_decision_sha256": decision.get(
            "measurement_quality_decision_sha256"
        ),
        "noise_ledger_sha256": decision.get("noise_ledger_sha256"),
        "forecast_correction_sha256": _sha256(forecast_path),
        "decision_sha256": _sha256(decision_path),
    }
    for field, expected in exact_bindings.items():
        if commitment.get(field) != expected:
            raise FastChargeV5PairwiseError(
                f"V8 cell prediction commitment does not bind {field}"
            )
    if decision.get("forecast_correction_sha256") != _sha256(forecast_path):
        raise FastChargeV5PairwiseError("V8 cell decision does not bind its forecast")
    if manifest.get("issuance_id") != decision.get("issuance_id"):
        raise FastChargeV5PairwiseError(
            "V8 cell manifest and decision issuance identities differ"
        )
    return decision, {
        "issuance_id": decision["issuance_id"],
        "cell_id": decision["cell_id"],
        "manufacturing_batch_id": decision["manufacturing_batch_id"],
        "quality_activated": bool(decision["stability"]["quality_activated"]),
        "exact_v5_fallback": bool(decision["exact_v5_fallback"]),
        "decision_sha256": _sha256(decision_path),
        "forecast_correction_sha256": _sha256(forecast_path),
        "prediction_commitment_sha256": _sha256(commitment_path),
        "cell_manifest_sha256": _sha256(directory / "manifest.json"),
    }


def _validate_execution_config(
    config: Mapping[str, object], protocol: Mapping[str, object]
) -> None:
    if config.get("schema_version") != (
        "lifetwin.fastcharge_v8_measurement_stability.execution_config.template.v1"
    ):
        raise FastChargeV5PairwiseError(
            "V8 cohort compilation requires the registered execution config"
        )
    gate = config.get("stability_gate")
    stage_b = protocol["stage_b_outcome_free_stability_issuance"]
    registered = stage_b["nonzero_update_requires_all"]
    if not isinstance(gate, dict) or not (
        gate.get("draw_count") == stage_b["monte_carlo_measurement_draw_count"]
        and gate.get("seed_derivation") == stage_b["monte_carlo_seed_derivation"]
        and gate.get("stable_correction") == stage_b["stable_correction"]
        and gate.get("minimum_measurement_resampled_activation_probability")
        == registered["minimum_measurement_resampled_activation_probability"]
        and gate.get("minimum_measurement_resampled_correction_sign_probability")
        == registered["minimum_measurement_resampled_correction_sign_probability"]
        and gate.get("maximum_p95_endpoint_effective_correction_deviation_pp")
        == registered["maximum_p95_endpoint_effective_correction_deviation_pp"]
    ):
        raise FastChargeV5PairwiseError(
            "V8 execution stability gate differs from the blind protocol"
        )


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    protocol_path = Path(args.protocol).resolve()
    candidate_path = Path(args.candidate).resolve()
    issuance_root = Path(args.issuance_root).resolve()
    output = Path(args.output_directory).resolve()
    config = _load_object(config_path, "config")
    protocol = _load_object(protocol_path, "protocol")
    _validate_execution_config(config, protocol)
    config_sha256 = _sha256(config_path)
    protocol_sha256 = _sha256(protocol_path)
    candidate_sha256 = _sha256(candidate_path)
    parent = config.get("parent_protocol")
    candidate = config.get("frozen_v7_rule")
    if not isinstance(parent, dict) or parent.get("sha256") != protocol_sha256:
        raise FastChargeV5PairwiseError(
            "V8 config does not bind the supplied blind protocol"
        )
    if not isinstance(candidate, dict) or candidate.get("sha256") != candidate_sha256:
        raise FastChargeV5PairwiseError(
            "V8 config does not bind the supplied V7 candidate"
        )
    if not issuance_root.is_dir():
        raise FastChargeV5PairwiseError("V8 issuance root must be a directory")
    if any(path.is_file() for path in issuance_root.iterdir()):
        raise FastChargeV5PairwiseError(
            "V8 issuance root may contain only cell issuance directories"
        )
    directories = sorted(path for path in issuance_root.iterdir() if path.is_dir())
    if not directories:
        raise FastChargeV5PairwiseError("V8 issuance root contains no cell issuances")
    decisions: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for directory in directories:
        decision, index = _load_verified_issuance(directory)
        if (
            decision.get("config_sha256") != config_sha256
            or decision.get("candidate_sha256") != candidate_sha256
        ):
            raise FastChargeV5PairwiseError(
                "V8 cell issuance belongs to another config or candidate"
            )
        decisions.append(decision)
        index_rows.append(index)
    readiness = v8.cohort_readiness_decision(decisions, protocol)

    _prepare_output(output)
    index_path = output / "cohort_issuance_index.csv"
    pd.DataFrame(index_rows).sort_values("cell_id", kind="stable").to_csv(
        index_path,
        index=False,
        lineterminator="\n",
    )
    decision_path = output / "cohort_decision.json"
    result = {
        "schema_version": (
            "lifetwin.fastcharge_v8.measurement_stability.cohort_result.v1"
        ),
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "candidate_sha256": candidate_sha256,
        "cohort_issuance_index_sha256": _sha256(index_path),
        "readiness": readiness,
        "decision": (
            "stage_c_single_open_authorized"
            if readiness["stage_c_outcome_opening_authorized"]
            else "stage_c_opening_blocked_retain_v5"
        ),
        "future_outcomes_read": False,
        "model_accuracy_evidence_created": False,
        "v5_champion_changed": False,
    }
    _write_json(result, decision_path)
    commitment_path = output / "cohort_prediction_commitment.json"
    _write_json(
        {
            "schema_version": (
                "lifetwin.fastcharge_v8.measurement_stability.cohort_commitment.v1"
            ),
            "config_sha256": config_sha256,
            "protocol_sha256": protocol_sha256,
            "candidate_sha256": candidate_sha256,
            "cohort_issuance_index_sha256": _sha256(index_path),
            "cohort_decision_sha256": _sha256(decision_path),
            "stage_c_outcome_opening_authorized": readiness[
                "stage_c_outcome_opening_authorized"
            ],
            "future_outcomes_read": False,
            "v5_champion_changed": False,
        },
        commitment_path,
    )
    artifacts = {
        path.name: {"sha256": _sha256(path), "byte_count": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    _write_json(
        {
            "schema_version": (
                "lifetwin.fastcharge_v8.measurement_stability.cohort_manifest.v1"
            ),
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--issuance-root", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
