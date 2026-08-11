"""Evaluate one hash-frozen, outcome-free V9 end-to-end replicate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments import fastcharge_v9_end_to_end_stability as v9
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiments/v9_end_to_end_correlated_stability_execution.template.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FastChargeV5PairwiseError(f"Cannot read V9 {label}: {path}") from error
    if not isinstance(value, dict):
        raise FastChargeV5PairwiseError(f"V9 {label} must be a JSON object")
    return value


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    ledger_path = Path(args.replicate_ledger)
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"V9 output directory must be new or empty: {output}")
    config = _load_json(config_path, "execution config")
    if config.get("status") != (
        "frozen_for_real_v9_execution_before_target_outcome_access"
    ):
        raise FastChargeV5PairwiseError(
            "V9 real evaluation requires a separately frozen execution config; "
            "the distributed template cannot be executed"
        )
    protocol_path = _bound_path(config, "parent_protocol")
    candidate_path = _bound_path(config, "frozen_v7_rule")
    _bound_path(config, "frozen_v5_selection")
    _bound_path(config, "frozen_v5_core_config")
    protocol = _load_json(protocol_path, "parent protocol")
    candidate = _load_json(candidate_path, "V7 candidate")
    _assert_protocol_gate(config, protocol)
    try:
        ledger = pd.read_csv(ledger_path)
    except (OSError, pd.errors.ParserError) as error:
        raise FastChargeV5PairwiseError(
            f"Cannot read V9 replicate ledger: {ledger_path}"
        ) from error
    correction, status, metrics = v9.evaluate_end_to_end_stability(
        ledger,
        candidate,
        config,
        protocol_sha256=_sha256(protocol_path),
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "draw_metrics.csv"
    correction_path = output / "forecast_correction.csv"
    metrics.to_csv(metrics_path, index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(
        {
            "forecast_cycle": np.arange(101, 301, dtype=int),
            "v9_effective_correction_pp": correction,
        }
    ).to_csv(correction_path, index=False, lineterminator="\n", float_format="%.17g")
    decision = {
        **status,
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "replicate_ledger_sha256": _sha256(ledger_path),
        "draw_metrics_sha256": _sha256(metrics_path),
        "forecast_correction_sha256": _sha256(correction_path),
        "decision": (
            "v9_end_to_end_stable_correction_qualified"
            if bool(status["quality_activated"])
            else "exact_v5_fallback_issued"
        ),
        "stage_c_outcome_opening_authorized": False,
        "future_outcomes_read": False,
        "model_accuracy_evidence_created": False,
        "v5_champion_changed": False,
    }
    decision_path = output / "decision.json"
    _write_json(decision, decision_path)
    manifest = {
        "schema_version": "lifetwin.fastcharge_v9.cell_issuance_manifest.v1",
        "artifacts": {
            path.name: {"sha256": _sha256(path), "byte_count": path.stat().st_size}
            for path in (metrics_path, correction_path, decision_path)
        },
    }
    _write_json(manifest, output / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _bound_path(config: dict[str, object], field: str) -> Path:
    record = config.get(field)
    if not isinstance(record, dict):
        raise FastChargeV5PairwiseError(f"V9 config lacks binding: {field}")
    path = _resolve(str(record.get("path", "")))
    if _sha256(path) != str(record.get("sha256", "")):
        raise FastChargeV5PairwiseError(f"V9 frozen source hash changed: {field}")
    return path


def _assert_protocol_gate(
    config: dict[str, object], protocol: dict[str, object]
) -> None:
    registered = protocol["stage_b_end_to_end_outcome_free_replay"][
        "nonzero_update_requires_all"
    ]
    observed = config["stability_gate"]
    for field, value in registered.items():
        if field == "unperturbed_v7_gate_activated":
            continue
        if observed.get(field) != value:
            raise FastChargeV5PairwiseError(
                f"V9 execution gate differs from protocol: {field}"
            )
    if int(observed["draw_count"]) != int(
        protocol["stage_b_end_to_end_outcome_free_replay"]["monte_carlo_draw_count"]
    ):
        raise FastChargeV5PairwiseError("V9 execution draw count changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--replicate-ledger", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
