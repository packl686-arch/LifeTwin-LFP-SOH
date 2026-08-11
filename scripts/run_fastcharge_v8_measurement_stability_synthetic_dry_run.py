"""Run the outcome-free V8 measurement-stability software dry run."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import scipy

from lifetwin.experiments import fastcharge_v8_measurement_stability as v8
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/experiments/v8_measurement_stability_synthetic_dry_run.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/fastcharge-v8-measurement-stability-synthetic-dry-run"
)
IMPLEMENTATION_PATH = (
    ROOT / "src/lifetwin/experiments/fastcharge_v8_measurement_stability.py"
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


def _measurement_row(
    *,
    role: str,
    physical_cell_id: str,
    landmark_cycle: int,
    repeat_index: int,
    retention_pct: float,
    tester_id: str,
    chamber_id: str,
    measurement_date: date,
    reference_channel_id: str = "",
    bridge_id: str = "",
) -> dict[str, object]:
    return {
        "record_role": role,
        "physical_cell_id": physical_cell_id,
        "landmark_cycle": landmark_cycle,
        "repeat_index": repeat_index,
        "retention_pct": retention_pct,
        "tester_id": tester_id,
        "temperature_chamber_id": chamber_id,
        "measurement_date": measurement_date.isoformat(),
        "reference_channel_id": reference_channel_id,
        "bridge_id": bridge_id,
    }


def build_synthetic_measurements(config: dict[str, object]) -> pd.DataFrame:
    settings = config["synthetic_fixture"]
    rng = np.random.default_rng(int(settings["seed"]))
    testers = [str(value) for value in settings["tester_ids"]]
    chambers = {
        str(key): str(value)
        for key, value in settings["temperature_chamber_by_tester"].items()
    }
    sigma = float(settings["cell_repeat_sigma_pp"])
    rows: list[dict[str, object]] = []
    start = date(2026, 1, 5)
    physical_cell_count = int(settings["physical_cell_count"])
    for cell_index in range(physical_cell_count):
        tester = testers[cell_index % len(testers)]
        chamber = chambers[tester]
        measurement_date = start + timedelta(days=cell_index % 4)
        for landmark in config["measurement_contract"]["required_landmarks"]:
            base = 100.0 - 0.008 * float(landmark) - 0.01 * cell_index
            for repeat_index in range(
                int(config["measurement_contract"]["minimum_repeats_per_cell_landmark"])
            ):
                rows.append(
                    _measurement_row(
                        role="cell_repeat",
                        physical_cell_id=f"SYNTH_CELL_{cell_index:02d}",
                        landmark_cycle=int(landmark),
                        repeat_index=repeat_index,
                        retention_pct=float(base + rng.normal(0.0, sigma)),
                        tester_id=tester,
                        chamber_id=chamber,
                        measurement_date=measurement_date,
                    )
                )

    reference_dates = int(settings["daily_reference_date_count"])
    for tester_index, tester in enumerate(testers):
        chamber = chambers[tester]
        for day_index in range(reference_dates):
            measurement_date = start + timedelta(days=day_index)
            daily_shift = 0.002 * (day_index - (reference_dates - 1) / 2.0)
            for repeat_index in range(int(settings["daily_reference_repeats"])):
                rows.append(
                    _measurement_row(
                        role="daily_reference",
                        physical_cell_id="",
                        landmark_cycle=0,
                        repeat_index=repeat_index,
                        retention_pct=float(
                            99.5
                            - 0.01 * tester_index
                            + daily_shift
                            + rng.normal(0.0, sigma / 2.0)
                        ),
                        tester_id=tester,
                        chamber_id=chamber,
                        measurement_date=measurement_date,
                        reference_channel_id=f"REF_{tester}",
                    )
                )

    for bridge_index in range(int(settings["tester_bridge_id_count"])):
        base = 98.0 - 0.01 * bridge_index
        for tester_index, tester in enumerate(testers):
            chamber = chambers[tester]
            offset = (
                float(settings["tester_b_bridge_offset_pp"])
                if tester_index == 1
                else 0.0
            )
            for repeat_index in range(int(settings["tester_bridge_repeats"])):
                rows.append(
                    _measurement_row(
                        role="tester_bridge",
                        physical_cell_id=f"BRIDGE_CELL_{bridge_index}",
                        landmark_cycle=100,
                        repeat_index=repeat_index,
                        retention_pct=float(
                            base + offset + rng.normal(0.0, sigma / 2.0)
                        ),
                        tester_id=tester,
                        chamber_id=chamber,
                        measurement_date=start + timedelta(days=bridge_index),
                        bridge_id=f"BRIDGE_{bridge_index}",
                    )
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    output = Path(args.output_directory)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = _resolve(str(config["parent_protocol"]["path"]))
    candidate_path = _resolve(str(config["frozen_v7_rule"]["path"]))
    for source, path in (
        (config["parent_protocol"], parent_path),
        (config["frozen_v7_rule"], candidate_path),
    ):
        observed = _sha256(path)
        if observed != str(source["sha256"]):
            raise FastChargeV5PairwiseError(
                f"V8 synthetic dry-run source hash changed: {path.name}"
            )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    measurements = build_synthetic_measurements(config)
    scores, ledger, quality = v8.characterize_measurement_noise(measurements, config)
    model = v8.measurement_noise_model(
        ledger,
        tester_id="TESTER_A",
        temperature_chamber_id="CHAMBER_1",
    )
    history_cycles = np.arange(61, 101, dtype=float)
    history_residuals = 0.006 * (history_cycles - 60.0)
    future_cycles = np.arange(101, 301, dtype=float)
    previous_center = 100.0 - 0.01 * future_cycles
    current_center = previous_center + 0.002 * (future_cycles - 100.0)
    correction, stability = v8.measurement_stability_update(
        history_cycles,
        history_residuals,
        future_cycles,
        previous_center,
        current_center,
        candidate,
        config["stability_gate"],
        model,
        protocol_sha256=_sha256(config_path),
        cell_id="SYNTH_STABLE_CELL",
        measurement_quality_passed=bool(quality["measurement_quality_passed"]),
    )
    fallback, fallback_status = v8.measurement_stability_update(
        history_cycles,
        history_residuals,
        future_cycles,
        previous_center,
        current_center,
        candidate,
        config["stability_gate"],
        None,
        protocol_sha256=_sha256(config_path),
        cell_id="SYNTH_MISSING_MAPPING_CELL",
        measurement_quality_passed=True,
    )
    result = {
        "schema_version": (
            "lifetwin.fastcharge_v8_measurement_stability.synthetic_dry_run.result.v1"
        ),
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_sha256": _sha256(config_path),
        "parent_protocol_sha256": _sha256(parent_path),
        "frozen_v7_rule_sha256": _sha256(candidate_path),
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
        "measurement_fixture": {
            "synthetic_only": True,
            "row_count": len(measurements),
            "physical_cell_count": int(
                measurements.loc[
                    measurements["record_role"] == "cell_repeat",
                    "physical_cell_id",
                ].nunique()
            ),
            "future_outcome_columns_present": False,
        },
        "measurement_quality": quality,
        "stable_path": {
            **stability,
            "mean_absolute_correction_pp": float(np.mean(np.abs(correction))),
            "endpoint_correction_pp": float(correction[-1]),
        },
        "missing_mapping_fallback": {
            **fallback_status,
            "exact_zero_correction": bool(np.all(fallback == 0.0)),
        },
        "decision": (
            "software_dry_run_passed_without_real_model_evidence"
            if bool(quality["measurement_quality_passed"])
            and bool(stability["quality_activated"])
            and bool(np.all(fallback == 0.0))
            else "software_dry_run_failed"
        ),
        "v5_champion_changed": False,
        "real_model_evidence_created": False,
        "claim_boundaries": config["claim_boundaries"],
    }
    _write_csv(scores, output / "noise_candidate_scores.csv")
    _write_csv(ledger, output / "noise_ledger.csv")
    _write_json(result, output / "decision.json")
    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
            }
    _write_json(
        {
            "schema_version": (
                "lifetwin.fastcharge_v8_measurement_stability.synthetic_dry_run.manifest.v1"
            ),
            "experiment_id": config["experiment_id"],
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
