"""Run the V9 correlated perturb-and-refit synthetic software exercise."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

# Keep the repeated sklearn fits deterministic in restricted Windows runners.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import sklearn

import lifetwin.experiments.fastcharge_v5_pairwise as v5
from lifetwin.experiments import fastcharge_v9_end_to_end_stability as v9
from lifetwin.experiments.fastcharge_safe_prior_v2 import (
    _core_config,
    load_fastcharge_safe_prior_v2_config,
)
from lifetwin.experiments.fastcharge_trajectory_portability import (
    _normalization_capacity,
    _retention,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs/experiments/v9_end_to_end_correlated_stability_synthetic_dry_run.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v9-end-to-end-synthetic-dry-run"
IMPLEMENTATION_PATH = (
    ROOT / "src/lifetwin/experiments/fastcharge_v9_end_to_end_stability.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*values: np.ndarray, metadata: str = "") -> str:
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
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


def build_synthetic_inputs(
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create complete reference histories and a target that ends at cycle 100."""

    fixture = config["synthetic_fixture"]
    count = int(fixture["training_cell_count"])
    nominal = float(fixture["nominal_capacity_ah"])
    rows: list[dict[str, object]] = []
    cycles = np.arange(1, 301, dtype=float)
    for index in range(count):
        rate = 0.0065 + 0.00032 * index
        curvature = 0.000012 + 0.0000005 * (index % 5)
        retention = (
            100.0
            - rate * cycles
            - curvature * np.power(cycles, 1.55)
            + 0.012 * np.sin(cycles / (11.0 + index % 3))
        )
        for cycle, value in zip(cycles.astype(int), retention, strict=True):
            rows.append(
                _cycle_row(
                    cell_id=f"SYNTH_V9_TRAIN_{index:02d}",
                    cycle=cycle,
                    retention_pct=float(value),
                    nominal_capacity_ah=nominal,
                    cell_offset=index,
                )
            )

    target_cycles = np.arange(1, 101, dtype=float)
    target = 100.0 - 0.0082 * target_cycles - 0.000014 * np.power(target_cycles, 1.55)
    acceleration = float(fixture["target_acceleration_after_cycle_60_pp_per_cycle"])
    target -= acceleration * np.maximum(target_cycles - 60.0, 0.0)
    target += 0.008 * np.sin(target_cycles / 12.0)
    target_rows = [
        _cycle_row(
            cell_id=str(fixture["target_cell_id"]),
            cycle=int(cycle),
            retention_pct=float(value),
            nominal_capacity_ah=nominal,
            cell_offset=7,
        )
        for cycle, value in zip(target_cycles, target, strict=True)
    ]
    training = pd.DataFrame(rows)
    target_prefix = pd.DataFrame(target_rows)
    if int(target_prefix["cycle_index"].max()) != 100:
        raise AssertionError(
            "Synthetic V9 target accidentally contains a future suffix"
        )
    return training, target_prefix


def _cycle_row(
    *,
    cell_id: str,
    cycle: int,
    retention_pct: float,
    nominal_capacity_ah: float,
    cell_offset: int,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "cycle_index": cycle,
        "discharge_capacity_ah": nominal_capacity_ah * retention_pct / 100.0,
        "internal_resistance_ohm": (
            0.030 + 0.0000045 * cycle + 0.00005 * (cell_offset % 4)
        ),
        "temperature_max_c": (29.0 + 0.035 * (cell_offset % 5) + 0.0015 * cycle),
        "charge_time_s": 3500.0 + 0.42 * cycle + 2.0 * (cell_offset % 3),
        "energy_efficiency": 0.985 - 0.000012 * cycle - 0.00002 * (cell_offset % 2),
    }


def perturb_cycle_frame(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    *,
    draw_index: int,
    protocol_sha256: str,
    common_capacity_bias_pp: float,
    common_capacity_drift_pp_per_cycle: float,
) -> pd.DataFrame:
    if draw_index == 0:
        return frame.copy()
    capacity = config["correlated_error_model"]["capacity"]
    other = config["correlated_error_model"]["other_channels"]
    result: list[pd.DataFrame] = []
    for cell_id, group in frame.groupby("cell_id", sort=True):
        cell = group.sort_values("cycle_index", kind="stable").copy()
        seed_material = f"{protocol_sha256}|{draw_index}|{cell_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        count = len(cell)
        ar = _stationary_ar1(
            rng,
            count,
            rho=float(capacity["ar1_rho"]),
            stationary_sigma=float(capacity["ar1_stationary_sigma_pp"]),
        )
        iid_distribution = str(capacity.get("iid_distribution", "gaussian"))
        iid_scale = float(capacity["iid_sigma_pp"])
        if iid_distribution == "gaussian":
            iid = rng.normal(0.0, iid_scale, size=count)
        elif iid_distribution == "student_t":
            degrees = float(capacity["iid_degrees_of_freedom"])
            if degrees <= 2.0:
                raise v5.FastChargeV5PairwiseError(
                    "V9 Student-t IID perturbations require df > 2"
                )
            iid = rng.standard_t(degrees, size=count) * iid_scale
        else:
            raise v5.FastChargeV5PairwiseError(
                f"Unknown V9 IID perturbation distribution: {iid_distribution}"
            )
        spikes = (
            rng.random(count) < float(capacity["spike_probability_per_cycle"])
        ) * rng.normal(0.0, float(capacity["spike_sigma_pp"]), size=count)
        cycles = cell["cycle_index"].to_numpy(dtype=float)
        capacity_error_pp = (
            common_capacity_bias_pp
            + common_capacity_drift_pp_per_cycle * (cycles - 1.0)
            + ar
            + iid
            + spikes
        )
        normalization = float(
            np.median(cell.loc[cell["cycle_index"] <= 5, "discharge_capacity_ah"])
        )
        cell["discharge_capacity_ah"] += normalization * capacity_error_pp / 100.0
        cell["internal_resistance_ohm"] *= np.exp(
            rng.normal(
                0.0,
                float(other["log_internal_resistance_iid_sigma"]),
                size=count,
            )
        )
        cell["temperature_max_c"] += rng.normal(
            0.0, float(other["temperature_iid_sigma_c"]), size=count
        )
        cell["charge_time_s"] *= np.exp(
            rng.normal(0.0, float(other["log_charge_time_iid_sigma"]), size=count)
        )
        cell["energy_efficiency"] += rng.normal(
            0.0, float(other["energy_efficiency_iid_sigma"]), size=count
        )
        result.append(cell)
    return pd.concat(result, ignore_index=True)


def _stationary_ar1(
    rng: np.random.Generator,
    count: int,
    *,
    rho: float,
    stationary_sigma: float,
) -> np.ndarray:
    if not 0.0 <= rho < 1.0 or stationary_sigma < 0.0:
        raise v5.FastChargeV5PairwiseError("Invalid V9 synthetic AR(1) parameters")
    values = np.empty(count, dtype=float)
    values[0] = rng.normal(0.0, stationary_sigma)
    innovation_sigma = stationary_sigma * np.sqrt(1.0 - rho * rho)
    for index in range(1, count):
        values[index] = rho * values[index - 1] + rng.normal(0.0, innovation_sigma)
    return values


def build_end_to_end_ledger(
    config: Mapping[str, object],
    training: pd.DataFrame,
    target_prefix: pd.DataFrame,
    *,
    protocol_sha256: str,
    selection: Mapping[str, object],
    core: Mapping[str, object],
) -> pd.DataFrame:
    model = config["model_contract"]
    capacity = config["correlated_error_model"]["capacity"]
    draw_count = int(config["stability_gate"]["draw_count"])
    spec = v5.ModelSpec(
        str(model["model_id"]),
        str(model["family"]),
        tuple((str(key), value) for key, value in sorted(model["parameters"].items())),
    )
    if str(selection["selected_model_spec"]["model_id"]) != spec.model_id:
        raise v5.FastChargeV5PairwiseError("V9 model differs from frozen V5 selection")
    rows: list[dict[str, object]] = []
    for draw_index in range(draw_count + 1):
        common_rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"{protocol_sha256}|common|{draw_index}".encode("utf-8")
                ).digest()[:8],
                "little",
            )
        )
        common_bias = (
            0.0
            if draw_index == 0
            else float(common_rng.normal(0.0, float(capacity["common_bias_sigma_pp"])))
        )
        common_drift = (
            0.0
            if draw_index == 0
            else float(
                common_rng.normal(
                    0.0,
                    float(capacity["linear_drift_slope_sigma_pp_per_cycle"]),
                )
            )
        )
        perturbed_training = perturb_cycle_frame(
            training,
            config,
            draw_index=draw_index,
            protocol_sha256=protocol_sha256,
            common_capacity_bias_pp=common_bias,
            common_capacity_drift_pp_per_cycle=common_drift,
        )
        perturbed_target = perturb_cycle_frame(
            target_prefix,
            config,
            draw_index=draw_index,
            protocol_sha256=protocol_sha256,
            common_capacity_bias_pp=common_bias,
            common_capacity_drift_pp_per_cycle=common_drift,
        )
        cells = v5._validated_cells(perturbed_training, required_support=300)
        predictions: dict[int, tuple[np.ndarray, dict[str, object], str]] = {}
        for prefix_cycle in (60, 100):
            matrix, labels, _ = v5.build_pairwise_training_matrix(
                cells,
                prefix_cycle,
                300,
                core,
                anchor_stride=int(model["anchor_stride"]),
            )
            estimator = v5.make_estimator(
                spec,
                pairwise=True,
                random_state=int(model["random_state"]),
            ).fit(matrix, labels)
            resources = v5._cell_resources(cells, prefix_cycle, core)
            target = perturbed_target.loc[
                perturbed_target["cycle_index"] <= prefix_cycle
            ].reset_index(drop=True)
            prediction, audit = v5.predict_pairwise_trajectory(
                estimator,
                target,
                cells,
                prefix_cycle,
                300,
                core,
                aggregation=str(model["aggregation"]),
                neighbor_count=int(model["reference_count"]),
                reference_resources=resources,
            )
            source_hash = _array_sha256(
                matrix,
                labels,
                target[
                    [
                        "cycle_index",
                        "discharge_capacity_ah",
                        "internal_resistance_ohm",
                        "temperature_max_c",
                        "charge_time_s",
                        "energy_efficiency",
                    ]
                ].to_numpy(dtype=float),
                metadata=f"{spec.model_id}|{prefix_cycle}|{draw_index}",
            )
            predictions[prefix_cycle] = prediction, audit, source_hash

        normalization = _normalization_capacity(perturbed_target)
        observed = _retention(perturbed_target, normalization)[60:100]
        observed_hash = _array_sha256(
            perturbed_target[
                [
                    "cycle_index",
                    "discharge_capacity_ah",
                    "internal_resistance_ohm",
                    "temperature_max_c",
                    "charge_time_s",
                    "energy_efficiency",
                ]
            ].to_numpy(dtype=float),
            metadata=f"observed|{draw_index}",
        )
        _append_role_rows(
            rows,
            config,
            draw_index=draw_index,
            role="p100_observed_prefix",
            cycles=range(61, 101),
            values=observed,
            references=[],
            source_sha256=observed_hash,
        )
        for prefix_cycle, role in ((60, "p60_v5_center"), (100, "p100_v5_center")):
            prediction, audit, source_hash = predictions[prefix_cycle]
            _append_role_rows(
                rows,
                config,
                draw_index=draw_index,
                role=role,
                cycles=range(prefix_cycle + 1, 301),
                values=prediction,
                references=list(audit["reference_cell_ids"]),
                source_sha256=source_hash,
            )
    return pd.DataFrame(rows, columns=v9.LEDGER_COLUMNS)


def _append_role_rows(
    rows: list[dict[str, object]],
    config: Mapping[str, object],
    *,
    draw_index: int,
    role: str,
    cycles: range,
    values: np.ndarray,
    references: list[str],
    source_sha256: str,
) -> None:
    fixture = config["synthetic_fixture"]
    reference_json = json.dumps(references, separators=(",", ":"))
    for cycle, value in zip(cycles, values, strict=True):
        rows.append(
            {
                "schema_version": v9.LEDGER_SCHEMA_VERSION,
                "issuance_id": "SYNTH_V9_ISSUANCE_001",
                "cell_id": str(fixture["target_cell_id"]),
                "manufacturing_batch_id": str(fixture["manufacturing_batch_id"]),
                "draw_index": draw_index,
                "trajectory_role": role,
                "cycle_index": cycle,
                "retention_pct": float(value),
                "reference_cell_ids_json": reference_json,
                "source_sha256": source_sha256,
            }
        )


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V9 output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
                f"V9 frozen source hash changed: {path.name}"
            )
        sources[name] = path
    candidate = json.loads(sources["frozen_v7_rule"].read_text(encoding="utf-8"))
    selection = json.loads(sources["frozen_v5_selection"].read_text(encoding="utf-8"))
    v2_config = load_fastcharge_safe_prior_v2_config(sources["frozen_v5_core_config"])
    core = _core_config(v2_config)
    training, target_prefix = build_synthetic_inputs(config)
    protocol_sha256 = _sha256(sources["parent_protocol"])
    ledger = build_end_to_end_ledger(
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
    stress_ledger = _stress_negative_control(ledger)
    stress_correction, stress_status, _ = v9.evaluate_end_to_end_stability(
        stress_ledger,
        candidate,
        config,
        protocol_sha256=protocol_sha256,
    )
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "replicate_ledger.csv"
    metrics_path = output / "draw_metrics.csv"
    correction_path = output / "stable_correction.csv"
    _write_csv(ledger, ledger_path)
    _write_csv(draw_metrics, metrics_path)
    _write_csv(
        pd.DataFrame(
            {
                "forecast_cycle": np.arange(101, 301, dtype=int),
                "v9_effective_correction_pp": correction,
            }
        ),
        correction_path,
    )
    decision = {
        **status,
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_sha256": _sha256(config_path),
        "parent_protocol_sha256": protocol_sha256,
        "frozen_v7_rule_sha256": _sha256(sources["frozen_v7_rule"]),
        "frozen_v5_selection_sha256": _sha256(sources["frozen_v5_selection"]),
        "runtime_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
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
        "synthetic_fixture": {
            "training_cell_count": int(
                config["synthetic_fixture"]["training_cell_count"]
            ),
            "target_prefix_last_cycle": int(target_prefix["cycle_index"].max()),
            "target_future_rows_generated": 0,
            "replicate_ledger_rows": len(ledger),
            "correlated_components": [
                "iid",
                "common_bias",
                "ar1",
                "linear_drift",
                "rare_spike",
            ],
            "historical_v5_refit_each_draw": True,
            "reference_reselection_each_draw": True,
        },
        "artifacts": {
            "replicate_ledger_sha256": _sha256(ledger_path),
            "draw_metrics_sha256": _sha256(metrics_path),
            "stable_correction_sha256": _sha256(correction_path),
        },
        "stress_negative_control": {
            "construction": (
                "software_only_reference_reselection_and_center_drift_stress"
            ),
            "quality_activated": bool(stress_status["quality_activated"]),
            "reasons": stress_status["reasons"],
            "exact_zero_correction": bool(np.all(stress_correction == 0.0)),
        },
        "decision": (
            "synthetic_end_to_end_software_dry_run_passed_without_model_evidence"
            if bool(status["quality_activated"])
            and not bool(stress_status["quality_activated"])
            and bool(np.all(stress_correction == 0.0))
            else "synthetic_end_to_end_software_dry_run_failed_closed"
        ),
        "real_model_evidence_created": False,
        "claim_boundaries": config["claim_boundaries"],
    }
    decision_path = output / "decision.json"
    _write_json(decision, decision_path)
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
                "lifetwin.fastcharge_v9.end_to_end_synthetic_manifest.v1"
            ),
            "experiment_id": config["experiment_id"],
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _stress_negative_control(ledger: pd.DataFrame) -> pd.DataFrame:
    """Create a labelled software-only case that must trigger exact fallback."""

    stress = ledger.copy()
    stochastic = stress["draw_index"] > 0
    center = stress["trajectory_role"].isin(("p60_v5_center", "p100_v5_center"))
    mask = stochastic & center
    direction = np.where(stress.loc[mask, "draw_index"].to_numpy() % 2, 1.0, -1.0)
    progress = (stress.loc[mask, "cycle_index"].to_numpy(dtype=float) - 100.0) / 200.0
    stress.loc[mask, "retention_pct"] += direction * 0.25 * np.maximum(progress, 0.0)
    stress_references = json.dumps(
        [f"SYNTH_STRESS_REFERENCE_{index:02d}" for index in range(12)],
        separators=(",", ":"),
    )
    stress.loc[mask, "reference_cell_ids_json"] = stress_references
    for draw_index in sorted(stress.loc[mask, "draw_index"].unique()):
        for role in ("p60_v5_center", "p100_v5_center"):
            role_mask = (stress["draw_index"] == draw_index) & (
                stress["trajectory_role"] == role
            )
            stress.loc[role_mask, "source_sha256"] = hashlib.sha256(
                f"v9-stress-negative-control|{draw_index}|{role}".encode("utf-8")
            ).hexdigest()
    return stress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
