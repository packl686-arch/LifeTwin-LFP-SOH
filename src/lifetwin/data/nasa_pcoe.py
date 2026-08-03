from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lifetwin.data.schema import (
    validate_cell_labels,
    validate_cycle_summary,
)


NASA_PCOE_DATASET_ID = "NASA_PCOE_LI_ION_AGING_DERIVED_CSV_V1"
NASA_PCOE_SOURCE_URL = "https://catalog.data.gov/dataset/li-ion-battery-aging-datasets"
NASA_PCOE_SOURCE_IDENTIFIER = "DASHLINK_133"
NASA_PCOE_SOURCE_LICENSE = "not_specified"
NASA_PCOE_CHEMISTRY = "unspecified_li_ion"
NASA_PCOE_EVIDENCE_ROLE = "cross_chemistry_accelerated_cycling_stress_only"
NASA_PCOE_CLAIM_BOUNDARY = (
    "not_lfp_not_calendar_aging_not_long_term_not_hithium_validation"
)
NASA_PCOE_NOMINAL_CAPACITY_AH = 2.0
NASA_PCOE_EOL_SOH_FRACTION = 0.70
NASA_PCOE_AMBIENT_TEMPERATURE_C = 24.0
NASA_PCOE_MAX_CSV_FIELD_BYTES = 512 * 1024 * 1024

# These identities pin the four third-party CSV conversions supplied for the
# benchmark. They are not hashes of the original NASA MAT archive.
NASA_PCOE_CSV_IDENTITIES: dict[str, dict[str, object]] = {
    "B0005": {
        "filename": "B0005.csv",
        "size_bytes": 49_218_466,
        "sha256": ("d74b6352fde77fcb55543df48180914ca92d56d36320d70e0ebfcd57696b6105"),
        "charge_count": 170,
        "discharge_count": 168,
        "impedance_count": 278,
        "discharge_cutoff_voltage_v": 2.7,
        "first_eol_cycle": 125,
        "campaign_id": "NASA_PCOE_FY08Q4_B0005_B0007_SYNCHRONIZED",
    },
    "B0006": {
        "filename": "B0006.csv",
        "size_bytes": 49_410_002,
        "sha256": ("d544bdcfdf053861cc96736dd25b91a7de99fa91d1a6877aba3e05bb6a5d97c9"),
        "charge_count": 170,
        "discharge_count": 168,
        "impedance_count": 278,
        "discharge_cutoff_voltage_v": 2.5,
        "first_eol_cycle": 109,
        "campaign_id": "NASA_PCOE_FY08Q4_B0005_B0007_SYNCHRONIZED",
    },
    "B0007": {
        "filename": "B0007.csv",
        "size_bytes": 49_943_430,
        "sha256": ("251b6a074702fc07991db86c1760db843db967d6648f01f4270337d94461fd80"),
        "charge_count": 170,
        "discharge_count": 168,
        "impedance_count": 278,
        "discharge_cutoff_voltage_v": 2.2,
        "first_eol_cycle": None,
        "campaign_id": "NASA_PCOE_FY08Q4_B0005_B0007_SYNCHRONIZED",
    },
    "B0018": {
        "filename": "B0018.csv",
        "size_bytes": 26_358_323,
        "sha256": ("9ce1516d47b3cb2a4a03d9a6c671fdbc6e703468795cbe5ee772605989ac011f"),
        "charge_count": 134,
        "discharge_count": 132,
        "impedance_count": 53,
        "discharge_cutoff_voltage_v": 2.5,
        "first_eol_cycle": 97,
        "campaign_id": "NASA_PCOE_FY08Q4_B0018_LATER_RUN",
    },
}

NASA_PCOE_OUTER_COLUMNS = ("type", "temp", "time", "data")
NASA_PCOE_ALLOWED_TYPES = ("charge", "discharge", "impedance")
NASA_PCOE_CHARGE_FIELDS = (
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_charge",
    "Voltage_charge",
    "Time",
)
NASA_PCOE_DISCHARGE_FIELDS = (
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Time",
    "Capacity",
)

NASA_PCOE_CYCLE_COLUMNS = (
    "dataset_id",
    "cell_id",
    "batch_id",
    "campaign_id",
    "protocol_id",
    "cycle_index",
    "event_time",
    "elapsed_days",
    "ambient_temperature_c",
    "discharge_capacity_ah",
    "nominal_capacity_ah",
    "soh_nominal_fraction",
    "soh_initial_fraction",
    "discharge_cutoff_voltage_v",
    "sample_count",
    "discharge_duration_s",
    "voltage_start_v",
    "voltage_end_v",
    "voltage_min_v",
    "voltage_max_v",
    "current_mean_a",
    "temperature_avg_c",
    "temperature_max_c",
    "temperature_rise_c",
    "integrated_discharge_capacity_ah",
    "integrated_discharge_energy_wh",
    "capacity_integration_ratio",
    "time_to_3p8_v_s",
    "time_to_3p6_v_s",
    "time_to_3p4_v_s",
    "common_window_3p8_to_3p4_duration_s",
    "voltage_at_0p5_ah_v",
    "voltage_at_1p0_ah_v",
    "mean_dv_dq_0p5_to_1p0_v_per_ah",
    "chemistry",
    "source_identifier",
    "source_url",
    "source_license",
    "source_format",
    "source_file",
    "source_sha256",
    "evidence_role",
    "long_term_validation_eligible",
    "claim_boundary",
)

NASA_PCOE_LABEL_COLUMNS = (
    "dataset_id",
    "cell_id",
    "cycle_life",
    "event_observed",
    "is_censored",
    "last_observed_cycle",
    "eol_threshold",
    "eol_capacity_ah",
    "event_definition",
    "chemistry",
    "evidence_role",
    "long_term_validation_eligible",
    "claim_boundary",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _NanLiteralRewriter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.nan_count = 0

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id.casefold() == "nan":
            self.nan_count += 1
            return ast.copy_location(ast.Constant(value=math.nan), node)
        raise ValueError(f"Unsupported name in NASA payload: {node.id}")


def parse_nasa_pcoe_payload(text: str) -> tuple[dict[str, object], int]:
    """Parse a CSV payload as data-only Python literals without using eval()."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("NASA payload must be a non-empty literal string")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("NASA payload is not valid literal syntax") from exc
    rewriter = _NanLiteralRewriter()
    try:
        tree = rewriter.visit(tree)
        ast.fix_missing_locations(tree)
        value = ast.literal_eval(tree)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise ValueError("NASA payload contains a non-literal expression") from exc
    if not isinstance(value, dict):
        raise ValueError("NASA payload must decode to a dictionary")
    return value, rewriter.nan_count


def _numeric_vector(
    payload: dict[str, object],
    field: str,
    *,
    allow_missing: bool,
) -> np.ndarray:
    value = payload.get(field)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"NASA payload field {field} must be a non-empty sequence")
    numeric: list[float] = []
    for item in value:
        if item is None and allow_missing:
            numeric.append(math.nan)
            continue
        if isinstance(item, bool):
            raise ValueError(f"NASA payload field {field} contains a boolean")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"NASA payload field {field} contains a non-numeric value"
            ) from exc
        if math.isnan(number) and allow_missing:
            numeric.append(number)
            continue
        if not math.isfinite(number):
            raise ValueError(
                f"NASA payload field {field} contains an unsupported non-finite value"
            )
        numeric.append(number)
    result = np.asarray(numeric, dtype=float)
    if not allow_missing and not np.isfinite(result).all():
        raise ValueError(f"NASA payload field {field} must be finite")
    return result


def _validate_signal_lengths(
    payload: dict[str, object],
    fields: tuple[str, ...],
    *,
    allow_missing: bool,
) -> dict[str, np.ndarray]:
    if set(payload) != set(fields):
        raise ValueError(
            "Unexpected NASA payload fields: "
            f"expected={list(fields)}, found={sorted(payload)}"
        )
    arrays = {
        field: _numeric_vector(payload, field, allow_missing=allow_missing)
        for field in fields
    }
    signal_fields = tuple(field for field in fields if field != "Capacity")
    lengths = {len(arrays[field]) for field in signal_fields}
    if len(lengths) != 1:
        raise ValueError("NASA signal arrays must have identical lengths")
    return arrays


def _parse_event_time(value: str, *, row_number: int) -> datetime:
    try:
        event_time = datetime.fromisoformat(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"NASA row {row_number} has an invalid event timestamp"
        ) from exc
    if event_time.tzinfo is not None:
        raise ValueError("NASA event timestamps must be timezone-naive source values")
    return event_time


def _protocol_id(cutoff_voltage_v: float) -> str:
    cutoff = f"{cutoff_voltage_v:.1f}".replace(".", "P")
    return f"NASA_PCOE_DISCHARGE_2A_CUTOFF_{cutoff}V"


def _first_downward_crossing_time(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    threshold_v: float,
) -> float:
    crossings = np.flatnonzero(voltage_v <= threshold_v)
    if crossings.size == 0:
        raise ValueError(f"NASA discharge curve does not reach {threshold_v:.1f} V")
    index = int(crossings[0])
    if index == 0:
        return float(time_s[0])
    time_before = float(time_s[index - 1])
    time_after = float(time_s[index])
    voltage_before = float(voltage_v[index - 1])
    voltage_after = float(voltage_v[index])
    voltage_drop = voltage_before - voltage_after
    if abs(voltage_drop) <= 1e-12:
        return time_after
    fraction = np.clip(
        (voltage_before - threshold_v) / voltage_drop,
        0.0,
        1.0,
    )
    return float(time_before + float(fraction) * (time_after - time_before))


def _discharge_curve_features(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    *,
    reported_capacity_ah: float,
) -> dict[str, float]:
    elapsed_s = time_s - float(time_s[0])
    delta_t_s = np.diff(elapsed_s)
    current_magnitude_a = np.abs(current_a)
    interval_current_a = 0.5 * (current_magnitude_a[:-1] + current_magnitude_a[1:])
    interval_capacity_ah = interval_current_a * delta_t_s / 3_600.0
    cumulative_capacity_ah = np.concatenate(
        (np.asarray([0.0]), np.cumsum(interval_capacity_ah))
    )
    integrated_capacity_ah = float(cumulative_capacity_ah[-1])
    interval_power_w = 0.5 * (
        voltage_v[:-1] * current_magnitude_a[:-1]
        + voltage_v[1:] * current_magnitude_a[1:]
    )
    integrated_energy_wh = float(np.sum(interval_power_w * delta_t_s) / 3_600.0)
    if integrated_capacity_ah <= 0.0 or integrated_energy_wh <= 0.0:
        raise ValueError("NASA discharge curve must contain positive charge and energy")

    crossing_times = {
        threshold: _first_downward_crossing_time(
            elapsed_s,
            voltage_v,
            threshold,
        )
        for threshold in (3.8, 3.6, 3.4)
    }

    strictly_increasing = np.concatenate(
        (
            np.asarray([True]),
            np.diff(cumulative_capacity_ah) > 1e-12,
        )
    )
    capacity_axis = cumulative_capacity_ah[strictly_increasing]
    voltage_axis = voltage_v[strictly_increasing]
    if capacity_axis.size < 2 or capacity_axis[-1] < 1.0 - 1e-9:
        raise ValueError("NASA discharge curve must support the common 1.0 Ah window")
    voltage_0p5_ah = float(np.interp(0.5, capacity_axis, voltage_axis))
    voltage_1p0_ah = float(np.interp(1.0, capacity_axis, voltage_axis))

    return {
        "temperature_rise_c": float(np.max(temperature_c) - temperature_c[0]),
        "integrated_discharge_capacity_ah": integrated_capacity_ah,
        "integrated_discharge_energy_wh": integrated_energy_wh,
        "capacity_integration_ratio": integrated_capacity_ah / reported_capacity_ah,
        "time_to_3p8_v_s": crossing_times[3.8],
        "time_to_3p6_v_s": crossing_times[3.6],
        "time_to_3p4_v_s": crossing_times[3.4],
        "common_window_3p8_to_3p4_duration_s": (
            crossing_times[3.4] - crossing_times[3.8]
        ),
        "voltage_at_0p5_ah_v": voltage_0p5_ah,
        "voltage_at_1p0_ah_v": voltage_1p0_ah,
        "mean_dv_dq_0p5_to_1p0_v_per_ah": ((voltage_1p0_ah - voltage_0p5_ah) / 0.5),
    }


def _read_nasa_pcoe_cell_csv(
    path: Path,
    *,
    cell_id: str,
    identity: dict[str, object],
    verify_source_identity: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"NASA CSV does not exist: {path}")
    source_size = path.stat().st_size
    source_sha256 = _sha256(path)
    expected_filename = str(identity.get("filename", f"{cell_id}.csv"))
    if path.name != expected_filename:
        raise ValueError(
            f"NASA filename mismatch for {cell_id}: expected {expected_filename}"
        )
    if verify_source_identity:
        expected_size = int(identity["size_bytes"])
        expected_sha256 = str(identity["sha256"]).casefold()
        if source_size != expected_size:
            raise ValueError(
                f"NASA {cell_id} byte-size mismatch: expected {expected_size}, "
                f"found {source_size}"
            )
        if source_sha256 != expected_sha256:
            raise ValueError(
                f"NASA {cell_id} SHA-256 mismatch: expected {expected_sha256}, "
                f"found {source_sha256}"
            )

    default_cutoffs = {
        "B0005": 2.7,
        "B0006": 2.5,
        "B0007": 2.2,
        "B0018": 2.5,
    }
    cutoff_voltage_v = float(
        identity.get("discharge_cutoff_voltage_v", default_cutoffs[cell_id])
    )
    default_campaign = (
        "NASA_PCOE_FY08Q4_B0005_B0007_SYNCHRONIZED"
        if cell_id in {"B0005", "B0006", "B0007"}
        else "NASA_PCOE_FY08Q4_B0018_LATER_RUN"
    )
    campaign_id = str(identity.get("campaign_id", default_campaign))
    counts = {operation: 0 for operation in NASA_PCOE_ALLOWED_TYPES}
    empty_impedance_count = 0
    nonempty_impedance_count = 0
    nan_literal_count = 0
    nan_by_operation: dict[str, int] = {operation: 0 for operation in counts}
    short_charge_record_count = 0
    all_event_times: list[datetime] = []
    discharge_rows: list[dict[str, object]] = []

    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"Could not open NASA CSV: {path}") from exc
    with stream:
        csv.field_size_limit(NASA_PCOE_MAX_CSV_FIELD_BYTES)
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != NASA_PCOE_OUTER_COLUMNS:
            raise ValueError(
                "NASA CSV must have the exact outer schema and order "
                f"{list(NASA_PCOE_OUTER_COLUMNS)}"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"NASA row {row_number} violates the four-column schema"
                )
            operation = str(row["type"]).strip().casefold()
            if operation not in counts:
                raise ValueError(
                    f"NASA row {row_number} has unsupported operation {operation!r}"
                )
            try:
                ambient_temperature_c = float(row["temp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"NASA row {row_number} has an invalid ambient temperature"
                ) from exc
            if not math.isfinite(ambient_temperature_c):
                raise ValueError("NASA ambient temperature must be finite")
            event_time = _parse_event_time(row["time"], row_number=row_number)
            if all_event_times and event_time <= all_event_times[-1]:
                raise ValueError(
                    "NASA top-level event timestamps must be strictly increasing"
                )
            all_event_times.append(event_time)

            payload, row_nan_count = parse_nasa_pcoe_payload(row["data"])
            nan_literal_count += row_nan_count
            nan_by_operation[operation] += row_nan_count
            counts[operation] += 1

            if operation == "impedance":
                if payload:
                    nonempty_impedance_count += 1
                else:
                    empty_impedance_count += 1
                continue
            if operation == "charge":
                arrays = _validate_signal_lengths(
                    payload,
                    NASA_PCOE_CHARGE_FIELDS,
                    allow_missing=True,
                )
                if len(arrays["Time"]) <= 5:
                    short_charge_record_count += 1
                continue

            arrays = _validate_signal_lengths(
                payload,
                NASA_PCOE_DISCHARGE_FIELDS,
                allow_missing=False,
            )
            capacity_values = arrays["Capacity"]
            if len(capacity_values) != 1:
                raise ValueError(
                    "NASA discharge Capacity must contain exactly one value"
                )
            capacity_ah = float(capacity_values[0])
            if capacity_ah <= 0.0:
                raise ValueError("NASA discharge Capacity must be positive")
            time_s = arrays["Time"]
            if len(time_s) < 2 or np.any(np.diff(time_s) < 0.0):
                raise ValueError("NASA discharge Time must be nondecreasing")
            voltage = arrays["Voltage_measured"]
            current = arrays["Current_measured"]
            temperature = arrays["Temperature_measured"]
            curve_features = _discharge_curve_features(
                time_s,
                voltage,
                current,
                temperature,
                reported_capacity_ah=capacity_ah,
            )
            discharge_rows.append(
                {
                    "dataset_id": NASA_PCOE_DATASET_ID,
                    "cell_id": cell_id,
                    "batch_id": "NASA_PCOE_FY08Q4",
                    "campaign_id": campaign_id,
                    "protocol_id": _protocol_id(cutoff_voltage_v),
                    "cycle_index": len(discharge_rows) + 1,
                    "event_time": event_time.isoformat(sep=" "),
                    "elapsed_days": math.nan,
                    "ambient_temperature_c": ambient_temperature_c,
                    "discharge_capacity_ah": capacity_ah,
                    "nominal_capacity_ah": NASA_PCOE_NOMINAL_CAPACITY_AH,
                    "soh_nominal_fraction": (
                        capacity_ah / NASA_PCOE_NOMINAL_CAPACITY_AH
                    ),
                    "soh_initial_fraction": math.nan,
                    "discharge_cutoff_voltage_v": cutoff_voltage_v,
                    "sample_count": len(time_s),
                    "discharge_duration_s": float(time_s[-1] - time_s[0]),
                    "voltage_start_v": float(voltage[0]),
                    "voltage_end_v": float(voltage[-1]),
                    "voltage_min_v": float(np.min(voltage)),
                    "voltage_max_v": float(np.max(voltage)),
                    "current_mean_a": float(np.mean(current)),
                    "temperature_avg_c": float(np.mean(temperature)),
                    "temperature_max_c": float(np.max(temperature)),
                    **curve_features,
                    "chemistry": NASA_PCOE_CHEMISTRY,
                    "source_identifier": NASA_PCOE_SOURCE_IDENTIFIER,
                    "source_url": NASA_PCOE_SOURCE_URL,
                    "source_license": NASA_PCOE_SOURCE_LICENSE,
                    "source_format": "third_party_csv_conversion_of_nasa_mat",
                    "source_file": path.name,
                    "source_sha256": source_sha256,
                    "evidence_role": NASA_PCOE_EVIDENCE_ROLE,
                    "long_term_validation_eligible": False,
                    "claim_boundary": NASA_PCOE_CLAIM_BOUNDARY,
                }
            )

    if not discharge_rows:
        raise ValueError(f"NASA {cell_id} has no discharge records")
    if verify_source_identity:
        for operation in NASA_PCOE_ALLOWED_TYPES:
            count_key = f"{operation}_count"
            if count_key not in identity:
                continue
            expected = int(identity[count_key])
            if counts[operation] != expected:
                raise ValueError(
                    f"NASA {cell_id} {operation} count mismatch: "
                    f"expected {expected}, found {counts[operation]}"
                )

    cycles = pd.DataFrame(discharge_rows)
    event_times = pd.to_datetime(cycles["event_time"], errors="raise")
    cycles["elapsed_days"] = (
        event_times - event_times.iloc[0]
    ).dt.total_seconds() / 86_400.0
    initial_capacity = float(cycles["discharge_capacity_ah"].iloc[:5].median())
    cycles["soh_initial_fraction"] = cycles["discharge_capacity_ah"] / initial_capacity
    cycles = cycles[list(NASA_PCOE_CYCLE_COLUMNS)]

    inventory = {
        "dataset_id": NASA_PCOE_DATASET_ID,
        "cell_id": cell_id,
        "campaign_id": campaign_id,
        "source_file": path.name,
        "source_path": str(path.resolve()),
        "source_size_bytes": source_size,
        "source_sha256": source_sha256,
        "identity_verified": bool(verify_source_identity),
        "charge_count": counts["charge"],
        "discharge_count": counts["discharge"],
        "impedance_count": counts["impedance"],
        "empty_impedance_count": empty_impedance_count,
        "nonempty_impedance_count": nonempty_impedance_count,
        "impedance_payload_available": nonempty_impedance_count > 0,
        "nan_literal_count": nan_literal_count,
        "charge_nan_literal_count": nan_by_operation["charge"],
        "discharge_nan_literal_count": nan_by_operation["discharge"],
        "short_charge_record_count": short_charge_record_count,
        "first_event_time": all_event_times[0].isoformat(sep=" "),
        "last_event_time": all_event_times[-1].isoformat(sep=" "),
        "discharge_cutoff_voltage_v": cutoff_voltage_v,
        "chemistry": NASA_PCOE_CHEMISTRY,
        "source_license": NASA_PCOE_SOURCE_LICENSE,
    }
    return cycles, inventory


def validate_nasa_pcoe_cycle_summary(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != NASA_PCOE_CYCLE_COLUMNS:
        raise ValueError("NASA canonical cycle columns or order changed")
    validate_cycle_summary(frame)
    if frame[list(NASA_PCOE_CYCLE_COLUMNS)].isna().any().any():
        raise ValueError("NASA canonical cycle summary cannot contain null values")
    expected_constants: dict[str, object] = {
        "dataset_id": NASA_PCOE_DATASET_ID,
        "nominal_capacity_ah": NASA_PCOE_NOMINAL_CAPACITY_AH,
        "chemistry": NASA_PCOE_CHEMISTRY,
        "source_identifier": NASA_PCOE_SOURCE_IDENTIFIER,
        "source_url": NASA_PCOE_SOURCE_URL,
        "source_license": NASA_PCOE_SOURCE_LICENSE,
        "evidence_role": NASA_PCOE_EVIDENCE_ROLE,
        "long_term_validation_eligible": False,
        "claim_boundary": NASA_PCOE_CLAIM_BOUNDARY,
    }
    for column, expected in expected_constants.items():
        if set(frame[column].tolist()) != {expected}:
            raise ValueError(f"Unexpected NASA canonical values for {column}")
    numeric_columns = (
        "elapsed_days",
        "ambient_temperature_c",
        "discharge_capacity_ah",
        "nominal_capacity_ah",
        "soh_nominal_fraction",
        "soh_initial_fraction",
        "discharge_cutoff_voltage_v",
        "sample_count",
        "discharge_duration_s",
        "voltage_start_v",
        "voltage_end_v",
        "voltage_min_v",
        "voltage_max_v",
        "current_mean_a",
        "temperature_avg_c",
        "temperature_max_c",
        "temperature_rise_c",
        "integrated_discharge_capacity_ah",
        "integrated_discharge_energy_wh",
        "capacity_integration_ratio",
        "time_to_3p8_v_s",
        "time_to_3p6_v_s",
        "time_to_3p4_v_s",
        "common_window_3p8_to_3p4_duration_s",
        "voltage_at_0p5_ah_v",
        "voltage_at_1p0_ah_v",
        "mean_dv_dq_0p5_to_1p0_v_per_ah",
    )
    numeric = frame[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("NASA canonical numeric values must be finite")
    if (numeric["sample_count"] < 2).any():
        raise ValueError("NASA discharge records require at least two samples")
    if (numeric["elapsed_days"] < 0.0).any():
        raise ValueError("NASA elapsed_days cannot be negative")
    if (numeric["integrated_discharge_capacity_ah"] <= 0.0).any():
        raise ValueError("NASA integrated discharge capacity must be positive")
    if (numeric["integrated_discharge_energy_wh"] <= 0.0).any():
        raise ValueError("NASA integrated discharge energy must be positive")
    if (numeric["temperature_rise_c"] < -1e-12).any():
        raise ValueError("NASA temperature rise cannot be negative")
    if not numeric["capacity_integration_ratio"].between(0.5, 1.5).all():
        raise ValueError("NASA integrated capacity is inconsistent with the label")
    if not np.allclose(
        numeric["capacity_integration_ratio"],
        numeric["integrated_discharge_capacity_ah"] / numeric["discharge_capacity_ah"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("NASA capacity integration ratio is inconsistent")
    if not (
        (numeric["time_to_3p8_v_s"] <= numeric["time_to_3p6_v_s"] + 1e-12)
        & (numeric["time_to_3p6_v_s"] <= numeric["time_to_3p4_v_s"] + 1e-12)
    ).all():
        raise ValueError("NASA voltage crossing times must be ordered")
    crossing_columns = ("time_to_3p8_v_s", "time_to_3p6_v_s", "time_to_3p4_v_s")
    if (numeric.loc[:, crossing_columns] < -1e-12).any().any() or numeric.loc[
        :, crossing_columns
    ].gt(numeric["discharge_duration_s"] + 1e-12, axis="index").any().any():
        raise ValueError("NASA voltage crossing time lies outside the discharge")
    if not np.allclose(
        numeric["common_window_3p8_to_3p4_duration_s"],
        numeric["time_to_3p4_v_s"] - numeric["time_to_3p8_v_s"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("NASA common-voltage-window duration is inconsistent")
    if not np.allclose(
        numeric["mean_dv_dq_0p5_to_1p0_v_per_ah"],
        (numeric["voltage_at_1p0_ah_v"] - numeric["voltage_at_0p5_ah_v"]) / 0.5,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("NASA common-capacity-window dV/dQ is inconsistent")
    for column in ("voltage_at_0p5_ah_v", "voltage_at_1p0_ah_v"):
        if not (
            (numeric[column] >= numeric["voltage_min_v"] - 1e-12)
            & (numeric[column] <= numeric["voltage_max_v"] + 1e-12)
        ).all():
            raise ValueError(f"NASA {column} lies outside the observed voltage range")
    if (numeric["mean_dv_dq_0p5_to_1p0_v_per_ah"] > 1e-12).any():
        raise ValueError("NASA common-capacity-window dV/dQ must be non-positive")
    if not np.allclose(
        numeric["soh_nominal_fraction"],
        numeric["discharge_capacity_ah"] / NASA_PCOE_NOMINAL_CAPACITY_AH,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("NASA nominal SOH is inconsistent with discharge capacity")
    for cell_id, cell in frame.groupby("cell_id", sort=True):
        ordered = cell.sort_values("cycle_index", kind="stable")
        expected_cycles = list(range(1, len(ordered) + 1))
        if ordered["cycle_index"].astype(int).tolist() != expected_cycles:
            raise ValueError(f"NASA {cell_id} cycle indices must be contiguous")
        if not ordered["elapsed_days"].is_monotonic_increasing:
            raise ValueError(f"NASA {cell_id} elapsed_days must be monotonic")
        initial_capacity = float(ordered["discharge_capacity_ah"].iloc[:5].median())
        if not np.allclose(
            ordered["soh_initial_fraction"],
            ordered["discharge_capacity_ah"] / initial_capacity,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"NASA {cell_id} initial-normalized SOH is inconsistent")
        if ordered["protocol_id"].nunique() != 1:
            raise ValueError(f"NASA {cell_id} cannot span discharge protocols")
        if ordered["discharge_cutoff_voltage_v"].nunique() != 1:
            raise ValueError(f"NASA {cell_id} cannot span discharge cutoffs")


def _build_nasa_pcoe_labels(
    cycles: pd.DataFrame,
    *,
    eol_soh_fraction: float,
) -> pd.DataFrame:
    if not 0.0 < eol_soh_fraction < 1.0:
        raise ValueError("NASA EOL SOH fraction must lie strictly between zero and one")
    eol_capacity_ah = NASA_PCOE_NOMINAL_CAPACITY_AH * eol_soh_fraction
    rows: list[dict[str, object]] = []
    for (dataset_id, cell_id), cell in cycles.groupby(
        ["dataset_id", "cell_id"], sort=True
    ):
        ordered = cell.sort_values("cycle_index", kind="stable")
        crossing = ordered.loc[ordered["discharge_capacity_ah"] <= eol_capacity_ah]
        event_observed = not crossing.empty
        cycle_life = int(
            crossing.iloc[0]["cycle_index"]
            if event_observed
            else ordered.iloc[-1]["cycle_index"]
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "cell_id": cell_id,
                "cycle_life": cycle_life,
                "event_observed": event_observed,
                "is_censored": not event_observed,
                "last_observed_cycle": int(ordered["cycle_index"].max()),
                "eol_threshold": eol_soh_fraction,
                "eol_capacity_ah": eol_capacity_ah,
                "event_definition": (
                    "first observed discharge capacity at or below "
                    f"{eol_capacity_ah:.6g} Ah"
                ),
                "chemistry": NASA_PCOE_CHEMISTRY,
                "evidence_role": NASA_PCOE_EVIDENCE_ROLE,
                "long_term_validation_eligible": False,
                "claim_boundary": NASA_PCOE_CLAIM_BOUNDARY,
            }
        )
    labels = pd.DataFrame(rows)[list(NASA_PCOE_LABEL_COLUMNS)]
    validate_cell_labels(labels)
    return labels


def nasa_pcoe_cycle_summary_sha256(frame: pd.DataFrame) -> str:
    validate_nasa_pcoe_cycle_summary(frame)
    normalized = frame.sort_values(
        ["dataset_id", "cell_id", "cycle_index"], kind="stable"
    )
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_nasa_pcoe_frames(
    source_directory: str | Path,
    *,
    verify_source_identity: bool = True,
    eol_soh_fraction: float = NASA_PCOE_EOL_SOH_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize the pinned four-cell CSV bundle into leakage-safe tables."""
    source_root = Path(source_directory)
    if not source_root.is_dir():
        raise FileNotFoundError(f"NASA source directory does not exist: {source_root}")
    normalized_identities: list[tuple[str, str, dict[str, object]]] = []
    for key, identity in NASA_PCOE_CSV_IDENTITIES.items():
        filename = str(
            identity.get(
                "filename",
                key if str(key).casefold().endswith(".csv") else f"{key}.csv",
            )
        )
        cell_id = str(identity.get("cell_id", Path(filename).stem))
        normalized_identities.append((cell_id, filename, identity))
    expected_names = {filename for _, filename, _ in normalized_identities}
    observed_names = {path.name for path in source_root.glob("*.csv")}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "NASA CSV bundle must contain exactly the pinned four files: "
            f"missing={missing}, unexpected={unexpected}"
        )

    cycle_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for cell_id, filename, identity in normalized_identities:
        path = source_root / filename
        cell_cycles, inventory = _read_nasa_pcoe_cell_csv(
            path,
            cell_id=cell_id,
            identity=identity,
            verify_source_identity=verify_source_identity,
        )
        cycle_frames.append(cell_cycles)
        inventory_rows.append(inventory)

    cycles = pd.concat(cycle_frames, ignore_index=True)
    cycles = cycles.sort_values(["cell_id", "cycle_index"], kind="stable").reset_index(
        drop=True
    )
    validate_nasa_pcoe_cycle_summary(cycles)
    labels = _build_nasa_pcoe_labels(
        cycles,
        eol_soh_fraction=eol_soh_fraction,
    )
    inventory = (
        pd.DataFrame(inventory_rows)
        .sort_values("cell_id", kind="stable")
        .reset_index(drop=True)
    )

    if verify_source_identity and all(
        "first_eol_cycle" in identity for _, _, identity in normalized_identities
    ):
        expected_events = {
            cell_id: identity["first_eol_cycle"]
            for cell_id, _, identity in normalized_identities
        }
        observed_events = {
            str(row.cell_id): (
                int(row.cycle_life) if bool(row.event_observed) else None
            )
            for row in labels.itertuples(index=False)
        }
        if observed_events != expected_events:
            raise ValueError(
                "NASA EOL crossing audit changed: "
                f"expected={expected_events}, found={observed_events}"
            )

    manifest_items = [
        {
            "cell_id": str(row.cell_id),
            "source_file": str(row.source_file),
            "source_size_bytes": int(row.source_size_bytes),
            "source_sha256": str(row.source_sha256),
        }
        for row in inventory.itertuples(index=False)
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest_items,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audit: dict[str, Any] = {
        "status": "passed",
        "dataset_id": NASA_PCOE_DATASET_ID,
        "warning": (
            "Third-party CSV conversion of the NASA PCoE accelerated cycling "
            "dataset. This is a cross-domain stress benchmark, not LFP calendar-"
            "aging, Hithium-product, stationary-storage, or 15-25 year evidence."
        ),
        "source": {
            "directory": str(source_root.resolve()),
            "identifier": NASA_PCOE_SOURCE_IDENTIFIER,
            "url": NASA_PCOE_SOURCE_URL,
            "license": NASA_PCOE_SOURCE_LICENSE,
            "original_format": "MATLAB MAT",
            "provided_format": "third-party CSV conversion",
            "identity_verified": bool(verify_source_identity),
            "manifest_sha256": manifest_sha256,
        },
        "counts": {
            "physical_cell_count": int(cycles["cell_id"].nunique()),
            "discharge_cycle_count": len(cycles),
            "charge_record_count": int(inventory["charge_count"].sum()),
            "impedance_record_count": int(inventory["impedance_count"].sum()),
            "empty_impedance_record_count": int(
                inventory["empty_impedance_count"].sum()
            ),
            "nan_literal_count": int(inventory["nan_literal_count"].sum()),
            "right_censored_cell_count": int(labels["is_censored"].sum()),
        },
        "quality_flags": {
            "impedance_payload_lost_in_conversion": bool(
                not inventory["impedance_payload_available"].any()
            ),
            "charge_nan_values_present": bool(
                inventory["charge_nan_literal_count"].sum() > 0
            ),
            "short_charge_records_present": bool(
                inventory["short_charge_record_count"].sum() > 0
            ),
            "different_discharge_cutoffs_present": bool(
                inventory["discharge_cutoff_voltage_v"].nunique() > 1
            ),
            "capacity_monotonicity_imposed": False,
        },
        "claim_guardrails": {
            "chemistry": NASA_PCOE_CHEMISTRY,
            "chemistry_authoritatively_documented": False,
            "lfp_claim_allowed": False,
            "calendar_aging": False,
            "long_term_validation_eligible": False,
            "industrial_evidence": False,
            "hithium_product_validation": False,
            "raw_csv_redistribution_allowed": False,
            "raw_csv_redistribution_reason": "license_not_specified",
        },
        "canonical_cycle_summary_sha256": nasa_pcoe_cycle_summary_sha256(cycles),
    }
    return cycles, labels, inventory, audit
