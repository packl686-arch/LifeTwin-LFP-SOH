from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest

import lifetwin.data.nasa_pcoe as nasa_pcoe
from lifetwin.data.schema import validate_cell_labels, validate_cycle_summary


_CAPACITY_BY_CELL = {
    "B0005": [1.90, 1.60, 1.39],
    "B0006": [2.00, 1.50, 1.35],
    "B0007": [1.90, 1.60, 1.45],
    "B0018": [1.85, 1.41, 1.30],
}


def _discharge_payload(capacity_ah: float) -> dict[str, list[float]]:
    return {
        "Voltage_measured": [4.10, 3.50, 2.70],
        "Current_measured": [-2.00, -2.01, -0.01],
        "Temperature_measured": [24.0, 27.0, 29.0],
        "Current_load": [-2.0, -2.0, 0.0],
        "Voltage_load": [3.95, 3.35, 0.0],
        "Time": [0.0, 1_500.0, 3_000.0],
        "Capacity": [capacity_ah],
    }


def _write_cell_csv(path: Path, capacities: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["type", "temp", "time", "data"])
        writer.writeheader()
        for cycle_index, capacity in enumerate(capacities, start=1):
            writer.writerow(
                {
                    "type": "discharge",
                    "temp": 24,
                    "time": f"2008-01-{cycle_index:02d} 12:00:00",
                    "data": repr(_discharge_payload(capacity)),
                }
            )
        writer.writerow(
            {
                "type": "impedance",
                "temp": 24,
                "time": "2008-01-04 12:00:00",
                "data": "{}",
            }
        )


def _write_source_bundle(source: Path) -> dict[str, dict[str, object]]:
    source.mkdir()
    identities: dict[str, dict[str, object]] = {}
    for cell_id, capacities in _CAPACITY_BY_CELL.items():
        path = source / f"{cell_id}.csv"
        _write_cell_csv(path, capacities)
        payload = path.read_bytes()
        identities[cell_id] = {
            "filename": path.name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "charge_count": 0,
            "discharge_count": len(capacities),
            "impedance_count": 1,
            "first_eol_cycle": None if cell_id == "B0007" else 3,
        }
    return identities


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = tmp_path / "nasa-pcoe"
    identities = _write_source_bundle(source)
    monkeypatch.setattr(nasa_pcoe, "NASA_PCOE_CSV_IDENTITIES", identities)
    return nasa_pcoe.prepare_nasa_pcoe_frames(
        source,
        verify_source_identity=True,
    )


def test_payload_parser_allows_only_literal_nan() -> None:
    parsed, nan_count = nasa_pcoe.parse_nasa_pcoe_payload(
        "{'Voltage_measured': [4.2, nan], "
        "'Current_measured': [1.5, nan], 'Capacity': [1.9]}"
    )

    assert nan_count == 2
    assert math.isnan(parsed["Voltage_measured"][1])
    assert math.isnan(parsed["Current_measured"][1])
    assert parsed["Capacity"] == [1.9]

    with pytest.raises(ValueError):
        nasa_pcoe.parse_nasa_pcoe_payload("{'value': len([1, 2, 3])}")
    with pytest.raises(ValueError):
        nasa_pcoe.parse_nasa_pcoe_payload("{'value': unknown_name}")


def test_prepare_builds_canonical_cycles_and_right_censored_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycles, labels, inventory, audit = _prepare(tmp_path, monkeypatch)

    required_cycle_columns = {
        "dataset_id",
        "cell_id",
        "batch_id",
        "protocol_id",
        "cycle_index",
        "discharge_capacity_ah",
    }
    required_label_columns = {
        "dataset_id",
        "cell_id",
        "cycle_life",
        "is_censored",
        "eol_threshold",
    }
    assert required_cycle_columns <= set(cycles.columns)
    assert {
        "integrated_discharge_capacity_ah",
        "integrated_discharge_energy_wh",
        "common_window_3p8_to_3p4_duration_s",
        "voltage_at_1p0_ah_v",
    } <= set(cycles.columns)
    assert required_label_columns <= set(labels.columns)
    assert len(cycles) == 12
    assert cycles["cell_id"].nunique() == 4
    assert len(labels) == 4
    assert len(inventory) == 4
    assert isinstance(audit, dict)

    validate_cycle_summary(cycles)
    validate_cell_labels(labels)
    for _, cell in cycles.groupby("cell_id", sort=False):
        assert cell.sort_values("cycle_index")["cycle_index"].tolist() == [1, 2, 3]

    b0007 = labels.loc[labels["cell_id"].astype(str).str.endswith("B0007")]
    assert len(b0007) == 1
    assert bool(b0007.iloc[0]["is_censored"])
    assert int(b0007.iloc[0]["cycle_life"]) == 3
    assert float(b0007.iloc[0]["eol_threshold"]) == pytest.approx(0.70)

    observed = labels.loc[~labels["is_censored"].astype(bool), "cell_id"].astype(str)
    assert {value[-5:] for value in observed} == {"B0005", "B0006", "B0018"}


def test_empty_impedance_payload_never_becomes_a_resistance_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycles, _, inventory, audit = _prepare(tmp_path, monkeypatch)

    if "internal_resistance_ohm" in cycles:
        assert cycles["internal_resistance_ohm"].isna().all()
    resistance_columns = [
        column
        for column in cycles.columns
        if "resistance" in column.casefold() or "impedance" in column.casefold()
    ]
    for column in resistance_columns:
        numeric = pd.to_numeric(cycles[column], errors="coerce")
        assert numeric.isna().all()

    audit_text = repr(audit).casefold()
    inventory_text = " ".join(str(column).casefold() for column in inventory.columns)
    assert "impedance" in audit_text or "impedance" in inventory_text


def test_cycle_summary_fingerprint_is_row_order_invariant_and_value_sensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycles, _, _, _ = _prepare(tmp_path, monkeypatch)

    expected = nasa_pcoe.nasa_pcoe_cycle_summary_sha256(cycles)
    shuffled = cycles.sample(frac=1.0, random_state=17).reset_index(drop=True)
    assert nasa_pcoe.nasa_pcoe_cycle_summary_sha256(shuffled) == expected

    changed = cycles.copy()
    changed.loc[changed.index[0], "discharge_capacity_ah"] += 0.001
    cell_id = changed.loc[changed.index[0], "cell_id"]
    cell_mask = changed["cell_id"] == cell_id
    cell = changed.loc[cell_mask].sort_values("cycle_index", kind="stable")
    initial_capacity = float(cell["discharge_capacity_ah"].iloc[:5].median())
    changed.loc[cell_mask, "soh_nominal_fraction"] = (
        changed.loc[cell_mask, "discharge_capacity_ah"] / 2.0
    )
    changed.loc[cell_mask, "soh_initial_fraction"] = (
        changed.loc[cell_mask, "discharge_capacity_ah"] / initial_capacity
    )
    changed.loc[changed.index[0], "capacity_integration_ratio"] = (
        changed.loc[changed.index[0], "integrated_discharge_capacity_ah"]
        / changed.loc[changed.index[0], "discharge_capacity_ah"]
    )
    assert nasa_pcoe.nasa_pcoe_cycle_summary_sha256(changed) != expected


def test_curve_feature_validator_rejects_impossible_crossing_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycles, _, _, _ = _prepare(tmp_path, monkeypatch)
    attacked = cycles.copy()
    attacked.loc[attacked.index[0], "time_to_3p4_v_s"] = (
        attacked.loc[attacked.index[0], "discharge_duration_s"] + 1.0
    )

    with pytest.raises(ValueError, match="outside the discharge"):
        nasa_pcoe.validate_nasa_pcoe_cycle_summary(attacked)
