from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_synthetic as synthetic
from scripts import run_synthetic_long_horizon_identifiability as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v1.json"
)


@pytest.fixture()
def protocol() -> synthetic.ValidatedSyntheticProtocol:
    return synthetic.load_frozen_protocol_config(CONFIG_PATH)


@pytest.fixture()
def scratch_root() -> Path:
    root = PROJECT_ROOT / "artifacts" / "test-scratch" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_ordinary_cluster_ids_are_deterministic_opaque_and_unique(
    protocol: synthetic.ValidatedSyntheticProtocol,
) -> None:
    identifiers = {
        runner._ordinary_cluster_id(protocol, partition, family, index)
        for partition in synthetic.PARTITION_NAMES
        for family in synthetic.TRUTH_FAMILY_IDS
        for index in range(3)
    }
    assert len(identifiers) == 60
    assert all(value.startswith("c_") and len(value) == 34 for value in identifiers)
    assert all(
        token not in value
        for value in identifiers
        for token in (*synthetic.PARTITION_NAMES, *synthetic.TRUTH_FAMILY_IDS)
    )
    assert runner._ordinary_cluster_id(
        protocol, "test", "single_power", 2
    ) == runner._ordinary_cluster_id(protocol, "test", "single_power", 2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_json_ready_rejects_every_nonfinite_number(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        runner._json_ready({"value": value})


def _freeze_environment(source_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "git_status_porcelain": "",
        "git_commit": "frozen-commit",
        "source_sha256": dict(source_hashes),
        "source_tree_sha256": runner._source_tree_sha256(source_hashes),
    }


def test_environment_freeze_rejects_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_hashes = {"frozen.py": "a" * 64}
    environment = _freeze_environment(source_hashes)

    def git_output(arguments: tuple[str, ...]) -> str:
        if arguments[0] == "status":
            return " M frozen.py"
        return "frozen-commit"

    monkeypatch.setattr(runner, "_git_output", git_output)
    monkeypatch.setattr(
        runner, "_current_source_hashes", lambda: dict(source_hashes)
    )

    with pytest.raises(RuntimeError, match="worktree is no longer clean"):
        runner._verify_environment_freeze(environment)


def test_environment_freeze_rejects_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_hashes = {"frozen.py": "a" * 64}
    environment = _freeze_environment(source_hashes)
    monkeypatch.setattr(
        runner,
        "_git_output",
        lambda arguments: "" if arguments[0] == "status" else "frozen-commit",
    )
    monkeypatch.setattr(
        runner,
        "_current_source_hashes",
        lambda: {"frozen.py": "b" * 64},
    )

    with pytest.raises(RuntimeError, match="source files changed: frozen.py"):
        runner._verify_environment_freeze(environment)


def test_environment_freeze_rejects_head_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_hashes = {"frozen.py": "a" * 64}
    environment = _freeze_environment(source_hashes)
    monkeypatch.setattr(
        runner,
        "_git_output",
        lambda arguments: "" if arguments[0] == "status" else "new-commit",
    )
    monkeypatch.setattr(
        runner, "_current_source_hashes", lambda: dict(source_hashes)
    )

    with pytest.raises(RuntimeError, match="git HEAD changed"):
        runner._verify_environment_freeze(environment)


def test_generation_phase_keeps_truth_in_separate_directory(
    protocol: synthetic.ValidatedSyntheticProtocol,
    scratch_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reduced = replace(
        protocol,
        cluster_counts_per_truth_family=tuple(
            (partition, 1) for partition in synthetic.PARTITION_NAMES
        ),
    )
    one_pair = synthetic.generate_matched_pair_packs(
        protocol, zero_based_pair_index=0
    )
    monkeypatch.setattr(runner, "load_frozen_protocol_config", lambda path: reduced)
    monkeypatch.setattr(
        runner,
        "generate_all_matched_pair_packs",
        lambda selected: one_pair,
    )
    work = scratch_root / "label_free"
    sealed = scratch_root / "sealed"

    result = runner.generate_phase(CONFIG_PATH, work, sealed)

    assert result["ordinary_cluster_count"] == 20
    assert result["matched_cluster_count"] == 2
    assert (work / "prefix_pack.csv").is_file()
    assert (work / "forecast_coordinates.csv").is_file()
    assert (work / "truth_commitment.json").is_file()
    assert not (work / "truth_pack.csv").exists()
    assert not (work / "matched_prefix_pairs.csv").exists()
    assert (sealed / "truth_pack.csv").is_file()
    assert (sealed / "matched_prefix_pairs.csv").is_file()
    commitment = runner._read_commitment(
        work / "truth_commitment.json", runner.TRUTH_COMMITMENT_KEYS
    )
    assert commitment["truth_values_withheld_until_prediction_commitment"] is True
    assert commitment["truth_pack_row_count"] == 22 * len(protocol.forecast_days)
    assert commitment["truth_pack_byte_sha256"] == runner._sha256_path(
        sealed / "truth_pack.csv"
    )


def _fake_prediction_results(
    tasks: list[
        tuple[str, str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    ],
    config_path: Path,
    workers: int,
):
    del config_path, workers
    for partition, cluster_id, prefix_days, observed, forecast_days in tasks:
        prefix = pd.DataFrame(
            {
                "protocol_id": synthetic.FROZEN_PROTOCOL_ID,
                "partition": partition,
                "cluster_id": cluster_id,
                "prefix_day": prefix_days,
                "observed_retention_pct": observed,
            },
            columns=synthetic.PREFIX_COLUMNS,
        )
        prefix_hash = synthetic.canonical_prefix_content_sha256(prefix)
        prediction_rows = []
        for day in forecast_days:
            prediction_rows.append(
                {
                    "protocol_id": synthetic.FROZEN_PROTOCOL_ID,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "forecast_day": day,
                    "candidate_point_forecast_pct": 90.0,
                    "persistence_forecast_pct": 91.0,
                    "sqrt_time_forecast_pct": 90.5,
                    "bounded_power_forecast_pct": 90.0,
                    "structure_envelope_lower_pct": 89.0,
                    "structure_envelope_upper_pct": 91.0,
                    "canonical_prefix_content_sha256": prefix_hash,
                }
            )
        diagnostic_rows = []
        for model_id in synthetic.STRUCTURE_MEMBER_IDS:
            diagnostic_rows.append(
                {
                    "protocol_id": synthetic.FROZEN_PROTOCOL_ID,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "model_id": model_id,
                    "variant_id": "unit_test",
                    "fit_status": "succeeded",
                    "credible_variant": True,
                    "prefix_rmse_pp": 0.1,
                    "prefix_max_abs_residual_pp": 0.2,
                    "forecast_min_pct": 89.0,
                    "forecast_max_pct": 91.0,
                    "canonical_prefix_content_sha256": prefix_hash,
                }
            )
        yield (
            pd.DataFrame(prediction_rows, columns=synthetic.PREDICTION_COLUMNS),
            pd.DataFrame(
                diagnostic_rows, columns=synthetic.MEMBER_DIAGNOSTIC_COLUMNS
            ),
        )


def test_prediction_phase_has_no_truth_path_and_writes_exact_commitment(
    protocol: synthetic.ValidatedSyntheticProtocol,
    scratch_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = scratch_root / "label_free"
    sealed = scratch_root / "sealed"
    work.mkdir()
    sealed.mkdir()
    prefix_rows = []
    coordinate_rows = []
    for index, partition in enumerate(
        (*synthetic.PARTITION_NAMES, synthetic.MATCHED_PARTITION)
    ):
        cluster_id = f"opaque_{index}"
        for day in protocol.prefix_days:
            prefix_rows.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "prefix_day": day,
                    "observed_retention_pct": 100.0 - 0.001 * day,
                }
            )
        for day in protocol.forecast_days:
            coordinate_rows.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "forecast_day": day,
                }
            )
    prefix = pd.DataFrame(prefix_rows, columns=synthetic.PREFIX_COLUMNS)
    coordinates = pd.DataFrame(
        coordinate_rows, columns=synthetic.FORECAST_COORDINATE_COLUMNS
    )
    runner._write_csv(work / "prefix_pack.csv", prefix, synthetic.PREFIX_COLUMNS)
    runner._write_csv(
        work / "forecast_coordinates.csv",
        coordinates,
        synthetic.FORECAST_COORDINATE_COLUMNS,
    )
    runner._write_json(
        work / "truth_commitment.json",
        {
            "protocol_id": protocol.protocol_id,
            "config_sha256": protocol.config_sha256,
            "truth_pack_byte_sha256": "0" * 64,
            "truth_pack_row_count": 40,
            "created_utc": "2026-07-22T00:00:00Z",
            "truth_values_withheld_until_prediction_commitment": True,
        },
    )
    sealed_truth = sealed / "truth_pack.csv"
    sealed_truth.write_bytes(b"must not be opened")
    opened: list[Path] = []
    original_read_bytes = Path.read_bytes

    def audited_read_bytes(path: Path) -> bytes:
        opened.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(runner, "_iter_prediction_results", _fake_prediction_results)
    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)

    result = runner.prediction_phase(CONFIG_PATH, work, workers=1)

    assert result["cluster_count"] == 5
    assert sealed_truth.resolve() not in opened
    commitment = runner._read_commitment(
        work / "prediction_commitment.json",
        runner.PREDICTION_COMMITMENT_KEYS,
    )
    assert commitment["truth_pack_opened_before_commitment"] is False
    assert commitment["row_counts"] == {
        "prefix_pack": 60,
        "forecast_coordinates": 40,
        "prediction_bundle": 40,
        "decision_bundle": 5,
        "member_fit_diagnostics": 30,
    }
    for filename, key in (
        ("prefix_pack.csv", "prefix_pack_byte_sha256"),
        ("forecast_coordinates.csv", "forecast_coordinates_byte_sha256"),
        ("prediction_bundle.csv", "prediction_bundle_byte_sha256"),
        ("decision_bundle.csv", "decision_bundle_byte_sha256"),
        (
            "member_fit_diagnostics.csv",
            "member_fit_diagnostics_byte_sha256",
        ),
    ):
        assert commitment[key] == hashlib.sha256(
            (work / filename).read_bytes()
        ).hexdigest()
