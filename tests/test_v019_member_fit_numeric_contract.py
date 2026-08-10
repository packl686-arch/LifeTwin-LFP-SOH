from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_csv_bytes,
    canonicalize_frame,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
)
from lifetwin.experiments.calendar_long_horizon_v019_contract import (
    load_v024_contract_view,
)
from lifetwin.experiments import calendar_long_horizon_v018_partition as v018_partition
from lifetwin.experiments import calendar_long_horizon_v019_partition as v019_partition
from lifetwin.experiments.calendar_long_horizon_v019_protocol import V024_PROTOCOL_ID
from lifetwin.experiments.calendar_long_horizon_v019_numeric_contract import (
    DIAGNOSTIC_COLUMNS,
    FORECAST_COLUMNS,
    V024MemberFitNumericContractError,
    validate_decision_bundle_numeric_contract,
    validate_member_fit_numeric_contract,
)


def _member_frames(
    *,
    partitions: tuple[str, ...] = ("center_development",),
    cluster_ids: tuple[str, ...] = ("fixture-cluster-0000",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(partitions) != len(cluster_ids):
        raise ValueError("Fixture partitions and cluster IDs differ")
    records: list[dict[str, object]] = []
    forecasts: list[dict[str, object]] = []
    for partition, cluster_id in zip(partitions, cluster_ids, strict=True):
        content_hash = (cluster_id.encode("ascii").hex() + "0" * 64)[:64]
        for variant_index, (model_id, variant_id) in enumerate(FROZEN_VARIANT_KEYS):
            succeeded = variant_index != 0
            records.append(
                {
                    "protocol_id": V024_PROTOCOL_ID,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "model_id": model_id,
                    "variant_id": variant_id,
                    "parameters_json": '{"p":1.0}' if succeeded else "{}",
                    "fit_status": "succeeded" if succeeded else "failed",
                    "credible_variant": succeeded,
                    "prefix_rmse_pp": 0.2 if succeeded else np.nan,
                    "prefix_max_abs_residual_pp": 0.3 if succeeded else np.nan,
                    "parameter_boundary_hit_fraction": 0.0 if succeeded else np.nan,
                    "canonical_prefix_content_sha256": content_hash,
                }
            )
            for day in FORECAST_DAYS:
                forecasts.append(
                    {
                        "protocol_id": V024_PROTOCOL_ID,
                        "partition": partition,
                        "cluster_id": cluster_id,
                        "model_id": model_id,
                        "variant_id": variant_id,
                        "forecast_day": day,
                        "raw_forecast_retention_pct": 90.0 if succeeded else np.nan,
                        "canonical_prefix_content_sha256": content_hash,
                    }
                )
    return (
        pd.DataFrame(records, columns=DIAGNOSTIC_COLUMNS),
        pd.DataFrame(forecasts, columns=FORECAST_COLUMNS),
    )


def test_blanket_finite_gate_conflicts_with_registered_failed_variant_mask() -> None:
    diagnostics, forecasts = _member_frames()
    with pytest.raises(
        v018_partition.V023PartitionContractError,
        match="member_fit_diagnostics.csv contains a nonfinite numeric value",
    ):
        v018_partition._require_finite_numeric(
            diagnostics,
            filename="member_fit_diagnostics.csv",
        )
    validate_member_fit_numeric_contract(diagnostics, forecasts)


def test_candidate_contract_survives_real_formal_canonicalization() -> None:
    cluster_ids = tuple(f"center-development-{index:04d}" for index in range(600))
    diagnostics, forecasts = _member_frames(
        partitions=("center_development",) * len(cluster_ids),
        cluster_ids=cluster_ids,
    )
    contract = load_v024_contract_view().artifacts
    diagnostic_schema = replace(
        contract.csv_schema("member_fit_diagnostics.csv"),
        required_rows=len(diagnostics),
        expected_partition="center_development",
    )
    forecast_schema = replace(
        contract.csv_schema("member_forecast_bundle.csv"),
        required_rows=len(forecasts),
        expected_partition="center_development",
    )
    canonical_diagnostics = canonicalize_frame(
        diagnostics,
        diagnostic_schema,
        contract,
        formal=True,
    )
    canonical_forecasts = canonicalize_frame(
        forecasts,
        forecast_schema,
        contract,
        formal=True,
    )
    validate_member_fit_numeric_contract(canonical_diagnostics, canonical_forecasts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("positive_infinity", "contains infinity"),
        ("negative_infinity", "contains infinity"),
        ("failed_metric_finite", "structural NaN"),
        ("succeeded_metric_nan", "succeeded variant has nonfinite"),
        ("failed_forecast_finite", "structural NaN raw forecasts"),
        ("succeeded_forecast_nan", "succeeded variant has a nonfinite raw forecast"),
        ("numeric_string", "numeric dtype"),
        ("invalid_status", "fit_status must be"),
        ("credible_failed", "failed variant was declared credible"),
        ("credible_objective_drift", "credible_variant differs"),
        ("boundary_out_of_range", r"outside \[0, 1\]"),
        ("duplicate_diagnostic_key", "duplicate variant keys"),
        ("missing_diagnostic_key", "exact frozen 86-variant"),
        ("missing_forecast_key", "exact eight-row forecast grid"),
        ("duplicate_forecast_key", "duplicate forecast keys"),
        ("wrong_forecast_day", "invalid forecast day"),
        ("content_hash_drift", "content hashes differ"),
        ("failed_parameters_retained", "retained fitted parameters"),
        ("nonfinite_parameter", "nonfinite or unsupported"),
        ("entire_metric_column_degenerate", "entirely degenerate"),
    ),
)
def test_member_fit_mutation_matrix_rejects_exact_violation(
    mutation: str,
    message: str,
) -> None:
    diagnostics, forecasts = _member_frames()
    failed = diagnostics["fit_status"].eq("failed")
    succeeded = diagnostics["fit_status"].eq("succeeded")
    failed_forecast = forecasts["variant_id"].eq(FROZEN_VARIANT_KEYS[0][1])
    succeeded_forecast = ~failed_forecast

    if mutation == "positive_infinity":
        diagnostics.loc[succeeded.idxmax(), "prefix_rmse_pp"] = np.inf
    elif mutation == "negative_infinity":
        diagnostics.loc[succeeded.idxmax(), "prefix_rmse_pp"] = -np.inf
    elif mutation == "failed_metric_finite":
        diagnostics.loc[failed, "prefix_rmse_pp"] = 0.0
    elif mutation == "succeeded_metric_nan":
        diagnostics.loc[succeeded.idxmax(), "prefix_rmse_pp"] = np.nan
    elif mutation == "failed_forecast_finite":
        forecasts.loc[failed_forecast, "raw_forecast_retention_pct"] = 90.0
    elif mutation == "succeeded_forecast_nan":
        forecasts.loc[succeeded_forecast.idxmax(), "raw_forecast_retention_pct"] = (
            np.nan
        )
    elif mutation == "numeric_string":
        diagnostics["prefix_rmse_pp"] = diagnostics["prefix_rmse_pp"].astype(object)
        diagnostics.loc[succeeded.idxmax(), "prefix_rmse_pp"] = "0.2"
    elif mutation == "invalid_status":
        diagnostics.loc[failed, "fit_status"] = "not-fitted"
    elif mutation == "credible_failed":
        diagnostics.loc[failed, "credible_variant"] = True
    elif mutation == "credible_objective_drift":
        diagnostics.loc[succeeded.idxmax(), "credible_variant"] = False
    elif mutation == "boundary_out_of_range":
        diagnostics.loc[succeeded.idxmax(), "parameter_boundary_hit_fraction"] = 1.1
    elif mutation == "duplicate_diagnostic_key":
        diagnostics.iloc[1] = diagnostics.iloc[0]
    elif mutation == "missing_diagnostic_key":
        diagnostics = diagnostics.iloc[:-1].reset_index(drop=True)
    elif mutation == "missing_forecast_key":
        forecasts = forecasts.iloc[:-1].reset_index(drop=True)
    elif mutation == "duplicate_forecast_key":
        forecasts.iloc[1] = forecasts.iloc[0]
    elif mutation == "wrong_forecast_day":
        forecasts.loc[0, "forecast_day"] = 999.0
    elif mutation == "content_hash_drift":
        forecasts.loc[0, "canonical_prefix_content_sha256"] = "f" * 64
    elif mutation == "failed_parameters_retained":
        diagnostics.loc[failed, "parameters_json"] = '{"p":1.0}'
    elif mutation == "nonfinite_parameter":
        diagnostics.loc[succeeded.idxmax(), "parameters_json"] = '{"p":NaN}'
    elif mutation == "entire_metric_column_degenerate":
        diagnostics["prefix_rmse_pp"] = np.nan
    else:  # pragma: no cover - the parameter registry is closed above
        raise AssertionError(mutation)

    with pytest.raises(V024MemberFitNumericContractError, match=message):
        validate_member_fit_numeric_contract(diagnostics, forecasts)


def test_preserved_decision_contract_rejects_rank_and_issuance_drift() -> None:
    feature = pd.DataFrame(
        {
            "partition": ["test"],
            "cluster_id": ["fixture-cluster-0000"],
            "hard_eligible": [True],
            "all_features_finite": [True],
        }
    )
    risk = pd.DataFrame(
        {
            "partition": ["test", "test"],
            "cluster_id": ["fixture-cluster-0000"] * 2,
            "score_id": ["prefix_only", "visible_stress"],
            "raw_risk_score": [0.1, 0.2],
        }
    )
    decision = pd.DataFrame(
        {
            "partition": ["test", "test"],
            "cluster_id": ["fixture-cluster-0000"] * 2,
            "arm": ["prefix_only", "visible_stress"],
            "raw_risk_score": [0.1, 0.2],
            "hard_eligible": [True, True],
            "issuance_rank": [1.0, 1.0],
            "issued": [True, True],
        }
    )
    validate_decision_bundle_numeric_contract(
        decision,
        feature,
        risk,
        primary_issue_counts={"test": 1},
    )
    drifted = decision.copy()
    drifted.loc[0, "issued"] = False
    with pytest.raises(ValueError, match="Issued flags do not match"):
        validate_decision_bundle_numeric_contract(
            drifted,
            feature,
            risk,
            primary_issue_counts={"test": 1},
        )


def _exact_cardinality_frames() -> dict[str, pd.DataFrame]:
    partition_names: list[str] = []
    cluster_ids: list[str] = []
    for partition, counts in v019_partition.PARTITION_COUNTS.items():
        for index in range(counts["clusters"]):
            partition_names.append(partition)
            cluster_ids.append(f"{partition}-{index:04d}")
    cluster_count = len(cluster_ids)
    variant_count = len(FROZEN_VARIANT_KEYS)
    diagnostic_count = cluster_count * variant_count
    diagnostic_cluster_index = np.repeat(np.arange(cluster_count), variant_count)
    variant_index = np.tile(np.arange(variant_count), cluster_count)
    succeeded = variant_index != 0
    model_ids = np.asarray([key[0] for key in FROZEN_VARIANT_KEYS], dtype=object)
    variant_ids = np.asarray([key[1] for key in FROZEN_VARIANT_KEYS], dtype=object)
    content_hashes = np.asarray(
        [f"{index:064x}" for index in range(cluster_count)], dtype=object
    )
    diagnostic_hashes = content_hashes[diagnostic_cluster_index]
    protocol_category = lambda count: pd.Categorical.from_codes(  # noqa: E731
        np.zeros(count, dtype=np.int8), categories=[V024_PROTOCOL_ID]
    )
    diagnostics = pd.DataFrame(
        {
            "protocol_id": protocol_category(diagnostic_count),
            "partition": pd.Categorical(
                np.asarray(partition_names, dtype=object)[diagnostic_cluster_index]
            ),
            "cluster_id": pd.Categorical(
                np.asarray(cluster_ids, dtype=object)[diagnostic_cluster_index]
            ),
            "model_id": pd.Categorical(model_ids[variant_index]),
            "variant_id": pd.Categorical(variant_ids[variant_index]),
            "parameters_json": pd.Categorical(np.where(succeeded, '{"p":1.0}', "{}")),
            "fit_status": pd.Categorical(np.where(succeeded, "succeeded", "failed")),
            "credible_variant": succeeded,
            "prefix_rmse_pp": np.where(succeeded, 0.2, np.nan),
            "prefix_max_abs_residual_pp": np.where(succeeded, 0.3, np.nan),
            "parameter_boundary_hit_fraction": np.where(succeeded, 0.0, np.nan),
            "canonical_prefix_content_sha256": pd.Categorical(diagnostic_hashes),
        },
        columns=DIAGNOSTIC_COLUMNS,
    )
    forecast_index = np.repeat(np.arange(diagnostic_count), len(FORECAST_DAYS))
    forecast_succeeded = succeeded[forecast_index]
    forecasts = pd.DataFrame(
        {
            "protocol_id": protocol_category(len(forecast_index)),
            "partition": pd.Categorical(
                np.asarray(partition_names, dtype=object)[
                    diagnostic_cluster_index[forecast_index]
                ]
            ),
            "cluster_id": pd.Categorical(
                np.asarray(cluster_ids, dtype=object)[
                    diagnostic_cluster_index[forecast_index]
                ]
            ),
            "model_id": pd.Categorical(model_ids[variant_index[forecast_index]]),
            "variant_id": pd.Categorical(variant_ids[variant_index[forecast_index]]),
            "forecast_day": np.tile(
                np.asarray(FORECAST_DAYS, dtype=float), diagnostic_count
            ),
            "raw_forecast_retention_pct": np.where(forecast_succeeded, 90.0, np.nan),
            "canonical_prefix_content_sha256": pd.Categorical(
                diagnostic_hashes[forecast_index]
            ),
        },
        columns=FORECAST_COLUMNS,
    )
    partition_array = np.asarray(partition_names, dtype=object)
    cluster_array = np.asarray(cluster_ids, dtype=object)
    prefix_cluster_index = np.repeat(np.arange(cluster_count), len(PREFIX_DAYS))
    coordinate_cluster_index = np.repeat(np.arange(cluster_count), len(FORECAST_DAYS))
    prefix = pd.DataFrame(
        {
            "protocol_id": protocol_category(len(prefix_cluster_index)),
            "partition": pd.Categorical(partition_array[prefix_cluster_index]),
            "cluster_id": pd.Categorical(cluster_array[prefix_cluster_index]),
            "prefix_day": np.tile(np.asarray(PREFIX_DAYS, dtype=float), cluster_count),
            "observed_retention_pct": np.full(len(prefix_cluster_index), 99.0),
        }
    )
    coordinates = pd.DataFrame(
        {
            "protocol_id": protocol_category(len(coordinate_cluster_index)),
            "partition": pd.Categorical(partition_array[coordinate_cluster_index]),
            "cluster_id": pd.Categorical(cluster_array[coordinate_cluster_index]),
            "forecast_day": np.tile(
                np.asarray(FORECAST_DAYS, dtype=float), cluster_count
            ),
        }
    )
    operating_values: dict[str, object] = {
        "protocol_id": protocol_category(cluster_count),
        "partition": pd.Categorical(partition_array),
        "cluster_id": pd.Categorical(cluster_array),
    }
    contract = load_v024_contract_view().artifacts
    operating_schema = contract.csv_schema("operating_pack.csv")
    for column in operating_schema.columns[3:]:
        if column.endswith("temperature_c"):
            value = 25.0
        elif column.endswith("soc_fraction") or column.endswith("dod_fraction"):
            value = 0.5
        elif column.endswith("efc_per_year"):
            value = 250.0
        else:
            value = 0.0
        operating_values[column] = np.full(cluster_count, value)
    operating = pd.DataFrame(operating_values, columns=operating_schema.columns)
    frames = {
        "prefix_pack.csv": prefix,
        "forecast_coordinates.csv": coordinates,
        "operating_pack.csv": operating,
        "member_fit_diagnostics.csv": diagnostics,
        "member_forecast_bundle.csv": forecasts,
    }
    assert {name: len(frame) for name, frame in frames.items()} == dict(
        v019_partition.WHOLE_COUNTS
    )
    expected_cluster_keys = set(zip(partition_names, cluster_ids, strict=True))
    for filename, frame in frames.items():
        schema = contract.csv_schema(filename)
        assert tuple(frame.columns) == schema.columns
        assert not frame.duplicated(list(schema.key)).any()
        assert (
            set(
                frame.loc[:, ["partition", "cluster_id"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            == expected_cluster_keys
        )
        if filename not in {
            "member_fit_diagnostics.csv",
            "member_forecast_bundle.csv",
        }:
            numeric = frame.select_dtypes(include=[np.number])
            assert np.isfinite(numeric.to_numpy(float)).all()
    assert set(prefix["prefix_day"].unique()) == set(PREFIX_DAYS)
    assert set(coordinates["forecast_day"].unique()) == set(FORECAST_DAYS)
    assert set(forecasts["forecast_day"].unique()) == set(FORECAST_DAYS)
    for partition, counts in v019_partition.PARTITION_COUNTS.items():
        assert {
            filename: int(frame["partition"].eq(partition).sum())
            for filename, frame in frames.items()
        } == {filename: counts[filename] for filename in v019_partition.INPUT_FILENAMES}
    return frames


def test_exact_cardinality_physical_whole_and_partition_roundtrip(
    tmp_path: Path,
) -> None:
    frames = _exact_cardinality_frames()
    contract = load_v024_contract_view().artifacts
    for filename in v019_partition.INPUT_FILENAMES:
        raw = canonical_csv_bytes(
            frames[filename],
            contract.csv_schema(filename),
            contract,
            formal=True,
        )
        (tmp_path / filename).write_bytes(raw)
    whole = v019_partition.validate_whole_bundle_from_root(tmp_path, contract)
    assert type(whole) is v019_partition.WholeBundleValidated
    assert all(size > 0 for size in whole.source_sizes.values())
    for partition, expected in v019_partition.PARTITION_COUNTS.items():
        view = v019_partition.derive_partition_view(
            whole,
            partition=partition,
            contract=contract,
        )
        consumed = v019_partition.consume_partition_frames(view, contract=contract)
        assert {name: len(frame) for name, frame in consumed.items()} == {
            name: expected[name] for name in v019_partition.INPUT_FILENAMES
        }
