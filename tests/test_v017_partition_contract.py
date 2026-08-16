from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_fit import FROZEN_VARIANT_KEYS
from lifetwin.experiments.calendar_long_horizon_v015_io import canonicalize_frame
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import make_probe_state
from lifetwin.experiments.calendar_long_horizon_v017_contract import (
    load_v022_contract_view,
)
from lifetwin.experiments import calendar_long_horizon_v017_partition as partition
from lifetwin.experiments import calendar_long_horizon_v017_runner as runner
from lifetwin.experiments.calendar_long_horizon_v017_protocol import (
    V022_ONLY_ATTEMPT_ID,
    V022_PROTOCOL_ID,
)


@pytest.fixture(scope="module")
def full_prefix() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for partition_name, counts in partition.PARTITION_COUNTS.items():
        for cluster_index in range(counts["clusters"]):
            cluster_id = f"{partition_name}-{cluster_index:04d}"
            for day in PREFIX_DAYS:
                records.append(
                    {
                        "protocol_id": V022_PROTOCOL_ID,
                        "partition": partition_name,
                        "cluster_id": cluster_id,
                        "prefix_day": day,
                        "observed_retention_pct": 100.0
                        - 0.8 * math.sqrt(day / 365.25),
                    }
                )
    contract = load_v022_contract_view().artifacts
    return canonicalize_frame(
        pd.DataFrame(records, columns=contract.csv_schema("prefix_pack.csv").columns),
        contract.csv_schema("prefix_pack.csv"),
        contract,
        formal=True,
    )


@pytest.fixture(scope="module")
def center_frames() -> dict[str, pd.DataFrame]:
    contract = load_v022_contract_view().artifacts
    name = "center_development"
    clusters = [f"{name}-{index:04d}" for index in range(600)]
    prefix = pd.DataFrame(
        {
            "protocol_id": V022_PROTOCOL_ID,
            "partition": name,
            "cluster_id": np.repeat(clusters, len(PREFIX_DAYS)),
            "prefix_day": np.tile(PREFIX_DAYS, len(clusters)),
            "observed_retention_pct": np.tile(
                [100.0 - 0.8 * math.sqrt(day / 365.25) for day in PREFIX_DAYS],
                len(clusters),
            ),
        },
        columns=contract.csv_schema("prefix_pack.csv").columns,
    )
    coordinates = pd.DataFrame(
        {
            "protocol_id": V022_PROTOCOL_ID,
            "partition": name,
            "cluster_id": np.repeat(clusters, len(FORECAST_DAYS)),
            "forecast_day": np.tile(FORECAST_DAYS, len(clusters)),
        },
        columns=contract.csv_schema("forecast_coordinates.csv").columns,
    )
    operating = pd.DataFrame(
        {
            "protocol_id": [V022_PROTOCOL_ID] * len(clusters),
            "partition": [name] * len(clusters),
            "cluster_id": clusters,
            **{
                field: [value] * len(clusters)
                for field, value in zip(
                    REAL_OPERATING_FIELDS,
                    (25.0, 0.5, 0.55, 250.0, 31.0, 0.6, 0.65, 300.0),
                    strict=True,
                )
            },
            **{
                field: [float(index)] * len(clusters)
                for index, field in enumerate(PLACEBO_FIELDS)
            },
        },
        columns=contract.csv_schema("operating_pack.csv").columns,
    )

    diagnostics = pd.DataFrame(
        [
            {
                "protocol_id": V022_PROTOCOL_ID,
                "partition": name,
                "cluster_id": cluster_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "parameters_json": "{}",
                "fit_status": "success",
                "credible_variant": True,
                "prefix_rmse_pp": 0.1,
                "prefix_max_abs_residual_pp": 0.2,
                "parameter_boundary_hit_fraction": 0.0,
                "canonical_prefix_content_sha256": "0" * 64,
            }
            for cluster_id in clusters
            for model_id, variant_id in FROZEN_VARIANT_KEYS
        ],
        columns=contract.csv_schema("member_fit_diagnostics.csv").columns,
    )
    forecasts = pd.DataFrame(
        [
            {
                "protocol_id": V022_PROTOCOL_ID,
                "partition": name,
                "cluster_id": cluster_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "forecast_day": day,
                "raw_forecast_retention_pct": 95.0,
                "canonical_prefix_content_sha256": "0" * 64,
            }
            for cluster_id in clusters
            for model_id, variant_id in FROZEN_VARIANT_KEYS
            for day in FORECAST_DAYS
        ],
        columns=contract.csv_schema("member_forecast_bundle.csv").columns,
    )
    unsorted = {
        "prefix_pack.csv": prefix,
        "forecast_coordinates.csv": coordinates,
        "operating_pack.csv": operating,
        "member_fit_diagnostics.csv": diagnostics,
        "member_forecast_bundle.csv": forecasts,
    }
    return {
        filename: frame.sort_values(
            list(contract.csv_schema(filename).key),
            kind="stable",
        ).reset_index(drop=True)
        for filename, frame in unsorted.items()
    }


def test_full_71400_prefix_contract_then_exact_7200_center_slice(
    full_prefix: pd.DataFrame,
) -> None:
    contract = load_v022_contract_view().artifacts
    full = full_prefix
    assert len(full) == 71_400
    center = full.loc[full["partition"].eq("center_development")].reset_index(
        drop=True
    )
    canonical, digest = partition._canonical_partition(
        center,
        filename="prefix_pack.csv",
        partition="center_development",
        required_rows=7_200,
        contract=contract,
    )
    assert len(canonical) == 7_200
    assert len(digest) == 64


def test_partition_contract_rejects_wrong_count_order_and_nonfinite(
    full_prefix: pd.DataFrame,
) -> None:
    contract = load_v022_contract_view().artifacts
    center = full_prefix.loc[
        lambda frame: frame["partition"].eq("center_development")
    ].reset_index(drop=True)
    kwargs = {
        "filename": "prefix_pack.csv",
        "partition": "center_development",
        "required_rows": 7_200,
        "contract": contract,
    }
    with pytest.raises(partition.V022PartitionContractError):
        partition._canonical_partition(center.iloc[:-1], **kwargs)
    wrong_order = center.copy()
    wrong_order.iloc[[0, 1]] = wrong_order.iloc[[1, 0]].to_numpy()
    with pytest.raises(partition.V022PartitionContractError, match="canonical order"):
        partition._canonical_partition(wrong_order, **kwargs)
    nonfinite = center.copy()
    nonfinite.loc[0, "observed_retention_pct"] = np.inf
    with pytest.raises(partition.V022PartitionContractError):
        partition._canonical_partition(nonfinite, **kwargs)


def test_real_apply_partition_uses_exact_five_table_capability(
    monkeypatch: pytest.MonkeyPatch,
    center_frames: dict[str, pd.DataFrame],
) -> None:
    contract = load_v022_contract_view().artifacts
    frames = center_frames
    whole = partition.WholeBundleValidated(
        issuer=partition._ISSUER,
        contract_hash=contract.config_byte_sha256,
        frames=frames,
        source_hashes={name: "1" * 64 for name in partition.INPUT_FILENAMES},
        source_sizes={name: 1 for name in partition.INPUT_FILENAMES},
    )
    observed: dict[str, object] = {}

    def spy(view, *, state, contract):
        observed["view_type"] = type(view)
        observed["partition"] = view.partition
        return "pipeline-sentinel"

    monkeypatch.setattr(
        runner,
        "recompute_validated_partition_with_state_v022",
        spy,
    )
    result, selected = runner._apply_partition(
        whole,
        partition="center_development",
        state=make_probe_state(1.0),
        view=load_v022_contract_view(),
    )
    assert result == "pipeline-sentinel"
    assert observed == {
        "view_type": partition.ValidatedPartitionView,
        "partition": "center_development",
    }
    assert {name: len(frame) for name, frame in selected.items()} == {
        name: partition.PARTITION_COUNTS["center_development"][name]
        for name in partition.INPUT_FILENAMES
    }
    for name, frame in selected.items():
        assert tuple(frame.columns) == contract.csv_schema(name).columns
        numeric = frame.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy(float)).all()


def test_capability_constructor_and_mutation_guards(
    center_frames: dict[str, pd.DataFrame],
) -> None:
    contract = load_v022_contract_view().artifacts
    with pytest.raises(partition.V022PartitionCapabilityError):
        partition.WholeBundleValidated(
            issuer=object(),
            contract_hash=contract.config_byte_sha256,
            frames={},
            source_hashes={},
            source_sizes={},
        )
    frames = center_frames
    whole = partition.WholeBundleValidated(
        issuer=partition._ISSUER,
        contract_hash=contract.config_byte_sha256,
        frames=frames,
        source_hashes={name: "1" * 64 for name in partition.INPUT_FILENAMES},
        source_sizes={name: 1 for name in partition.INPUT_FILENAMES},
    )
    view = partition.derive_partition_view(
        whole,
        partition="center_development",
        contract=contract,
    )
    view._frames["operating_pack.csv"].loc[0, "past_mean_temperature_c"] += 0.5
    with pytest.raises(partition.V022PartitionCapabilityError, match="mutated"):
        partition.consume_partition_frames(view, contract=contract)


def test_runner_binds_one_attempt_and_validates_partition_before_truth() -> None:
    tmp_path = (
        Path.cwd()
        / "artifacts"
        / "operator-evidence"
        / V022_ONLY_ATTEMPT_ID
        / "nonexistent-static-test-root"
    ).resolve()
    assert not tmp_path.exists()
    paths = runner.V022RunPaths.resolve(
        repo_root=tmp_path,
        label_free_root=tmp_path / "label",
        sealed_truth_root=tmp_path / "truth",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
    )
    with pytest.raises(runner.V022RunnerError, match="must equal"):
        runner.initialize_formal_attempt(paths=paths, attempt_id="v022-formal-a2")
    assert not paths.label_free_root.exists()
    assert not paths.sealed_truth_root.exists()
    assert not paths.score_root.exists()
    assert not paths.termination_root.exists()

    center_source = inspect.getsource(runner._fit_center_stage)
    risk_source = inspect.getsource(runner._fit_risk_stage)
    assert center_source.index("_apply_partition(") < center_source.index(
        "open_truth_for_phase("
    )
    assert risk_source.index("_apply_partition(") < risk_source.index(
        "open_truth_for_phase("
    )
    runner_source = inspect.getsource(runner)
    assert "formal=False" not in runner_source
    assert V022_ONLY_ATTEMPT_ID == "v022-formal-20260809-a1"
