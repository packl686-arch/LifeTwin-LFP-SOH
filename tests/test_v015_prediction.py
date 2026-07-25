from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v015_prediction as prediction,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_csv_bytes,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    StandardizerState,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
    FrozenLabelFreeState,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FORECAST_COORDINATE_COLUMNS,
    FROZEN_PROTOCOL_ID,
    OPERATING_COLUMNS,
    PLACEBO_FIELDS,
    PREFIX_COLUMNS,
    REAL_OPERATING_FIELDS,
)


FORECAST_DAYS = (
    1095.75,
    1461.0,
    1826.25,
    2556.75,
    3652.5,
    5478.75,
    7305.0,
    9131.25,
)


def _risk_state(feature_names: tuple[str, ...]) -> LogisticRiskState:
    dimension = len(feature_names)
    return LogisticRiskState(
        feature_names=feature_names,
        standardizer=StandardizerState(
            mean=(0.0,) * dimension,
            scale=(1.0,) * dimension,
            zero_variance=(False,) * dimension,
        ),
        intercept=-4.0,
        coefficients=(0.0,) * dimension,
    )


def _state() -> FrozenLabelFreeState:
    return FrozenLabelFreeState(
        center_beta=0.5,
        prefix_only_risk=_risk_state(PREFIX_FEATURE_NAMES),
        visible_stress_risk=_risk_state(VISIBLE_STRESS_FEATURE_NAMES),
        placebo_risk=_risk_state(PLACEBO_FEATURE_NAMES),
        arm_a_plus_s_plan_risk=_risk_state(ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
        strongest_single_feature_name=PREFIX_FEATURE_NAMES[0],
        strongest_single_feature_orientation=1,
        prefix_only_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        visible_stress_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        conformal=ConformalExpansionState(
            coverage=0.90,
            calibration_count=900,
            order_statistic_index=811,
            expansion_pp=1.0,
        ),
    )


def _fixture_tables(
    cluster_ids: tuple[str, ...] = ("fixture-z", "fixture-a"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix_records: list[dict[str, object]] = []
    coordinate_records: list[dict[str, object]] = []
    operating_records: list[dict[str, object]] = []
    for cluster_index, cluster_id in enumerate(cluster_ids):
        for day in reversed(PREFIX_DAYS):
            years = day / 365.25
            prefix_records.append(
                {
                    "protocol_id": FROZEN_PROTOCOL_ID,
                    "partition": "calibration",
                    "cluster_id": cluster_id,
                    "prefix_day": day,
                    "observed_retention_pct": (
                        100.0 - (0.72 + 0.03 * cluster_index) * math.sqrt(years)
                    ),
                }
            )
        for day in reversed(FORECAST_DAYS):
            coordinate_records.append(
                {
                    "protocol_id": FROZEN_PROTOCOL_ID,
                    "partition": "calibration",
                    "cluster_id": cluster_id,
                    "forecast_day": day,
                }
            )
        real = (
            24.0 + cluster_index,
            0.48,
            0.52,
            240.0,
            29.0 + cluster_index,
            0.58,
            0.62,
            310.0,
        )
        operating_records.append(
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                **dict(zip(REAL_OPERATING_FIELDS, real, strict=True)),
                **{
                    field: -0.7 + 0.1 * index + 0.01 * cluster_index
                    for index, field in enumerate(PLACEBO_FIELDS)
                },
            }
        )
    return (
        pd.DataFrame(prefix_records, columns=PREFIX_COLUMNS),
        pd.DataFrame(
            coordinate_records,
            columns=FORECAST_COORDINATE_COLUMNS,
        ),
        pd.DataFrame(
            list(reversed(operating_records)),
            columns=OPERATING_COLUMNS,
        ),
    )


def _fitted_bytes(
    result: prediction.V015FitResult,
) -> dict[str, bytes]:
    contract = load_artifact_contract()
    frames = {
        "member_fit_diagnostics.csv": result.member_fit_diagnostics,
        "member_forecast_bundle.csv": result.member_forecast_bundle,
    }
    return {
        filename: canonical_csv_bytes(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=False,
        )
        for filename, frame in frames.items()
    }


def _downstream_bytes(
    result: prediction.LabelFreePipelineResult,
) -> dict[str, bytes]:
    contract = load_artifact_contract()
    frames = {
        "prediction_bundle.csv": result.prediction_bundle,
        "risk_bundle.csv": result.primary_risk_bundle,
        "decision_bundle.csv": result.decision_bundle,
    }
    return {
        filename: canonical_csv_bytes(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=False,
        )
        for filename, frame in frames.items()
    }


@dataclass(frozen=True)
class _WorkerInvarianceResults:
    one_fit: prediction.V015FitResult
    six_fit: prediction.V015FitResult
    one_downstream: prediction.LabelFreePipelineResult
    six_downstream: prediction.LabelFreePipelineResult


@pytest.fixture(scope="module")
def worker_invariance_results() -> _WorkerInvarianceResults:
    prefix, coordinates, operating = _fixture_tables()
    one_fit = prediction.fit_structure_library_parallel(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        worker_count=1,
    )
    six_fit = prediction.fit_structure_library_parallel(
        prefix_pack=prefix.sample(frac=1.0, random_state=11),
        forecast_coordinates=coordinates.sample(frac=1.0, random_state=12),
        worker_count=6,
    )
    one_downstream = prediction.run_pipeline_from_fitted(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating,
        member_fit_diagnostics=one_fit.member_fit_diagnostics,
        member_forecast_bundle=one_fit.member_forecast_bundle,
        state=_state(),
    )
    six_downstream = prediction.run_pipeline_from_fitted(
        prefix_pack=prefix.sample(frac=1.0, random_state=21),
        forecast_coordinates=coordinates.sample(frac=1.0, random_state=22),
        operating_pack=operating.sample(frac=1.0, random_state=23),
        member_fit_diagnostics=six_fit.member_fit_diagnostics.sample(
            frac=1.0, random_state=24
        ),
        member_forecast_bundle=six_fit.member_forecast_bundle.sample(
            frac=1.0, random_state=25
        ),
        state=_state(),
    )
    return _WorkerInvarianceResults(
        one_fit=one_fit,
        six_fit=six_fit,
        one_downstream=one_downstream,
        six_downstream=six_downstream,
    )


def test_one_worker_and_six_workers_are_canonical_byte_identical(
    worker_invariance_results: _WorkerInvarianceResults,
) -> None:
    assert _fitted_bytes(worker_invariance_results.one_fit) == _fitted_bytes(
        worker_invariance_results.six_fit
    )
    assert _downstream_bytes(
        worker_invariance_results.one_downstream
    ) == _downstream_bytes(worker_invariance_results.six_downstream)


def test_outputs_are_canonically_sorted_after_parallel_collection(
    worker_invariance_results: _WorkerInvarianceResults,
) -> None:
    result = worker_invariance_results.six_downstream
    for frame, key in (
        (
            worker_invariance_results.six_fit.member_fit_diagnostics,
            ["partition", "cluster_id", "model_id", "variant_id"],
        ),
        (
            worker_invariance_results.six_fit.member_forecast_bundle,
            [
                "partition",
                "cluster_id",
                "model_id",
                "variant_id",
                "forecast_day",
            ],
        ),
        (
            result.prediction_bundle,
            ["partition", "cluster_id", "forecast_day"],
        ),
        (
            result.primary_risk_bundle,
            ["partition", "cluster_id", "score_id"],
        ),
        (
            result.decision_bundle,
            ["partition", "cluster_id", "arm"],
        ),
    ):
        expected = frame.sort_values(key, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(frame, expected)


def test_public_entry_points_have_no_hidden_information_channel() -> None:
    fit_parameters = tuple(
        inspect.signature(prediction.fit_structure_library_parallel).parameters
    )
    assert fit_parameters == (
        "prefix_pack",
        "forecast_coordinates",
        "worker_count",
    )
    formal_parameters = tuple(
        inspect.signature(prediction.fit_structure_library_formal).parameters
    )
    assert formal_parameters == (
        "prefix_pack",
        "forecast_coordinates",
    )
    apply_parameters = tuple(
        inspect.signature(prediction.run_pipeline_from_fitted).parameters
    )
    assert apply_parameters == (
        "prefix_pack",
        "forecast_coordinates",
        "operating_pack",
        "member_fit_diagnostics",
        "member_forecast_bundle",
        "state",
    )
    combined_parameters = tuple(
        inspect.signature(prediction.run_nonformal_label_free_prediction).parameters
    )
    assert combined_parameters == (
        "prefix_pack",
        "forecast_coordinates",
        "operating_pack",
        "state",
        "worker_count",
    )
    assert not hasattr(prediction, "run_formal_label_free_prediction")
    forbidden = ("truth", "path", "family", "pair", "side", "score")
    assert not any(
        token in parameter.lower()
        for parameter in (
            *fit_parameters,
            *formal_parameters,
            *apply_parameters,
            *combined_parameters,
        )
        for token in forbidden
    )


def test_formal_wrapper_hard_locks_six_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        prediction,
        "fit_structure_library_parallel",
        fake_run,
    )
    monkeypatch.setattr(
        prediction,
        "_canonical_input",
        lambda frame, **_: frame,
    )
    frame = pd.DataFrame()
    assert (
        prediction.fit_structure_library_formal(
            prefix_pack=frame,
            forecast_coordinates=frame,
        )
        is sentinel
    )
    assert captured["worker_count"] == 6
    with pytest.raises(TypeError):
        prediction.fit_structure_library_formal(
            prefix_pack=frame,
            forecast_coordinates=frame,
            worker_count=1,  # type: ignore[call-arg]
        )


def test_formal_wrapper_rejects_nonformal_member_counts_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, coordinates, _ = _fixture_tables(("fixture-one",))

    def forbidden_fit(**_: object) -> None:
        raise AssertionError("formal cardinality must fail before fitting")

    monkeypatch.setattr(
        prediction,
        "fit_structure_library_parallel",
        forbidden_fit,
    )
    with pytest.raises(prediction.V015PredictionError, match="row count"):
        prediction.fit_structure_library_formal(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
        )


@pytest.mark.parametrize("worker_count", [True, 0, -1, 1.5, 65])
def test_worker_count_is_strictly_validated(worker_count: object) -> None:
    prefix, coordinates, _ = _fixture_tables(("fixture-one",))
    with pytest.raises(prediction.V015PredictionError, match="worker_count"):
        prediction.fit_structure_library_parallel(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            worker_count=worker_count,  # type: ignore[arg-type]
        )


def test_duplicate_and_missing_input_clusters_are_rejected() -> None:
    prefix, coordinates, _ = _fixture_tables()
    duplicate = pd.concat([prefix, prefix.iloc[[0]]], ignore_index=True)
    with pytest.raises(prediction.V015PredictionError, match="duplicate"):
        prediction.fit_structure_library_parallel(
            prefix_pack=duplicate,
            forecast_coordinates=coordinates,
            worker_count=1,
        )

    missing = coordinates.loc[~coordinates["cluster_id"].eq("fixture-z")].reset_index(
        drop=True
    )
    with pytest.raises(prediction.V015PredictionError, match="cluster sets"):
        prediction.fit_structure_library_parallel(
            prefix_pack=prefix,
            forecast_coordinates=missing,
            worker_count=1,
        )


def test_worker_exception_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, coordinates, _ = _fixture_tables(("fixture-one",))
    shutdown_calls: list[tuple[bool, bool]] = []

    class FailingExecutor:
        def __init__(self, **_: object) -> None:
            pass

        def submit(self, *_: object) -> Future[object]:
            future: Future[object] = Future()
            future.set_exception(RuntimeError("worker exploded"))
            return future

        def shutdown(
            self,
            *,
            wait: bool,
            cancel_futures: bool,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(prediction, "ProcessPoolExecutor", FailingExecutor)
    with pytest.raises(
        prediction.V015PredictionError,
        match="worker failed",
    ):
        prediction.fit_structure_library_parallel(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            worker_count=1,
        )
    assert shutdown_calls == [(False, True)]


def test_keyboard_interrupt_cancels_bounded_work_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, coordinates, _ = _fixture_tables(
        tuple(f"fixture-{index}" for index in range(20))
    )
    shutdown_calls: list[tuple[bool, bool]] = []
    submit_count = 0

    class InterruptedExecutor:
        def __init__(self, **_: object) -> None:
            pass

        def submit(self, *_: object) -> Future[object]:
            nonlocal submit_count
            submit_count += 1
            future: Future[object] = Future()
            future.set_exception(KeyboardInterrupt())
            return future

        def shutdown(
            self,
            *,
            wait: bool,
            cancel_futures: bool,
        ) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(prediction, "ProcessPoolExecutor", InterruptedExecutor)
    with pytest.raises(KeyboardInterrupt):
        prediction.fit_structure_library_parallel(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            worker_count=3,
        )
    assert submit_count == 6
    assert shutdown_calls == [(False, True)]


def test_invalid_worker_output_cluster_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, coordinates, _ = _fixture_tables(("fixture-one",))

    class InvalidExecutor:
        def __init__(self, **_: object) -> None:
            pass

        def submit(self, *_: object) -> Future[object]:
            future: Future[object] = Future()
            future.set_result(object())
            return future

        def shutdown(
            self,
            *,
            wait: bool,
            cancel_futures: bool,
        ) -> None:
            assert not wait
            assert cancel_futures

    monkeypatch.setattr(prediction, "ProcessPoolExecutor", InvalidExecutor)
    with pytest.raises(
        prediction.V015PredictionError,
        match="invalid result type",
    ):
        prediction.fit_structure_library_parallel(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            worker_count=1,
        )


def test_apply_from_fitted_does_not_invoke_any_optimizer(
    worker_invariance_results: _WorkerInvarianceResults,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix, coordinates, operating = _fixture_tables()

    def forbidden_fit(*_: object, **__: object) -> None:
        raise AssertionError("fit must not run while applying frozen state")

    monkeypatch.setattr(prediction, "fit_structure_library", forbidden_fit)
    result = prediction.run_pipeline_from_fitted(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating,
        member_fit_diagnostics=(
            worker_invariance_results.one_fit.member_fit_diagnostics
        ),
        member_forecast_bundle=(
            worker_invariance_results.one_fit.member_forecast_bundle
        ),
        state=_state(),
    )
    assert _downstream_bytes(result) == _downstream_bytes(
        worker_invariance_results.one_downstream
    )
    assert np.isfinite(
        result.prediction_bundle["center_forecast_pct"].to_numpy(float)
    ).all()
