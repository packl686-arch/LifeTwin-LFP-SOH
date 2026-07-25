"""Deterministic fit-once prediction execution for frozen V0.15.

The worker boundary receives one cluster's prefix observations and declared
forecast coordinates.  Operating covariates and the frozen learned state stay
in the parent process.  Formal orchestration fits all clusters once, reuses
subsets of those committed fits during training, then applies the frozen state
without ever invoking the optimizer again.
"""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
import multiprocessing

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
    V015FitResult,
    fit_structure_library,
    validate_frozen_variant_keys,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonicalize_frame,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    FrozenLabelFreeState,
    LabelFreePipelineResult,
    recompute_label_free_pipeline,
)


_FORMAL_WORKER_COUNT = 6
_MAXIMUM_HAND_FIXTURE_WORKERS = 64


class V015PredictionError(RuntimeError):
    """Raised when the label-free prediction run cannot commit all outputs."""


@dataclass(frozen=True)
class V015PredictionResult:
    """Fitted members and all downstream label-free prediction artifacts."""

    member_fit_diagnostics: pd.DataFrame
    member_forecast_bundle: pd.DataFrame
    prediction_bundle: pd.DataFrame
    feature_bundle: pd.DataFrame
    primary_risk_bundle: pd.DataFrame
    decision_bundle: pd.DataFrame
    predictor_content_bundle: pd.DataFrame

    @property
    def risk_bundle(self) -> pd.DataFrame:
        """Return the externally committed name for the primary-risk table."""

        return self.primary_risk_bundle


@dataclass(frozen=True)
class _ClusterFitTask:
    key: tuple[str, str]
    prefix_pack: pd.DataFrame
    forecast_coordinates: pd.DataFrame


@dataclass(frozen=True)
class _ClusterFitOutput:
    key: tuple[str, str]
    member_fit_diagnostics: pd.DataFrame
    member_forecast_bundle: pd.DataFrame


def _fit_cluster_worker(task: _ClusterFitTask) -> _ClusterFitOutput:
    """Spawn-safe top-level worker that fits exactly one cluster."""

    result = fit_structure_library(
        task.prefix_pack,
        task.forecast_coordinates,
    )
    return _ClusterFitOutput(
        key=task.key,
        member_fit_diagnostics=result.member_fit_diagnostics,
        member_forecast_bundle=result.member_forecast_bundle,
    )


def _canonical_input(
    frame: pd.DataFrame,
    *,
    filename: str,
    contract: FrozenArtifactContract,
    formal: bool = False,
) -> pd.DataFrame:
    try:
        return canonicalize_frame(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=formal,
        )
    except V015ArtifactError as exc:
        raise V015PredictionError(str(exc)) from exc


def _cluster_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["partition"].astype(str),
            frame["cluster_id"].astype(str),
            strict=True,
        )
    )


def _prepare_fit_tasks(
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    contract: FrozenArtifactContract,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[_ClusterFitTask, ...],
]:
    prefix = _canonical_input(
        prefix_pack,
        filename="prefix_pack.csv",
        contract=contract,
    )
    coordinates = _canonical_input(
        forecast_coordinates,
        filename="forecast_coordinates.csv",
        contract=contract,
    )
    key_sets = (_cluster_keys(prefix), _cluster_keys(coordinates))
    if key_sets[0] != key_sets[1]:
        raise V015PredictionError("Prefix and coordinate cluster sets differ")

    prefix_groups = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in prefix.groupby(
            ["partition", "cluster_id"], sort=False
        )
    }
    coordinate_groups = {
        (str(partition), str(cluster_id)): group.reset_index(drop=True)
        for (partition, cluster_id), group in coordinates.groupby(
            ["partition", "cluster_id"], sort=False
        )
    }
    tasks: list[_ClusterFitTask] = []
    for partition, cluster_id in sorted(key_sets[0]):
        tasks.append(
            _ClusterFitTask(
                key=(partition, cluster_id),
                prefix_pack=prefix_groups[(partition, cluster_id)],
                forecast_coordinates=coordinate_groups[(partition, cluster_id)],
            )
        )
    return prefix, coordinates, tuple(tasks)


def _require_worker_output(
    task: _ClusterFitTask,
    output: object,
) -> _ClusterFitOutput:
    if not isinstance(output, _ClusterFitOutput):
        raise V015PredictionError("A fit worker returned an invalid result type")
    if output.key != task.key:
        raise V015PredictionError("A fit worker returned a different cluster key")
    for frame, label in (
        (output.member_fit_diagnostics, "diagnostics"),
        (output.member_forecast_bundle, "member forecasts"),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise V015PredictionError(f"A fit worker returned invalid {label}")
        if _cluster_keys(frame) != {task.key}:
            raise V015PredictionError(
                f"A fit worker returned missing or extra {label} clusters"
            )
    return output


def _collect_worker_outputs(
    tasks: tuple[_ClusterFitTask, ...],
    worker_count: int,
) -> tuple[_ClusterFitOutput, ...]:
    context = multiprocessing.get_context("spawn")
    try:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        )
    except Exception as exc:
        raise V015PredictionError(
            "The structure-fit worker pool could not be started"
        ) from exc
    futures: dict[Future[_ClusterFitOutput], _ClusterFitTask] = {}
    failed = False
    try:
        task_iterator = iter(tasks)
        maximum_in_flight = max(worker_count * 2, 1)
        for _ in range(min(maximum_in_flight, len(tasks))):
            task = next(task_iterator)
            futures[executor.submit(_fit_cluster_worker, task)] = task
        outputs: dict[tuple[str, str], _ClusterFitOutput] = {}
        while futures:
            completed, _ = wait(
                tuple(futures),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                task = futures.pop(future)
                output = _require_worker_output(task, future.result())
                if output.key in outputs:
                    raise V015PredictionError(
                        "Fit workers returned a duplicate cluster"
                    )
                outputs[output.key] = output
                try:
                    next_task = next(task_iterator)
                except StopIteration:
                    continue
                futures[executor.submit(_fit_cluster_worker, next_task)] = next_task
        expected = {task.key for task in tasks}
        if set(outputs) != expected:
            raise V015PredictionError("Fit workers returned an incomplete cluster set")
        return tuple(outputs[key] for key in sorted(expected))
    except KeyboardInterrupt:
        failed = True
        for future in futures:
            future.cancel()
        raise
    except BaseException as exc:
        failed = True
        for future in futures:
            future.cancel()
        if isinstance(exc, V015PredictionError):
            raise
        if not isinstance(exc, Exception):
            raise
        raise V015PredictionError(
            "A structure-fit worker failed; no prediction result was produced"
        ) from exc
    finally:
        executor.shutdown(wait=not failed, cancel_futures=failed)


def _concat_fitted_tables(
    outputs: tuple[_ClusterFitOutput, ...],
    *,
    contract: FrozenArtifactContract,
) -> V015FitResult:
    diagnostics = pd.concat(
        [output.member_fit_diagnostics for output in outputs],
        ignore_index=True,
    )
    forecasts = pd.concat(
        [output.member_forecast_bundle for output in outputs],
        ignore_index=True,
    )
    diagnostics = _canonical_input(
        diagnostics,
        filename="member_fit_diagnostics.csv",
        contract=contract,
    )
    forecasts = _canonical_input(
        forecasts,
        filename="member_forecast_bundle.csv",
        contract=contract,
    )
    expected = {output.key for output in outputs}
    if _cluster_keys(diagnostics) != expected or _cluster_keys(forecasts) != expected:
        raise V015PredictionError(
            "Canonical fitted tables contain a missing or extra cluster"
        )
    for key, group in diagnostics.groupby(["partition", "cluster_id"], sort=False):
        try:
            validate_frozen_variant_keys(
                zip(
                    group["model_id"].astype(str),
                    group["variant_id"].astype(str),
                    strict=True,
                ),
                context=f"parallel diagnostics {key}",
            )
        except ValueError as exc:
            raise V015PredictionError(str(exc)) from exc
    for key, group in forecasts.groupby(["partition", "cluster_id"], sort=False):
        distinct = tuple(
            group.loc[:, ["model_id", "variant_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        try:
            validate_frozen_variant_keys(
                distinct,
                context=f"parallel forecasts {key}",
            )
        except ValueError as exc:
            raise V015PredictionError(str(exc)) from exc
        sizes = group.groupby(["model_id", "variant_id"], sort=False).size()
        days = group.groupby(["model_id", "variant_id"], sort=False)[
            "forecast_day"
        ].agg(lambda values: tuple(sorted(values)))
        if (
            len(sizes) != len(FROZEN_VARIANT_KEYS)
            or not sizes.eq(len(contract.forecast_days)).all()
            or not all(value == contract.forecast_days for value in days)
        ):
            raise V015PredictionError(
                f"Parallel forecasts {key} have an incomplete frozen grid"
            )
    return V015FitResult(
        member_fit_diagnostics=diagnostics,
        member_forecast_bundle=forecasts,
    )


def _assemble_result(
    fitted: V015FitResult,
    downstream: LabelFreePipelineResult,
) -> V015PredictionResult:
    return V015PredictionResult(
        member_fit_diagnostics=fitted.member_fit_diagnostics,
        member_forecast_bundle=fitted.member_forecast_bundle,
        prediction_bundle=downstream.prediction_bundle,
        feature_bundle=downstream.feature_bundle,
        primary_risk_bundle=downstream.primary_risk_bundle,
        decision_bundle=downstream.decision_bundle,
        predictor_content_bundle=downstream.predictor_content_bundle,
    )


def _validate_worker_count(worker_count: object) -> int:
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or not 1 <= worker_count <= _MAXIMUM_HAND_FIXTURE_WORKERS
    ):
        raise V015PredictionError("worker_count must be an integer in [1, 64]")
    return worker_count


def fit_structure_library_parallel(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    worker_count: int,
) -> V015FitResult:
    """Fit every cluster exactly once in an isolated process pool."""

    worker_count = _validate_worker_count(worker_count)
    contract = load_artifact_contract()
    _, _, tasks = _prepare_fit_tasks(
        prefix_pack,
        forecast_coordinates,
        contract,
    )
    outputs = _collect_worker_outputs(tasks, worker_count)
    return _concat_fitted_tables(outputs, contract=contract)


def fit_structure_library_formal(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
) -> V015FitResult:
    """Run the sole formal structure fit with exactly six workers."""

    contract = load_artifact_contract()
    prefix = _canonical_input(
        prefix_pack,
        filename="prefix_pack.csv",
        contract=contract,
        formal=True,
    )
    coordinates = _canonical_input(
        forecast_coordinates,
        filename="forecast_coordinates.csv",
        contract=contract,
        formal=True,
    )
    return fit_structure_library_parallel(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        worker_count=_FORMAL_WORKER_COUNT,
    )


def run_pipeline_from_fitted(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    member_fit_diagnostics: pd.DataFrame,
    member_forecast_bundle: pd.DataFrame,
    state: FrozenLabelFreeState,
) -> LabelFreePipelineResult:
    """Apply frozen state to committed fits without invoking an optimizer."""

    try:
        return recompute_label_free_pipeline(
            prefix_pack=prefix_pack,
            forecast_coordinates=forecast_coordinates,
            operating_pack=operating_pack,
            member_fit_diagnostics=member_fit_diagnostics,
            member_forecast_bundle=member_forecast_bundle,
            state=state,
        )
    except Exception as exc:
        raise V015PredictionError(
            "The frozen label-free pipeline rejected committed fit artifacts"
        ) from exc


def run_nonformal_label_free_prediction(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    state: FrozenLabelFreeState,
    worker_count: int,
) -> V015PredictionResult:
    """Convenience path for hand fixtures; formal orchestration must not use it."""

    fitted = fit_structure_library_parallel(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        worker_count=worker_count,
    )
    downstream = run_pipeline_from_fitted(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        operating_pack=operating_pack,
        member_fit_diagnostics=fitted.member_fit_diagnostics,
        member_forecast_bundle=fitted.member_forecast_bundle,
        state=state,
    )
    return _assemble_result(fitted, downstream)


__all__ = [
    "V015PredictionError",
    "V015PredictionResult",
    "fit_structure_library_formal",
    "fit_structure_library_parallel",
    "run_nonformal_label_free_prediction",
    "run_pipeline_from_fitted",
]
