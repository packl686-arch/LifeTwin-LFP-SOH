"""Exercise result-blind failure observability around the V2.10 fit path."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from unittest import mock

import pandas as pd

from v210_diagnostic_resources import ResourceSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifetwin.experiments import (  # noqa: E402
    calendar_long_horizon_v015_prediction as v015_prediction,
)
from lifetwin.experiments import (  # noqa: E402
    calendar_long_horizon_v019_io as v019_io,
)
from lifetwin.experiments import (  # noqa: E402
    calendar_long_horizon_v019_prediction as v019_prediction,
)
from lifetwin.experiments.calendar_long_horizon_v019_io import (  # noqa: E402
    V024IOError,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (  # noqa: E402
    sanitized_structural_traceback,
)


_ABRUPT_EXIT_CODE = 71


class _InjectedPoolStartupError(RuntimeError):
    pass


class _InjectedWorkerError(RuntimeError):
    pass


class _InjectedSubmissionError(RuntimeError):
    pass


class _InjectedWaitError(RuntimeError):
    pass


class _InjectedShutdownError(RuntimeError):
    pass


class _FakeFuture:
    def __init__(self, *, result: object = None, error: BaseException | None = None):
        self._result = result
        self._error = error

    def result(self) -> object:
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self) -> bool:
        return True


class _FakeExecutor:
    def __init__(
        self,
        *,
        future: _FakeFuture,
        submit_error: BaseException | None = None,
        shutdown_error: BaseException | None = None,
    ) -> None:
        self._future = future
        self._submit_error = submit_error
        self._shutdown_error = shutdown_error

    def submit(self, function: object, task: object) -> _FakeFuture:
        del function, task
        if self._submit_error is not None:
            raise self._submit_error
        return self._future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait, cancel_futures
        if self._shutdown_error is not None:
            raise self._shutdown_error


def _task() -> v015_prediction._ClusterFitTask:
    return v015_prediction._ClusterFitTask(
        key=("calibration", "diagnostic"),
        prefix_pack=pd.DataFrame(),
        forecast_coordinates=pd.DataFrame(),
    )


def _valid_output() -> v015_prediction._ClusterFitOutput:
    frame = pd.DataFrame(
        {"partition": ["calibration"], "cluster_id": ["diagnostic"]}
    )
    return v015_prediction._ClusterFitOutput(
        key=("calibration", "diagnostic"),
        member_fit_diagnostics=frame,
        member_forecast_bundle=frame.copy(),
    )


def _fake_wait(
    futures: tuple[_FakeFuture, ...],
    *,
    return_when: object,
) -> tuple[set[_FakeFuture], set[_FakeFuture]]:
    del return_when
    return {futures[0]}, set(futures[1:])


def _capture_error(function: object) -> BaseException:
    try:
        function()
    except BaseException as error:
        return error
    raise RuntimeError("The injected failure did not occur")


def _io_failure() -> BaseException:
    with mock.patch.object(
        v019_io,
        "_extract_fresh_generation_frames_for_formal_fit_v024",
        side_effect=V024IOError(),
    ):
        return _capture_error(
            lambda: v019_prediction.fit_verified_generation_bundle_v024(object())
        )


def _pool_startup_failure() -> BaseException:
    def fail_startup(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise _InjectedPoolStartupError()

    with mock.patch.object(
        v015_prediction,
        "ProcessPoolExecutor",
        side_effect=fail_startup,
    ):
        return _capture_error(
            lambda: v015_prediction._collect_worker_outputs((_task(),), 1)
        )


def _executor_failure(
    *,
    future: _FakeFuture,
    submit_error: BaseException | None = None,
    wait_error: BaseException | None = None,
    shutdown_error: BaseException | None = None,
) -> BaseException:
    executor = _FakeExecutor(
        future=future,
        submit_error=submit_error,
        shutdown_error=shutdown_error,
    )
    wait_probe = wait_error if wait_error is not None else _fake_wait
    with (
        mock.patch.object(
            v015_prediction,
            "ProcessPoolExecutor",
            return_value=executor,
        ),
        mock.patch.object(v015_prediction, "wait", side_effect=wait_probe),
    ):
        return _capture_error(
            lambda: v015_prediction._collect_worker_outputs((_task(),), 1)
        )


def _worker_exception_failure() -> BaseException:
    return _executor_failure(future=_FakeFuture(error=_InjectedWorkerError()))


def _submission_failure() -> BaseException:
    return _executor_failure(
        future=_FakeFuture(),
        submit_error=_InjectedSubmissionError(),
    )


def _wait_failure() -> BaseException:
    return _executor_failure(
        future=_FakeFuture(),
        wait_error=_InjectedWaitError(),
    )


def _broken_pool_failure() -> BaseException:
    return _executor_failure(future=_FakeFuture(error=BrokenProcessPool()))


def _invalid_output_failure() -> BaseException:
    return _executor_failure(future=_FakeFuture(result=object()))


def _shutdown_failure() -> BaseException:
    return _executor_failure(
        future=_FakeFuture(result=_valid_output()),
        shutdown_error=_InjectedShutdownError(),
    )


def _abrupt_worker() -> None:
    time.sleep(0.5)
    os._exit(_ABRUPT_EXIT_CODE)


def _observe_abrupt_worker_exit() -> tuple[BaseException, list[int]]:
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
    processes: list[multiprocessing.Process] = []
    try:
        future = executor.submit(_abrupt_worker)
        error = _capture_error(future.result)
        processes = list(executor._processes.values())
        for process in processes:
            process.join(timeout=5.0)
        exit_codes = sorted(
            int(process.exitcode)
            for process in processes
            if process.exitcode is not None
        )
        return error, exit_codes
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _safe_error_identity(error: BaseException) -> dict[str, object]:
    payload = json.loads(
        sanitized_structural_traceback(error, repo_root=ROOT).decode("ascii")
    )
    return {
        "exception_chain": [
            {
                "exception_class": entry["exception_class"],
                "relationship": entry["relationship"],
                "frame_count": len(entry["frames"]),
            }
            for entry in payload["exception_chain"]
        ],
        "classification_mode": payload["classification_mode"],
        "attempt_disposition": payload["attempt_disposition"],
        "reason_code": payload["reason_code"],
        "exception_chain_truncated": payload["exception_chain_truncated"],
    }


def _case_result(
    *,
    case: str,
    phase: str,
    expected_chain: list[str],
    probe: object,
    observe_exit_code: bool = False,
    expected_runtime_phase: str | None = None,
) -> dict[str, object]:
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    worker_exit_codes: list[int] = []
    try:
        error = probe()
        if observe_exit_code:
            abrupt_error, worker_exit_codes = _observe_abrupt_worker_exit()
            abrupt_class = type(abrupt_error).__name__
        else:
            abrupt_class = None
    finally:
        resource_telemetry = sampler.stop()
    identity = _safe_error_identity(error)
    runtime_telemetry = v015_prediction.result_blind_worker_failure_telemetry(error)
    observed_chain = [
        item["exception_class"] for item in identity["exception_chain"]
    ]
    exit_code_ok = (
        not observe_exit_code
        or (
            abrupt_class == "BrokenProcessPool"
            and worker_exit_codes == [_ABRUPT_EXIT_CODE]
        )
    )
    runtime_phase_ok = (
        (expected_runtime_phase is None and runtime_telemetry is None)
        or (
            runtime_telemetry is not None
            and runtime_telemetry["phase"] == expected_runtime_phase
        )
    )
    return {
        "case": case,
        "phase": phase,
        "status": (
            "expected_failure_observed"
            if observed_chain == expected_chain and exit_code_ok and runtime_phase_ok
            else "unexpected_observation"
        ),
        **identity,
        "runtime_failure_telemetry": runtime_telemetry,
        "elapsed_seconds": time.perf_counter() - started,
        "worker_exit_codes": worker_exit_codes,
        "worker_exit_code_source": (
            "independent_spawn_abrupt_exit_calibration"
            if observe_exit_code
            else "not_applicable"
        ),
        "resource_telemetry": resource_telemetry,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    cases = [
        _case_result(
            case="verified_bundle_io",
            phase="verified_generation_extraction",
            expected_chain=["V024PredictionError", "V024IOError"],
            probe=_io_failure,
        ),
        _case_result(
            case="pool_startup",
            phase="process_pool_construction",
            expected_chain=[
                "V015WorkerPoolStartupError",
                "_InjectedPoolStartupError",
            ],
            probe=_pool_startup_failure,
            expected_runtime_phase="process_pool_construction",
        ),
        _case_result(
            case="worker_submission",
            phase="worker_submission",
            expected_chain=[
                "V015WorkerSubmissionError",
                "_InjectedSubmissionError",
            ],
            probe=_submission_failure,
            expected_runtime_phase="worker_submission",
        ),
        _case_result(
            case="worker_completion_wait",
            phase="worker_completion_wait",
            expected_chain=["V015WorkerWaitError", "_InjectedWaitError"],
            probe=_wait_failure,
            expected_runtime_phase="worker_completion_wait",
        ),
        _case_result(
            case="worker_exception",
            phase="worker_future_result",
            expected_chain=["V015WorkerExecutionError", "_InjectedWorkerError"],
            probe=_worker_exception_failure,
            expected_runtime_phase="worker_future_result",
        ),
        _case_result(
            case="broken_process_pool",
            phase="worker_future_result",
            expected_chain=["V015WorkerPoolBrokenError", "BrokenProcessPool"],
            probe=_broken_pool_failure,
            observe_exit_code=True,
            expected_runtime_phase="broken_process_pool",
        ),
        _case_result(
            case="invalid_worker_output",
            phase="worker_output_validation",
            expected_chain=["V015WorkerOutputError"],
            probe=_invalid_output_failure,
            expected_runtime_phase="worker_output_validation",
        ),
        _case_result(
            case="executor_shutdown",
            phase="process_pool_shutdown",
            expected_chain=["V015WorkerShutdownError", "_InjectedShutdownError"],
            probe=_shutdown_failure,
            expected_runtime_phase="process_pool_shutdown",
        ),
    ]
    passed = all(case["status"] == "expected_failure_observed" for case in cases)
    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "passed" if passed else "failed",
                "result_blind": True,
                "formal_inputs_used": False,
                "formal_rows_opened": False,
                "formal_seeds_used": False,
                "sealed_truth_opened": False,
                "cases": cases,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
