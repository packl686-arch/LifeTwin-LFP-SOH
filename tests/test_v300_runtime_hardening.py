from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_prediction as prediction


ROOT = Path(__file__).resolve().parents[1]


class _Future:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def result(self) -> object:
        raise self.error

    def cancel(self) -> bool:
        return True


class _Executor:
    def __init__(
        self,
        error: BaseException,
        *,
        exit_code: int | None = None,
        shutdown_error: BaseException | None = None,
    ) -> None:
        self.future = _Future(error)
        self._processes = (
            {} if exit_code is None else {1: SimpleNamespace(exitcode=exit_code)}
        )
        self.shutdown_error = shutdown_error

    def submit(self, function: object, task: object) -> _Future:
        del function, task
        return self.future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait, cancel_futures
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _task() -> prediction._ClusterFitTask:
    return prediction._ClusterFitTask(
        key=("calibration", "diagnostic"),
        prefix_pack=pd.DataFrame(),
        forecast_coordinates=pd.DataFrame(),
    )


def _wait(
    futures: tuple[_Future, ...],
    *,
    return_when: object,
) -> tuple[set[_Future], set[_Future]]:
    del return_when
    return {futures[0]}, set(futures[1:])


def _run_with_executor(executor: _Executor) -> None:
    with (
        mock.patch.object(
            prediction,
            "ProcessPoolExecutor",
            return_value=executor,
        ),
        mock.patch.object(prediction, "wait", side_effect=_wait),
    ):
        prediction._collect_worker_outputs((_task(),), 1)


def test_primary_worker_error_survives_secondary_shutdown_failure() -> None:
    executor = _Executor(
        RuntimeError("private worker detail"),
        shutdown_error=ValueError("private shutdown detail"),
    )

    with pytest.raises(prediction.V015WorkerExecutionError) as observed:
        _run_with_executor(executor)

    error = observed.value
    assert type(error.__cause__) is RuntimeError
    telemetry = prediction.result_blind_worker_failure_telemetry(error)
    assert telemetry == {
        "schema_version": "1.0.0",
        "phase": "worker_future_result",
        "exception_class": "V015WorkerExecutionError",
        "worker_exit_codes": [],
        "shutdown_exception_class": "ValueError",
    }
    assert "private" not in json.dumps(telemetry)


def test_broken_pool_telemetry_contains_exit_code_but_no_process_id() -> None:
    executor = _Executor(BrokenProcessPool(), exit_code=71)

    with pytest.raises(prediction.V015WorkerPoolBrokenError) as observed:
        _run_with_executor(executor)

    telemetry = prediction.result_blind_worker_failure_telemetry(observed.value)
    assert telemetry is not None
    assert telemetry["phase"] == "broken_process_pool"
    assert telemetry["worker_exit_codes"] == [71]
    assert "process_id" not in telemetry


@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("pwsh") is None,
    reason="PowerShell wrapper is Windows-specific",
)
def test_result_blind_wrapper_preserves_native_exit_and_hashes(tmp_path: Path) -> None:
    child = tmp_path / "child.py"
    child.write_text(
        "import sys\nprint('public-out')\nprint('public-err', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="ascii",
    )
    stdout = tmp_path / "stdout.txt"
    stderr = tmp_path / "stderr.txt"
    manifest = tmp_path / "exit.json"
    command = [
        "pwsh",
        "-File",
        str(ROOT / "scripts/run_result_blind_python.ps1"),
        "-Python",
        sys.executable,
        "-Script",
        str(child),
        "-WorkingDirectory",
        str(tmp_path),
        "-StdoutPath",
        str(stdout),
        "-StderrPath",
        str(stderr),
        "-ExitManifestPath",
        str(manifest),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 7, completed.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["wrapper_status"] == "completed"
    assert payload["process_exit_code"] == 7
    assert payload["timed_out"] is False
    assert payload["launch_exception_class"] is None
    assert payload["stdout_sha256"] == hashlib.sha256(stdout.read_bytes()).hexdigest()
    assert payload["stderr_sha256"] == hashlib.sha256(stderr.read_bytes()).hexdigest()
    assert "public-out" in stdout.read_text(encoding="utf-8")
    assert "public-err" in stderr.read_text(encoding="utf-8")

    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert json.loads(manifest.read_text(encoding="utf-8")) == payload


@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("pwsh") is None,
    reason="PowerShell wrapper is Windows-specific",
)
def test_result_blind_wrapper_enforces_process_tree_timeout(tmp_path: Path) -> None:
    child = tmp_path / "sleep.py"
    child.write_text("import time\ntime.sleep(30)\n", encoding="ascii")
    stdout = tmp_path / "stdout.txt"
    stderr = tmp_path / "stderr.txt"
    manifest = tmp_path / "exit.json"

    completed = subprocess.run(
        (
            "pwsh",
            "-File",
            str(ROOT / "scripts/run_result_blind_python.ps1"),
            "-Python",
            sys.executable,
            "-Script",
            str(child),
            "-WorkingDirectory",
            str(tmp_path),
            "-StdoutPath",
            str(stdout),
            "-StderrPath",
            str(stderr),
            "-ExitManifestPath",
            str(manifest),
            "-TimeoutSeconds",
            "1",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 124, completed.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["wrapper_status"] == "completed"
    assert payload["process_exit_code"] == 124
    assert payload["timed_out"] is True
