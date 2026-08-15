"""Verify the result-blind V2.10 expanded diagnostic record set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_SCALE_RUNS = {
    "scale-1024.json": (1024, 2),
    "scale-2048.json": (2048, 2),
    "scale-4096.json": (4096, 2),
    "scale-5950-a.json": (5950, 1),
    "scale-5950-b.json": (5950, 1),
}
_FAILURE_CASES = {
    "verified_bundle_io",
    "pool_startup",
    "worker_exception",
    "broken_process_pool",
    "invalid_worker_output",
    "executor_shutdown",
}
_BANNED_KEYS = {
    "cluster_id",
    "forecast_day",
    "message",
    "observed_retention_pct",
    "process_id",
    "seed",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid diagnostic JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Diagnostic JSON is not an object: {path.name}")
    return payload, raw


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def _verify_result_blind_keys(payload: dict[str, Any], *, filename: str) -> None:
    observed = _walk_keys(payload)
    forbidden = sorted(observed.intersection(_BANNED_KEYS))
    if forbidden:
        raise RuntimeError(f"Unsafe keys in {filename}: {forbidden}")


def _verify_resource_telemetry(payload: dict[str, Any], *, filename: str) -> None:
    telemetry = payload.get("resource_telemetry")
    if not isinstance(telemetry, dict):
        raise RuntimeError(f"Missing resource telemetry: {filename}")
    if telemetry.get("backend") != "windows_toolhelp_psapi_v1":
        raise RuntimeError(f"Unexpected resource backend: {filename}")
    integer_fields = (
        "sample_count",
        "sampling_error_count",
        "peak_process_tree_process_count",
        "peak_worker_process_count",
        "peak_process_tree_working_set_bytes",
        "peak_process_tree_private_bytes",
    )
    if any(not isinstance(telemetry.get(key), int) for key in integer_fields):
        raise RuntimeError(f"Invalid resource telemetry: {filename}")
    if telemetry["sample_count"] < 2 or telemetry["sampling_error_count"] != 0:
        raise RuntimeError(f"Incomplete resource telemetry: {filename}")


def _verify_hashes(payload: dict[str, Any], *, filename: str) -> dict[str, str]:
    hashes = payload.get("hashes")
    expected_keys = {
        "member_fit_diagnostics.csv",
        "member_forecast_bundle.csv",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_keys:
        raise RuntimeError(f"Unexpected output hash set: {filename}")
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in hashes.values()):
        raise RuntimeError(f"Invalid output hash: {filename}")
    return {str(key): str(value) for key, value in hashes.items()}


def _verify_scale_run(
    payload: dict[str, Any],
    *,
    filename: str,
    clusters: int,
    repeat: int,
) -> dict[str, str]:
    expected = {
        "schema_version": "1.0.0",
        "status": "passed",
        "phase": "completed",
        "clusters": clusters,
        "workers": 6,
        "repeat": repeat,
        "suite": "randomized",
        "worker_exit_codes": [],
        "diagnostic_rows": clusters * 86,
        "forecast_rows": clusters * 86 * 8,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Scale-run identity mismatch: {filename}")
    elapsed = payload.get("repeat_elapsed_seconds")
    if (
        not isinstance(elapsed, list)
        or len(elapsed) != repeat
        or any(not isinstance(value, (int, float)) or value <= 0 for value in elapsed)
    ):
        raise RuntimeError(f"Scale-run timing mismatch: {filename}")
    _verify_resource_telemetry(payload, filename=filename)
    if payload["resource_telemetry"]["peak_worker_process_count"] != 6:
        raise RuntimeError(f"Six-worker process tree was not observed: {filename}")
    _verify_result_blind_keys(payload, filename=filename)
    return _verify_hashes(payload, filename=filename)


def _verify_failure_matrix(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version": "1.0.0",
        "status": "passed",
        "result_blind": True,
        "formal_inputs_used": False,
        "formal_rows_opened": False,
        "formal_seeds_used": False,
        "sealed_truth_opened": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Failure-matrix boundary changed")
    cases = payload.get("cases")
    if not isinstance(cases, list) or {
        case.get("case") for case in cases if isinstance(case, dict)
    } != _FAILURE_CASES:
        raise RuntimeError("Failure-matrix cases changed")
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("Failure-matrix case is invalid")
        if case.get("status") != "expected_failure_observed":
            raise RuntimeError("Failure-matrix injection was not observed")
        _verify_resource_telemetry(case, filename="failure-matrix.json")
    abrupt = next(case for case in cases if case["case"] == "broken_process_pool")
    if abrupt.get("worker_exit_codes") != [71]:
        raise RuntimeError("Abrupt worker exit code changed")
    _verify_result_blind_keys(payload, filename="failure-matrix.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_root", type=Path)
    args = parser.parse_args()
    root = args.record_root.resolve()

    failure_matrix, failure_bytes = _load_json(root / "failure-matrix.json")
    _verify_failure_matrix(failure_matrix)
    file_hashes = {
        "failure-matrix.json": hashlib.sha256(failure_bytes).hexdigest()
    }
    output_hashes: dict[str, dict[str, str]] = {}
    completed_cluster_fits = 0
    for filename, (clusters, repeat) in _SCALE_RUNS.items():
        payload, raw = _load_json(root / filename)
        output_hashes[filename] = _verify_scale_run(
            payload,
            filename=filename,
            clusters=clusters,
            repeat=repeat,
        )
        completed_cluster_fits += clusters * repeat
        file_hashes[filename] = hashlib.sha256(raw).hexdigest()

    if output_hashes["scale-5950-a.json"] != output_hashes["scale-5950-b.json"]:
        raise RuntimeError("Independent full-scale output hashes differ")

    stderr_files = ["failure-matrix.stderr.txt"] + [
        filename.replace(".json", ".stderr.txt") for filename in _SCALE_RUNS
    ]
    for filename in stderr_files:
        if (root / filename).read_bytes() != b"":
            raise RuntimeError(f"Diagnostic stderr is not empty: {filename}")

    print(
        json.dumps(
            {
                "status": "V031_EXPANDED_DIAGNOSTICS_OK",
                "completed_cluster_fits": completed_cluster_fits,
                "formal_rows_opened": False,
                "formal_seeds_used": False,
                "sealed_truth_opened": False,
                "record_sha256": file_hashes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
