from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_result_blind_failure_matrix_covers_fit_lifecycle() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "diagnose_v210_failure_matrix.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["result_blind"] is True
    assert payload["formal_inputs_used"] is False
    assert payload["formal_rows_opened"] is False
    assert payload["formal_seeds_used"] is False
    assert payload["sealed_truth_opened"] is False

    cases = {case["case"]: case for case in payload["cases"]}
    assert set(cases) == {
        "verified_bundle_io",
        "pool_startup",
        "worker_exception",
        "broken_process_pool",
        "invalid_worker_output",
        "executor_shutdown",
    }
    assert all(
        case["status"] == "expected_failure_observed" for case in cases.values()
    )
    assert cases["verified_bundle_io"]["reason_code"] == (
        "INTEGRITY_ARTIFACT_HASH_MISMATCH"
    )
    assert cases["broken_process_pool"]["worker_exit_codes"] == [71]
    assert all(
        case["resource_telemetry"]["sample_count"] >= 2
        for case in cases.values()
    )

    lower_output = completed.stdout.lower()
    assert "sensitive" not in lower_output
    assert "cluster_id" not in lower_output
    assert "observed_retention_pct" not in lower_output
    assert "forecast_day" not in lower_output
    assert "process_id" not in lower_output
