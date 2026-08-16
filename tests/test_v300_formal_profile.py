from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.runtime_reliability_v300_protocol import (
    V300_CONFIG_BYTE_SHA256,
    V300_CONFIG_SEMANTIC_SHA256,
    V300_DESIGN_COMMIT,
    V300_DEVELOPMENT_SEED_ROOT,
    V300_EXPECTED_FAILURE_PHASES,
    V300_EXPECTED_JOBS,
    V300_FORMAL_SEED_ROOT,
    V300_ONLY_ATTEMPT_ID,
    V300_PROTOCOL_COMMIT,
    V300_PROTOCOL_ID,
    V300ProtocolError,
    load_v300_design,
)
from lifetwin.experiments.runtime_reliability_v300_runner import (
    V300IntegrityError,
    evaluate_v300_attempt,
    preflight_v300,
)
from scripts import diagnose_v210_fit_spawn as fit_probe


ROOT = Path(__file__).resolve().parents[1]


def _rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _hashes(clusters: int) -> dict[str, str]:
    return {
        name: hashlib.sha256(f"{clusters}:{name}".encode("ascii")).hexdigest()
        for name in (
            "member_fit_diagnostics.csv",
            "member_forecast_bundle.csv",
        )
    }


def _normal_record(job_id: str, clusters: int, workers: int) -> dict[str, object]:
    del job_id
    payload = {
        "schema_version": "1.0.0",
        "status": "passed",
        "phase": "completed",
        "clusters": clusters,
        "workers": workers,
        "repeat": 1,
        "suite": "mixed",
        "elapsed_seconds": 12.5 if clusters != 5950 else 6000.0,
        "repeat_elapsed_seconds": [12.5],
        "diagnostic_rows": clusters * 86,
        "forecast_rows": clusters * 86 * 8,
        "hashes": _hashes(clusters),
        "runtime_failure_telemetry": None,
        "worker_exit_codes": [],
        "resource_telemetry": {
            "backend": "windows_toolhelp_psapi_v1",
            "sample_count": 5,
            "sampling_error_count": 0,
            "peak_process_tree_process_count": workers + 1,
            "peak_worker_process_count": workers,
            "peak_process_tree_working_set_bytes": 1_500_000_000,
            "peak_process_tree_private_bytes": 2_000_000_000,
            "minimum_available_physical_memory_bytes": 2_000_000_000,
            "disk_free_start_bytes": 100_000_000_000,
            "disk_free_end_bytes": 99_000_000_000,
        },
        "formal_inputs_used": True,
        "formal_rows_opened": False,
        "formal_seeds_used": True,
        "sealed_truth_opened": False,
    }
    return {
        "payload": payload,
        "progress": deepcopy(payload),
        "exit_manifest": {
            "schema_version": "1.0.0",
            "wrapper_status": "completed",
            "process_exit_code": 0,
            "timed_out": False,
            "launch_exception_class": None,
            "started_utc": "2026-08-15T00:00:00Z",
            "finished_utc": "2026-08-15T00:00:01Z",
            "elapsed_seconds": 1.0,
            "script_sha256": "a" * 64,
            "stdout_byte_count": 1,
            "stdout_sha256": "b" * 64,
            "stderr_byte_count": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "wrapper_exit_code": 0,
        "stderr_byte_count": 0,
    }


def _records() -> dict[str, object]:
    return {
        job_id: _normal_record(job_id, clusters, workers)
        for job_id, clusters, workers in V300_EXPECTED_JOBS
    }


def _failure_matrix() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for case, phase in V300_EXPECTED_FAILURE_PHASES.items():
        cases.append(
            {
                "case": case,
                "status": "expected_failure_observed",
                "runtime_failure_telemetry": (
                    None
                    if phase is None
                    else {
                        "schema_version": "1.0.0",
                        "phase": phase,
                        "exception_class": "V015WorkerPoolError",
                        "worker_exit_codes": [],
                        "shutdown_exception_class": None,
                    }
                ),
            }
        )
    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "result_blind": True,
        "formal_inputs_used": False,
        "formal_rows_opened": False,
        "formal_seeds_used": False,
        "sealed_truth_opened": False,
        "cases": cases,
    }


def test_v300_protocol_loads_without_consuming_rng() -> None:
    before = np.random.get_state()
    design = load_v300_design()
    after = np.random.get_state()

    assert _rng_equal(before, after)
    assert design.protocol_id == V300_PROTOCOL_ID
    assert design.config_byte_sha256 == V300_CONFIG_BYTE_SHA256
    assert design.config_semantic_sha256 == V300_CONFIG_SEMANTIC_SHA256
    assert design.formal_seed_root == V300_FORMAL_SEED_ROOT
    assert design.development_seed_root == V300_DEVELOPMENT_SEED_ROOT
    assert tuple((job.job_id, job.clusters, job.workers) for job in design.jobs) == (
        V300_EXPECTED_JOBS
    )


def test_v300_protocol_commit_is_direct_child_of_design() -> None:
    assert (
        subprocess.run(
            ("git", "rev-parse", f"{V300_PROTOCOL_COMMIT}^"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == V300_DESIGN_COMMIT
    )
    changed = subprocess.run(
        ("git", "diff", "--name-only", V300_DESIGN_COMMIT, V300_PROTOCOL_COMMIT),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert set(changed) == {
        "configs/experiments/runtime_reliability_v3_0.json",
        "reports/runtime_reliability_v3_0_preregistration.md",
        "requirements/v300-formal.txt",
    }


def test_v300_loader_rejects_any_byte_drift(tmp_path: Path) -> None:
    source = load_v300_design().config_path
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(V300ProtocolError):
        load_v300_design(changed)


def test_v300_inputs_and_identity_are_disjoint_from_v210() -> None:
    design = load_v300_design()
    independence = design.raw["independence_contract"]
    assert independence["v2_10_formal_rows_forbidden"] is True
    assert independence["v2_10_sealed_truth_forbidden"] is True
    assert independence["v2_10_seed_or_artifact_reuse_forbidden"] is True
    assert design.attempt_root.name == V300_ONLY_ATTEMPT_ID
    assert not design.attempt_root.exists()
    assert not design.authorization_record.exists()


def test_mixed_development_fixture_is_seeded_and_deterministic() -> None:
    first_prefix, first_coordinates = fit_probe._fixture_tables(
        4,
        "mixed",
        seed_root=V300_DEVELOPMENT_SEED_ROOT,
        cluster_prefix="v300-development",
    )
    second_prefix, second_coordinates = fit_probe._fixture_tables(
        4,
        "mixed",
        seed_root=V300_DEVELOPMENT_SEED_ROOT,
        cluster_prefix="v300-development",
    )
    changed_prefix, _ = fit_probe._fixture_tables(
        4,
        "mixed",
        seed_root=V300_DEVELOPMENT_SEED_ROOT + 100,
        cluster_prefix="v300-development",
    )

    pd.testing.assert_frame_equal(first_prefix, second_prefix)
    pd.testing.assert_frame_equal(first_coordinates, second_coordinates)
    even = first_prefix[first_prefix["cluster_id"].str.endswith(("00000", "00002"))]
    changed_even = changed_prefix[
        changed_prefix["cluster_id"].str.endswith(("00000", "00002"))
    ]
    pd.testing.assert_frame_equal(
        even.reset_index(drop=True), changed_even.reset_index(drop=True)
    )
    assert not first_prefix.equals(changed_prefix)
    assert first_prefix["cluster_id"].str.startswith("v300-development-").all()


def test_v300_evaluator_accepts_only_all_conjunctive_gates() -> None:
    evaluation = evaluate_v300_attempt(
        load_v300_design(),
        _records(),
        _failure_matrix(),
    )

    assert evaluation.passed is True
    assert evaluation.disposition == "success"
    assert evaluation.failed_gates == ()
    assert all(evaluation.gates.values())


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    (
        ("hash", "hash_equivalence"),
        ("resource", "working_set_ceiling"),
        ("elapsed", "full_scale_elapsed"),
        ("stderr", "normal_record_contract"),
        ("payload", "normal_record_contract"),
        ("phase", "failure_matrix_exact_phases"),
    ),
)
def test_v300_evaluator_fails_closed(mutation: str, failed_gate: str) -> None:
    records = _records()
    matrix = _failure_matrix()
    if mutation == "hash":
        records["reference-96-parallel-b"]["payload"]["hashes"][
            "member_fit_diagnostics.csv"
        ] = "f" * 64
        records["reference-96-parallel-b"]["progress"] = deepcopy(
            records["reference-96-parallel-b"]["payload"]
        )
    elif mutation == "resource":
        records["full-5950-a"]["payload"]["resource_telemetry"][
            "peak_process_tree_working_set_bytes"
        ] = 2_415_919_105
        records["full-5950-a"]["progress"] = deepcopy(records["full-5950-a"]["payload"])
    elif mutation == "elapsed":
        records["full-5950-a"]["payload"]["elapsed_seconds"] = 7200.1
        records["full-5950-a"]["progress"] = deepcopy(records["full-5950-a"]["payload"])
    elif mutation == "stderr":
        records["reference-96-serial"]["stderr_byte_count"] = 1
    elif mutation == "payload":
        records["reference-96-serial"]["payload"]["cluster_id"] = "forbidden"
        records["reference-96-serial"]["progress"] = deepcopy(
            records["reference-96-serial"]["payload"]
        )
    else:
        matrix["cases"][0]["runtime_failure_telemetry"] = {"phase": "wrong_phase"}

    evaluation = evaluate_v300_attempt(load_v300_design(), records, matrix)

    assert evaluation.passed is False
    assert evaluation.disposition == "operational_failure"
    assert failed_gate in evaluation.failed_gates


def test_v300_cli_exposes_no_scientific_or_identity_override() -> None:
    text = (ROOT / "scripts" / "run_runtime_reliability_v300.py").read_text("utf-8")
    forbidden = (
        "--attempt-id",
        "--protocol-id",
        "--config",
        "--seed",
        "--clusters",
        "--workers",
        "--suite",
        "--threshold",
        "--root",
    )
    assert all(flag not in text for flag in forbidden)


def test_v300_execute_fails_before_freeze_without_creating_formal_root() -> None:
    design = load_v300_design()
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/run_runtime_reliability_v300.py"),
            "--execute",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["exception_class"] == "V300IntegrityError"
    assert not design.attempt_root.exists()
    assert not design.authorization_record.exists()


def test_v300_child_rejects_formal_seed_before_authorization(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/diagnose_v210_fit_spawn.py"),
            "--clusters",
            "96",
            "--workers",
            "1",
            "--repeat",
            "1",
            "--suite",
            "mixed",
            "--seed-root",
            str(V300_FORMAL_SEED_ROOT),
            "--cluster-prefix",
            "v300-formal-runtime",
            "--execution-profile",
            "v300-formal",
            "--progress-file",
            str(progress),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "fixed authorization record" in completed.stderr
    assert completed.stdout == ""
    assert not progress.exists()
    assert not load_v300_design().authorization_record.exists()


def test_v300_preflight_is_seed_free_across_freeze_lifecycle() -> None:
    design = load_v300_design()
    freeze_record_path = (
        ROOT / "reports" / "runtime_reliability_v3_0_freeze_record.json"
    )
    parent = subprocess.run(
        ("git", "rev-parse", "HEAD^"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    is_unlaunched_freeze = (
        freeze_record_path.is_file()
        and json.loads(freeze_record_path.read_text("utf-8")).get(
            "implementation_commit"
        )
        == parent
        and not design.attempt_root.exists()
    )

    before = np.random.get_state()
    if is_unlaunched_freeze:
        report = preflight_v300(ROOT)
        assert report.status in {"ready", "ready_pending_authorization"}
        assert not design.attempt_root.exists()
    else:
        with pytest.raises(V300IntegrityError):
            preflight_v300(ROOT)
    after = np.random.get_state()

    assert _rng_equal(before, after)
