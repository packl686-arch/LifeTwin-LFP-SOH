from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from lifetwin.private_artifacts import verify_completion_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_enterprise_dry_run_stops_before_locked_truth(tmp_path: Path) -> None:
    output = tmp_path / "dry_run"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_private_enterprise_synthetic_dry_run.py"),
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "dry_run_summary.json").read_text())
    assert summary["synthetic_only"] is True
    assert summary["hithium_data_accessed"] is False
    assert summary["locked_test_truth_opened"] is False
    assert summary["prefix_readiness_passed_before_truth_access"] is True
    assert summary["readiness_truth_vault_inputs_read"] is False
    assert summary["fallback_to_v3"] is True
    completion = json.loads((output / "dry_run_complete.json").read_text())
    verified = verify_completion_manifest(output, completion)
    assert verified["artifact_count"] == 14
