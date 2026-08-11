from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.private_artifacts import (
    PrivateArtifactError,
    _atomic_replace,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
    run_lock_path,
    verify_completion_manifest,
)


def test_atomic_private_artifact_writers_and_completion_seal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "private-run"
    frame = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
    json_path = output / "value.json"
    csv_path = output / "value.csv"
    parquet_path = output / "value.parquet"
    atomic_write_json({"private_only": True}, json_path)
    atomic_write_csv(frame, csv_path)
    atomic_write_parquet(frame, parquet_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "private_only": True
    }
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)

    manifest = build_completion_manifest(
        output,
        {"json": json_path, "csv": csv_path, "parquet": parquet_path},
        metadata={"experiment_id": "fixture"},
    )
    seal_path = output / "run_complete.json"
    atomic_write_json(manifest, seal_path)
    result = verify_completion_manifest(
        output,
        json.loads(seal_path.read_text(encoding="utf-8")),
        required_names=["json", "csv", "parquet"],
    )
    assert result["status"] == "passed"
    assert result["artifact_count"] == 3

    csv_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(PrivateArtifactError, match="size changed|bytes changed"):
        verify_completion_manifest(output, manifest)


def test_private_run_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    output = tmp_path / "private-run"
    lock = run_lock_path(output)
    with exclusive_run_lock(output):
        assert lock.is_file()
        with pytest.raises(PrivateArtifactError, match="already locked"):
            with exclusive_run_lock(output):
                pass
    assert not lock.exists()

    with pytest.raises(RuntimeError, match="fixture failure"):
        with exclusive_run_lock(output):
            raise RuntimeError("fixture failure")
    assert not lock.exists()


def test_atomic_replace_preserves_old_target_on_writer_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "stable.txt"
    target.write_text("old", encoding="utf-8")

    def _failing_writer(temporary: Path) -> None:
        temporary.write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        _atomic_replace(target, _failing_writer)
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_completion_manifest_rejects_outside_artifact(tmp_path: Path) -> None:
    output = tmp_path / "private-run"
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    with pytest.raises(PrivateArtifactError, match="outside"):
        build_completion_manifest(output, {"outside": outside})
