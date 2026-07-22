from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_v014_release_asset import (
    ARCHIVE_ROOT,
    build_release_asset,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    payloads = {"a.csv": b"a,b\n1,2\n", "result.json": b'{"status":"failure"}\n'}
    entries = []
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)
        entries.append(
            {"path": name, "byte_count": len(payload), "sha256": _sha256(payload)}
        )
    manifest = {
        "entries": entries,
        "file_count": len(entries),
        "byte_count": sum(len(payload) for payload in payloads.values()),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return source, manifest_path


def test_release_asset_is_deterministic_and_self_describing(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = build_release_asset(source, manifest, first)
    second_result = build_release_asset(source, manifest, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["sha256"] == second_result["sha256"]
    assert first.with_suffix(".zip.sha256").is_file()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            f"{ARCHIVE_ROOT}/a.csv",
            f"{ARCHIVE_ROOT}/result.json",
            f"{ARCHIVE_ROOT}/full_bundle_manifest.json",
        ]


def test_release_asset_rejects_source_drift_and_overwrite(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path)
    output = tmp_path / "asset.zip"
    build_release_asset(source, manifest, output)
    with pytest.raises(FileExistsError):
        build_release_asset(source, manifest, output)

    (source / "a.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Byte-count mismatch"):
        build_release_asset(source, manifest, tmp_path / "drifted.zip")
