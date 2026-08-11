"""Atomic local artifact primitives for private experiment runs.

The module contains no private measurements.  It prevents concurrent writers,
uses same-directory atomic replacement, and seals completed artifact sets with
byte hashes so interrupted runs cannot be mistaken for finished evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence
from uuid import uuid4

import pandas as pd

from lifetwin.experiments.nasa_prefix_loco import canonical_json_sha256


class PrivateArtifactError(RuntimeError):
    """Raised when a private run lock or artifact seal is invalid."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(value: object, path: str | Path) -> None:
    target = Path(path)

    def _write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    _atomic_replace(target, _write)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)

    def _write(temporary: Path) -> None:
        frame.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )

    _atomic_replace(target, _write)


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)

    def _write(temporary: Path) -> None:
        frame.to_parquet(temporary, index=False)

    _atomic_replace(target, _write)


def run_lock_path(output_directory: str | Path) -> Path:
    output = Path(output_directory)
    return output.parent / f".{output.name}.run.lock"


@contextmanager
def exclusive_run_lock(output_directory: str | Path) -> Iterator[Path]:
    """Create an exclusive process marker beside an output directory."""
    lock = run_lock_path(output_directory)
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "lifetwin.private_run_lock.v1",
        "pid": os.getpid(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(Path(output_directory).resolve()),
    }
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        detail = lock.read_text(encoding="utf-8", errors="replace")
        raise PrivateArtifactError(
            f"Private run is already locked: {lock}; owner={detail}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield lock
    finally:
        lock.unlink(missing_ok=True)


def build_completion_manifest(
    output_directory: str | Path,
    artifacts: Mapping[str, str | Path],
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output = Path(output_directory).resolve()
    rows = []
    for name, raw_path in sorted(artifacts.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise PrivateArtifactError(f"Completion artifact is missing: {path}")
        try:
            relative = path.relative_to(output).as_posix()
        except ValueError as exc:
            raise PrivateArtifactError(
                f"Completion artifact is outside the output directory: {path}"
            ) from exc
        rows.append(
            {
                "name": str(name),
                "path": relative,
                "byte_count": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "lifetwin.private_run_completion.v1",
        "status": "complete",
        "private_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": rows,
        "metadata": dict(metadata or {}),
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def verify_completion_manifest(
    output_directory: str | Path,
    manifest: Mapping[str, object],
    *,
    required_names: Sequence[str] | None = None,
) -> dict[str, object]:
    value = json.loads(json.dumps(dict(manifest), allow_nan=False))
    if value.get("schema_version") != "lifetwin.private_run_completion.v1":
        raise PrivateArtifactError("Private completion schema changed")
    if value.get("status") != "complete" or value.get("private_only") is not True:
        raise PrivateArtifactError("Private run is not marked complete")
    expected_hash = str(value.pop("manifest_content_sha256", ""))
    if canonical_json_sha256(value) != expected_hash:
        raise PrivateArtifactError("Private completion manifest content changed")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PrivateArtifactError("Private completion artifact list is empty")
    names = [str(item["name"]) for item in artifacts]
    if len(set(names)) != len(names):
        raise PrivateArtifactError("Private completion artifact names are duplicated")
    if required_names is not None and set(names) != set(required_names):
        raise PrivateArtifactError("Private completion artifact membership changed")
    output = Path(output_directory).resolve()
    for item in artifacts:
        path = (output / str(item["path"])).resolve()
        try:
            path.relative_to(output)
        except ValueError as exc:
            raise PrivateArtifactError("Unsafe completion artifact path") from exc
        if not path.is_file():
            raise PrivateArtifactError(f"Completed artifact is missing: {path}")
        if path.stat().st_size != int(item["byte_count"]):
            raise PrivateArtifactError(f"Completed artifact size changed: {path}")
        if file_sha256(path) != str(item["sha256"]):
            raise PrivateArtifactError(f"Completed artifact bytes changed: {path}")
    return {
        "status": "passed",
        "artifact_count": len(artifacts),
        "manifest_content_sha256": expected_hash,
    }


__all__ = [
    "PrivateArtifactError",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_parquet",
    "build_completion_manifest",
    "exclusive_run_lock",
    "file_sha256",
    "run_lock_path",
    "verify_completion_manifest",
]
