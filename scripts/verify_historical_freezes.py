"""Verify historical formal freezes directly from immutable Git objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


_FULL_GIT_HASH = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return completed.stdout.strip()


def freeze_commit_for_record(root: Path, record_path: Path) -> str:
    relative = record_path.resolve().relative_to(root.resolve()).as_posix()
    commits = _git(
        root,
        "log",
        "--diff-filter=A",
        "--format=%H",
        "--",
        relative,
    ).splitlines()
    if len(commits) != 1 or _FULL_GIT_HASH.fullmatch(commits[0]) is None:
        raise RuntimeError(f"Freeze record has no unique creation commit: {relative}")
    return commits[0]


def current_checkout_is_freeze(root: Path, record_path: Path) -> bool:
    return _git(root, "rev-parse", "HEAD") == freeze_commit_for_record(
        root,
        record_path,
    )


def _blob_bytes(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _source_tree_sha256(source_hashes: dict[str, str]) -> str:
    raw = (
        json.dumps(
            source_hashes,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def verify_freeze_record(root: Path, record_path: Path) -> dict[str, Any]:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    implementation_commit = payload.get("implementation_source_commit")
    source_hashes = payload.get("implementation_source_byte_hashes")
    if (
        not isinstance(implementation_commit, str)
        or _FULL_GIT_HASH.fullmatch(implementation_commit) is None
        or not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            for path, digest in source_hashes.items()
        )
    ):
        raise RuntimeError("Historical freeze record source identity is invalid")

    observed_hashes = {
        path: hashlib.sha256(_blob_bytes(root, implementation_commit, path)).hexdigest()
        for path in source_hashes
    }
    if observed_hashes != source_hashes:
        raise RuntimeError("Historical implementation source hash changed")
    if _source_tree_sha256(observed_hashes) != payload.get(
        "implementation_source_tree_sha256"
    ):
        raise RuntimeError("Historical implementation source tree hash changed")

    freeze_commit = freeze_commit_for_record(root, record_path)
    if _git(root, "rev-parse", f"{freeze_commit}^") != implementation_commit:
        raise RuntimeError("Historical freeze is not a direct implementation child")
    changed = sorted(
        line
        for line in _git(
            root,
            "diff",
            "--name-only",
            implementation_commit,
            freeze_commit,
        ).splitlines()
        if line
    )
    metadata_paths = payload.get("execution_metadata_paths")
    if not isinstance(metadata_paths, list) or changed != sorted(metadata_paths):
        raise RuntimeError("Historical freeze metadata allowlist changed")

    bound_files = (
        (payload.get("amendment_path"), payload.get("amendment_byte_sha256")),
        (payload.get("preregistration_path"), payload.get("preregistration_byte_sha256")),
        (payload.get("environment_lock_path"), payload.get("environment_lock_byte_sha256")),
        (payload.get("implementation_audit_path"), payload.get("implementation_audit_byte_sha256")),
    )
    for path, expected in bound_files:
        if (
            not isinstance(path, str)
            or not isinstance(expected, str)
            or hashlib.sha256(_blob_bytes(root, freeze_commit, path)).hexdigest()
            != expected
        ):
            raise RuntimeError("Historical freeze input hash changed")

    return {
        "status": "passed",
        "protocol_id": payload.get("protocol_id"),
        "freeze_commit": freeze_commit,
        "implementation_commit": implementation_commit,
        "source_file_count": len(source_hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = [
        verify_freeze_record(root, (root / record).resolve())
        for record in args.records
    ]
    print(json.dumps({"status": "passed", "records": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
