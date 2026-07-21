from __future__ import annotations

import argparse
from datetime import date
import fnmatch
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlparse


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _version_consistency(
    project_root: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    try:
        if int(manifest.get("schema_version", 1)) < 2:
            return {"status": "not_applicable_legacy_manifest"}
        pyproject = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        citation = (project_root / "CITATION.cff").read_text(encoding="utf-8")
        package_init = (project_root / "src/lifetwin/__init__.py").read_text(
            encoding="utf-8"
        )
        citation_version_match = re.search(
            r"(?m)^version:\s*[\"']?([^\s\"']+)", citation
        )
        citation_date_match = re.search(
            r"(?m)^date-released:\s*[\"']?([^\s\"']+)", citation
        )
        package_version_match = re.search(
            r"(?m)^__version__\s*=\s*[\"']([^\"']+)", package_init
        )
        release_id = str(manifest["release_id"])
        manifest_version = release_id.rsplit("_v", 1)[-1]
        versions = {
            "manifest": manifest_version,
            "pyproject": str(pyproject["project"]["version"]),
            "citation": (
                citation_version_match.group(1) if citation_version_match else None
            ),
            "package": (
                package_version_match.group(1) if package_version_match else None
            ),
        }
        dates = {
            "manifest": str(manifest["release_date"]),
            "citation": (citation_date_match.group(1) if citation_date_match else None),
        }
        version_values = list(versions.values())
        date_values = list(dates.values())
        valid_dates = all(value is not None for value in date_values)
        if valid_dates:
            for value in date_values:
                date.fromisoformat(str(value))
        passed = (
            all(value is not None for value in version_values)
            and len(set(version_values)) == 1
            and valid_dates
            and len(set(date_values)) == 1
        )
        return {
            "status": "passed" if passed else "failed",
            "versions": versions,
            "release_dates": dates,
        }
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_tracked_files(project_root: Path) -> list[Path] | None:
    """Return the index contents, or ``None`` outside a Git work tree."""
    completed = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return [
        project_root / relative
        for relative in completed.stdout.decode(
            "utf-8", errors="surrogateescape"
        ).split("\0")
        if relative
    ]


def _release_files(project_root: Path) -> tuple[list[Path], str]:
    tracked = _git_tracked_files(project_root)
    if tracked is not None:
        return tracked, "git_tracked_files"
    return [
        path
        for path in project_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ], "filesystem_fallback"


def _is_canonical_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        return False
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    return (
        not posix_path.is_absolute()
        and not windows_path.drive
        and value == posix_path.as_posix()
        and bool(posix_path.parts)
        and all(part not in {".", ".."} for part in posix_path.parts)
    )


def _unfrozen_tracked_allowlist(
    manifest: dict[str, object],
    tracked_regular_files: set[str],
) -> tuple[set[str], list[dict[str, object]]]:
    raw_allowlist = manifest.get("unfrozen_tracked_allowlist", [])
    if not isinstance(raw_allowlist, list):
        return set(), [
            {
                "reason": "must_be_an_array",
                "value": raw_allowlist,
            }
        ]

    allowlist: set[str] = set()
    errors: list[dict[str, object]] = []
    for index, value in enumerate(raw_allowlist):
        if not _is_canonical_relative_path(value):
            errors.append(
                {
                    "index": index,
                    "path": value,
                    "reason": "not_a_canonical_relative_path",
                }
            )
            continue
        assert isinstance(value, str)
        if value in allowlist:
            errors.append(
                {
                    "index": index,
                    "path": value,
                    "reason": "duplicate_path",
                }
            )
            continue
        if value not in tracked_regular_files:
            errors.append(
                {
                    "index": index,
                    "path": value,
                    "reason": "not_a_tracked_regular_file",
                }
            )
            continue
        allowlist.add(value)
    return allowlist, errors


def _broken_markdown_links(project_root: Path, files: list[Path]) -> list[str]:
    broken: list[str] = []
    resolved_root = project_root.resolve()
    for markdown in files:
        if markdown.suffix.lower() != ".md" or not markdown.is_file():
            continue
        text = markdown.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path_text:
                continue
            destination = (markdown.parent / path_text).resolve()
            line = text.count("\n", 0, match.start()) + 1
            try:
                destination.relative_to(resolved_root)
            except ValueError:
                broken.append(
                    f"{markdown.relative_to(project_root).as_posix()}:{line}:"
                    f" link escapes repository: {target}"
                )
                continue
            if not destination.exists():
                broken.append(
                    f"{markdown.relative_to(project_root).as_posix()}:{line}: {target}"
                )
    return sorted(broken)


def verify(project_root: Path) -> dict[str, object]:
    manifest_path = project_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_files, scan_mode = _release_files(project_root)
    release_file_set = {
        path.relative_to(project_root).as_posix()
        for path in release_files
        if path.is_file()
    }
    mismatches: list[dict[str, object]] = []
    for relative_path, expected in manifest["frozen_files_sha256"].items():
        path = project_root / relative_path
        observed = _sha256(path) if path.is_file() else None
        if observed != expected:
            mismatches.append(
                {
                    "path": relative_path,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                }
            )

    forbidden_matches: list[str] = []
    for path in release_files:
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        if any(
            fnmatch.fnmatch(relative, pattern)
            for pattern in manifest["forbidden_release_globs"]
        ):
            forbidden_matches.append(relative)

    oversized = [
        path.relative_to(project_root).as_posix()
        for path in release_files
        if path.is_file()
        and path.stat().st_size > int(manifest["maximum_file_size_bytes"])
    ]
    broken_links = _broken_markdown_links(project_root, release_files)
    version_consistency = _version_consistency(project_root, manifest)
    manifest_tracked = (
        "release_manifest.json" in release_file_set
        if scan_mode == "git_tracked_files"
        else None
    )
    untracked_frozen = (
        sorted(set(manifest["frozen_files_sha256"]) - release_file_set)
        if scan_mode == "git_tracked_files"
        else []
    )
    freeze_gate_value = manifest.get("require_all_tracked_files_frozen", False)
    freeze_gate_errors: list[dict[str, object]] = []
    if not isinstance(freeze_gate_value, bool):
        freeze_gate_errors.append(
            {
                "field": "require_all_tracked_files_frozen",
                "reason": "must_be_boolean",
                "value": freeze_gate_value,
            }
        )
    freeze_gate_required = freeze_gate_value is True
    allowlist: set[str] = set()
    allowlist_errors: list[dict[str, object]] = []
    unfrozen_tracked: list[str] = []
    if freeze_gate_required:
        if scan_mode != "git_tracked_files":
            freeze_gate_errors.append(
                {
                    "field": "require_all_tracked_files_frozen",
                    "reason": "git_tracked_files_unavailable",
                }
            )
        else:
            allowlist, allowlist_errors = _unfrozen_tracked_allowlist(
                manifest,
                release_file_set,
            )
            unfrozen_tracked = sorted(
                release_file_set
                - set(manifest["frozen_files_sha256"])
                - {"release_manifest.json"}
                - allowlist
            )
    result = {
        "release_id": manifest["release_id"],
        "status": (
            "passed"
            if not mismatches
            and not forbidden_matches
            and not oversized
            and not broken_links
            and manifest_tracked is not False
            and not untracked_frozen
            and not freeze_gate_errors
            and not allowlist_errors
            and not unfrozen_tracked
            and version_consistency["status"]
            in {
                "passed",
                "not_applicable_legacy_manifest",
            }
            else "failed"
        ),
        "scan_mode": scan_mode,
        "scanned_file_count": len(release_file_set),
        "manifest_tracked": manifest_tracked,
        "frozen_file_count": len(manifest["frozen_files_sha256"]),
        "untracked_frozen_files": untracked_frozen,
        "require_all_tracked_files_frozen": freeze_gate_required,
        "tracked_file_freeze_gate_errors": freeze_gate_errors,
        "unfrozen_tracked_allowlist": sorted(allowlist),
        "unfrozen_tracked_allowlist_errors": allowlist_errors,
        "unfrozen_tracked_files": unfrozen_tracked,
        "hash_mismatches": mismatches,
        "forbidden_files": sorted(forbidden_matches),
        "oversized_files": sorted(oversized),
        "broken_markdown_links": broken_links,
        "version_consistency": version_consistency,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the public release bundle.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = verify(args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
