from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _broken_markdown_links(project_root: Path) -> list[str]:
    broken: list[str] = []
    resolved_root = project_root.resolve()
    for markdown in project_root.rglob("*.md"):
        if ".git" in markdown.parts:
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
    for path in project_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(project_root).as_posix()
        if any(
            fnmatch.fnmatch(relative, pattern)
            for pattern in manifest["forbidden_release_globs"]
        ):
            forbidden_matches.append(relative)

    oversized = [
        path.relative_to(project_root).as_posix()
        for path in project_root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.stat().st_size > int(manifest["maximum_file_size_bytes"])
    ]
    broken_links = _broken_markdown_links(project_root)
    result = {
        "release_id": manifest["release_id"],
        "status": (
            "passed"
            if not mismatches
            and not forbidden_matches
            and not oversized
            and not broken_links
            else "failed"
        ),
        "frozen_file_count": len(manifest["frozen_files_sha256"]),
        "hash_mismatches": mismatches,
        "forbidden_files": sorted(forbidden_matches),
        "oversized_files": sorted(oversized),
        "broken_markdown_links": broken_links,
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
