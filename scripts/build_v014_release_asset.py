from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "artifacts/synthetic_long_horizon_identifiability_v1"
)
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "showcase/evidence_v014/synthetic_long_horizon_identifiability_v1/"
    "full_bundle_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "artifacts/lifetwin-v0.14.0-synthetic-long-horizon-full.zip"
)
ARCHIVE_ROOT = "synthetic_long_horizon_identifiability_v1"
FIXED_TIMESTAMP = (2026, 7, 22, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_entries(source: Path, manifest: dict[str, object]) -> list[Path]:
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Full-bundle manifest entries must be a list")

    expected_names: set[str] = set()
    verified: list[Path] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("Every full-bundle entry must be an object")
        name = raw.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Non-canonical full-bundle entry path: {name!r}")
        if name in expected_names:
            raise ValueError(f"Duplicate full-bundle entry: {name}")
        expected_names.add(name)

        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing full-bundle artifact: {path}")
        if path.stat().st_size != int(raw["byte_count"]):
            raise ValueError(f"Byte-count mismatch for {name}")
        if _sha256(path) != str(raw["sha256"]):
            raise ValueError(f"SHA-256 mismatch for {name}")
        verified.append(path)

    observed_names = {path.name for path in source.iterdir() if path.is_file()}
    if observed_names != expected_names:
        extra = sorted(observed_names - expected_names)
        missing = sorted(expected_names - observed_names)
        raise ValueError(
            f"Full-bundle file-set mismatch: extra={extra}, missing={missing}"
        )

    if len(verified) != int(manifest.get("file_count", -1)):
        raise ValueError("Full-bundle manifest file_count mismatch")
    if sum(path.stat().st_size for path in verified) != int(
        manifest.get("byte_count", -1)
    ):
        raise ValueError("Full-bundle manifest byte_count mismatch")
    return sorted(verified, key=lambda path: path.name)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build_release_asset(
    source: Path,
    manifest_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite release asset: {output}")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    entries = _verified_entries(source, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for path in entries:
                archive.writestr(
                    _zip_info(f"{ARCHIVE_ROOT}/{path.name}"),
                    path.read_bytes(),
                    compresslevel=9,
                )
            archive.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/full_bundle_manifest.json"),
                manifest_bytes,
                compresslevel=9,
            )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    digest = _sha256(output)
    checksum_path = output.with_suffix(f"{output.suffix}.sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {
        "status": "passed",
        "output": str(output),
        "byte_count": output.stat().st_size,
        "sha256": digest,
        "checksum_file": str(checksum_path),
        "source_file_count": len(entries),
        "archive_entry_count": len(entries) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic LifeTwin v0.14 full evidence asset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_release_asset(
        args.source.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
        overwrite=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
