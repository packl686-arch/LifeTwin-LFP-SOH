from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.private_artifacts import (
    atomic_write_json,
    atomic_write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
)
from lifetwin.validation.private_cycle_adapter import (
    build_private_cycle_blind_bundle,
    freeze_private_cycle_partitions,
    normalize_private_cycle_measurements,
    validate_private_cycle_adapter_config,
    validate_private_cycle_partition_manifest,
)


DEFAULT_CONFIG = Path("configs/validation/hithium_private_cycle_adapter_v1.json")


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.casefold() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.casefold() == ".csv":
        return pd.read_csv(source)
    raise ValueError("Private adapter input must be CSV or Parquet")


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite private blind-bundle artifacts: "
            + ", ".join(existing)
        )


def _freeze(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    config_path = output / "adapter_config.private.json"
    manifest_path = output / "partition_manifest.private.json"
    complete_path = output / "partition_freeze_complete.private.json"
    _ensure_available(
        [config_path, manifest_path, complete_path], overwrite=args.overwrite
    )
    config = validate_private_cycle_adapter_config(_load_json(args.config))
    metadata = _load_table(args.metadata)
    manifest = freeze_private_cycle_partitions(metadata, config)
    atomic_write_json(config, config_path)
    atomic_write_json(manifest, manifest_path)
    completion = build_completion_manifest(
        output,
        {"config": config_path, "partition_manifest": manifest_path},
        metadata={
            "operation": "metadata_only_partition_freeze",
            "measurement_values_read": False,
            "target_outcomes_read": False,
            "public_release_permitted": False,
        },
    )
    atomic_write_json(completion, complete_path)
    print(
        json.dumps(
            {
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "group_counts": manifest["group_counts"],
                "cell_count": manifest["cell_count"],
                "measurement_values_read": False,
            },
            indent=2,
        )
    )
    return 0


def _build(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    config_path = Path(args.config)
    partition_path = Path(args.partition_manifest)
    paths = {
        "development_trajectories": output / "development_trajectories.private.parquet",
        "calibration_prefixes": output / "calibration_prefixes.private.parquet",
        "calibration_truth_vault": output / "calibration_truth_vault.private.parquet",
        "locked_test_prefixes": output / "locked_test_prefixes.private.parquet",
        "locked_test_truth_vault": output / "locked_test_truth_vault.private.parquet",
        "bundle_manifest": output / "blind_bundle_manifest.private.json",
        "complete": output / "blind_bundle_complete.private.json",
    }
    _ensure_available(list(paths.values()), overwrite=args.overwrite)
    config = validate_private_cycle_adapter_config(_load_json(config_path))
    partition_manifest = validate_private_cycle_partition_manifest(
        _load_json(partition_path), config
    )
    measurements = _load_table(args.measurements)
    normalized = normalize_private_cycle_measurements(
        measurements, partition_manifest, config
    )
    frames, bundle_manifest = build_private_cycle_blind_bundle(
        normalized, partition_manifest, config
    )
    for name, frame in frames.items():
        atomic_write_parquet(frame, paths[name])
    atomic_write_json(bundle_manifest, paths["bundle_manifest"])
    completion = build_completion_manifest(
        output,
        {name: path for name, path in paths.items() if name != "complete"},
        metadata={
            "operation": "private_blind_bundle_build",
            "partition_manifest_content_sha256": partition_manifest[
                "manifest_content_sha256"
            ],
            "bundle_manifest_content_sha256": bundle_manifest[
                "manifest_content_sha256"
            ],
            "prediction_inputs_contain_target_suffix_outcomes": False,
            "truth_vault_requires_separate_access_control": True,
            "public_release_permitted": False,
        },
    )
    atomic_write_json(completion, paths["complete"])
    print(
        json.dumps(
            {
                "bundle_manifest_content_sha256": bundle_manifest[
                    "manifest_content_sha256"
                ],
                "artifact_row_counts": bundle_manifest["artifact_row_counts"],
                "truth_vault_requires_separate_access_control": True,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and build a batch-disjoint private cycle-aging bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-partitions")
    freeze.add_argument("metadata")
    freeze.add_argument("--config", default=str(DEFAULT_CONFIG))
    freeze.add_argument("--output-directory", required=True)
    freeze.add_argument("--overwrite", action="store_true")
    freeze.set_defaults(handler=_freeze)
    build = subparsers.add_parser("build-bundle")
    build.add_argument("measurements")
    build.add_argument("--config", required=True)
    build.add_argument("--partition-manifest", required=True)
    build.add_argument("--output-directory", required=True)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(handler=_build)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with exclusive_run_lock(Path(args.output_directory)):
        return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
