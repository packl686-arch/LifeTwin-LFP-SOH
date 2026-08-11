"""Audit Hithium private prediction prefixes without accepting truth vaults."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from lifetwin.validation.private_cycle_adapter import (
    validate_private_cycle_adapter_config,
    validate_private_cycle_bundle_manifest,
    validate_private_cycle_partition_manifest,
    verify_private_cycle_bundle_frame,
)
from lifetwin.validation.private_prefix_readiness import (
    audit_private_prefix_readiness,
    validate_private_prefix_readiness_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_CONFIG = (
    ROOT / "configs/validation/hithium_private_cycle_adapter_v1.json"
)
DEFAULT_READINESS_CONFIG = (
    ROOT / "configs/validation/hithium_private_prefix_readiness_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Private readiness input must be CSV or Parquet")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty private readiness output: {output}"
        )
    adapter = validate_private_cycle_adapter_config(
        _load_json(Path(args.adapter_config))
    )
    readiness = validate_private_prefix_readiness_config(
        _load_json(Path(args.readiness_config))
    )
    partition = validate_private_cycle_partition_manifest(
        _load_json(Path(args.partition_manifest)), adapter
    )
    bundle = validate_private_cycle_bundle_manifest(
        _load_json(Path(args.bundle_manifest)), adapter
    )
    development = _load_table(Path(args.development_trajectories))
    calibration = _load_table(Path(args.calibration_prefixes))
    locked = _load_table(Path(args.locked_test_prefixes))
    for name, frame in (
        ("development_trajectories", development),
        ("calibration_prefixes", calibration),
        ("locked_test_prefixes", locked),
    ):
        verify_private_cycle_bundle_frame(name, frame, bundle)
    drift, decision = audit_private_prefix_readiness(
        development,
        calibration,
        locked,
        partition,
        adapter,
        readiness,
    )
    output.mkdir(parents=True, exist_ok=True)
    drift_path = output / "prefix_support_distances.private.csv"
    decision_path = output / "readiness_decision.private.json"
    _write_csv(drift, drift_path)
    _write_json(decision, decision_path)
    _write_json(
        {
            "schema_version": "lifetwin.private_prefix_readiness.manifest.v1",
            "audit_id": readiness["audit_id"],
            "private_only": True,
            "truth_vault_inputs_read": False,
            "public_release_permitted": False,
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "byte_count": path.stat().st_size,
                }
                for path in (drift_path, decision_path)
            },
        },
        output / "readiness_manifest.private.json",
    )
    print(
        json.dumps(
            {
                "audit_id": readiness["audit_id"],
                "ready_to_issue_predictions": decision["ready_to_issue_predictions"],
                "truth_vault_inputs_read": False,
                "locked_test_truth_may_be_opened": False,
                "next_action": decision["next_action"],
                "private_output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_trajectories")
    parser.add_argument("calibration_prefixes")
    parser.add_argument("locked_test_prefixes")
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--adapter-config", default=str(DEFAULT_ADAPTER_CONFIG))
    parser.add_argument("--readiness-config", default=str(DEFAULT_READINESS_CONFIG))
    parser.add_argument("--output-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
