"""Shared V0.20 file registries for checkpoint production and verification."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


class V020CheckpointRegistryError(ValueError):
    """Raised when a checkpoint registry or a bound input byte changes."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMON_TRAINING_INPUTS = (
    "actual_analysis_hash_ledger_commitment.json",
    "fit_commitment.json",
    "forecast_coordinates.csv",
    "generation_plan_commitment.json",
    "member_fit_diagnostics.csv",
    "member_forecast_bundle.csv",
    "operating_pack.csv",
    "prefix_pack.csv",
    "truth_commitments.json",
)

CENTER_INPUT_FILENAMES = tuple(
    sorted((*_COMMON_TRAINING_INPUTS, "center_development_truth.csv"))
)
RISK_INPUT_FILENAMES = tuple(
    sorted(
        (
            *_COMMON_TRAINING_INPUTS,
            "center_state_checkpoint.json",
            "risk_development_truth.csv",
        )
    )
)
CALIBRATION_INPUT_FILENAMES = tuple(
    sorted(
        (
            *_COMMON_TRAINING_INPUTS,
            "center_state_checkpoint.json",
            "risk_state_checkpoint.json",
            "training_manifest.json",
            "calibration_mask_commitment.json",
            "calibration_truth.csv",
        )
    )
)

INPUT_FILENAMES_BY_STAGE = MappingProxyType(
    {
        "center_development": CENTER_INPUT_FILENAMES,
        "risk_development": RISK_INPUT_FILENAMES,
        "calibration": CALIBRATION_INPUT_FILENAMES,
    }
)

REVEAL_PREREQUISITES = MappingProxyType(
    {
        "risk_truth_opened": (
            "exposure_log.jsonl",
            "truth_commitments.json",
            "generation_plan_commitment.json",
            "actual_analysis_hash_ledger_commitment.json",
            "fit_commitment.json",
            "center_state_checkpoint.json",
        ),
        "calibration_truth_opened": (
            "exposure_log.jsonl",
            "truth_commitments.json",
            "generation_plan_commitment.json",
            "actual_analysis_hash_ledger_commitment.json",
            "fit_commitment.json",
            "center_state_checkpoint.json",
            "risk_state_checkpoint.json",
            "calibration_mask_commitment.json",
        ),
        "scoring_truth_opened": (
            "exposure_log.jsonl",
            "truth_commitments.json",
            "generation_plan_commitment.json",
            "actual_analysis_hash_ledger_commitment.json",
            "fit_commitment.json",
            "center_state_checkpoint.json",
            "risk_state_checkpoint.json",
            "calibration_mask_commitment.json",
            "prediction_commitment.json",
        ),
    }
)


def _registered_file(*, label_root: Path, sealed_root: Path, filename: str) -> Path:
    root = sealed_root if filename.endswith("_truth.csv") else label_root
    physical_root = root.resolve(strict=True)
    path = (physical_root / filename).resolve(strict=True)
    if path.parent != physical_root or not path.is_file():
        raise V020CheckpointRegistryError(f"unsafe registered input: {filename}")
    return path


def registered_input_hashes_v020(
    stage: str,
    *,
    label_root: str | Path,
    sealed_root: str | Path,
) -> dict[str, str]:
    """Hash exactly the registry shared by the producer and verifier."""

    try:
        filenames = INPUT_FILENAMES_BY_STAGE[stage]
    except KeyError as exc:
        raise V020CheckpointRegistryError(f"unknown checkpoint stage: {stage}") from exc
    label = Path(label_root)
    sealed = Path(sealed_root)
    return {
        filename: hashlib.sha256(
            _registered_file(
                label_root=label,
                sealed_root=sealed,
                filename=filename,
            ).read_bytes()
        ).hexdigest()
        for filename in filenames
    }


def verify_registered_input_hashes_v020(
    stage: str,
    value: object,
    *,
    label_root: str | Path,
    sealed_root: str | Path,
) -> dict[str, str]:
    """Fail closed on key-set drift or any post-checkpoint byte change."""

    try:
        filenames = INPUT_FILENAMES_BY_STAGE[stage]
    except KeyError as exc:
        raise V020CheckpointRegistryError(f"unknown checkpoint stage: {stage}") from exc
    if not isinstance(value, Mapping) or set(value) != set(filenames):
        raise V020CheckpointRegistryError(f"{stage} input registry changed")
    for filename in filenames:
        digest = value[filename]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise V020CheckpointRegistryError(f"invalid input hash: {filename}")
    observed = registered_input_hashes_v020(
        stage,
        label_root=label_root,
        sealed_root=sealed_root,
    )
    for filename in filenames:
        if observed[filename] != value[filename]:
            raise V020CheckpointRegistryError(
                f"input changed after checkpoint: {filename}"
            )
    return observed


__all__ = [
    "CALIBRATION_INPUT_FILENAMES",
    "CENTER_INPUT_FILENAMES",
    "INPUT_FILENAMES_BY_STAGE",
    "REVEAL_PREREQUISITES",
    "RISK_INPUT_FILENAMES",
    "V020CheckpointRegistryError",
    "registered_input_hashes_v020",
    "verify_registered_input_hashes_v020",
]
