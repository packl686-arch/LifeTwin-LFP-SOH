from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import shutil
import sys
import uuid

import pandas as pd

from lifetwin.atomic_publish import (
    AtomicPublishRetryExhausted,
    publish_directory,
)
from lifetwin.data.geisbauer_calendar import (
    GEISBAUER_CALENDAR_MEMBER_SHA256,
    GEISBAUER_CALENDAR_OBSERVATIONS_SHA256,
    geisbauer_calendar_observations_sha256,
    load_geisbauer_calendar_observations,
)
from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
)
from lifetwin.experiments.geisbauer_robustness_audit import (
    run_geisbauer_robustness_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_TARGET = Path("data/external/geisbauer_2022/LFP_Data.csv")
DEFAULT_EXTERNAL_PROTOCOL = Path(
    "configs/experiments/geisbauer_lfp_calendar_external_stress.json"
)
DEFAULT_AUDIT_PROTOCOL = Path(
    "configs/experiments/geisbauer_lfp_calendar_robustness_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/geisbauer_robustness_audit_v1")
EXPECTED_SOURCE_INPUT_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)
EXPECTED_TARGET_INPUT_SHA256 = GEISBAUER_CALENDAR_MEMBER_SHA256
OUTPUT_FILES = {
    "result": "result.json",
    "cell_paired_deltas": "cell_paired_deltas.csv",
    "cell_day_paired_deltas": "cell_day_paired_deltas.csv",
    "stratum_diagnostics": "stratum_diagnostics.csv",
    "leave_one_cell_out": "leave_one_cell_out.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/lifetwin/atomic_publish.py",
        PROJECT_ROOT / "src/lifetwin/data/naumann.py",
        PROJECT_ROOT / "src/lifetwin/data/geisbauer_calendar.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v2.py",
        PROJECT_ROOT / "src/lifetwin/models/calendar_v3_activation.py",
        PROJECT_ROOT / "src/lifetwin/experiments/geisbauer_external_stress.py",
        PROJECT_ROOT / "src/lifetwin/experiments/geisbauer_robustness_audit.py",
    )
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths
    }
    encoded = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "source_sha256": source_hashes,
        "source_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "python": sys.version,
        "packages": {
            package: importlib_metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return {"row_count": len(frame), "sha256": _sha256(path)}


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(
    source_path: Path,
    target_path: Path,
    external_protocol_path: Path,
    audit_protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            "The Geisbauer robustness runner never overwrites an evidence "
            f"directory: {output_dir}"
        )
    source_file_sha256 = _sha256(source_path)
    if source_file_sha256 != EXPECTED_SOURCE_INPUT_SHA256:
        raise ValueError(
            "Naumann source input SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_INPUT_SHA256}, found {source_file_sha256}"
        )
    target_file_sha256 = _sha256(target_path)
    if target_file_sha256 != EXPECTED_TARGET_INPUT_SHA256:
        raise ValueError(
            "Geisbauer target input SHA-256 mismatch: "
            f"expected {EXPECTED_TARGET_INPUT_SHA256}, found {target_file_sha256}"
        )
    source_observations = pd.read_csv(source_path)
    source_outcome_sha256 = canonical_naumann_outcome_sha256(source_observations)
    if source_outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError("Naumann source canonical outcome snapshot mismatch")
    target_observations, target_adapter_audit = load_geisbauer_calendar_observations(
        target_path
    )
    target_adapter_audit["source"]["provided_path"] = target_path.as_posix()
    target_outcome_sha256 = geisbauer_calendar_observations_sha256(target_observations)
    if target_outcome_sha256 != GEISBAUER_CALENDAR_OBSERVATIONS_SHA256:
        raise ValueError("Geisbauer adapter canonical fingerprint mismatch")

    external_protocol = json.loads(external_protocol_path.read_text(encoding="utf-8"))
    audit_protocol = json.loads(audit_protocol_path.read_text(encoding="utf-8"))
    result, cell_deltas, cell_day_deltas, strata, leave_one_out = (
        run_geisbauer_robustness_audit(
            source_observations,
            target_observations,
            external_protocol=external_protocol,
            audit_protocol=audit_protocol,
        )
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        frames = {
            "cell_paired_deltas": cell_deltas,
            "cell_day_paired_deltas": cell_day_deltas,
            "stratum_diagnostics": strata,
            "leave_one_cell_out": leave_one_out,
        }
        artifacts: dict[str, object] = {}
        for name, frame in frames.items():
            metadata = _write_csv(frame, staging / OUTPUT_FILES[name])
            artifacts[name] = {
                "path": (output_dir / OUTPUT_FILES[name]).as_posix(),
                **metadata,
            }
        result["provenance"] = {
            "source_path": source_path.as_posix(),
            "source_input_file_sha256": source_file_sha256,
            "source_canonical_outcome_sha256": source_outcome_sha256,
            "target_path": target_path.as_posix(),
            "target_input_file_sha256": target_file_sha256,
            "target_canonical_observations_sha256": target_outcome_sha256,
            "target_adapter_audit": target_adapter_audit,
            "external_protocol_path": external_protocol_path.as_posix(),
            "external_protocol_file_sha256": _sha256(external_protocol_path),
            "audit_protocol_path": audit_protocol_path.as_posix(),
            "audit_protocol_file_sha256": _sha256(audit_protocol_path),
            **_source_provenance(),
        }
        result["artifacts"] = artifacts
        _write_json(result, staging / OUTPUT_FILES["result"])
        publish_directory(staging, output_dir)
    except AtomicPublishRetryExhausted:
        raise
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the retrospective physical-cell robustness audit of the 120-day, "
            "60 C Geisbauer external stress check. This is not long-term validation."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--external-protocol", type=Path, default=DEFAULT_EXTERNAL_PROTOCOL
    )
    parser.add_argument("--audit-protocol", type=Path, default=DEFAULT_AUDIT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        args.source,
        args.target,
        args.external_protocol,
        args.audit_protocol,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "result": (args.output_dir / OUTPUT_FILES["result"]).as_posix(),
                "status": result["status"],
                "design_status": result["design_status"],
                "overall_paired_diagnostic": result["overall_paired_diagnostic"],
                "leave_one_cell_out": result["leave_one_cell_out"],
                "negative_transfer_diagnosis": result["negative_transfer_diagnosis"],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
