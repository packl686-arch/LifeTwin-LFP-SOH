from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.calendar_landmark_readiness import (
    run_landmark_readiness,
    validate_landmark_readiness_protocol,
)


DEFAULT_OBSERVATIONS = Path("data/interim/naumann_calendar_observations.csv")
DEFAULT_V3_CONFIG = Path(
    "configs/experiments/naumann_calendar_v3_activation_development.json"
)
DEFAULT_PROTOCOL = Path(
    "configs/experiments/naumann_calendar_landmark_readiness.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/calendar_landmark_readiness_v1")
EXPECTED_OBSERVATIONS_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_new_outputs(paths: list[Path]) -> None:
    existing = [path.as_posix() for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "The landmark runner never overwrites evidence artifacts; "
            f"existing={existing}"
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return {
        "path": path.as_posix(),
        "row_count": len(frame),
        "sha256": _sha256(path),
    }


def _write_json(payload: dict[str, object], path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def run(
    observations_path: Path,
    v3_config_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_paths = {
        "metrics": output_dir / "common_support_metrics.csv",
        "summary": output_dir / "landmark_summary.csv",
        "decision": output_dir / "decision.json",
    }
    _require_new_outputs(list(output_paths.values()))
    observations_file_sha256 = _sha256(observations_path)
    if observations_file_sha256 != EXPECTED_OBSERVATIONS_SHA256:
        raise ValueError(
            "Naumann observation SHA-256 mismatch: "
            f"expected {EXPECTED_OBSERVATIONS_SHA256}, "
            f"found {observations_file_sha256}"
        )

    protocol = validate_landmark_readiness_protocol(
        json.loads(protocol_path.read_text(encoding="utf-8"))
    )
    v3_config = json.loads(v3_config_path.read_text(encoding="utf-8"))
    observations = pd.read_csv(observations_path)
    metrics, summary, decision = run_landmark_readiness(
        observations,
        v3_config=v3_config,
        protocol=protocol,
    )
    decision["provenance"] = {
        "observations_path": observations_path.as_posix(),
        "observations_file_sha256": observations_file_sha256,
        "v3_config_path": v3_config_path.as_posix(),
        "v3_config_file_sha256": _sha256(v3_config_path),
        "protocol_path": protocol_path.as_posix(),
        "protocol_file_sha256": _sha256(protocol_path),
    }
    decision["artifacts"] = {
        "common_support_metrics": _write_csv(metrics, output_paths["metrics"]),
        "landmark_summary": _write_csv(summary, output_paths["summary"]),
    }
    _write_json(decision, output_paths["decision"])
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score frozen Calendar V3 predictions at identical future support for "
            "a retrospective landmark-readiness diagnostic."
        )
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--v3-config", type=Path, default=DEFAULT_V3_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    decision = run(
        args.observations,
        args.v3_config,
        args.protocol,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "decision": (args.output_dir / "decision.json").as_posix(),
                "status": decision["status"],
                "retrospective_signal_landmark": decision[
                    "retrospective_signal_landmark"
                ],
                "confirmed_earliest_landmark": decision[
                    "confirmed_earliest_landmark"
                ],
                "confirmation_status": decision["confirmation_status"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
