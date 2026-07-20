from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.audits.phase1_adversarial import (
    run_phase1_adversarial_audit,
    write_phase1_audit_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/naumann_calendar_v3_activation_development.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/phase1_adversarial_audit_v1"
OUTPUT_NAMES = (
    "phase1_adversarial_audit.json",
    "data_condition_audit.csv",
    "future_label_attack_cases.csv",
    "independent_metric_audit.csv",
    "ablation_audit.csv",
    "gate_boundary_cases.csv",
    "failure_condition_table.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LifeTwin Phase 1 adversarial evidence audit."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace audit files with the same names; unrelated files are untouched.",
    )
    args = parser.parse_args()

    existing = [args.output_dir / name for name in OUTPUT_NAMES if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Audit output already exists ({names}); choose a new directory or use --overwrite"
        )

    observations = pd.read_csv(args.data)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = run_phase1_adversarial_audit(
        observations,
        config=config,
        data_path=args.data,
    )
    paths = write_phase1_audit_bundle(bundle, args.output_dir)
    output = {
        "audit_execution_status": bundle.summary["audit_execution_status"],
        "model_validation_status": bundle.summary["model_validation_status"],
        "output_files": {
            name: path.resolve().as_posix() for name, path in paths.items()
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if bundle.summary["audit_execution_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
