from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.private_schedule_scenarios import (
    build_private_schedule_scenarios,
)
from lifetwin.private_artifacts import (
    atomic_write_json,
    atomic_write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
)


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build sealed outcome-free private operating scenarios."
    )
    parser.add_argument("target_prefixes")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Scenario output directory is not empty")
    with exclusive_run_lock(output):
        schedules, manifest = build_private_schedule_scenarios(
            pd.read_parquet(args.target_prefixes),
            _json(args.model_config),
            _json(args.scenario_config),
        )
        artifacts: dict[str, Path] = {}
        for scenario_id, schedule in schedules.items():
            path = output / f"forecast_schedule.{scenario_id}.private.parquet"
            atomic_write_parquet(schedule, path)
            artifacts[f"schedule_{scenario_id}"] = path
        manifest_path = output / "scenario_bundle_manifest.private.json"
        atomic_write_json(manifest, manifest_path)
        artifacts["manifest"] = manifest_path
        completion = build_completion_manifest(
            output,
            artifacts,
            metadata={
                "operation": "outcome_free_schedule_scenario_construction",
                "truth_vault_opened": False,
                "public_release_permitted": False,
            },
        )
        atomic_write_json(completion, output / "scenario_bundle_complete.private.json")
    print(
        json.dumps(
            {
                "scenario_count": manifest["scenario_count"],
                "truth_vault_opened": False,
                "manifest_content_sha256": manifest["manifest_content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
