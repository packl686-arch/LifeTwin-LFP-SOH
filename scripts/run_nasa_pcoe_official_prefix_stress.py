from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.nasa_official_prefix_stress import (
    NasaOfficialPrefixStressError,
    ensure_execution_authorized,
    execute_score_once,
    load_nasa_official_prefix_stress_config,
    predict_prefix_baselines,
    prepare_prefix_and_future_labels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/nasa_pcoe_official_bundle_prefix_stress_v1.json"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen official NASA auxiliary prefix stress protocol."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-gate")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("cycles", type=Path)
    prepare.add_argument("output_directory", type=Path)
    predict = subparsers.add_parser("predict")
    predict.add_argument("prefix_table", type=Path)
    predict.add_argument("output_directory", type=Path)
    score = subparsers.add_parser("score")
    score.add_argument("future_labels", type=Path)
    score.add_argument("predictions", type=Path)
    score.add_argument("prediction_manifest", type=Path)
    score.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    try:
        config = load_nasa_official_prefix_stress_config(args.config)
        ensure_execution_authorized(config)
        if args.command == "check-gate":
            print(json.dumps({"execution_allowed": True}))
            return 0
        if args.command == "prepare":
            cycles = pd.read_csv(args.cycles, float_precision="round_trip")
            prefixes, labels, audit = prepare_prefix_and_future_labels(cycles, config)
            _write_csv(args.output_directory / "prefix_inputs.csv", prefixes)
            _write_csv(args.output_directory / "future_labels.csv", labels)
            _write_json(args.output_directory / "prepare_audit.json", audit)
            return 0
        if args.command == "predict":
            prefixes = pd.read_csv(args.prefix_table, float_precision="round_trip")
            predictions, manifest = predict_prefix_baselines(prefixes, config)
            _write_csv(args.output_directory / "predictions.csv", predictions)
            _write_json(args.output_directory / "prediction_manifest.json", manifest)
            return 0
        labels = pd.read_csv(args.future_labels, float_precision="round_trip")
        predictions = pd.read_csv(args.predictions, float_precision="round_trip")
        manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
        execute_score_once(labels, predictions, manifest, config, args.output_directory)
        return 0
    except NasaOfficialPrefixStressError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "execution_allowed": False,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
