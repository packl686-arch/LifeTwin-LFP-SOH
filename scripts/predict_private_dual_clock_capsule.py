from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.experiments.private_dual_clock_prior_v3 import (
    predict_private_dual_clock_prior_capsule,
)
from lifetwin.private_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    exclusive_run_lock,
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, float_precision="round_trip")
    raise ValueError("Private prefix and forecast plan must be CSV or Parquet")


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite private inference artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional replay."
        )


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    metadata_path = Path(args.metadata_output)
    _ensure_available([output, metadata_path], overwrite=args.overwrite)
    prefix = _read_table(Path(args.prefix))
    plan = _read_table(Path(args.forecast_plan))
    if "forecast_equivalent_full_cycles" not in plan.columns:
        raise ValueError("Forecast plan lacks forecast_equivalent_full_cycles")
    elapsed = (
        plan["forecast_elapsed_days"].to_numpy(dtype=float)
        if "forecast_elapsed_days" in plan.columns
        else None
    )
    capsule = json.loads(Path(args.capsule).read_text(encoding="utf-8"))
    predictions, metadata = predict_private_dual_clock_prior_capsule(
        prefix,
        plan["forecast_equivalent_full_cycles"].to_numpy(dtype=float),
        capsule,
        forecast_elapsed_days=elapsed,
        strict_ood=not args.allow_diagnostic_ood,
    )
    atomic_write_csv(predictions, output)
    atomic_write_json(metadata, metadata_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a private dual-clock V3 model capsule on one canonical RPT "
            "prefix and a future EFC/time plan."
        )
    )
    parser.add_argument("capsule")
    parser.add_argument("prefix")
    parser.add_argument("forecast_plan")
    parser.add_argument("--output", default="private_predictions.csv")
    parser.add_argument("--metadata-output", default="private_prediction_metadata.json")
    parser.add_argument("--allow-diagnostic-ood", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with exclusive_run_lock(Path(args.output)):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
