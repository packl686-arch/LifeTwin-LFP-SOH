from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.fastcharge_portability import (
    build_fastcharge_prediction_inputs,
    prepare_fastcharge_portability_cycles,
)
from lifetwin.experiments.fastcharge_safe_prior_v2 import (
    load_fastcharge_safe_prior_v2_config,
    predict_fastcharge_safe_prior_v2,
    score_fastcharge_safe_prior_v2,
)


DEFAULT_CONFIG = Path("configs/experiments/fastcharge_lfp_safe_prior_v2.json")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/fastcharge-safe-prior-v2")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _paths(output_directory: Path) -> dict[str, Path]:
    return {
        "cycles": output_directory / "canonical_cycles.parquet",
        "data_audit": output_directory / "data_audit.json",
        "training": output_directory / "training_cycles.parquet",
        "prefixes": output_directory / "target_prefixes.parquet",
        "input_audit": output_directory / "prediction_input_audit.json",
        "calibration": output_directory / "calibration_quantiles.csv",
        "qualification": output_directory / "safe_pool_qualification.csv",
        "predictions": output_directory / "predictions.parquet",
        "manifest": output_directory / "prediction_manifest.json",
        "scores": output_directory / "scores.csv",
        "summary": output_directory / "score_summary.json",
    }


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite FastCharge safe-prior V2 artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional rerun."
        )


def _prepare(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    generated = [
        paths["cycles"],
        paths["data_audit"],
        paths["training"],
        paths["prefixes"],
        paths["input_audit"],
    ]
    _ensure_available(generated, overwrite=args.overwrite)
    config = load_fastcharge_safe_prior_v2_config(args.config)
    raw_cycles = pd.read_parquet(args.raw_cycles)
    cycles, data_audit = prepare_fastcharge_portability_cycles(
        raw_cycles,
        args.authoritative_crosswalk,
        config,
    )
    training, prefixes, input_audit = build_fastcharge_prediction_inputs(
        cycles,
        config,
    )
    _write_parquet(cycles, paths["cycles"])
    _write_json(data_audit, paths["data_audit"])
    _write_parquet(training, paths["training"])
    _write_parquet(prefixes, paths["prefixes"])
    _write_json(input_audit, paths["input_audit"])
    print(
        json.dumps(
            {"data_audit": data_audit, "prediction_input_audit": input_audit},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _predict(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    generated = [
        paths["calibration"],
        paths["qualification"],
        paths["predictions"],
        paths["manifest"],
    ]
    _ensure_available(generated, overwrite=args.overwrite)
    config = load_fastcharge_safe_prior_v2_config(args.config)
    training = pd.read_parquet(args.training_cycles)
    prefixes = pd.read_parquet(args.target_prefixes)
    predictions, manifest, calibration, qualification = (
        predict_fastcharge_safe_prior_v2(training, prefixes, config)
    )
    _write_csv(calibration, paths["calibration"])
    _write_csv(qualification, paths["qualification"])
    _write_parquet(predictions, paths["predictions"])
    _write_json(manifest, paths["manifest"])
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    generated = [paths["scores"], paths["summary"]]
    _ensure_available(generated, overwrite=args.overwrite)
    config = load_fastcharge_safe_prior_v2_config(args.config)
    cycles = pd.read_parquet(args.canonical_cycles)
    predictions = pd.read_parquet(args.predictions)
    manifest = json.loads(Path(args.prediction_manifest).read_text(encoding="utf-8"))
    scores, summary = score_fastcharge_safe_prior_v2(
        cycles,
        predictions,
        manifest,
        config,
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen FastCharge V2 safe-prior protocol in explicit "
            "prepare, predict, and score phases"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("raw_cycles")
    prepare.add_argument("authoritative_crosswalk")
    _common(prepare)
    prepare.set_defaults(handler=_prepare)

    predict = subparsers.add_parser("predict")
    predict.add_argument("training_cycles")
    predict.add_argument("target_prefixes")
    _common(predict)
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument("canonical_cycles")
    score.add_argument("predictions")
    score.add_argument("prediction_manifest")
    _common(score)
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
