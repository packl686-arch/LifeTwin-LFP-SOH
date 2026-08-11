from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import canonical_frame_sha256
from lifetwin.experiments.private_cycle_prior_v2 import (
    default_private_cycle_prior_v2_config,
    predict_private_cycle_prior_v2,
    score_private_cycle_prior_v2,
    train_private_cycle_prior_capsule,
    validate_private_cycle_prior_v2_config,
)
from lifetwin.private_artifacts import (
    atomic_write_csv as _write_csv,
    atomic_write_json as _write_json,
    atomic_write_parquet as _write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
)


DEFAULT_INPUT_DIRECTORY = Path("artifacts/snl-lfp-rpt-loco-v1")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/private-cycle-prior-v2")


def _paths(output_directory: Path) -> dict[str, Path]:
    return {
        "config": output_directory / "private_config.json",
        "predictions": output_directory / "predictions.parquet",
        "decisions": output_directory / "model_decisions.parquet",
        "decisions_csv": output_directory / "model_decisions.csv",
        "manifest": output_directory / "prediction_manifest.json",
        "scores": output_directory / "scores.csv",
        "summary": output_directory / "score_summary.json",
        "capsule": output_directory / "model_capsule.private.json",
        "complete": output_directory / "run_complete.json",
    }


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite private cycle-prior V2 artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional private replay."
        )


def _load_config(path: str | None) -> dict[str, object]:
    if path is None:
        return validate_private_cycle_prior_v2_config(
            default_private_cycle_prior_v2_config()
        )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_private_cycle_prior_v2_config(value)


def _predict(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    generated = [
        paths["config"],
        paths["predictions"],
        paths["decisions"],
        paths["decisions_csv"],
        paths["manifest"],
    ]
    _ensure_available(generated, overwrite=args.overwrite)
    config = _load_config(args.config)
    references = pd.read_parquet(args.references)
    prefixes = pd.read_parquet(args.target_prefixes)
    predictions, decisions, manifest = predict_private_cycle_prior_v2(
        references, prefixes, config
    )
    _write_json(config, paths["config"])
    _write_parquet(predictions, paths["predictions"])
    _write_parquet(decisions, paths["decisions"])
    _write_csv(decisions, paths["decisions_csv"])
    _write_json(manifest, paths["manifest"])
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    _ensure_available(
        [paths["scores"], paths["summary"]], overwrite=args.overwrite
    )
    config = _load_config(args.config)
    truth = pd.read_parquet(args.target_truth)
    predictions = pd.read_parquet(args.predictions)
    decisions = pd.read_parquet(args.model_decisions)
    manifest = json.loads(
        Path(args.prediction_manifest).read_text(encoding="utf-8")
    )
    scores, summary = score_private_cycle_prior_v2(
        truth, predictions, decisions, manifest, config
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _train_capsule(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    _ensure_available([paths["capsule"]], overwrite=args.overwrite)
    config = _load_config(args.config)
    trajectories = pd.read_parquet(args.rpt_trajectories)
    training_hash = canonical_frame_sha256(
        trajectories.loc[:, RPT_TRAJECTORY_COLUMNS], RPT_TRAJECTORY_COLUMNS
    )
    capsule = train_private_cycle_prior_capsule(
        trajectories,
        config,
        training_identity={
            "dataset_id": str(config["dataset_id"]),
            "canonical_rpt_trajectory_sha256": training_hash,
            "training_row_count": len(trajectories),
            "raw_measurements_embedded": False,
        },
    )
    _write_json(capsule, paths["capsule"])
    print(
        json.dumps(
            {
                "capsule_content_sha256": capsule["capsule_content_sha256"],
                "landmarks": sorted(capsule["landmark_models"]),
                "private_only": True,
                "public_release_permitted": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _run_all(args: argparse.Namespace) -> int:
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    _ensure_available(list(paths.values()), overwrite=args.overwrite)
    config = _load_config(args.config)

    # Prediction is completed and persisted before held-condition suffix truth is read.
    references = pd.read_parquet(input_directory / "outer_fold_references.parquet")
    prefixes = pd.read_parquet(input_directory / "target_prefixes.parquet")
    predictions, decisions, manifest = predict_private_cycle_prior_v2(
        references, prefixes, config
    )
    _write_json(config, paths["config"])
    _write_parquet(predictions, paths["predictions"])
    _write_parquet(decisions, paths["decisions"])
    _write_csv(decisions, paths["decisions_csv"])
    _write_json(manifest, paths["manifest"])

    truth = pd.read_parquet(input_directory / "target_truth.parquet")
    scores, summary = score_private_cycle_prior_v2(
        truth, predictions, decisions, manifest, config
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["summary"])

    trajectories = pd.read_parquet(input_directory / "rpt_trajectories.parquet")
    training_hash = canonical_frame_sha256(
        trajectories.loc[:, RPT_TRAJECTORY_COLUMNS], RPT_TRAJECTORY_COLUMNS
    )
    capsule = train_private_cycle_prior_capsule(
        trajectories,
        config,
        training_identity={
            "dataset_id": str(config["dataset_id"]),
            "canonical_rpt_trajectory_sha256": training_hash,
            "training_row_count": len(trajectories),
            "raw_measurements_embedded": False,
        },
    )
    _write_json(capsule, paths["capsule"])
    completed_artifacts = {
        name: path for name, path in paths.items() if name != "complete"
    }
    completion = build_completion_manifest(
        output_directory,
        completed_artifacts,
        metadata={
            "experiment_id": str(config["experiment_id"]),
            "dataset_id": str(config["dataset_id"]),
            "prediction_manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
            "score_summary_content_sha256": summary["summary_content_sha256"],
            "capsule_content_sha256": capsule["capsule_content_sha256"],
            "public_release_permitted": False,
        },
    )
    _write_json(completion, paths["complete"])
    print(
        json.dumps(
            {
                "score_summary": summary,
                "capsule_content_sha256": capsule["capsule_content_sha256"],
                "private_only": True,
                "public_release_permitted": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the private hierarchical cycle-prior V2 pipeline. All generated "
            "model and result artifacts remain private."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("references")
    predict.add_argument("target_prefixes")
    _common(predict)
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument("target_truth")
    score.add_argument("predictions")
    score.add_argument("model_decisions")
    score.add_argument("prediction_manifest")
    _common(score)
    score.set_defaults(handler=_score)

    train = subparsers.add_parser("train-capsule")
    train.add_argument("rpt_trajectories")
    _common(train)
    train.set_defaults(handler=_train_capsule)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--input-directory", default=str(DEFAULT_INPUT_DIRECTORY))
    _common(run_all)
    run_all.set_defaults(handler=_run_all)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with exclusive_run_lock(Path(args.output_directory)):
        return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
