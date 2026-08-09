from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import lifetwin.experiments.private_dual_clock_prior_v3 as v3_implementation
import lifetwin.models.hierarchical_cycle_prior as prior_implementation
from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import canonical_frame_sha256
from lifetwin.experiments.private_dual_clock_prior_v3 import (
    default_private_dual_clock_prior_v3_config,
    predict_private_dual_clock_prior_v3,
    score_private_dual_clock_prior_v3,
    train_private_dual_clock_prior_capsule,
    validate_private_dual_clock_prior_v3_config,
)
from lifetwin.private_artifacts import (
    atomic_write_csv as _write_csv,
    atomic_write_json as _write_json,
    atomic_write_parquet as _write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
    file_sha256,
    verify_completion_manifest,
)


DEFAULT_INPUT_DIRECTORY = Path("artifacts/snl-lfp-rpt-loco-v1")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/private-dual-clock-prior-v3")


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
            "Refusing to overwrite private dual-clock V3 artifacts: "
            + ", ".join(existing)
            + ". Pass --overwrite only for an intentional private replay."
        )


def _load_config(path: str | None) -> dict[str, object]:
    if path is None:
        return validate_private_dual_clock_prior_v3_config(
            default_private_dual_clock_prior_v3_config()
        )
    return validate_private_dual_clock_prior_v3_config(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _training_identity(
    trajectories: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset_id": str(config["dataset_id"]),
        "canonical_rpt_trajectory_sha256": canonical_frame_sha256(
            trajectories.loc[:, RPT_TRAJECTORY_COLUMNS], RPT_TRAJECTORY_COLUMNS
        ),
        "training_row_count": len(trajectories),
        "raw_measurements_embedded": False,
    }


def _predict(args: argparse.Namespace) -> int:
    paths = _paths(Path(args.output_directory))
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
    predictions, decisions, manifest = predict_private_dual_clock_prior_v3(
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
    paths = _paths(Path(args.output_directory))
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
    scores, summary = score_private_dual_clock_prior_v3(
        truth, predictions, decisions, manifest, config
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["summary"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _train_capsule(args: argparse.Namespace) -> int:
    paths = _paths(Path(args.output_directory))
    _ensure_available([paths["capsule"]], overwrite=args.overwrite)
    config = _load_config(args.config)
    trajectories = pd.read_parquet(args.rpt_trajectories)
    capsule = train_private_dual_clock_prior_capsule(
        trajectories,
        config,
        training_identity=_training_identity(trajectories, config),
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
        )
    )
    return 0


def _run_all(args: argparse.Namespace) -> int:
    input_directory = Path(args.input_directory)
    paths = _paths(Path(args.output_directory))
    _ensure_available(list(paths.values()), overwrite=args.overwrite)
    config = _load_config(args.config)

    # Freeze predictions before opening the held-condition capacity suffix table.
    references = pd.read_parquet(input_directory / "outer_fold_references.parquet")
    prefixes = pd.read_parquet(input_directory / "target_prefixes.parquet")
    predictions, decisions, manifest = predict_private_dual_clock_prior_v3(
        references, prefixes, config
    )
    _write_json(config, paths["config"])
    _write_parquet(predictions, paths["predictions"])
    _write_parquet(decisions, paths["decisions"])
    _write_csv(decisions, paths["decisions_csv"])
    _write_json(manifest, paths["manifest"])

    truth = pd.read_parquet(input_directory / "target_truth.parquet")
    scores, summary = score_private_dual_clock_prior_v3(
        truth, predictions, decisions, manifest, config
    )
    _write_csv(scores, paths["scores"])
    _write_json(summary, paths["summary"])

    trajectories = pd.read_parquet(input_directory / "rpt_trajectories.parquet")
    capsule = train_private_dual_clock_prior_capsule(
        trajectories,
        config,
        training_identity=_training_identity(trajectories, config),
    )
    _write_json(capsule, paths["capsule"])
    completed_artifacts = {
        name: path for name, path in paths.items() if name != "complete"
    }
    completion = build_completion_manifest(
        Path(args.output_directory),
        completed_artifacts,
        metadata={
            "experiment_id": str(config["experiment_id"]),
            "dataset_id": str(config["dataset_id"]),
            "prediction_manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
            "score_summary_content_sha256": summary["summary_content_sha256"],
            "capsule_content_sha256": capsule["capsule_content_sha256"],
            "implementation_file_sha256": file_sha256(
                Path(v3_implementation.__file__)
            ),
            "prior_implementation_file_sha256": file_sha256(
                Path(prior_implementation.__file__)
            ),
            "runner_file_sha256": file_sha256(Path(__file__)),
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


def _verify_run(args: argparse.Namespace) -> int:
    output_directory = Path(args.output_directory)
    paths = _paths(output_directory)
    manifest = json.loads(paths["complete"].read_text(encoding="utf-8"))
    required = [name for name in paths if name != "complete"]
    result = verify_completion_manifest(
        output_directory,
        manifest,
        required_names=required,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the private dual-clock V3 protocol. Generated measurements, "
            "scores, and model capsules are private-only artifacts."
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
    verify = subparsers.add_parser("verify-run")
    verify.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    verify.set_defaults(handler=_verify_run, read_only=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "read_only", False):
        return int(args.handler(args))
    with exclusive_run_lock(Path(args.output_directory)):
        return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
