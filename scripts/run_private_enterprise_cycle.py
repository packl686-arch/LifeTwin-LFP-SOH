from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import lifetwin.experiments.private_enterprise_cycle as enterprise_impl
from lifetwin.experiments.private_enterprise_cycle import (
    default_private_enterprise_v3_config,
    predict_private_enterprise_cycle,
    score_private_enterprise_cycle,
)
from lifetwin.experiments.private_schedule_v4 import (
    BOUNDED_SCHEDULE_MODE_ID,
    ELAPSED_SCHEDULE_MODE_ID,
    SCHEDULE_MODE_ID,
    canonicalize_private_forecast_schedule,
)
from lifetwin.private_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    build_completion_manifest,
    exclusive_run_lock,
    file_sha256,
    verify_completion_manifest,
)


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite private enterprise artifacts: " + ", ".join(existing)
        )


def _prediction_paths(output: Path) -> dict[str, Path]:
    return {
        "model_config": output / "model_config.private.json",
        "capsule": output / "model_capsule.private.json",
        "predictions": output / "predictions.private.parquet",
        "decisions": output / "decisions.private.parquet",
        "decisions_csv": output / "decisions.private.csv",
        "manifest": output / "prediction_manifest.private.json",
        "complete": output / "prediction_complete.private.json",
    }


def _predict(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    paths = _prediction_paths(output)
    schedule_path = output / "forecast_schedule.private.parquet"
    requested_paths = list(paths.values())
    if args.forecast_schedule:
        requested_paths.append(schedule_path)
    _ensure_available(requested_paths, overwrite=args.overwrite)
    adapter = _json(args.adapter_config)
    bundle = _json(args.bundle_manifest)
    model_config = (
        _json(args.model_config)
        if args.model_config
        else default_private_enterprise_v3_config(adapter)
    )
    development = pd.read_parquet(args.development_trajectories)
    prefixes = pd.read_parquet(args.target_prefixes)
    schedule = (
        canonicalize_private_forecast_schedule(pd.read_parquet(args.forecast_schedule))
        if args.forecast_schedule
        else None
    )
    selected_schedule_mode = (
        args.schedule_mode or ELAPSED_SCHEDULE_MODE_ID
        if schedule is not None
        else args.schedule_mode
    )
    predictions, decisions, capsule, manifest = predict_private_enterprise_cycle(
        development,
        prefixes,
        bundle,
        adapter,
        model_config,
        forecast_schedule=schedule,
        schedule_mode_id=selected_schedule_mode,
    )
    atomic_write_json(model_config, paths["model_config"])
    atomic_write_json(capsule, paths["capsule"])
    atomic_write_parquet(predictions, paths["predictions"])
    atomic_write_parquet(decisions, paths["decisions"])
    atomic_write_csv(decisions, paths["decisions_csv"])
    atomic_write_json(manifest, paths["manifest"])
    completion_artifacts = {
        name: path for name, path in paths.items() if name != "complete"
    }
    if schedule is not None:
        atomic_write_parquet(schedule, schedule_path)
        completion_artifacts["forecast_schedule"] = schedule_path
    completion = build_completion_manifest(
        output,
        completion_artifacts,
        metadata={
            "experiment_id": "private_enterprise_cycle_v1",
            "operation": "truth_free_prediction",
            "partition": manifest["partition"],
            "prediction_manifest_content_sha256": manifest["manifest_content_sha256"],
            "truth_vault_opened": False,
            "future_schedule_assumption": manifest["future_schedule_assumption"],
            "prediction_mode_id": manifest.get("prediction_mode_id"),
            "implementation_file_sha256": file_sha256(Path(enterprise_impl.__file__)),
            "runner_file_sha256": file_sha256(Path(__file__)),
            "public_release_permitted": False,
        },
    )
    atomic_write_json(completion, paths["complete"])
    print(
        json.dumps(
            {
                "partition": manifest["partition"],
                "prediction_row_count": manifest["prediction_row_count"],
                "decision_row_count": manifest["decision_row_count"],
                "issued_count": int(decisions["issued"].sum()),
                "truth_vault_opened": False,
                "manifest_content_sha256": manifest["manifest_content_sha256"],
            },
            indent=2,
        )
    )
    return 0


def _score(args: argparse.Namespace) -> int:
    prediction_directory = Path(args.prediction_directory)
    prediction_paths = _prediction_paths(prediction_directory)
    prediction_manifest = _json(prediction_paths["manifest"])
    schedule_path = prediction_directory / "forecast_schedule.private.parquet"
    required_names = [name for name in prediction_paths if name != "complete"]
    if prediction_manifest.get("future_schedule_assumption") == (
        "declared_piecewise_operating_schedule"
    ):
        required_names.append("forecast_schedule")
    prediction_completion = _json(prediction_paths["complete"])
    verified = verify_completion_manifest(
        prediction_directory,
        prediction_completion,
        required_names=required_names,
    )
    output = Path(args.output_directory)
    paths = {
        "scores": output / "scores.private.csv",
        "summary": output / "score_summary.private.json",
        "complete": output / "score_complete.private.json",
    }
    _ensure_available(list(paths.values()), overwrite=args.overwrite)
    truth = pd.read_parquet(args.truth_vault)
    predictions = pd.read_parquet(prediction_paths["predictions"])
    decisions = pd.read_parquet(prediction_paths["decisions"])
    schedule = (
        canonicalize_private_forecast_schedule(pd.read_parquet(schedule_path))
        if "forecast_schedule" in required_names
        else None
    )
    bundle = _json(args.bundle_manifest)
    adapter = _json(args.adapter_config)
    model_config = _json(prediction_paths["model_config"])
    scores, summary = score_private_enterprise_cycle(
        truth,
        predictions,
        decisions,
        prediction_manifest,
        bundle,
        adapter,
        model_config,
        forecast_schedule=schedule,
    )
    atomic_write_csv(scores, paths["scores"])
    atomic_write_json(summary, paths["summary"])
    completion = build_completion_manifest(
        output,
        {name: path for name, path in paths.items() if name != "complete"},
        metadata={
            "experiment_id": "private_enterprise_cycle_v1",
            "operation": "post_freeze_truth_linkage",
            "partition": summary["partition"],
            "prediction_completion_manifest_sha256": verified[
                "manifest_content_sha256"
            ],
            "score_summary_content_sha256": summary["summary_content_sha256"],
            "implementation_file_sha256": file_sha256(Path(enterprise_impl.__file__)),
            "runner_file_sha256": file_sha256(Path(__file__)),
            "public_release_permitted": False,
        },
    )
    atomic_write_json(completion, paths["complete"])
    print(json.dumps(summary["summary_by_landmark"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run truth-isolated private enterprise V3 prediction and scoring."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("development_trajectories")
    predict.add_argument("target_prefixes")
    predict.add_argument("--adapter-config", required=True)
    predict.add_argument("--bundle-manifest", required=True)
    predict.add_argument("--model-config")
    predict.add_argument(
        "--forecast-schedule",
        help=(
            "Outcome-free Parquet schedule. Omit to retain the frozen "
            "constant-prefix-duty V3 assumption."
        ),
    )
    predict.add_argument(
        "--schedule-mode",
        choices=(
            ELAPSED_SCHEDULE_MODE_ID,
            BOUNDED_SCHEDULE_MODE_ID,
            SCHEDULE_MODE_ID,
        ),
        help=(
            "Schedule-aware predictor. Defaults to the conservative V4.1 "
            "explicit-elapsed mode when --forecast-schedule is present."
        ),
    )
    predict.add_argument("--output-directory", required=True)
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(handler=_predict)
    score = subparsers.add_parser("score")
    score.add_argument("truth_vault")
    score.add_argument("--prediction-directory", required=True)
    score.add_argument("--adapter-config", required=True)
    score.add_argument("--bundle-manifest", required=True)
    score.add_argument("--output-directory", required=True)
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with exclusive_run_lock(Path(args.output_directory)):
        return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
