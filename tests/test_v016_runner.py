from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from lifetwin.experiments import calendar_long_horizon_v016_prediction as prediction
from lifetwin.experiments import calendar_long_horizon_v016_runner as runner


def test_formal_api_has_no_scientific_or_generation_override_surface() -> None:
    parameters = set(inspect.signature(runner.run_formal_attempt).parameters)
    assert parameters == {
        "attempt_id",
        "label_free_root",
        "sealed_truth_root",
        "score_root",
        "termination_root",
        "repo_root",
    }
    forbidden = (
        "seed",
        "count",
        "worker",
        "threshold",
        "family",
        "mapping",
        "analysis_root",
    )
    assert not any(token in name for token in forbidden for name in parameters)

    prediction_parameters = set(
        inspect.signature(prediction.run_isolated_prediction_process_v021).parameters
    )
    assert prediction_parameters == {
        "label_free_root",
        "attempt_id",
        "repo_root",
    }


def test_prediction_module_has_no_generation_or_collision_import() -> None:
    tree = ast.parse(inspect.getsource(prediction))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        token in module
        for module in imports
        for token in ("v016_generation", "v016_collision", "v016_actual_ledger")
    )
    source = inspect.getsource(prediction)
    assert "seed_root" not in source
    assert "family_id" not in source
    assert "ordinary_family_lookup" not in source


def test_cli_prediction_branch_lazily_imports_only_prediction_adapter() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_calendar_long_horizon_v016.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    top_level_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module.startswith("lifetwin.") for module in top_level_modules)
    source = script.read_text(encoding="utf-8")
    branch_start = source.index('if args.internal_stage == "prediction":')
    prediction_branch = source[
        branch_start : source.index(
            "\n    attempt_id = _require_attempt_id",
            branch_start,
        )
    ]
    assert "calendar_long_horizon_v016_prediction" in prediction_branch
    assert "calendar_long_horizon_v016_runner" not in prediction_branch
    assert "--seed" not in source
    assert "--worker" not in source
    assert "--threshold" not in source


def test_prediction_subprocess_arguments_are_capability_minimal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def capture(arguments: object, *, context: str) -> None:
        observed["arguments"] = tuple(arguments)  # type: ignore[arg-type]
        observed["context"] = context

    monkeypatch.setattr(runner, "_run_checked_process", capture)
    runner._launch_prediction_process(
        label_free_root=tmp_path / "label",
        attempt_id="v021-fixture",
        repo_root=tmp_path / "repo",
    )
    arguments = observed["arguments"]
    assert isinstance(arguments, tuple)
    assert arguments[2:] == (
        "--internal-stage",
        "prediction",
        "--label-free-root",
        str(tmp_path / "label"),
        "--attempt-id",
        "v021-fixture",
        "--repo-root",
        str(tmp_path / "repo"),
    )
    joined = " ".join(arguments)
    assert "sealed-truth" not in joined
    assert "score-root" not in joined
    assert "termination-root" not in joined
    assert "seed" not in joined
    assert "family" not in joined


def test_attempt_id_is_rejected_before_environment_or_first_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = runner.V021RunPaths.resolve(
        repo_root=tmp_path / "repo",
        label_free_root=tmp_path / "label",
        sealed_truth_root=tmp_path / "sealed",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
    )
    monkeypatch.setattr(
        runner,
        "verify_formal_environment",
        lambda _root: pytest.fail("environment must not be inspected"),
    )
    with pytest.raises(runner.V021RunnerError, match="attempt_id must match"):
        runner.initialize_formal_attempt(
            paths=paths,
            attempt_id="v016-not-formal",
        )
    assert not paths.ledger_path.exists()


def test_run_paths_require_four_disjoint_artifact_trees(tmp_path: Path) -> None:
    with pytest.raises(runner.V021RunnerError, match="pairwise disjoint"):
        runner.V021RunPaths.resolve(
            repo_root=tmp_path / "repo",
            label_free_root=tmp_path / "attempt",
            sealed_truth_root=tmp_path / "sealed",
            score_root=tmp_path / "score",
            termination_root=tmp_path / "attempt" / "terminal",
        )


def test_run_paths_reject_reparse_in_supplied_artifact_traversal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "junction"
    parent.mkdir()
    monkeypatch.setattr(
        runner,
        "_is_reparse",
        lambda path: Path(os.path.abspath(path)) == parent,
    )
    with pytest.raises(runner.V021RunnerError, match="traverses a reparse"):
        runner.V021RunPaths.resolve(
            repo_root=tmp_path / "repo",
            label_free_root=parent / "label",
            sealed_truth_root=tmp_path / "sealed",
            score_root=tmp_path / "score",
            termination_root=tmp_path / "termination",
        )


def test_empty_artifact_root_rejects_reparse_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = (tmp_path / "parent").resolve()
    root = parent / "root"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "_is_reparse",
        lambda path: path.resolve() == parent,
    )
    with pytest.raises(runner.V021RunnerError, match="traverses a reparse"):
        runner._prepare_empty_physical_root(root, context="label-free")


def test_score_root_recheck_rejects_reparse_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = (tmp_path / "parent").resolve()
    root = parent / "score"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "_is_reparse",
        lambda path: path.resolve() == parent,
    )
    with pytest.raises(
        runner.V021RunnerError,
        match="physical empty directory",
    ):
        runner._require_empty_score_root(root)


def test_latent_truth_join_is_exact_on_opaque_id_and_day() -> None:
    rows = []
    for cluster_id, offset in (("opaque-b", 10.0), ("opaque-a", 0.0)):
        for day_index, day in enumerate(runner.FORECAST_DAYS):
            rows.append(
                {
                    "partition": "calibration",
                    "cluster_id": cluster_id,
                    "forecast_day": day,
                    "latent_retention_pct": offset + day_index,
                    "noisy_retention_pct": -9999.0,
                }
            )
    frame = runner.pd.DataFrame(reversed(rows))
    matrix = runner._latent_truth_matrix(
        frame,
        partition="calibration",
        cluster_ids=("opaque-a", "opaque-b"),
    )
    assert matrix[0].tolist() == list(range(len(runner.FORECAST_DAYS)))
    assert matrix[1].tolist() == [
        10.0 + index for index in range(len(runner.FORECAST_DAYS))
    ]
    broken = frame.loc[
        ~(
            frame["cluster_id"].eq("opaque-a")
            & frame["forecast_day"].eq(runner.FORECAST_DAYS[-1])
        )
    ]
    with pytest.raises(runner.V021RunnerError, match="exact forecast_day grid"):
        runner._latent_truth_matrix(
            broken,
            partition="calibration",
            cluster_ids=("opaque-a", "opaque-b"),
        )


def test_attempt_id_regex_is_the_provenance_v021_namespace() -> None:
    assert runner._FORMAL_ATTEMPT_ID.pattern == (
        r"^v021-[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$"
    )
    assert re.fullmatch(runner._FORMAL_ATTEMPT_ID, "v021-run.001")
    assert re.fullmatch(runner._FORMAL_ATTEMPT_ID, "v016-run.001") is None


def test_formal_runner_executes_frozen_lifecycle_in_exact_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    truth_hash = "1" * 64
    actual_hash = "2" * 64
    prediction_hash = "3" * 64
    environment = SimpleNamespace(git_commit="4" * 40)
    view = SimpleNamespace(
        design_status="implementation_frozen",
        artifacts=object(),
    )
    identity = SimpleNamespace(attempt_id="v021-lifecycle")
    generation_progress = SimpleNamespace(
        completed_phase="truth_committed",
        pending_phase=None,
        truth_commitments_byte_sha256=truth_hash,
    )
    final_progress = SimpleNamespace(
        completed_phase="scoring_completed",
        pending_phase=None,
        terminal_failed=False,
    )
    progress = iter((generation_progress, final_progress))
    monkeypatch.setattr(
        runner,
        "initialize_formal_attempt",
        lambda **_kwargs: (environment, view, identity),
    )
    monkeypatch.setattr(
        runner,
        "_launch_generation_process",
        lambda _paths: calls.append("generation"),
    )
    monkeypatch.setattr(
        runner,
        "validate_formal_exposure_log",
        lambda *_args: {"v021-lifecycle": next(progress)},
    )
    monkeypatch.setattr(
        runner,
        "_file_hash",
        lambda _path: truth_hash,
    )

    def actual(**_kwargs: object) -> str:
        calls.append("actual")
        return actual_hash

    def fit(**_kwargs: object) -> tuple[dict[str, object], str]:
        calls.append("fit")
        return {}, "5" * 64

    def train(**_kwargs: object) -> object:
        calls.append("training")
        return object()

    def predict(**_kwargs: object) -> tuple[str, object, dict[str, object], object]:
        calls.append("prediction")
        return prediction_hash, object(), {}, object()

    def score(**_kwargs: object) -> str:
        calls.append("scoring")
        return "ok"

    monkeypatch.setattr(runner, "_commit_actual_analysis_hash_ledger", actual)
    monkeypatch.setattr(runner, "_fit_structure_stage", fit)
    monkeypatch.setattr(runner, "_fit_training_stages", train)
    monkeypatch.setattr(runner, "_prediction_and_commitment", predict)
    monkeypatch.setattr(runner, "_score_and_write", score)
    monkeypatch.setattr(
        runner,
        "_publish_preprediction_failure",
        lambda **_kwargs: pytest.fail("successful run cannot publish terminal"),
    )
    result = runner.run_formal_attempt(
        attempt_id="v021-lifecycle",
        label_free_root=tmp_path / "label",
        sealed_truth_root=tmp_path / "sealed",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
        repo_root=tmp_path / "repo",
    )
    assert calls == [
        "generation",
        "actual",
        "fit",
        "training",
        "prediction",
        "scoring",
    ]
    assert result.actual_analysis_hash_ledger_commitment_byte_sha256 == (actual_hash)
    assert result.prediction_commitment_byte_sha256 == prediction_hash
    assert result.score_status == "ok"


def test_preprediction_failure_publishes_only_to_termination_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = runner.V021RunPaths.resolve(
        repo_root=tmp_path / "repo",
        label_free_root=tmp_path / "label",
        sealed_truth_root=tmp_path / "sealed",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
    )
    paths.label_free_root.mkdir()
    paths.termination_root.mkdir()
    progress = SimpleNamespace(
        prediction_commitment_byte_sha256=None,
    )
    monkeypatch.setattr(
        runner,
        "validate_formal_exposure_log",
        lambda *_args: {"v021-terminal": progress},
    )
    context = object()
    monkeypatch.setattr(
        runner.TerminalContext,
        "from_progress",
        lambda *_args, **_kwargs: context,
    )
    observed: dict[str, object] = {}

    def publish(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(runner, "publish_terminal", publish)
    error = RuntimeError("before prediction")
    runner._publish_preprediction_failure(
        error=error,
        paths=paths,
        view=SimpleNamespace(artifacts=object()),
        attempt_id="v021-terminal",
        attempt_created_utc="2026-07-26T00:00:00Z",
    )
    assert observed["termination_root"] == paths.termination_root
    assert observed["label_free_artifact_root"] == paths.label_free_root
    assert observed["context"] is context
    assert observed["error"] is error


def test_postcommit_failure_cannot_publish_preprediction_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = runner.V021RunPaths.resolve(
        repo_root=tmp_path / "repo",
        label_free_root=tmp_path / "label",
        sealed_truth_root=tmp_path / "sealed",
        score_root=tmp_path / "score",
        termination_root=tmp_path / "termination",
    )
    progress = SimpleNamespace(
        prediction_commitment_byte_sha256="9" * 64,
    )
    monkeypatch.setattr(
        runner,
        "validate_formal_exposure_log",
        lambda *_args: {"v021-postcommit": progress},
    )
    monkeypatch.setattr(
        runner,
        "publish_terminal",
        lambda **_kwargs: pytest.fail(
            "postcommit failures cannot publish a preprediction terminal"
        ),
    )
    runner._publish_preprediction_failure(
        error=RuntimeError("after prediction commitment"),
        paths=paths,
        view=SimpleNamespace(artifacts=object()),
        attempt_id="v021-postcommit",
        attempt_created_utc="2026-07-26T00:00:00Z",
    )
