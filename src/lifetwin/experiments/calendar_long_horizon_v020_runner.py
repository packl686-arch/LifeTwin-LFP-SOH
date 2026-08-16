"""Fixed V2.5 profile over the shared V0.20 lifecycle implementation."""

from __future__ import annotations

from pathlib import Path

from lifetwin.experiments.calendar_long_horizon_v019_prediction import (
    V024PredictionWriteResult,
    run_isolated_prediction_process_v024,
)
from lifetwin.experiments.calendar_long_horizon_v019_runner import (
    V024FormalRunResult,
    run_formal_attempt,
    run_isolated_generation_stage,
)
from lifetwin.experiments.calendar_long_horizon_v020_checkpoint_registry import (
    INPUT_FILENAMES_BY_STAGE,
)
from lifetwin.experiments.calendar_long_horizon_v020_contract import (
    load_v025_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v020_environment import (
    verify_formal_environment_v025,
    verify_prediction_environment_v025,
)
from lifetwin.experiments.calendar_long_horizon_v020_protocol import (
    V025_DESIGN_STATUS,
    V025_ONLY_ATTEMPT_ID,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FORMAL_SCRIPT = _PROJECT_ROOT / "scripts" / "run_calendar_long_horizon_v020.py"


def run_isolated_generation_stage_v025(
    *,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
) -> None:
    """Run generation with the one authenticated V2.5 identity."""

    run_isolated_generation_stage(
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        _contract_view=load_v025_contract_view(),
        _environment_verifier=verify_formal_environment_v025,
        _implementable_status=V025_DESIGN_STATUS,
    )


def run_isolated_prediction_process_v025(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    repo_root: str | Path,
) -> V024PredictionWriteResult:
    """Run the truth-incapable prediction capsule under the V2.5 attester."""

    if attempt_id != V025_ONLY_ATTEMPT_ID:
        raise ValueError("Prediction attempt differs from the fixed V2.5 identity")
    return run_isolated_prediction_process_v024(
        label_free_root=label_free_root,
        attempt_id=attempt_id,
        repo_root=repo_root,
        _environment_verifier=verify_prediction_environment_v025,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
    )


def run_formal_attempt_v025(
    *,
    attempt_id: str,
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
    score_root: str | Path,
    termination_root: str | Path,
    repo_root: str | Path = _PROJECT_ROOT,
) -> V024FormalRunResult:
    """Execute the sole V2.5 formal profile without a scientific override."""

    return run_formal_attempt(
        attempt_id=attempt_id,
        label_free_root=label_free_root,
        sealed_truth_root=sealed_truth_root,
        score_root=score_root,
        termination_root=termination_root,
        repo_root=repo_root,
        _contract_view=load_v025_contract_view(),
        _formal_attempt_id=V025_ONLY_ATTEMPT_ID,
        _formal_script=_FORMAL_SCRIPT,
        _environment_verifier=verify_formal_environment_v025,
        _implementable_status=V025_DESIGN_STATUS,
        _input_filenames_by_stage=INPUT_FILENAMES_BY_STAGE,
    )


__all__ = [
    "run_formal_attempt_v025",
    "run_isolated_generation_stage_v025",
    "run_isolated_prediction_process_v025",
]
