from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.private_enterprise_cycle import SCORE_COLUMNS
from lifetwin.experiments.private_schedule_v4 import (
    BOUNDED_SCHEDULE_MODE_ID,
    ELAPSED_SCHEDULE_MODE_ID,
    SCHEDULE_MODE_ID,
)
from lifetwin.experiments.private_schedule_v4_gates import (
    PrivateScheduleV4GateError,
    evaluate_private_schedule_v4_gates,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "configs/experiments/private_enterprise_schedule_v4_preregistered.json"
)
AMENDMENT = ROOT / "configs/experiments/private_enterprise_schedule_v4_1_amendment.json"
BOUNDED_PREREGISTRATION = (
    ROOT / "configs/experiments/private_enterprise_schedule_v4_2_preregistered.json"
)


def _scores(value: float) -> pd.DataFrame:
    rows = []
    for landmark in (3, 4):
        for condition in ("condition_a", "condition_b"):
            rows.append(
                {
                    "experiment_id": "private_enterprise_cycle_v1",
                    "adapter_id": "adapter",
                    "dataset_id": "private",
                    "partition": "calibration",
                    "cell_id": f"{condition}_{landmark}",
                    "condition_id": condition,
                    "landmark_visit_count": landmark,
                    "future_observation_count": 3,
                    "trajectory_iae_pp": value,
                    "trajectory_mae_pp": value,
                    "trajectory_rmse_pp": value,
                    "endpoint_absolute_error_pp": value,
                    "pointwise_interval_coverage": 0.95,
                    "simultaneous_trajectory_covered": True,
                    "mean_full_interval_width_pp": 2.0,
                }
            )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def _summary(
    value: float,
    *,
    candidate: bool,
    mode_id: str = SCHEDULE_MODE_ID,
    scores: pd.DataFrame | None = None,
) -> dict[str, object]:
    score_frame = _scores(value) if scores is None else scores
    landmark = {
        str(item): {
            "condition_equal_trajectory_iae_pp": value,
            "issued_fraction": 0.9,
            "condition_equal_pointwise_interval_coverage": 0.95,
        }
        for item in (3, 4)
    }
    result: dict[str, object] = {
        "schema_version": "lifetwin.private_enterprise_cycle.score_summary.v1",
        "experiment_id": "private_enterprise_cycle_v1",
        "adapter_id": "adapter",
        "dataset_id": "private",
        "partition": "calibration",
        "evidence_role": "private_batch_disjoint_calibration",
        "summary_by_landmark": landmark,
        "score_rows_sha256": canonical_frame_sha256(score_frame, SCORE_COLUMNS),
    }
    if candidate:
        result.update(
            {
                "prediction_mode_id": mode_id,
                "schedule_role": "deployment_candidate",
                "primary_evidence_eligible": True,
            }
        )
    result["summary_content_sha256"] = canonical_json_sha256(result)
    return result


def _preregistration() -> dict[str, object]:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_schedule_v4_promotion_requires_every_frozen_gate() -> None:
    result = evaluate_private_schedule_v4_gates(
        _scores(1.0),
        _scores(0.8),
        _summary(1.0, candidate=False),
        _summary(0.8, candidate=True),
        _preregistration(),
    )
    assert result["promote_v4"] is True
    assert all(row["passed"] for row in result["by_landmark"].values())
    assert len(result["result_content_sha256"]) == 64


def test_schedule_v4_1_amendment_binds_elapsed_only_candidate() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    result = evaluate_private_schedule_v4_gates(
        _scores(1.0),
        _scores(0.8),
        _summary(1.0, candidate=False),
        _summary(
            0.8,
            candidate=True,
            mode_id=ELAPSED_SCHEDULE_MODE_ID,
        ),
        amendment,
    )
    assert result["candidate"] == ELAPSED_SCHEDULE_MODE_ID
    assert result["promote_candidate"] is True


def test_schedule_v4_1_amendment_rejects_full_v4_result() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    with pytest.raises(PrivateScheduleV4GateError, match="preregistered"):
        evaluate_private_schedule_v4_gates(
            _scores(1.0),
            _scores(0.8),
            _summary(1.0, candidate=False),
            _summary(0.8, candidate=True),
            amendment,
        )


def test_schedule_v4_2_preregistration_binds_bounded_candidate() -> None:
    protocol = json.loads(BOUNDED_PREREGISTRATION.read_text(encoding="utf-8"))
    result = evaluate_private_schedule_v4_gates(
        _scores(1.0),
        _scores(0.8),
        _summary(1.0, candidate=False),
        _summary(0.8, candidate=True, mode_id=BOUNDED_SCHEDULE_MODE_ID),
        protocol,
    )
    assert result["candidate"] == BOUNDED_SCHEDULE_MODE_ID
    assert result["promote_candidate"] is True


def test_schedule_v4_missing_condition_cannot_pass_by_abstaining() -> None:
    candidate = _scores(0.8)
    candidate = candidate.loc[candidate["condition_id"] != "condition_b"].reset_index(
        drop=True
    )
    result = evaluate_private_schedule_v4_gates(
        _scores(1.0),
        candidate,
        _summary(1.0, candidate=False),
        _summary(0.8, candidate=True, scores=candidate),
        _preregistration(),
    )
    assert result["promote_v4"] is False
    assert result["by_landmark"]["3"]["missing_candidate_conditions"] == ["condition_b"]


def test_schedule_v4_oracle_result_is_ineligible_for_promotion() -> None:
    candidate = _summary(0.8, candidate=True)
    candidate["schedule_role"] = "oracle_upper_bound"
    candidate["primary_evidence_eligible"] = False
    candidate.pop("summary_content_sha256")
    candidate["summary_content_sha256"] = canonical_json_sha256(candidate)
    with pytest.raises(PrivateScheduleV4GateError, match="Oracle"):
        evaluate_private_schedule_v4_gates(
            _scores(1.0),
            _scores(0.8),
            _summary(1.0, candidate=False),
            candidate,
            _preregistration(),
        )


def test_schedule_gate_rejects_rehashed_summary_with_changed_scores() -> None:
    candidate_scores = _scores(0.8)
    candidate_summary = _summary(0.8, candidate=True, scores=candidate_scores)
    candidate_scores.loc[0, "trajectory_iae_pp"] = 0.01
    with pytest.raises(PrivateScheduleV4GateError, match="score rows changed"):
        evaluate_private_schedule_v4_gates(
            _scores(1.0),
            candidate_scores,
            _summary(1.0, candidate=False),
            candidate_summary,
            _preregistration(),
        )


def test_schedule_gate_rejects_duplicate_score_keys() -> None:
    candidate_scores = pd.concat([_scores(0.8), _scores(0.8).iloc[[0]]])
    candidate_scores = candidate_scores.loc[:, SCORE_COLUMNS].reset_index(drop=True)
    with pytest.raises(PrivateScheduleV4GateError, match="duplicated"):
        evaluate_private_schedule_v4_gates(
            _scores(1.0),
            candidate_scores,
            _summary(1.0, candidate=False),
            _summary(0.8, candidate=True, scores=candidate_scores),
            _preregistration(),
        )
