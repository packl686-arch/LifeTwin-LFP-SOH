from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from lifetwin.experiments.private_enterprise_cycle import (
    predict_private_enterprise_cycle,
    score_private_enterprise_cycle,
)
from lifetwin.experiments.private_schedule_v4 import (
    ELAPSED_SCHEDULE_MODE_ID,
    SCHEDULE_MODE_ID,
)
from lifetwin.experiments.snl_rpt_loco import _trajectory_iae


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_api_has_no_truth_or_future_capacity_argument() -> None:
    prediction_parameters = set(
        inspect.signature(predict_private_enterprise_cycle).parameters
    )
    assert "truth_vault" not in prediction_parameters
    assert "target_truth" not in prediction_parameters
    assert "future_capacity" not in prediction_parameters
    assert "capacity_suffix" not in prediction_parameters
    assert "truth_vault" in inspect.signature(score_private_enterprise_cycle).parameters


def test_trajectory_iae_matches_independent_hand_integral() -> None:
    actual_x = np.asarray([1.0, 3.0])
    actual = np.asarray([100.0, 100.0])
    predicted = np.asarray([99.0, 97.0])
    # Anchor error is zero at x=0: area = 0.5*1 + 2*2 = 4.5; 4.5/3 = 1.5.
    assert _trajectory_iae(0.0, actual_x, actual, predicted) == pytest.approx(1.5)


def test_v4_1_amendment_cannot_silently_relabel_full_v4() -> None:
    amendment = json.loads(
        (
            ROOT / "configs/experiments/private_enterprise_schedule_v4_1_amendment.json"
        ).read_text(encoding="utf-8")
    )
    assert amendment["candidate"] == ELAPSED_SCHEDULE_MODE_ID
    assert amendment["retained_negative_control"] == SCHEDULE_MODE_ID
    assert (
        amendment["model_contract"]["future_condition_delta_applied_to_prediction"]
        is False
    )
