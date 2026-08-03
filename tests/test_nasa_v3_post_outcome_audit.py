from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    BASE_MODEL_IDS,
    build_nasa_dynamic_gate_fold_table,
    load_nasa_dynamic_gate_config,
)
from lifetwin.experiments.nasa_evidence_weighted_moe_v3 import (
    V3_MODEL_ID,
    load_nasa_evidence_weighted_moe_config,
    predict_nasa_evidence_weighted_moe,
    score_nasa_evidence_weighted_moe,
)
from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    DATASET_ID,
    SCORE_END_CYCLE,
)
from lifetwin.experiments.nasa_v3_post_outcome_audit import (
    ABLATION_SCORE_COLUMNS,
    EVIDENCE_COLUMNS,
    NasaV3PostOutcomeAuditError,
    audit_nasa_v3_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cycles() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    parameters = {
        "B0005": (1.88, 0.22, 0.035),
        "B0006": (2.02, 0.28, 0.055),
        "B0007": (1.91, 0.18, 0.025),
        "B0018": (1.87, 0.30, 0.060),
    }
    for cell_id, (initial, terminal_loss, curvature) in parameters.items():
        cutoff = CELL_CUTOFFS[cell_id]
        for cycle_index in range(1, SCORE_END_CYCLE + 1):
            progress = (cycle_index - 1) / (SCORE_END_CYCLE - 1)
            loss = terminal_loss * progress + curvature * np.sqrt(progress)
            capacity = initial * (1.0 - loss + 0.004 * np.sin(cycle_index / 7.0))
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "discharge_capacity_ah": float(capacity),
                    "discharge_cutoff_voltage_v": cutoff,
                    "common_window_3p8_to_3p4_duration_s": float(
                        1_800.0 * capacity / initial
                    ),
                    "voltage_at_1p0_ah_v": float(
                        3.48 - 0.10 * progress + 0.025 * (cutoff - 2.5)
                    ),
                    "temperature_rise_c": float(14.0 + 3.0 * progress),
                }
            )
    return pd.DataFrame(rows)


def test_post_outcome_audit_replays_inputs_and_keeps_oracle_explicit() -> None:
    cycles = _cycles()
    v2_config = load_nasa_dynamic_gate_config(
        PROJECT_ROOT / "configs/experiments/nasa_dynamic_gate_v2.json"
    )
    v3_config = load_nasa_evidence_weighted_moe_config(
        PROJECT_ROOT / "configs/experiments/nasa_evidence_weighted_moe_v3.json"
    )
    folds = build_nasa_dynamic_gate_fold_table(cycles, v2_config)
    predictions, manifest = predict_nasa_evidence_weighted_moe(
        folds,
        v2_config,
        v3_config,
    )
    scores, score_summary = score_nasa_evidence_weighted_moe(
        cycles,
        predictions,
        manifest,
        v2_config,
        v3_config,
    )
    ablation, prefix_summary, evidence, summary = audit_nasa_v3_result(
        cycles,
        predictions,
        manifest,
        scores,
        score_summary,
        v2_config,
        v3_config,
    )

    assert tuple(ablation.columns) == ABLATION_SCORE_COLUMNS
    assert tuple(evidence.columns) == EVIDENCE_COLUMNS
    assert len(evidence) == 16
    assert len(prefix_summary) == 24
    assert summary["evidence_role"] == "post_outcome_diagnostic_not_model_selection"
    pivot = ablation.pivot(
        index=["held_out_cell_id", "prefix_cycle"],
        columns="audit_model_id",
        values="trajectory_mae_pp",
    )
    realized_best_base = (
        scores.loc[scores["model_id"].isin(BASE_MODEL_IDS)]
        .groupby(["held_out_cell_id", "prefix_cycle"], sort=True)["trajectory_mae_pp"]
        .min()
    )
    np.testing.assert_allclose(
        pivot["hindsight_oracle_base_expert"],
        realized_best_base,
        rtol=0.0,
        atol=1e-12,
    )
    v3_scores = scores.loc[scores["model_id"] == V3_MODEL_ID].set_index(
        ["held_out_cell_id", "prefix_cycle"]
    )
    np.testing.assert_allclose(
        evidence.set_index(["held_out_cell_id", "prefix_cycle"])[
            "v3_trajectory_mae_pp"
        ],
        v3_scores["trajectory_mae_pp"],
        rtol=0.0,
        atol=1e-12,
    )

    attacked_scores = scores.copy()
    attacked_scores.loc[0, "trajectory_mae_pp"] += 0.01
    with pytest.raises(NasaV3PostOutcomeAuditError, match="score table changed"):
        audit_nasa_v3_result(
            cycles,
            predictions,
            manifest,
            attacked_scores,
            score_summary,
            v2_config,
            v3_config,
        )
