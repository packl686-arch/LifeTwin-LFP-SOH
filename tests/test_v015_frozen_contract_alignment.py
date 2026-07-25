from __future__ import annotations

import json
from pathlib import Path

from lifetwin.experiments.calendar_long_horizon_v015_analysis import (
    BOOTSTRAP_RESAMPLES,
    RANDOM_RANKING_COUNT,
    RISK_SCORE_IDS,
    STRESS_PERMUTATIONS,
)
from lifetwin.experiments.calendar_long_horizon_v015_scoring import (
    REQUIRED_FORMAL_NON_SCORE_ARTIFACTS,
    REQUIRED_SCORE_ARTIFACTS,
    SCORE_REPORT_KEYS,
)


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = (
    _ROOT / "configs" / "experiments" / "synthetic_long_horizon_identifiability_v2.json"
)


def _config() -> dict[str, object]:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_formal_artifact_registries_cover_the_frozen_protocol() -> None:
    config = _config()
    artifacts = config["firewall_and_artifacts"]

    assert tuple(artifacts["required_score_artifacts"]) == (REQUIRED_SCORE_ARTIFACTS)

    frozen_non_score = {
        filename
        for key in (
            "required_label_free_artifacts",
            "required_frozen_state_artifacts",
            "required_prediction_artifacts",
            "required_sealed_artifacts",
            "required_attempt_artifacts",
        )
        for filename in artifacts[key]
    }
    implementation_checkpoints = {
        "fit_commitment.json",
        "center_state_checkpoint.json",
        "risk_state_checkpoint.json",
        "model_state_commitment.json",
    }
    assert set(REQUIRED_FORMAL_NON_SCORE_ARTIFACTS) == (
        frozen_non_score | implementation_checkpoints
    )
    assert len(REQUIRED_FORMAL_NON_SCORE_ARTIFACTS) == len(
        set(REQUIRED_FORMAL_NON_SCORE_ARTIFACTS)
    )


def test_score_report_surface_covers_every_mandatory_reporting_clause() -> None:
    mandatory = _config()["reporting"]["mandatory"]
    assert len(mandatory) == 12

    mandatory_report_sections = {
        "test_primary_estimates",
        "policy_comparison",
        "risk_coverage_curves_secondary",
        "coverage_metrics",
        "isotonic_calibration_diagnostics",
        "structure_diagnostics",
        "test_audit_distribution_shift",
        "gate_evaluations",
        "stress_plan_summary",
        "stress_permutation_summary",
        "protocol_deviations",
    }
    assert mandatory_report_sections.issubset(SCORE_REPORT_KEYS)


def test_formal_analysis_counts_and_risk_heads_match_the_freeze() -> None:
    config = _config()
    risk_schema = config["firewall_and_artifacts"]["artifact_schemas"][
        "risk_bundle.csv"
    ]

    assert tuple(risk_schema["required_score_ids"]) == RISK_SCORE_IDS
    assert BOOTSTRAP_RESAMPLES == config["endpoints"]["bootstrap"]["resamples"]
    assert RANDOM_RANKING_COUNT == 10_000
    assert STRESS_PERMUTATIONS == 10_000
