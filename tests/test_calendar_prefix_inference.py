from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from unittest.mock import patch

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

import lifetwin.inference.calendar_prefix as calendar_prefix
from lifetwin import atomic_publish
from lifetwin import cli
from lifetwin.experiments.calendar_v4_hybrid_development import (
    CALIBRATION_CONDITION_IDS,
    TEST_CONDITION_IDS,
    TRAINING_CONDITION_IDS,
    fit_calendar_v4_reference_state,
)
from lifetwin.inference.calendar_prefix import (
    CalendarPrefixRequestError,
    LOCKED_CALENDAR_ELAPSED_DAYS,
    predict_calendar_prefix,
    validate_calendar_prefix_request,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/naumann_calendar_v4_hybrid_development.json"
)
SCHEMA_PATH = (
    PROJECT_ROOT / "configs/inference/calendar_prefix_request.schema.json"
)
REFERENCE_PATH = (
    PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv"
)
FALLBACK_REQUEST_PATH = (
    PROJECT_ROOT
    / "showcase/product_demo/naumann_t40_soc37_5_request.json"
)
SPECIALIST_REQUEST_PATH = (
    PROJECT_ROOT
    / "showcase/product_demo/naumann_t40_soc12_5_request.json"
)
FROZEN_V4_PATH = (
    PROJECT_ROOT
    / "showcase/evidence_v011/v4/label_free_predictions.csv"
)
FROZEN_V4_SHA256 = (
    "ac0bd25154954a603eab6dbbbcfd3a1f281b4ec59d0ce5207a95c015a87c4d0c"
)


@pytest.fixture(scope="module")
def prefix_assets(observations: pd.DataFrame) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    requests = {
        "fallback": json.loads(FALLBACK_REQUEST_PATH.read_text(encoding="utf-8")),
        "specialist": json.loads(
            SPECIALIST_REQUEST_PATH.read_text(encoding="utf-8")
        ),
    }
    state = fit_calendar_v4_reference_state(observations, config=config)
    return {
        "config": config,
        "schema": schema,
        "requests": requests,
        "state": state,
    }


def _predict_with_state(
    request: dict[str, object],
    *,
    observations: pd.DataFrame,
    assets: dict[str, object],
    state: object | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    selected_state = assets["state"] if state is None else state
    with patch.object(
        calendar_prefix,
        "fit_calendar_v4_reference_state",
        return_value=selected_state,
    ):
        return predict_calendar_prefix(
            request,
            reference_observations=observations,
            config=assets["config"],
            schema=assets["schema"],
        )


@pytest.fixture(scope="module")
def golden_prefix_results(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
) -> dict[str, tuple[dict[str, object], pd.DataFrame]]:
    requests = prefix_assets["requests"]
    assert isinstance(requests, dict)
    return {
        name: _predict_with_state(
            request,
            observations=observations,
            assets=prefix_assets,
        )
        for name, request in requests.items()
    }


def test_fallback_golden_case_has_diagnostic_interval_but_refuses_operations(
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    decision, forecast = golden_prefix_results["fallback"]

    assert decision["status"] == (
        "research_forecast_generated_operationally_abstained"
    )
    assert decision["mean_prediction"] == {
        "status": "available",
        "route": "hierarchical_power_fallback",
        "fallback_reasons": ["specialist_gate_not_ready"],
        "activation_gate_ready": False,
        "failure_type": None,
    }
    assert decision["diagnostic_interval"]["status"] == "available"
    assert decision["diagnostic_interval"]["requested_coverage"] == 0.8
    assert decision["diagnostic_interval"]["formal_coverage_claim_allowed"] is False
    assert decision["diagnostic_interval"]["calibration_condition_count"] == 5
    assert decision["diagnostic_interval"]["calibration_order_statistic_rank"] == 5
    assert decision["diagnostic_interval"]["calibration_multiplier"] == pytest.approx(
        2.1698424743004083
    )
    assert decision["support"]["prefix_supported"] is True
    assert decision["support"]["support_boundary_tolerance_days"] == 1e-12
    assert decision["support"]["prefix_reference_mode"] == (
        "same_temperature_soc_interpolation"
    )
    assert decision["operational_decision"] == {
        "issuance_status": "abstained",
        "abstention_reasons": [
            "calibration_evidence_not_independent",
            "independent_long_term_evidence_missing",
        ],
        "lower_pct": None,
        "upper_pct": None,
    }

    assert len(forecast) == 25
    assert forecast["forecast_index"].tolist() == list(range(10, 35))
    assert set(forecast["mean_route"]) == {"hierarchical_power_fallback"}
    assert set(forecast["diagnostic_interval_status"]) == {"available"}
    assert forecast[
        [
            "predicted_capacity_retention_pct",
            "predictive_sd_pp",
            "diagnostic_lower_pct",
            "diagnostic_upper_pct",
        ]
    ].notna().all().all()
    assert (
        forecast["diagnostic_lower_pct"]
        <= forecast["predicted_capacity_retention_pct"]
    ).all()
    assert (
        forecast["predicted_capacity_retention_pct"]
        <= forecast["diagnostic_upper_pct"]
    ).all()
    assert set(forecast["operational_issuance_status"]) == {"abstained"}


def test_specialist_golden_case_fails_closed_without_route_calibration(
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    decision, forecast = golden_prefix_results["specialist"]

    assert decision["status"] == (
        "research_forecast_generated_operationally_abstained"
    )
    assert decision["mean_prediction"] == {
        "status": "available",
        "route": "hierarchical_activation_residual",
        "fallback_reasons": [],
        "activation_gate_ready": True,
        "failure_type": None,
    }
    assert decision["diagnostic_interval"] == {
        "status": "unavailable",
        "role": "retrospective_reused_data_diagnostic_only",
        "requested_coverage": 0.8,
        "formal_coverage_claim_allowed": False,
        "calibration_condition_count": 1,
        "calibration_order_statistic_rank": 2,
        "calibration_multiplier": None,
        "abstention_reasons": ["insufficient_same_route_calibration"],
    }
    assert decision["operational_decision"]["issuance_status"] == "abstained"
    assert decision["operational_decision"]["lower_pct"] is None
    assert decision["operational_decision"]["upper_pct"] is None
    assert decision["support"]["prefix_supported"] is True
    assert decision["support"]["prefix_reference_mode"] == (
        "same_temperature_soc_interpolation"
    )
    assert {
        "calibration_unavailable",
        "calibration_evidence_not_independent",
        "insufficient_same_route_calibration",
        "independent_long_term_evidence_missing",
        "interval_width_invalid",
    }.issubset(decision["operational_decision"]["abstention_reasons"])

    assert len(forecast) == 25
    assert set(forecast["mean_route"]) == {
        "hierarchical_activation_residual"
    }
    assert forecast["predicted_capacity_retention_pct"].notna().all()
    assert set(forecast["diagnostic_interval_status"]) == {"unavailable"}
    assert set(forecast["diagnostic_abstention_reasons"]) == {
        "insufficient_same_route_calibration"
    }
    assert forecast[
        ["diagnostic_lower_pct", "diagnostic_upper_pct"]
    ].isna().all().all()
    assert set(forecast["operational_issuance_status"]) == {"abstained"}


def test_future_outcome_field_is_rejected_by_closed_schema(
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["forecast"][0]["capacity_retention_pct"] = 90.0

    with pytest.raises(
        CalendarPrefixRequestError,
        match="Request schema validation failed",
    ):
        validate_calendar_prefix_request(
            request,
            schema=prefix_assets["schema"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", "-schema-only-rejection"),
        ("requested_coverage", 0.8000000000001),
        ("temperature_c", 121.0),
    ],
)
def test_semantic_validator_matches_schema_when_schema_is_omitted(
    field: str,
    value: object,
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request[field] = value

    for schema in (None, prefix_assets["schema"]):
        with pytest.raises(CalendarPrefixRequestError):
            validate_calendar_prefix_request(request, schema=schema)


def test_schema_and_runtime_both_lock_the_initial_time(
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["prefix"][0]["elapsed_days"] = 1.0
    for schema in (None, prefix_assets["schema"]):
        with pytest.raises(CalendarPrefixRequestError):
            validate_calendar_prefix_request(request, schema=schema)


def test_schema_and_runtime_both_lock_initial_retention(
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["prefix"][0]["capacity_retention_pct"] = 99.0
    validator = Draft202012Validator(prefix_assets["schema"])
    assert list(validator.iter_errors(request))
    for schema in (None, prefix_assets["schema"]):
        with pytest.raises(CalendarPrefixRequestError):
            validate_calendar_prefix_request(request, schema=schema)


def test_schema_is_valid_and_uses_the_runtime_locked_time_grid(
    prefix_assets: dict[str, object],
) -> None:
    schema = prefix_assets["schema"]
    Draft202012Validator.check_schema(schema)
    prefix_times = [
        item["properties"]["elapsed_days"]["const"]
        for item in schema["properties"]["prefix"]["prefixItems"]
    ]
    forecast_times = [
        item["properties"]["elapsed_days"]["const"]
        for item in schema["properties"]["forecast"]["prefixItems"]
    ]
    assert tuple([*prefix_times, *forecast_times]) == (
        LOCKED_CALENDAR_ELAPSED_DAYS
    )


def test_json_schema_integral_numbers_are_normalized_to_integer_indices(
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["prefix"][1]["observation_index"] = 1.0
    request["forecast"][0]["forecast_index"] = 10.0

    for schema in (None, prefix_assets["schema"]):
        parsed = validate_calendar_prefix_request(request, schema=schema)
        assert parsed["prefix"][1]["observation_index"] == 1
        assert isinstance(parsed["prefix"][1]["observation_index"], int)
        assert parsed["forecast"][0]["forecast_index"] == 10
        assert isinstance(parsed["forecast"][0]["forecast_index"], int)


def _attack_request(
    original: dict[str, object],
    attack: str,
) -> dict[str, object]:
    request = copy.deepcopy(original)
    if attack == "prefix_out_of_order":
        request["prefix"][1], request["prefix"][2] = (
            request["prefix"][2],
            request["prefix"][1],
        )
    elif attack == "forecast_out_of_order":
        request["forecast"][0], request["forecast"][1] = (
            request["forecast"][1],
            request["forecast"][0],
        )
    elif attack == "prefix_duplicate":
        request["prefix"][2] = copy.deepcopy(request["prefix"][1])
    elif attack == "forecast_duplicate":
        request["forecast"][1] = copy.deepcopy(request["forecast"][0])
    elif attack == "prefix_too_short":
        request["prefix"].pop()
    elif attack == "forecast_too_short":
        request["forecast"].pop()
    elif attack == "nan":
        request["prefix"][5]["capacity_retention_pct"] = float("nan")
    elif attack == "wrong_statistical_unit":
        request["statistical_unit"] = "individual_cell"
    elif attack == "fifteen_year_coordinate":
        request["forecast"][-1]["elapsed_days"] = 15.0 * 365.0
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(f"Unknown attack: {attack}")
    return request


@pytest.mark.parametrize(
    "attack",
    [
        "prefix_out_of_order",
        "forecast_out_of_order",
        "prefix_duplicate",
        "forecast_duplicate",
        "prefix_too_short",
        "forecast_too_short",
        "nan",
        "wrong_statistical_unit",
        "fifteen_year_coordinate",
    ],
)
def test_request_contract_rejects_order_count_value_and_horizon_attacks(
    attack: str,
    prefix_assets: dict[str, object],
) -> None:
    attacked = _attack_request(prefix_assets["requests"]["fallback"], attack)
    with pytest.raises(CalendarPrefixRequestError):
        validate_calendar_prefix_request(
            attacked,
            schema=prefix_assets["schema"],
        )


def test_out_of_domain_request_exposes_no_numeric_prediction(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["request_id"] = "outside_training_hull"
    request["temperature_c"] = 100.0
    request["storage_soc_fraction"] = 1.0

    decision, forecast = _predict_with_state(
        request,
        observations=observations,
        assets=prefix_assets,
    )

    assert decision["status"] == "request_rejected_domain_unsupported"
    assert decision["mean_prediction"]["status"] == "unavailable"
    assert decision["mean_prediction"]["route"] == "unavailable"
    assert decision["support"]["domain_supported"] is False
    assert decision["diagnostic_interval"]["status"] == "unavailable"
    assert decision["operational_decision"]["issuance_status"] == "abstained"
    assert "domain_unsupported" in decision["operational_decision"][
        "abstention_reasons"
    ]
    assert decision["operational_decision"]["lower_pct"] is None
    assert decision["operational_decision"]["upper_pct"] is None

    assert len(forecast) == 25
    assert set(forecast["mean_route"]) == {"unavailable"}
    assert set(forecast["domain_supported"]) == {False}
    assert set(forecast["operational_issuance_status"]) == {"abstained"}
    assert forecast[
        [
            "predicted_capacity_retention_pct",
            "predictive_sd_pp",
            "residual_correction_pp",
            "diagnostic_lower_pct",
            "diagnostic_upper_pct",
        ]
    ].isna().all().all()


@pytest.mark.parametrize(
    "retention",
    [
        [100.0, *([0.0] * 9)],
        [100.0, 110.0, 0.0, 110.0, 0.0, 110.0, 0.0, 110.0, 0.0, 110.0],
        [
            100.0,
            97.186,
            97.669,
            96.363,
            96.819,
            95.465,
            95.871,
            94.084,
            94.503,
            93.199,
        ],
    ],
)
def test_extreme_schema_valid_prefixes_fail_closed_without_numeric_output(
    retention: list[float],
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["request_id"] = "extreme_prefix"
    for row, value in zip(request["prefix"], retention, strict=True):
        row["capacity_retention_pct"] = value

    decision, forecast = _predict_with_state(
        request,
        observations=observations,
        assets=prefix_assets,
    )

    assert decision["status"] == (
        "request_rejected_prefix_outside_reference_support"
    )
    assert decision["mean_prediction"]["status"] == "unavailable"
    assert decision["support"]["domain_supported"] is True
    assert decision["support"]["prefix_supported"] is False
    assert decision["operational_decision"]["issuance_status"] == "abstained"
    assert "prefix_outside_reference_support" in decision[
        "operational_decision"
    ]["abstention_reasons"]
    assert forecast[
        [
            "predicted_capacity_retention_pct",
            "predictive_sd_pp",
            "residual_correction_pp",
            "diagnostic_lower_pct",
            "diagnostic_upper_pct",
        ]
    ].isna().all().all()


def test_prefix_from_the_wrong_stress_condition_is_rejected(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
) -> None:
    request = copy.deepcopy(prefix_assets["requests"]["fallback"])
    request["request_id"] = "stress_mismatched_prefix"
    request["temperature_c"] = 60.0
    request["storage_soc_fraction"] = 1.0
    source = observations.loc[
        observations["condition_id"].astype(str).eq("NAUMANN_CAL_T0_SOC50")
        & pd.to_numeric(observations["checkup_index"]).lt(10)
    ].sort_values("checkup_index", kind="stable")
    for row, value in zip(
        request["prefix"],
        source["capacity_retention_pct"].to_numpy(dtype=float),
        strict=True,
    ):
        row["capacity_retention_pct"] = float(value)

    decision, forecast = _predict_with_state(
        request,
        observations=observations,
        assets=prefix_assets,
    )

    assert decision["status"] == (
        "request_rejected_prefix_outside_reference_support"
    )
    assert decision["support"]["prefix_reference_mode"] == (
        "exact_reference_condition"
    )
    assert decision["support"]["prefix_supported"] is False
    assert decision["support"]["prefix_max_point_deviation_pp"] > 0.5
    assert forecast["predicted_capacity_retention_pct"].isna().all()


def test_supported_prefix_model_failure_becomes_structured_abstention(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fit(*_: object, **__: object) -> object:
        raise RuntimeError("injected fit failure")

    monkeypatch.setattr(calendar_prefix, "predict_calendar_v4_condition", fail_fit)
    decision, forecast = _predict_with_state(
        prefix_assets["requests"]["fallback"],
        observations=observations,
        assets=prefix_assets,
    )

    assert decision["status"] == "request_rejected_model_fit_failed"
    assert decision["mean_prediction"]["status"] == "unavailable"
    assert decision["mean_prediction"]["failure_type"] == "RuntimeError"
    assert decision["support"]["prefix_supported"] is True
    assert decision["operational_decision"]["issuance_status"] == "abstained"
    assert forecast["predicted_capacity_retention_pct"].isna().all()


def _permute_test_future_outcomes(observations: pd.DataFrame) -> pd.DataFrame:
    attacked = observations.copy(deep=True)
    for condition_id in TEST_CONDITION_IDS:
        condition = attacked["condition_id"].astype(str).eq(condition_id)
        future = condition & pd.to_numeric(attacked["checkup_index"]).ge(10)
        row_indices = attacked.index[future].to_numpy()
        retention = attacked.loc[
            row_indices, "capacity_retention_pct"
        ].to_numpy(dtype=float)[::-1]
        initial_capacity = float(
            attacked.loc[
                condition & pd.to_numeric(attacked["checkup_index"]).eq(0),
                "capacity_ah",
            ].iloc[0]
        )
        attacked.loc[row_indices, "capacity_retention_pct"] = retention
        attacked.loc[row_indices, "capacity_loss_pct"] = 100.0 - retention
        attacked.loc[row_indices, "capacity_ah"] = initial_capacity * retention / 100.0
    return attacked


def test_reference_fit_and_api_ignore_all_test_condition_future_outcomes(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
) -> None:
    attacked_observations = _permute_test_future_outcomes(observations)
    attacked_state = fit_calendar_v4_reference_state(
        attacked_observations,
        config=prefix_assets["config"],
    )
    original_state = prefix_assets["state"]

    assert attacked_state.training_state_sha256 == (
        original_state.training_state_sha256
    )
    assert attacked_state.calibration_state_sha256 == (
        original_state.calibration_state_sha256
    )
    pd.testing.assert_frame_equal(
        attacked_state.residual_crossfit,
        original_state.residual_crossfit,
    )
    pd.testing.assert_frame_equal(
        attacked_state.calibration_condition_scores,
        original_state.calibration_condition_scores,
    )

    request = prefix_assets["requests"]["fallback"]
    original_decision, original_forecast = _predict_with_state(
        request,
        observations=observations,
        assets=prefix_assets,
        state=original_state,
    )
    attacked_decision, attacked_forecast = _predict_with_state(
        request,
        observations=attacked_observations,
        assets=prefix_assets,
        state=attacked_state,
    )
    assert attacked_decision == original_decision
    pd.testing.assert_frame_equal(attacked_forecast, original_forecast)


def test_reference_fit_and_api_accept_redacted_test_future_outcomes(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    redacted = observations.copy(deep=True)
    test_future = redacted["condition_id"].astype(str).isin(TEST_CONDITION_IDS) & (
        pd.to_numeric(redacted["checkup_index"]).ge(10)
    )
    outcome_columns = [
        "capacity_ah",
        "capacity_retention_pct",
        "capacity_loss_pct",
        "resistance_dc_ohm",
        "resistance_growth_pct",
    ]
    redacted.loc[test_future, outcome_columns] = float("nan")

    redacted_state = fit_calendar_v4_reference_state(
        redacted,
        config=prefix_assets["config"],
    )
    original_state = prefix_assets["state"]
    assert redacted_state.training_state_sha256 == (
        original_state.training_state_sha256
    )
    assert redacted_state.calibration_state_sha256 == (
        original_state.calibration_state_sha256
    )

    decision, forecast = predict_calendar_prefix(
        prefix_assets["requests"]["fallback"],
        reference_observations=redacted,
        config=prefix_assets["config"],
        schema=prefix_assets["schema"],
    )
    expected_decision, expected_forecast = golden_prefix_results["fallback"]
    assert decision == expected_decision
    pd.testing.assert_frame_equal(forecast, expected_forecast)

    reference_only = redacted.loc[
        ~redacted["condition_id"].astype(str).isin(TEST_CONDITION_IDS)
    ].copy()
    reduced_decision, reduced_forecast = predict_calendar_prefix(
        prefix_assets["requests"]["fallback"],
        reference_observations=reference_only,
        config=prefix_assets["config"],
        schema=prefix_assets["schema"],
    )
    assert reduced_decision == expected_decision
    pd.testing.assert_frame_equal(reduced_forecast, expected_forecast)


@pytest.mark.parametrize("attack", ["double_time_grid", "change_stress_identity"])
def test_reference_fit_rejects_noncanonical_reference_coordinates(
    attack: str,
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
) -> None:
    attacked = observations.copy(deep=True)
    reference_ids = set(TRAINING_CONDITION_IDS) | set(CALIBRATION_CONDITION_IDS)
    reference_rows = attacked["condition_id"].astype(str).isin(reference_ids)
    if attack == "double_time_grid":
        for column in ("elapsed_time_s", "elapsed_hours", "elapsed_days"):
            attacked.loc[reference_rows, column] = (
                pd.to_numeric(attacked.loc[reference_rows, column]) * 2.0
            )
        expected = "reference time axis mismatch"
    else:
        condition = attacked["condition_id"].astype(str).eq(
            "NAUMANN_CAL_T60_SOC100"
        )
        attacked.loc[condition, "temperature_c"] = 80.0
        expected = "reference identity mismatch"

    with pytest.raises(ValueError, match=expected):
        fit_calendar_v4_reference_state(
            attacked,
            config=prefix_assets["config"],
        )


def test_prediction_content_hash_is_independent_and_deterministic(
    observations: pd.DataFrame,
    prefix_assets: dict[str, object],
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    first_decision, first_forecast = golden_prefix_results["fallback"]
    second_decision, second_forecast = _predict_with_state(
        prefix_assets["requests"]["fallback"],
        observations=observations,
        assets=prefix_assets,
    )
    pd.testing.assert_frame_equal(first_forecast, second_forecast)
    assert first_decision == second_decision

    canonical_bytes = first_forecast.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    independent_hash = hashlib.sha256(canonical_bytes).hexdigest()
    assert first_decision["provenance"]["forecast_content_sha256"] == (
        independent_hash
    )
    assert len(independent_hash) == 64


def _cli_args(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        request=str(FALLBACK_REQUEST_PATH),
        reference=str(REFERENCE_PATH),
        config=str(CONFIG_PATH),
        schema=str(SCHEMA_PATH),
        output_dir=str(output_dir),
    )


def test_cli_packages_both_golden_decisions_without_changing_the_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    by_request_id = {
        result[0]["request_id"]: result
        for result in golden_prefix_results.values()
    }

    def fake_predict(
        request: dict[str, object],
        **_: object,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        decision, forecast = by_request_id[request["request_id"]]
        return copy.deepcopy(decision), forecast.copy(deep=True)

    monkeypatch.setattr(cli, "predict_calendar_prefix", fake_predict)
    cases = [
        (
            FALLBACK_REQUEST_PATH,
            "hierarchical_power_fallback",
            "available",
        ),
        (
            SPECIALIST_REQUEST_PATH,
            "hierarchical_activation_residual",
            "unavailable",
        ),
    ]
    for request_path, route, interval_status in cases:
        output_dir = tmp_path / request_path.stem
        args = _cli_args(output_dir)
        args.request = str(request_path)
        assert cli._calendar_prefix_predict(args) == 0

        decision_path = output_dir / "decision.json"
        forecast_path = output_dir / "forecast.csv"
        assert sorted(path.name for path in output_dir.iterdir()) == [
            "decision.json",
            "forecast.csv",
        ]
        written_decision = json.loads(decision_path.read_text(encoding="utf-8"))
        expected_decision, expected_forecast = by_request_id[
            written_decision["request_id"]
        ]

        assert written_decision["mean_prediction"]["route"] == route
        assert written_decision["diagnostic_interval"]["status"] == (
            interval_status
        )
        assert written_decision["operational_decision"]["issuance_status"] == (
            "abstained"
        )
        assert written_decision["artifacts"]["forecast"]["row_count"] == 25
        assert written_decision["artifacts"]["forecast"]["sha256"] == (
            hashlib.sha256(forecast_path.read_bytes()).hexdigest()
        )
        assert written_decision["provenance"]["forecast_content_sha256"] == (
            expected_decision["provenance"]["forecast_content_sha256"]
        )
        expected_bytes = expected_forecast.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        ).encode("utf-8")
        assert forecast_path.read_bytes() == expected_bytes


def test_cli_retries_transient_windows_publish_without_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    output_dir = tmp_path / "retry-bundle"
    expected_decision, expected_forecast = golden_prefix_results["fallback"]
    real_replace = os.replace
    predict_calls = 0
    replace_calls = 0
    sleeps: list[float] = []

    def counted_predict(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        nonlocal predict_calls
        predict_calls += 1
        return copy.deepcopy(expected_decision), expected_forecast.copy(deep=True)

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            error = PermissionError(13, "access denied")
            error.winerror = 5
            raise error
        real_replace(source, destination)

    monkeypatch.setattr(cli, "predict_calendar_prefix", counted_predict)
    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    assert cli._calendar_prefix_predict(_cli_args(output_dir)) == 0

    assert predict_calls == 1
    assert replace_calls == 2
    assert sleeps == [0.05]
    assert output_dir.is_dir()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "decision.json",
        "forecast.csv",
    ]


def test_cli_non_winerror5_publish_failure_is_immediate_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    output_dir = tmp_path / "nonretryable-bundle"
    expected_decision, expected_forecast = golden_prefix_results["fallback"]
    predict_calls = 0
    replace_calls = 0
    sleeps: list[float] = []

    def counted_predict(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        nonlocal predict_calls
        predict_calls += 1
        return copy.deepcopy(expected_decision), expected_forecast.copy(deep=True)

    def denied_replace(_source: Path, _destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        error = PermissionError(13, "sharing violation")
        error.winerror = 32
        raise error

    monkeypatch.setattr(cli, "predict_calendar_prefix", counted_predict)
    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", denied_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    with pytest.raises(PermissionError) as captured:
        cli._calendar_prefix_predict(_cli_args(output_dir))

    assert captured.value.winerror == 32
    assert predict_calls == 1
    assert replace_calls == 1
    assert sleeps == []
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".nonretryable-bundle.staging-*"))


def test_cli_publish_retry_exhaustion_retains_complete_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_prefix_results: dict[
        str, tuple[dict[str, object], pd.DataFrame]
    ],
) -> None:
    output_dir = tmp_path / "exhausted-bundle"
    expected_decision, expected_forecast = golden_prefix_results["fallback"]
    predict_calls = 0
    replace_calls = 0
    sleeps: list[float] = []

    def counted_predict(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        nonlocal predict_calls
        predict_calls += 1
        return copy.deepcopy(expected_decision), expected_forecast.copy(deep=True)

    def denied_replace(_source: Path, _destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        error = PermissionError(13, "access denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(cli, "predict_calendar_prefix", counted_predict)
    monkeypatch.setattr(atomic_publish.sys, "platform", "win32")
    monkeypatch.setattr(atomic_publish.os, "replace", denied_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    with pytest.raises(
        atomic_publish.AtomicPublishRetryExhausted,
        match="exhausted after 7 attempts",
    ) as captured:
        cli._calendar_prefix_predict(_cli_args(output_dir))

    retained = list(tmp_path.glob(".exhausted-bundle.staging-*"))
    assert captured.value.destination == output_dir
    assert captured.value.attempts == 7
    assert predict_calls == 1
    assert replace_calls == 7
    assert sleeps == list(atomic_publish.RETRY_DELAYS_SECONDS)
    assert not output_dir.exists()
    assert len(retained) == 1
    assert sorted(path.name for path in retained[0].iterdir()) == [
        "decision.json",
        "forecast.csv",
    ]


def test_source_checkout_cli_runs_outside_repo_with_default_reference_paths(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "real-cli-bundle"
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_path, environment.get("PYTHONPATH", ""))
        if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lifetwin.cli",
            "calendar-prefix-predict",
            "--request",
            str(FALLBACK_REQUEST_PATH),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    summary = json.loads(completed.stdout)
    decision_path = output_dir / "decision.json"
    forecast_path = output_dir / "forecast.csv"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    forecast = pd.read_csv(forecast_path)

    assert summary["status"] == (
        "research_forecast_generated_operationally_abstained"
    )
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    assert decision["lifetwin_version"] == project["project"]["version"]
    assert decision["support"]["prefix_supported"] is True
    assert decision["artifacts"]["forecast"]["row_count"] == 25
    assert decision["artifacts"]["forecast"]["sha256"] == hashlib.sha256(
        forecast_path.read_bytes()
    ).hexdigest()
    assert len(forecast) == 25

    assert project["project"]["scripts"]["lifetwin"] == "lifetwin.cli:main"


def test_cli_never_overwrites_an_existing_evidence_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing-bundle"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="never overwrites"):
        cli._calendar_prefix_predict(_cli_args(output_dir))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output_dir.iterdir()) == [sentinel]


def test_cli_failure_removes_staging_and_never_publishes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "failed-bundle"
    forecast = pd.DataFrame({"forecast_index": [10], "value": [99.0]})
    decision = {
        "provenance": {"forecast_content_sha256": "0" * 64},
    }

    monkeypatch.setattr(
        cli,
        "predict_calendar_prefix",
        lambda *args, **kwargs: (copy.deepcopy(decision), forecast.copy()),
    )
    with pytest.raises(
        RuntimeError,
        match="Written forecast disagrees with the in-memory freeze",
    ):
        cli._calendar_prefix_predict(_cli_args(output_dir))

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".failed-bundle.staging-*"))


def test_existing_v4_prediction_evidence_bytes_remain_frozen() -> None:
    assert hashlib.sha256(FROZEN_V4_PATH.read_bytes()).hexdigest() == (
        FROZEN_V4_SHA256
    )
