from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v015_training as v015_training,
)
from lifetwin.experiments import (
    calendar_long_horizon_v016_prediction_capsule as capsule,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    canonical_json_bytes,
    load_artifact_contract,
    predictor_content_hashes,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    StandardizerState,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
    FrozenLabelFreeState,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
)
from lifetwin.experiments.calendar_long_horizon_v016_pipeline import (
    _recompute_label_free_pipeline_hand_fixture_v021,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    DEFAULT_V021_AMENDMENT_PATH,
    load_v021_design,
)


_CONFIG_SHA256 = "a" * 64
_CREATED_UTC = "2026-07-26T03:00:00Z"


def _risk_state(
    feature_names: tuple[str, ...],
    coefficients: tuple[float, ...] | None = None,
) -> LogisticRiskState:
    dimension = len(feature_names)
    return LogisticRiskState(
        feature_names=feature_names,
        standardizer=StandardizerState(
            mean=(0.0,) * dimension,
            scale=(1.0,) * dimension,
            zero_variance=(False,) * dimension,
        ),
        intercept=-4.0,
        coefficients=coefficients or (0.0,) * dimension,
    )


def _prediction_state() -> capsule.PredictionState:
    prefix_coefficients = [0.0] * len(PREFIX_FEATURE_NAMES)
    prefix_coefficients[0] = 0.1
    visible_coefficients = (
        *prefix_coefficients,
        *(0.0 for _ in REAL_OPERATING_FIELDS),
    )
    return capsule.PredictionState(
        center_beta=0.5,
        prefix_only_risk=_risk_state(
            PREFIX_FEATURE_NAMES,
            tuple(prefix_coefficients),
        ),
        visible_stress_risk=_risk_state(
            VISIBLE_STRESS_FEATURE_NAMES,
            tuple(visible_coefficients),
        ),
        placebo_risk=_risk_state(PLACEBO_FEATURE_NAMES),
        arm_a_plus_s_plan_risk=_risk_state(ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
        strongest_single_feature_name=PREFIX_FEATURE_NAMES[0],
        strongest_single_feature_orientation=1,
        prefix_only_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        visible_stress_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        conformal=ConformalExpansionState(
            coverage=0.9,
            calibration_count=900,
            order_statistic_index=811,
            expansion_pp=1.0,
        ),
    )


def _training_state(
    prediction_state: capsule.PredictionState,
) -> v015_training.FrozenTrainingState:
    return v015_training.FrozenTrainingState(
        center=v015_training.CenterDevelopmentState(beta=prediction_state.center_beta),
        risk=v015_training.RiskDevelopmentState(
            prefix_only_risk=prediction_state.prefix_only_risk,
            visible_stress_risk=prediction_state.visible_stress_risk,
            placebo_risk=prediction_state.placebo_risk,
            arm_a_plus_s_plan_risk=prediction_state.arm_a_plus_s_plan_risk,
            strongest_single_feature_name=(
                prediction_state.strongest_single_feature_name
            ),
            strongest_single_feature_orientation=(
                prediction_state.strongest_single_feature_orientation
            ),
            strongest_single_feature_auroc=0.5,
            development_cluster_count=600,
            eligible_cluster_count=600,
            positive_label_count=300,
            negative_label_count=300,
        ),
        calibration=v015_training.CalibrationDevelopmentState(
            prefix_only_isotonic=prediction_state.prefix_only_isotonic,
            visible_stress_isotonic=prediction_state.visible_stress_isotonic,
            conformal=prediction_state.conformal,
            selected_mean_baseline="target_prefix_persistence",
            mean_baseline_iae_pp=(
                ("target_prefix_persistence", 1.0),
                ("target_prefix_sqrt_time", 2.0),
                ("target_prefix_bounded_power_law", 3.0),
            ),
            calibration_cluster_count=900,
            positive_label_count=450,
            negative_label_count=450,
        ),
    )


def _model_state_payload() -> dict[str, object]:
    prediction_state = _prediction_state()
    payload = v015_training.build_model_state_payload(
        _training_state(prediction_state),
        center_development_input_hashes={"center_arrays.bin": "1" * 64},
        risk_development_input_hashes={"risk_arrays.bin": "2" * 64},
        calibration_input_hashes={"calibration_arrays.bin": "3" * 64},
        software_versions=v015_training.default_software_versions(),
        created_utc=_CREATED_UTC,
    )
    payload["protocol_id"] = capsule.V021_PROTOCOL_ID
    payload["config_sha256"] = _CONFIG_SHA256
    return payload


def _v021_contract() -> FrozenArtifactContract:
    design = load_v021_design()
    return replace(
        load_artifact_contract(),
        protocol_id=capsule.V021_PROTOCOL_ID,
        config_path=DEFAULT_V021_AMENDMENT_PATH,
        config_byte_sha256=design.config_byte_sha256,
    )


def _identity_frame(
    filename: str,
    records: list[dict[str, object]],
) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        columns=_v021_contract().csv_schema(filename).columns,
    )


@pytest.fixture(scope="module")
def hand_fixture() -> tuple[pd.DataFrame, ...]:
    cluster_id = "v021-capsule-fixture"
    prefix = _identity_frame(
        "prefix_pack.csv",
        [
            {
                "protocol_id": capsule.V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                "prefix_day": day,
                "observed_retention_pct": (100.0 - 0.8 * math.sqrt(day / 365.25)),
            }
            for day in PREFIX_DAYS
        ],
    )
    coordinates = _identity_frame(
        "forecast_coordinates.csv",
        [
            {
                "protocol_id": capsule.V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                "forecast_day": day,
            }
            for day in FORECAST_DAYS
        ],
    )
    operating = _identity_frame(
        "operating_pack.csv",
        [
            {
                "protocol_id": capsule.V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                **dict(
                    zip(
                        REAL_OPERATING_FIELDS,
                        (
                            25.0,
                            0.5,
                            0.55,
                            250.0,
                            31.0,
                            0.6,
                            0.65,
                            300.0,
                        ),
                        strict=True,
                    )
                ),
                **{
                    name: -0.8 + 0.2 * index
                    for index, name in enumerate(PLACEBO_FIELDS)
                },
            }
        ],
    )
    fitted = fit_structure_library(
        prefix.assign(protocol_id=FROZEN_PROTOCOL_ID),
        coordinates.assign(protocol_id=FROZEN_PROTOCOL_ID),
    )
    hashes = predictor_content_hashes(prefix, coordinates, operating.iloc[0])
    diagnostics = fitted.member_fit_diagnostics.assign(
        protocol_id=capsule.V021_PROTOCOL_ID,
        canonical_prefix_content_sha256=hashes.arm_a,
    )
    forecasts = fitted.member_forecast_bundle.assign(
        protocol_id=capsule.V021_PROTOCOL_ID,
        canonical_prefix_content_sha256=hashes.arm_a,
    )
    return prefix, coordinates, operating, diagnostics, forecasts


def test_isolated_prediction_call_time_imports_only_safe_capabilities() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    module_prefix = "lifetwin.experiments.calendar_long_horizon_"
    script = f"""
import json
import sys
sys.path.insert(0, {str(source_root)!r})
from lifetwin.experiments import calendar_long_horizon_v016_prediction as prediction
try:
    prediction.run_isolated_prediction_process_v021(
        label_free_root="definitely-not-a-v021-attempt",
        attempt_id="capsule-import-probe",
        repo_root="definitely-not-a-repository",
    )
except Exception:
    pass
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith({module_prefix!r})
)))
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        cwd=Path(__file__).resolve().parents[1],
    )
    imported = set(json.loads(completed.stdout))
    stem = "lifetwin.experiments.calendar_long_horizon_"
    forbidden_suffixes = {
        "synthetic",
        "v015_fit",
        "v015_io",
        "v015_pipeline",
        "v015_protocol",
        "v015_training",
        "v016_actual_ledger_io",
        "v016_analysis",
        "v016_collision",
        "v016_contract",
        "v016_environment",
        "v016_firewall",
        "v016_fit",
        "v016_generation",
        "v016_io",
        "v016_pipeline",
        "v016_protocol",
        "v016_provenance",
        "v016_runner",
        "v016_scoring",
        "v016_signals",
        "v016_state",
        "v016_terminal",
        "v016_training",
    }
    forbidden = {f"{stem}{suffix}" for suffix in forbidden_suffixes}
    assert imported.isdisjoint(forbidden), sorted(imported & forbidden)
    assert {
        f"{stem}v015_model",
        f"{stem}v016_ledger",
        f"{stem}v016_prediction",
        f"{stem}v016_prediction_capsule",
        f"{stem}v016_prediction_environment",
    }.issubset(imported)


def test_capsule_decoder_matches_independent_frozen_state_codec() -> None:
    payload = _model_state_payload()
    raw = canonical_json_bytes(payload)
    decoded = capsule.decode_prediction_state(
        raw,
        expected_config_sha256=_CONFIG_SHA256,
    )
    assert decoded.state == _prediction_state()
    assert {
        phase: dict(hashes) for phase, hashes in decoded.input_byte_hashes.items()
    } == payload["input_byte_hashes"]
    assert decoded.model_state_byte_sha256 == hashlib.sha256(raw).hexdigest()


def test_capsule_decoder_fails_closed_on_identity_schema_and_json_tampering() -> None:
    raw = canonical_json_bytes(_model_state_payload())
    with pytest.raises(capsule.V021PredictionCapsuleError):
        capsule.decode_prediction_state(
            raw,
            expected_config_sha256="b" * 64,
        )

    extra = _model_state_payload()
    extra["unexpected_capability"] = "truth"
    with pytest.raises(capsule.V021PredictionCapsuleError):
        capsule.decode_prediction_state(
            canonical_json_bytes(extra),
            expected_config_sha256=_CONFIG_SHA256,
        )

    duplicate = raw.replace(
        b'"protocol_id":',
        b'"protocol_id": "duplicate",\n  "protocol_id":',
        1,
    )
    with pytest.raises(capsule.V021PredictionCapsuleError):
        capsule.decode_prediction_state(
            duplicate,
            expected_config_sha256=_CONFIG_SHA256,
        )

    feature_order = _model_state_payload()
    feature_order["feature_orders"]["prefix_only"].reverse()  # type: ignore[index]
    with pytest.raises(capsule.V021PredictionCapsuleError):
        capsule.decode_prediction_state(
            canonical_json_bytes(feature_order),
            expected_config_sha256=_CONFIG_SHA256,
        )


def test_capsule_core_is_exactly_equivalent_to_existing_v021_pipeline(
    hand_fixture: tuple[pd.DataFrame, ...],
) -> None:
    frames = tuple(frame.copy(deep=True) for frame in hand_fixture)
    originals = tuple(frame.copy(deep=True) for frame in frames)
    state = _prediction_state()
    reference_state = FrozenLabelFreeState(
        center_beta=state.center_beta,
        prefix_only_risk=state.prefix_only_risk,
        visible_stress_risk=state.visible_stress_risk,
        placebo_risk=state.placebo_risk,
        arm_a_plus_s_plan_risk=state.arm_a_plus_s_plan_risk,
        strongest_single_feature_name=state.strongest_single_feature_name,
        strongest_single_feature_orientation=(
            state.strongest_single_feature_orientation
        ),
        prefix_only_isotonic=state.prefix_only_isotonic,
        visible_stress_isotonic=state.visible_stress_isotonic,
        conformal=state.conformal,
    )
    expected = _recompute_label_free_pipeline_hand_fixture_v021(
        prefix_pack=frames[0],
        forecast_coordinates=frames[1],
        operating_pack=frames[2],
        member_fit_diagnostics=frames[3],
        member_forecast_bundle=frames[4],
        state=reference_state,
        contract=_v021_contract(),
    )
    observed = capsule.recompute_prediction_pipeline(
        prefix_pack=frames[0],
        forecast_coordinates=frames[1],
        operating_pack=frames[2],
        member_fit_diagnostics=frames[3],
        member_forecast_bundle=frames[4],
        state=state,
        formal=False,
    )

    for frame, original in zip(frames, originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)
    for field in (
        "prediction_bundle",
        "feature_bundle",
        "primary_risk_bundle",
        "decision_bundle",
        "predictor_content_bundle",
    ):
        pd.testing.assert_frame_equal(
            getattr(observed, field),
            getattr(expected, field),
        )


def test_writer_rejects_an_unbound_recomputed_result(
    hand_fixture: tuple[pd.DataFrame, ...],
) -> None:
    result = capsule.recompute_prediction_pipeline(
        prefix_pack=hand_fixture[0].copy(deep=True),
        forecast_coordinates=hand_fixture[1].copy(deep=True),
        operating_pack=hand_fixture[2].copy(deep=True),
        member_fit_diagnostics=hand_fixture[3].copy(deep=True),
        member_forecast_bundle=hand_fixture[4].copy(deep=True),
        state=_prediction_state(),
        formal=False,
    )

    with pytest.raises(
        capsule.V021PredictionCapsuleError,
        match="not issued for this exact sealed bundle",
    ):
        capsule._bound_prediction_output_bytes(object(), result)  # type: ignore[arg-type]


def test_writer_binding_is_external_to_replaceable_result_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encode(
        frame: pd.DataFrame,
        filename: str,
        *,
        formal: bool,
    ) -> bytes:
        assert formal is True
        return f"{filename}:{frame.iloc[0, 0]}".encode("ascii")

    monkeypatch.setattr(capsule, "canonical_csv_bytes", encode)
    frame = pd.DataFrame({"value": [1]})
    result = capsule.PredictionPipelineResult(
        prediction_bundle=frame.copy(deep=True),
        feature_bundle=frame.copy(deep=True),
        primary_risk_bundle=frame.copy(deep=True),
        decision_bundle=frame.copy(deep=True),
        predictor_content_bundle=frame.copy(deep=True),
    )
    bundle = object()
    raw_by_name = {
        filename: encode(
            getattr(result, attribute),
            filename,
            formal=True,
        )
        for filename, attribute in capsule._PIPELINE_OUTPUT_FIELDS
    }
    binding = (
        result,
        bundle,
        tuple(
            (
                filename,
                hashlib.sha256(raw_by_name[filename]).hexdigest(),
            )
            for filename, _ in capsule._PIPELINE_OUTPUT_FIELDS
        ),
    )
    capsule._PIPELINE_RESULT_BINDINGS[id(result)] = binding  # type: ignore[assignment]
    try:
        assert (
            capsule._bound_prediction_output_bytes(  # type: ignore[arg-type]
                bundle,
                result,
            )
            == raw_by_name
        )

        forged = replace(
            result,
            prediction_bundle=pd.DataFrame({"value": [2]}),
        )
        with pytest.raises(
            capsule.V021PredictionCapsuleError,
            match="not issued for this exact sealed bundle",
        ):
            capsule._bound_prediction_output_bytes(  # type: ignore[arg-type]
                bundle,
                forged,
            )

        result.prediction_bundle.loc[0, "value"] = 2
        with pytest.raises(
            capsule.V021PredictionCapsuleError,
            match="changed after capsule execution",
        ):
            capsule._bound_prediction_output_bytes(  # type: ignore[arg-type]
                bundle,
                result,
            )
    finally:
        capsule._PIPELINE_RESULT_BINDINGS.pop(id(result), None)


def test_capsule_core_rejects_committed_prefix_hash_tampering(
    hand_fixture: tuple[pd.DataFrame, ...],
) -> None:
    diagnostics = hand_fixture[3].copy(deep=True)
    diagnostics.loc[0, "canonical_prefix_content_sha256"] = "0" * 64
    with pytest.raises(capsule.V021PredictionCapsuleError):
        capsule.recompute_prediction_pipeline(
            prefix_pack=hand_fixture[0].copy(deep=True),
            forecast_coordinates=hand_fixture[1].copy(deep=True),
            operating_pack=hand_fixture[2].copy(deep=True),
            member_fit_diagnostics=diagnostics,
            member_forecast_bundle=hand_fixture[4].copy(deep=True),
            state=_prediction_state(),
            formal=False,
        )


def test_capsule_core_rejects_succeeded_member_forecast_formula_tampering(
    hand_fixture: tuple[pd.DataFrame, ...],
) -> None:
    diagnostics = hand_fixture[3].copy(deep=True)
    forecasts = hand_fixture[4].copy(deep=True)
    succeeded = diagnostics.loc[diagnostics["fit_status"].eq("succeeded")]
    assert not succeeded.empty
    committed = succeeded.iloc[0]
    variant_rows = forecasts.loc[
        forecasts["partition"].eq(committed["partition"])
        & forecasts["cluster_id"].eq(committed["cluster_id"])
        & forecasts["model_id"].eq(committed["model_id"])
        & forecasts["variant_id"].eq(committed["variant_id"])
    ].sort_values("forecast_day", kind="stable")
    assert len(variant_rows) == len(FORECAST_DAYS)
    target = variant_rows.index[0]
    forecasts.loc[target, "raw_forecast_retention_pct"] = (
        float(forecasts.loc[target, "raw_forecast_retention_pct"]) + 0.1
    )

    with pytest.raises(
        capsule.V021PredictionCapsuleError,
        match=("raw_forecast_retention_pct differs from frozen formula recomputation"),
    ):
        capsule.recompute_prediction_pipeline(
            prefix_pack=hand_fixture[0].copy(deep=True),
            forecast_coordinates=hand_fixture[1].copy(deep=True),
            operating_pack=hand_fixture[2].copy(deep=True),
            member_fit_diagnostics=diagnostics,
            member_forecast_bundle=forecasts,
            state=_prediction_state(),
            formal=False,
        )


@pytest.mark.parametrize(
    "field",
    (
        "prefix_rmse_pp",
        "parameter_boundary_hit_fraction",
    ),
)
def test_capsule_core_rejects_succeeded_member_metric_tampering(
    hand_fixture: tuple[pd.DataFrame, ...],
    field: str,
) -> None:
    diagnostics = hand_fixture[3].copy(deep=True)
    succeeded = diagnostics.index[diagnostics["fit_status"].eq("succeeded")]
    assert len(succeeded) > 0
    target = succeeded[0]
    original = float(diagnostics.loc[target, field])
    delta = -0.1 if original > 0.9 else 0.1
    diagnostics.loc[target, field] = original + delta

    with pytest.raises(
        capsule.V021PredictionCapsuleError,
        match=rf"{field} differs from frozen formula recomputation",
    ):
        capsule.recompute_prediction_pipeline(
            prefix_pack=hand_fixture[0].copy(deep=True),
            forecast_coordinates=hand_fixture[1].copy(deep=True),
            operating_pack=hand_fixture[2].copy(deep=True),
            member_fit_diagnostics=diagnostics,
            member_forecast_bundle=hand_fixture[4].copy(deep=True),
            state=_prediction_state(),
            formal=False,
        )
