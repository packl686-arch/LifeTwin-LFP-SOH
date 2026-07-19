from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.attia import (
    ATTIA_DATASET_ID,
    ATTIA_OUTCOME_COLUMNS,
    EXPECTED_CELL_IDS,
    attia_outcome_artifact_sha256,
    validate_attia_outcome_pack,
)
from lifetwin.experiments.external_validation import (
    EXTERNAL_PREDICTION_SCHEMA_VERSION,
    NULL_MODEL_NAME,
    SOURCE_MODEL_NAME,
    external_prediction_artifact_sha256,
)


EXPERIMENT_ID = "attia_crosswalk_permutation_sensitivity_v1"
ANALYSIS_ROLE = "mapping_sensitivity_not_external_validation"
FROZEN_CONFIG_SHA256 = (
    "c25f2aacab343109abfb484cce88943cb2e582c059120e3274cd90e08016a194"
)
FROZEN_PREDICTION_RAW_SHA256 = (
    "8df939c9530e28be17b0711b995f87a3d53d5b7caf34aa41309a5eb520467e06"
)
FROZEN_PREDICTION_CANONICAL_SHA256 = (
    "c5f31b9c9b3401998f714833694a0301985494fe27c1c81a490b6d2ba6975beb"
)
FROZEN_OUTCOME_RAW_SHA256 = (
    "337b0f3ab252d904dcf74bdc8273de96fffcc81fb02f53da7df92bbb9b53b1b7"
)
FROZEN_OUTCOME_CANONICAL_SHA256 = (
    "8a673980a723969da63d1cd8c7ac97772880286a6ba970e925bbdf179aab0f4f"
)
FROZEN_REFERENCE_VALIDATION_RAW_SHA256 = (
    "b3d0286aa7765fcffa113f6ecacb39844c117bae9dbc82d0329483e175f88258"
)
FROZEN_SEED = 20260719
FROZEN_RESAMPLES = 200_000
FROZEN_SIGNAL_GATE_THRESHOLDS = {
    "maximum_delta_nll": 0.0,
    "minimum_mape_improvement_fraction": 0.1,
    "minimum_protocols_with_mape_improvement": 8,
}
STATISTIC_NAMES = (
    "protocol_balanced_delta_nll",
    "protocol_balanced_delta_mape",
    "protocols_with_mape_improvement",
)
LOCAL_PERMUTATIONS = tuple(itertools.permutations(range(5)))
LOCAL_PERMUTATION_INDEX = np.asarray(LOCAL_PERMUTATIONS, dtype=np.intp)
JOINT_MAPPING_SPACE_SIZE = math.factorial(5) ** 9

PREDICTION_COLUMNS = (
    "dataset_id",
    "cell_id",
    "test_id",
    "protocol_id",
    "source_cell_id",
    "log10_delta_q_variance",
    "prediction_schema_version",
    f"{SOURCE_MODEL_NAME}_log_location",
    f"{SOURCE_MODEL_NAME}_predictive_sigma",
    f"{SOURCE_MODEL_NAME}_p10",
    f"{SOURCE_MODEL_NAME}_p50",
    f"{SOURCE_MODEL_NAME}_p90",
    f"{NULL_MODEL_NAME}_log_location",
    f"{NULL_MODEL_NAME}_predictive_sigma",
    f"{NULL_MODEL_NAME}_p10",
    f"{NULL_MODEL_NAME}_p50",
    f"{NULL_MODEL_NAME}_p90",
)

_TOP_LEVEL_CONFIG_KEYS = {
    "experiment_id",
    "analysis_role",
    "inputs",
    "cohort",
    "permutation",
    "preregistered_statistics",
    "signal_gate_thresholds",
    "reporting",
    "claim_boundary",
}
_HASH_KEYS = {
    "predictions": {"path", "raw_sha256", "canonical_sha256", "schema_version"},
    "outcomes": {"path", "raw_sha256", "canonical_sha256", "schema_version"},
    "reference_validation": {"path", "raw_sha256", "expected_status"},
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_attia_crosswalk_sensitivity_config(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Attia crosswalk sensitivity config: {config_path}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("Attia crosswalk sensitivity config must be a JSON object")
    _validate_config(config)
    return config, file_sha256(config_path)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(
            f"Unexpected {label} keys: missing={missing}, unexpected={unexpected}"
        )


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _validate_config(config: Mapping[str, object]) -> None:
    _require_exact_keys(config, _TOP_LEVEL_CONFIG_KEYS, label="top-level config")
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError(f"Unexpected sensitivity experiment_id: {config['experiment_id']}")
    if config["analysis_role"] != ANALYSIS_ROLE:
        raise ValueError(f"Unexpected sensitivity analysis_role: {config['analysis_role']}")

    inputs = config["inputs"]
    if not isinstance(inputs, Mapping):
        raise ValueError("Sensitivity inputs must be an object")
    _require_exact_keys(inputs, set(_HASH_KEYS), label="input")
    for name, keys in _HASH_KEYS.items():
        item = inputs[name]
        if not isinstance(item, Mapping):
            raise ValueError(f"Sensitivity input {name} must be an object")
        _require_exact_keys(item, keys, label=f"{name} input")
        if not isinstance(item["path"], str) or not item["path"]:
            raise ValueError(f"Sensitivity input {name} path must be non-empty")
        _require_sha256(item["raw_sha256"], label=f"{name} raw_sha256")
        if "canonical_sha256" in item:
            _require_sha256(
                item["canonical_sha256"],
                label=f"{name} canonical_sha256",
            )
    frozen_hashes = {
        ("predictions", "raw_sha256"): FROZEN_PREDICTION_RAW_SHA256,
        ("predictions", "canonical_sha256"): FROZEN_PREDICTION_CANONICAL_SHA256,
        ("outcomes", "raw_sha256"): FROZEN_OUTCOME_RAW_SHA256,
        ("outcomes", "canonical_sha256"): FROZEN_OUTCOME_CANONICAL_SHA256,
        (
            "reference_validation",
            "raw_sha256",
        ): FROZEN_REFERENCE_VALIDATION_RAW_SHA256,
    }
    for (input_name, hash_name), expected in frozen_hashes.items():
        if inputs[input_name][hash_name] != expected:
            raise ValueError(
                f"Sensitivity config {input_name} {hash_name} is not frozen"
            )
    if inputs["predictions"]["schema_version"] != EXTERNAL_PREDICTION_SCHEMA_VERSION:
        raise ValueError("Sensitivity config must bind the v2 prediction schema")
    if inputs["outcomes"]["schema_version"] != "attia_validation45_outcomes_v2":
        raise ValueError("Sensitivity config must bind the v2 outcome schema")
    if inputs["reference_validation"]["expected_status"] != (
        "external_signal_gate_failed"
    ):
        raise ValueError("Sensitivity config must preserve the frozen negative result")

    cohort = config["cohort"]
    if not isinstance(cohort, Mapping):
        raise ValueError("Sensitivity cohort must be an object")
    _require_exact_keys(
        cohort,
        {
            "dataset_id",
            "cell_count",
            "protocol_count",
            "cells_per_protocol",
            "protocol_ids",
            "cell_order",
        },
        label="cohort",
    )
    if (
        cohort["dataset_id"] != ATTIA_DATASET_ID
        or cohort["cell_count"] != 45
        or cohort["protocol_count"] != 9
        or cohort["cells_per_protocol"] != 5
        or cohort["cell_order"] != "lexicographic_cell_id"
    ):
        raise ValueError("Sensitivity cohort must remain the frozen Attia 9x5 cohort")
    protocol_ids = cohort["protocol_ids"]
    if (
        not isinstance(protocol_ids, list)
        or len(protocol_ids) != 9
        or protocol_ids != sorted(set(str(item) for item in protocol_ids))
    ):
        raise ValueError("Sensitivity protocol_ids must be nine sorted unique strings")

    permutation = config["permutation"]
    if not isinstance(permutation, Mapping):
        raise ValueError("Sensitivity permutation design must be an object")
    _require_exact_keys(
        permutation,
        {
            "unit",
            "permuted_bundle_columns",
            "fixed_structure",
            "local_enumeration",
            "local_permutations_per_protocol",
            "joint_mapping_space_size",
            "joint_sampling",
            "with_replacement",
            "rng",
            "seed",
            "resamples",
        },
        label="permutation design",
    )
    expected_design = {
        "unit": "within_protocol_outcome_identity_bundle",
        "permuted_bundle_columns": ["replicate_id", "cycle_life"],
        "fixed_structure": "nine_protocols_by_five_cells",
        "local_enumeration": "exhaustive_all_5_factorial_assignments",
        "local_permutations_per_protocol": len(LOCAL_PERMUTATIONS),
        "joint_mapping_space_size": JOINT_MAPPING_SPACE_SIZE,
        "joint_sampling": "independent_uniform_local_assignment_per_protocol",
        "with_replacement": True,
        "rng": "numpy_PCG64",
    }
    for key, expected in expected_design.items():
        if permutation[key] != expected:
            raise ValueError(f"Unexpected permutation design value for {key}")
    seed = permutation["seed"]
    resamples = permutation["resamples"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Permutation seed must be a non-negative integer")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or not 1 <= resamples <= 10_000_000
    ):
        raise ValueError("Permutation resamples must be between 1 and 10,000,000")

    statistics = config["preregistered_statistics"]
    if not isinstance(statistics, list) or len(statistics) != len(STATISTIC_NAMES):
        raise ValueError("Sensitivity statistics must contain the three fixed statistics")
    observed_names: list[str] = []
    for statistic in statistics:
        if not isinstance(statistic, Mapping):
            raise ValueError("Each sensitivity statistic must be an object")
        _require_exact_keys(
            statistic,
            {"name", "definition", "direction"},
            label="statistic",
        )
        observed_names.append(str(statistic["name"]))
        if not isinstance(statistic["definition"], str) or not statistic["definition"]:
            raise ValueError("Sensitivity statistic definitions must be non-empty")
    if tuple(observed_names) != STATISTIC_NAMES:
        raise ValueError("Sensitivity statistic names or order changed")
    if [item["direction"] for item in statistics] != [
        "negative_is_better",
        "negative_is_better",
        "higher_is_better",
    ]:
        raise ValueError("Sensitivity statistic directions changed")

    thresholds = config["signal_gate_thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("Signal-gate thresholds must be an object")
    _require_exact_keys(
        thresholds,
        {
            "maximum_delta_nll",
            "minimum_mape_improvement_fraction",
            "minimum_protocols_with_mape_improvement",
        },
        label="signal-gate threshold",
    )
    numeric_thresholds = [
        thresholds["maximum_delta_nll"],
        thresholds["minimum_mape_improvement_fraction"],
        thresholds["minimum_protocols_with_mape_improvement"],
    ]
    if any(isinstance(value, bool) for value in numeric_thresholds):
        raise ValueError("Signal-gate thresholds must be numeric")
    try:
        finite_thresholds = np.asarray(numeric_thresholds, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Signal-gate thresholds must be numeric") from exc
    if not np.isfinite(finite_thresholds).all():
        raise ValueError("Signal-gate thresholds must be finite")

    reporting = config["reporting"]
    if not isinstance(reporting, Mapping):
        raise ValueError("Sensitivity reporting design must be an object")
    _require_exact_keys(
        reporting,
        {"quantiles", "quantile_method", "position_definition"},
        label="reporting",
    )
    quantiles = reporting["quantiles"]
    if (
        not isinstance(quantiles, list)
        or not quantiles
        or any(isinstance(value, bool) for value in quantiles)
    ):
        raise ValueError("Reporting quantiles must be a non-empty numeric list")
    try:
        numeric_quantiles = [float(value) for value in quantiles]
    except (TypeError, ValueError) as exc:
        raise ValueError("Reporting quantiles must be numeric") from exc
    if (
        numeric_quantiles != sorted(set(numeric_quantiles))
        or numeric_quantiles[0] != 0.0
        or numeric_quantiles[-1] != 1.0
        or any(value < 0.0 or value > 1.0 for value in numeric_quantiles)
    ):
        raise ValueError("Reporting quantiles must be sorted, unique, and span [0, 1]")
    if reporting["quantile_method"] != "linear":
        raise ValueError("Sensitivity quantile method must remain linear")
    if reporting["position_definition"] != (
        "empirical_fractions_strictly_below_and_at_or_below_observed"
    ):
        raise ValueError("Sensitivity position definition changed")

    boundary = config["claim_boundary"]
    if not isinstance(boundary, list) or len(boundary) != 4 or set(boundary) != {
        "mapping_sensitivity_only",
        "not_a_new_external_validation",
        "not_a_p_value_or_significance_proof",
        "no_retraining_no_threshold_tuning_no_calibration",
    }:
        raise ValueError("Sensitivity claim boundary must remain explicit")


def _validate_production_preregistration(
    config: Mapping[str, object],
    config_sha256: str,
) -> None:
    if config_sha256 != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "Attia crosswalk sensitivity preregistration config SHA-256 mismatch: "
            f"expected {FROZEN_CONFIG_SHA256}, found {config_sha256}"
        )
    if config["permutation"]["seed"] != FROZEN_SEED:
        raise ValueError("Production Attia sensitivity seed drifted")
    if config["permutation"]["resamples"] != FROZEN_RESAMPLES:
        raise ValueError("Production Attia sensitivity resample count drifted")
    if config["signal_gate_thresholds"] != FROZEN_SIGNAL_GATE_THRESHOLDS:
        raise ValueError("Production Attia sensitivity signal-gate thresholds drifted")


def _validate_raw_hash(
    observed: str,
    expected: object,
    *,
    label: str,
) -> None:
    expected_hash = _require_sha256(expected, label=f"expected {label} raw SHA-256")
    if observed != expected_hash:
        raise ValueError(
            f"Attia sensitivity {label} raw SHA-256 mismatch: "
            f"expected {expected_hash}, found {observed}"
        )


def _validate_predictions(
    predictions: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, str]:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            "Attia sensitivity requires the exact v2 prediction schema and column order"
        )
    canonical_sha256 = external_prediction_artifact_sha256(predictions)
    expected = str(config["inputs"]["predictions"]["canonical_sha256"])
    if canonical_sha256 != expected:
        raise ValueError(
            "Attia sensitivity prediction canonical SHA-256 mismatch: "
            f"expected {expected}, found {canonical_sha256}"
        )
    if len(predictions) != 45:
        raise ValueError("Attia sensitivity requires exactly 45 frozen predictions")
    if set(predictions["dataset_id"].astype(str)) != {ATTIA_DATASET_ID}:
        raise ValueError("Unexpected Attia prediction dataset identity")
    if set(predictions["cell_id"].astype(str)) != EXPECTED_CELL_IDS:
        raise ValueError("Unexpected Attia prediction cell identities")
    if predictions["test_id"].isna().any() or predictions["test_id"].duplicated().any():
        raise ValueError("Attia prediction test identities must be unique and non-null")
    expected_tests = predictions["cell_id"].astype(str) + "_CYCLING"
    if not predictions["test_id"].astype(str).equals(expected_tests):
        raise ValueError("Attia prediction test identities disagree with cell identities")
    expected_sources = predictions["cell_id"].astype(str).str.replace(
        "CLO_B4C",
        "b4c",
        regex=False,
    )
    if not predictions["source_cell_id"].astype(str).equals(expected_sources):
        raise ValueError("Attia prediction source-cell identities disagree")

    numeric_columns = [
        "log10_delta_q_variance",
        *[
            f"{model}_{suffix}"
            for model in (SOURCE_MODEL_NAME, NULL_MODEL_NAME)
            for suffix in (
                "log_location",
                "predictive_sigma",
                "p10",
                "p50",
                "p90",
            )
        ],
    ]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Attia prediction values must be finite numeric values")
    for model in (SOURCE_MODEL_NAME, NULL_MODEL_NAME):
        if (numeric[f"{model}_predictive_sigma"] <= 0.0).any():
            raise ValueError("Attia predictive sigma values must be positive")
        quantiles = numeric[
            [f"{model}_p10", f"{model}_p50", f"{model}_p90"]
        ].to_numpy()
        if not ((quantiles[:, 0] < quantiles[:, 1]) & (quantiles[:, 1] < quantiles[:, 2])).all():
            raise ValueError("Attia predictive quantiles must be strictly ordered")

    ordered = predictions.sort_values("cell_id", kind="stable").reset_index(drop=True)
    return ordered, canonical_sha256


def _validate_outcomes(
    outcomes: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, str]:
    if set(outcomes.columns) != set(ATTIA_OUTCOME_COLUMNS):
        raise ValueError("Attia sensitivity requires the exact v2 outcome schema")
    validate_attia_outcome_pack(outcomes)
    canonical_sha256 = attia_outcome_artifact_sha256(outcomes)
    expected = str(config["inputs"]["outcomes"]["canonical_sha256"])
    if canonical_sha256 != expected:
        raise ValueError(
            "Attia sensitivity outcome canonical SHA-256 mismatch: "
            f"expected {expected}, found {canonical_sha256}"
        )
    ordered = outcomes.sort_values("cell_id", kind="stable").reset_index(drop=True)
    return ordered, canonical_sha256


def _validate_cohort_join(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: Mapping[str, object],
) -> list[str]:
    protocol_ids = list(config["cohort"]["protocol_ids"])
    observed_prediction_protocols = sorted(predictions["protocol_id"].astype(str).unique())
    observed_outcome_protocols = sorted(outcomes["protocol_id"].astype(str).unique())
    if observed_prediction_protocols != protocol_ids:
        raise ValueError("Frozen prediction protocol identities disagree with config")
    if observed_outcome_protocols != protocol_ids:
        raise ValueError("Frozen outcome protocol identities disagree with config")
    for frame, label in ((predictions, "prediction"), (outcomes, "outcome")):
        counts = frame["protocol_id"].astype(str).value_counts()
        if len(counts) != 9 or not counts.eq(5).all():
            raise ValueError(f"Attia {label} structure must remain nine by five")

    identity_columns = ["cell_id", "dataset_id", "test_id", "source_cell_id", "protocol_id"]
    prediction_identity = predictions[identity_columns].astype(str).set_index("cell_id")
    outcome_identity = outcomes[identity_columns].astype(str).set_index("cell_id")
    if not prediction_identity.equals(outcome_identity):
        raise ValueError("Prediction and outcome protocol or cell identities disagree")
    return protocol_ids


def _validate_reference_validation(
    reference: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, float | int | bool | str]:
    expected_status = config["inputs"]["reference_validation"]["expected_status"]
    if reference.get("status") != expected_status:
        raise ValueError("Frozen Attia reference validation status changed")
    try:
        firewall = reference["prediction_firewall"]
        metrics = reference["protocol_balanced_primary_metrics"]
        signal_gate = reference["signal_gate"]
        cohort = reference["cohort"]
        source_metrics = metrics[SOURCE_MODEL_NAME]
        null_metrics = metrics[NULL_MODEL_NAME]
    except (KeyError, TypeError) as exc:
        raise ValueError("Frozen Attia reference validation schema is incomplete") from exc
    if firewall["verified_prediction_sha256"] != (
        config["inputs"]["predictions"]["canonical_sha256"]
    ) or firewall["verified_outcome_sha256"] != (
        config["inputs"]["outcomes"]["canonical_sha256"]
    ):
        raise ValueError("Frozen validation firewall hashes disagree with sensitivity inputs")
    if signal_gate.get("status") != "failed" or signal_gate.get("thresholds") != (
        config["signal_gate_thresholds"]
    ):
        raise ValueError("Frozen validation signal gate or thresholds changed")
    if (
        cohort.get("cell_count") != 45
        or cohort.get("protocol_count") != 9
        or cohort.get("cells_per_protocol") != [5]
    ):
        raise ValueError("Frozen validation cohort identity changed")
    try:
        source_nll = float(source_metrics["mean_negative_log_likelihood"])
        null_nll = float(null_metrics["mean_negative_log_likelihood"])
        source_mape = float(source_metrics["mape_fraction"])
        null_mape = float(null_metrics["mape_fraction"])
        delta_nll = float(metrics["delta_nll"])
        improvement_fraction = float(metrics["mape_improvement_fraction"])
        improved_protocols = int(metrics["protocols_with_mape_improvement"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frozen validation primary metrics are invalid") from exc
    numeric = np.asarray(
        [source_nll, null_nll, source_mape, null_mape, delta_nll, improvement_fraction],
        dtype=float,
    )
    if not np.isfinite(numeric).all() or null_mape <= 0.0:
        raise ValueError("Frozen validation primary metrics must be finite")
    return {
        "status": str(reference["status"]),
        "source_nll": source_nll,
        "null_nll": null_nll,
        "source_mape": source_mape,
        "null_mape": null_mape,
        "delta_nll": delta_nll,
        "delta_mape": source_mape - null_mape,
        "mape_improvement_fraction": improvement_fraction,
        "protocols_with_mape_improvement": improved_protocols,
        "signal_gate_passed": False,
    }


def _lognormal_nll(
    log_location: np.ndarray,
    sigma: np.ndarray,
    lifetime: np.ndarray,
) -> np.ndarray:
    log_lifetime = np.log(lifetime)
    z_score = (log_lifetime - log_location) / sigma
    return (
        log_lifetime
        + np.log(sigma)
        + 0.5 * z_score**2
        + 0.5 * math.log(2.0 * math.pi)
    )


def _local_model_metrics(
    predictions: pd.DataFrame,
    lifetime_assignments: np.ndarray,
    *,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    location = predictions[f"{model}_log_location"].to_numpy(dtype=float)[None, :]
    sigma = predictions[f"{model}_predictive_sigma"].to_numpy(dtype=float)[None, :]
    median = predictions[f"{model}_p50"].to_numpy(dtype=float)[None, :]
    nll = _lognormal_nll(location, sigma, lifetime_assignments).mean(axis=1)
    mape = (np.abs(median - lifetime_assignments) / lifetime_assignments).mean(axis=1)
    return nll, mape


def _gate_passed(
    delta_nll: np.ndarray | float,
    mape_improvement_fraction: np.ndarray | float,
    improved_protocols: np.ndarray | int,
    thresholds: Mapping[str, object],
) -> np.ndarray:
    return (
        (np.asarray(delta_nll) < float(thresholds["maximum_delta_nll"]))
        & (
            np.asarray(mape_improvement_fraction)
            >= float(thresholds["minimum_mape_improvement_fraction"])
        )
        & (
            np.asarray(improved_protocols)
            >= int(thresholds["minimum_protocols_with_mape_improvement"])
        )
    )


def _quantile_key(value: float) -> str:
    return format(value, ".6g")


def _summarize_draws(
    draws: np.ndarray,
    *,
    observed: float,
    quantiles: Sequence[float],
    quantile_method: str,
) -> dict[str, object]:
    below_count = int(np.count_nonzero(draws < observed))
    at_or_below_count = int(np.count_nonzero(draws <= observed))
    values = np.quantile(draws, quantiles, method=quantile_method)
    count = len(draws)
    return {
        "observed": float(observed),
        "observed_position": {
            "strictly_below_count": below_count,
            "strictly_below_fraction": below_count / count,
            "at_or_below_count": at_or_below_count,
            "at_or_below_fraction": at_or_below_count / count,
            "midrank_fraction": (below_count + at_or_below_count) / (2.0 * count),
        },
        "quantiles": {
            _quantile_key(float(level)): float(value)
            for level, value in zip(quantiles, values, strict=True)
        },
        "sampled_range": [float(np.min(draws)), float(np.max(draws))],
    }


def _draw_digest(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, values in named_arrays:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        if np.issubdtype(values.dtype, np.integer):
            normalized = np.asarray(values, dtype="<i8")
        elif np.issubdtype(values.dtype, np.bool_):
            normalized = np.asarray(values, dtype=np.uint8)
        else:
            normalized = np.asarray(values, dtype="<f8")
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def analyze_attia_crosswalk_sensitivity(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    reference_validation: Mapping[str, object],
    config: Mapping[str, object],
    *,
    prediction_raw_sha256: str,
    outcome_raw_sha256: str,
    reference_validation_raw_sha256: str,
    config_sha256: str | None = None,
) -> dict[str, object]:
    """Quantify within-protocol outcome-mapping sensitivity without refitting."""
    _validate_config(config)
    _validate_raw_hash(
        prediction_raw_sha256,
        config["inputs"]["predictions"]["raw_sha256"],
        label="prediction",
    )
    _validate_raw_hash(
        outcome_raw_sha256,
        config["inputs"]["outcomes"]["raw_sha256"],
        label="outcome",
    )
    _validate_raw_hash(
        reference_validation_raw_sha256,
        config["inputs"]["reference_validation"]["raw_sha256"],
        label="reference validation",
    )
    ordered_predictions, prediction_canonical_sha256 = _validate_predictions(
        predictions,
        config,
    )
    ordered_outcomes, outcome_canonical_sha256 = _validate_outcomes(outcomes, config)
    protocol_ids = _validate_cohort_join(
        ordered_predictions,
        ordered_outcomes,
        config,
    )
    frozen_reference = _validate_reference_validation(reference_validation, config)

    local_states: list[dict[str, object]] = []
    serialized_protocols: list[dict[str, object]] = []
    for protocol_id in protocol_ids:
        protocol_predictions = ordered_predictions.loc[
            ordered_predictions["protocol_id"].astype(str) == protocol_id
        ].sort_values("cell_id", kind="stable")
        protocol_outcomes = ordered_outcomes.set_index("cell_id").loc[
            protocol_predictions["cell_id"]
        ]
        lifetimes = protocol_outcomes["cycle_life"].to_numpy(dtype=float)
        lifetime_assignments = lifetimes[LOCAL_PERMUTATION_INDEX]
        source_nll, source_mape = _local_model_metrics(
            protocol_predictions,
            lifetime_assignments,
            model=SOURCE_MODEL_NAME,
        )
        null_nll, null_mape = _local_model_metrics(
            protocol_predictions,
            lifetime_assignments,
            model=NULL_MODEL_NAME,
        )
        delta_nll = source_nll - null_nll
        delta_mape = source_mape - null_mape
        improved = source_mape < null_mape
        if np.ptp(null_nll) > 1e-12 or np.ptp(null_mape) > 1e-12:
            raise ValueError(
                "Pinned null-model protocol metrics must be invariant to identity mapping"
            )
        state = {
            "protocol_id": protocol_id,
            "source_nll": source_nll,
            "null_nll": null_nll,
            "source_mape": source_mape,
            "null_mape": null_mape,
            "delta_nll": delta_nll,
            "delta_mape": delta_mape,
            "improved": improved,
        }
        local_states.append(state)
        serialized_protocols.append(
            {
                "protocol_id": protocol_id,
                "cell_order": protocol_predictions["cell_id"].astype(str).tolist(),
                "observed_outcome_identity_order": [
                    {
                        "cell_id": str(row.cell_id),
                        "replicate_id": str(row.replicate_id),
                        "cycle_life": int(row.cycle_life),
                    }
                    for row in protocol_outcomes.reset_index().itertuples(index=False)
                ],
                "local_permutation_count": len(LOCAL_PERMUTATIONS),
                "observed": {
                    "delta_nll": float(delta_nll[0]),
                    "delta_mape": float(delta_mape[0]),
                    "mape_improved": bool(improved[0]),
                },
                "exact_local_support": {
                    "delta_nll": [float(delta_nll.min()), float(delta_nll.max())],
                    "delta_mape": [
                        float(delta_mape.min()),
                        float(delta_mape.max()),
                    ],
                    "mape_improvement_possible_values": sorted(
                        set(bool(value) for value in improved.tolist())
                    ),
                },
            }
        )

    protocol_count = len(local_states)
    observed_source_nll = float(
        np.mean([state["source_nll"][0] for state in local_states])
    )
    observed_null_nll = float(
        np.mean([state["null_nll"][0] for state in local_states])
    )
    observed_source_mape = float(
        np.mean([state["source_mape"][0] for state in local_states])
    )
    observed_null_mape = float(
        np.mean([state["null_mape"][0] for state in local_states])
    )
    observed_delta_nll = observed_source_nll - observed_null_nll
    observed_delta_mape = observed_source_mape - observed_null_mape
    observed_mape_improvement = 1.0 - observed_source_mape / observed_null_mape
    observed_improved_protocols = int(
        sum(bool(state["improved"][0]) for state in local_states)
    )
    thresholds = config["signal_gate_thresholds"]
    observed_gate_passed = bool(
        _gate_passed(
            observed_delta_nll,
            observed_mape_improvement,
            observed_improved_protocols,
            thresholds,
        )
    )
    recalculated = {
        "source_nll": observed_source_nll,
        "null_nll": observed_null_nll,
        "source_mape": observed_source_mape,
        "null_mape": observed_null_mape,
        "delta_nll": observed_delta_nll,
        "delta_mape": observed_delta_mape,
        "mape_improvement_fraction": observed_mape_improvement,
    }
    for name, value in recalculated.items():
        if not math.isclose(
            value,
            float(frozen_reference[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Recalculated observed {name} disagrees with frozen score")
    if (
        observed_improved_protocols
        != frozen_reference["protocols_with_mape_improvement"]
        or observed_gate_passed != frozen_reference["signal_gate_passed"]
    ):
        raise ValueError("Recalculated observed conclusion disagrees with frozen score")

    exact_delta_nll_best = float(
        np.mean([np.min(state["delta_nll"]) for state in local_states])
    )
    exact_delta_nll_worst = float(
        np.mean([np.max(state["delta_nll"]) for state in local_states])
    )
    exact_delta_mape_best = float(
        np.mean([np.min(state["delta_mape"]) for state in local_states])
    )
    exact_delta_mape_worst = float(
        np.mean([np.max(state["delta_mape"]) for state in local_states])
    )
    exact_improved_protocols_worst = int(
        sum(int(np.min(state["improved"])) for state in local_states)
    )
    exact_improved_protocols_best = int(
        sum(int(np.max(state["improved"])) for state in local_states)
    )
    exact_source_mape_best = float(
        np.mean([np.min(state["source_mape"]) for state in local_states])
    )
    exact_source_mape_worst = float(
        np.mean([np.max(state["source_mape"]) for state in local_states])
    )
    exact_mape_improvement_best = 1.0 - exact_source_mape_best / observed_null_mape
    exact_mape_improvement_worst = 1.0 - exact_source_mape_worst / observed_null_mape
    impossibility_reasons: list[str] = []
    if exact_delta_nll_best >= float(thresholds["maximum_delta_nll"]):
        impossibility_reasons.append("best_case_delta_nll_fails_frozen_threshold")
    if exact_mape_improvement_best < float(
        thresholds["minimum_mape_improvement_fraction"]
    ):
        impossibility_reasons.append(
            "best_case_relative_mape_improvement_fails_frozen_threshold"
        )
    if exact_improved_protocols_best < int(
        thresholds["minimum_protocols_with_mape_improvement"]
    ):
        impossibility_reasons.append(
            "best_case_improved_protocol_count_fails_frozen_threshold"
        )
    exact_gate_pass_impossible = bool(impossibility_reasons)

    permutation = config["permutation"]
    resamples = int(permutation["resamples"])
    seed = int(permutation["seed"])
    rng = np.random.Generator(np.random.PCG64(seed))
    assignment_indices = rng.integers(
        0,
        len(LOCAL_PERMUTATIONS),
        size=(resamples, protocol_count),
        dtype=np.uint8,
    )
    sampled_delta_nll = np.zeros(resamples, dtype=float)
    sampled_delta_mape = np.zeros(resamples, dtype=float)
    sampled_source_mape = np.zeros(resamples, dtype=float)
    sampled_null_mape = np.zeros(resamples, dtype=float)
    sampled_improved_protocols = np.zeros(resamples, dtype=np.int16)
    for index, state in enumerate(local_states):
        choices = assignment_indices[:, index].astype(np.intp)
        sampled_delta_nll += state["delta_nll"][choices]
        sampled_delta_mape += state["delta_mape"][choices]
        sampled_source_mape += state["source_mape"][choices]
        sampled_null_mape += state["null_mape"][choices]
        sampled_improved_protocols += state["improved"][choices].astype(np.int16)
    sampled_delta_nll /= protocol_count
    sampled_delta_mape /= protocol_count
    sampled_source_mape /= protocol_count
    sampled_null_mape /= protocol_count
    sampled_mape_improvement = 1.0 - sampled_source_mape / sampled_null_mape
    sampled_gate_passed = _gate_passed(
        sampled_delta_nll,
        sampled_mape_improvement,
        sampled_improved_protocols,
        thresholds,
    )
    conclusion_flipped = sampled_gate_passed != observed_gate_passed
    quantiles = [float(value) for value in config["reporting"]["quantiles"]]
    quantile_method = str(config["reporting"]["quantile_method"])

    exact_support = {
        "protocol_balanced_delta_nll": {
            "best_case": exact_delta_nll_best,
            "worst_case": exact_delta_nll_worst,
            "interval": [exact_delta_nll_best, exact_delta_nll_worst],
        },
        "protocol_balanced_delta_mape": {
            "best_case": exact_delta_mape_best,
            "worst_case": exact_delta_mape_worst,
            "interval": [exact_delta_mape_best, exact_delta_mape_worst],
        },
        "protocols_with_mape_improvement": {
            "worst_case": exact_improved_protocols_worst,
            "best_case": exact_improved_protocols_best,
            "interval": [
                exact_improved_protocols_worst,
                exact_improved_protocols_best,
            ],
        },
        "mape_improvement_fraction_for_frozen_gate": {
            "best_case": exact_mape_improvement_best,
            "worst_case": exact_mape_improvement_worst,
            "interval": [
                exact_mape_improvement_worst,
                exact_mape_improvement_best,
            ],
        },
        "null_model_metric_invariance_verified": True,
        "frozen_signal_gate_pass_impossible": exact_gate_pass_impossible,
        "impossibility_reasons": impossibility_reasons,
        "exact_negative_conclusion_flip_fraction": (
            0.0 if exact_gate_pass_impossible and not observed_gate_passed else None
        ),
    }

    result: dict[str, object] = {
        "status": "mapping_sensitivity_complete",
        "experiment_id": EXPERIMENT_ID,
        "analysis_role": ANALYSIS_ROLE,
        "preregistration": {
            "config_sha256": config_sha256,
            "seed": seed,
            "resamples": resamples,
            "rng": permutation["rng"],
            "sampling": permutation["joint_sampling"],
            "statistics": list(STATISTIC_NAMES),
            "quantiles": quantiles,
            "quantile_method": quantile_method,
            "signal_gate_thresholds_unchanged": dict(thresholds),
        },
        "input_firewall": {
            "prediction_raw_sha256": prediction_raw_sha256,
            "prediction_canonical_sha256": prediction_canonical_sha256,
            "prediction_schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
            "outcome_raw_sha256": outcome_raw_sha256,
            "outcome_canonical_sha256": outcome_canonical_sha256,
            "outcome_schema_version": "attia_validation45_outcomes_v2",
            "reference_validation_raw_sha256": reference_validation_raw_sha256,
            "reference_validation_status": frozen_reference["status"],
            "protocol_identity_verified_before_analysis": True,
            "row_order_normalized_by": "stable_lexicographic_cell_id",
        },
        "cohort": {
            "cell_count": 45,
            "protocol_count": 9,
            "cells_per_protocol": 5,
            "protocol_ids": protocol_ids,
        },
        "mapping_space": {
            "permuted_identity_bundle": ["replicate_id", "cycle_life"],
            "fixed_structure": "nine_protocols_by_five_cells",
            "local_permutations_per_protocol": len(LOCAL_PERMUTATIONS),
            "local_permutations_exhaustively_enumerated": True,
            "joint_mapping_space_size": JOINT_MAPPING_SPACE_SIZE,
            "joint_distribution_sampled": True,
            "models_refit": False,
            "thresholds_tuned": False,
            "source_or_target_artifacts_modified": False,
        },
        "observed_mapping": {
            "reference_status": frozen_reference["status"],
            "source_model_protocol_balanced_nll": observed_source_nll,
            "null_model_protocol_balanced_nll": observed_null_nll,
            "source_model_protocol_balanced_mape": observed_source_mape,
            "null_model_protocol_balanced_mape": observed_null_mape,
            "protocol_balanced_delta_nll": observed_delta_nll,
            "protocol_balanced_delta_mape": observed_delta_mape,
            "mape_improvement_fraction": observed_mape_improvement,
            "protocols_with_mape_improvement": observed_improved_protocols,
            "frozen_signal_gate_passed": observed_gate_passed,
        },
        "exact_marginal_support": exact_support,
        "monte_carlo": {
            "assignment_index_sha256": hashlib.sha256(
                assignment_indices.tobytes(order="C")
            ).hexdigest(),
            "sample_statistics_sha256": _draw_digest(
                [
                    ("delta_nll", sampled_delta_nll),
                    ("delta_mape", sampled_delta_mape),
                    ("improved_protocols", sampled_improved_protocols),
                    ("mape_improvement_fraction", sampled_mape_improvement),
                    ("signal_gate_passed", sampled_gate_passed),
                ]
            ),
            "protocol_balanced_delta_nll": _summarize_draws(
                sampled_delta_nll,
                observed=observed_delta_nll,
                quantiles=quantiles,
                quantile_method=quantile_method,
            ),
            "protocol_balanced_delta_mape": _summarize_draws(
                sampled_delta_mape,
                observed=observed_delta_mape,
                quantiles=quantiles,
                quantile_method=quantile_method,
            ),
            "protocols_with_mape_improvement": _summarize_draws(
                sampled_improved_protocols.astype(float),
                observed=float(observed_improved_protocols),
                quantiles=quantiles,
                quantile_method=quantile_method,
            ),
            "frozen_signal_gate_pass_count": int(sampled_gate_passed.sum()),
            "frozen_signal_gate_pass_fraction": float(sampled_gate_passed.mean()),
            "negative_conclusion_flip_count": int(conclusion_flipped.sum()),
            "negative_conclusion_flip_fraction": float(conclusion_flipped.mean()),
        },
        "per_protocol_local_enumeration": serialized_protocols,
        "interpretation": {
            "mapping_conclusion": (
                "The frozen negative external signal conclusion cannot flip under any "
                "allowed within-protocol reassignment of the five outcome identities."
                if exact_gate_pass_impossible
                else (
                    "Marginal bounds alone do not prove infeasibility or joint "
                    "feasibility of the frozen signal gate."
                )
            ),
            "scope": "source-derived cell-to-outcome crosswalk sensitivity only",
            "not_a_new_external_validation": True,
            "not_a_p_value_or_significance_proof": True,
            "no_retraining_no_threshold_tuning_no_calibration": True,
        },
    }
    return result


def run_attia_crosswalk_sensitivity_from_config(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    config, config_sha256 = load_attia_crosswalk_sensitivity_config(config_path)
    _validate_production_preregistration(config, config_sha256)
    root = Path(project_root)
    prediction_path = root / str(config["inputs"]["predictions"]["path"])
    outcome_path = root / str(config["inputs"]["outcomes"]["path"])
    reference_path = root / str(config["inputs"]["reference_validation"]["path"])
    for path, label in (
        (prediction_path, "prediction"),
        (outcome_path, "outcome"),
        (reference_path, "reference validation"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Attia sensitivity {label} input: {path}")
    prediction_raw_sha256 = file_sha256(prediction_path)
    outcome_raw_sha256 = file_sha256(outcome_path)
    reference_raw_sha256 = file_sha256(reference_path)
    _validate_raw_hash(
        prediction_raw_sha256,
        config["inputs"]["predictions"]["raw_sha256"],
        label="prediction",
    )
    _validate_raw_hash(
        outcome_raw_sha256,
        config["inputs"]["outcomes"]["raw_sha256"],
        label="outcome",
    )
    _validate_raw_hash(
        reference_raw_sha256,
        config["inputs"]["reference_validation"]["raw_sha256"],
        label="reference validation",
    )
    predictions = pd.read_csv(prediction_path)
    outcomes = pd.read_csv(outcome_path)
    try:
        reference_validation = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read frozen Attia reference validation") from exc
    if not isinstance(reference_validation, dict):
        raise ValueError("Frozen Attia reference validation must be a JSON object")
    result = analyze_attia_crosswalk_sensitivity(
        predictions,
        outcomes,
        reference_validation,
        config,
        prediction_raw_sha256=prediction_raw_sha256,
        outcome_raw_sha256=outcome_raw_sha256,
        reference_validation_raw_sha256=reference_raw_sha256,
        config_sha256=config_sha256,
    )
    try:
        relative_config_path = Path(config_path).resolve().relative_to(root.resolve())
        recorded_config_path = relative_config_path.as_posix()
    except ValueError:
        recorded_config_path = str(Path(config_path).resolve())
    result["provenance"] = {
        "config_path": recorded_config_path,
        "config_sha256": config_sha256,
        "implementation_path": "src/lifetwin/experiments/attia_crosswalk_sensitivity.py",
        "implementation_sha256": file_sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    return result
