from __future__ import annotations

import json
from pathlib import Path

from lifetwin.experiments.calendar_long_horizon_v015_prediction import (
    V015PredictionError,
)
from lifetwin.experiments.calendar_long_horizon_v019_fit import V024FitError
from lifetwin.experiments.calendar_long_horizon_v019_io import V024IOError
from lifetwin.experiments.calendar_long_horizon_v019_prediction import (
    V024PredictionError,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (
    ClassificationMode,
    TerminalDisposition,
    TerminalReason,
    _validate_structural_traceback,
    classify_terminal_exception,
    sanitized_structural_traceback,
)


ROOT = Path(__file__).resolve().parents[1]


def _fit_failure_chain() -> V024PredictionError:
    try:
        try:
            try:
                raise RuntimeError("sensitive worker detail")
            except RuntimeError as cause:
                raise V015PredictionError("worker failed") from cause
        except V015PredictionError as cause:
            raise V024FitError("fit boundary failed") from cause
    except V024FitError as cause:
        try:
            raise V024PredictionError("prediction boundary failed") from cause
        except V024PredictionError as error:
            return error


def _io_failure_chain() -> V024PredictionError:
    try:
        raise V024IOError("sensitive artifact detail")
    except V024IOError as cause:
        try:
            raise V024PredictionError("prediction boundary failed") from cause
        except V024PredictionError as error:
            return error


def test_structural_traceback_preserves_sanitized_exception_chain() -> None:
    payload = json.loads(
        sanitized_structural_traceback(_fit_failure_chain(), repo_root=ROOT)
    )

    assert payload["schema_version"] == "1.1.0"
    assert payload["exception_chain_truncated"] is False
    assert [entry["exception_class"] for entry in payload["exception_chain"]] == [
        "V024PredictionError",
        "V024FitError",
        "V015PredictionError",
        "RuntimeError",
    ]
    assert [entry["relationship"] for entry in payload["exception_chain"]] == [
        "outer",
        "cause",
        "cause",
        "cause",
    ]
    assert payload["frames"] == payload["exception_chain"][0]["frames"]
    assert all(entry["frames"] for entry in payload["exception_chain"])
    assert b"sensitive" not in json.dumps(payload).encode("ascii")
    _validate_structural_traceback(
        payload,
        expected_reason=TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION,
        expected_mode=ClassificationMode.UNKNOWN_DEFAULT,
        expected_disposition=TerminalDisposition.UNKNOWN,
        expected_message=(
            "An unclassified exception terminated the attempt before prediction."
        ),
    )


def test_nested_io_capability_failure_is_proven_integrity() -> None:
    error = _io_failure_chain()
    classified = classify_terminal_exception(error)
    payload = json.loads(sanitized_structural_traceback(error, repo_root=ROOT))

    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert classified.reason is TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH
    assert [entry["exception_class"] for entry in payload["exception_chain"]] == [
        "V024PredictionError",
        "V024IOError",
    ]
    assert payload["reason_code"] == "INTEGRITY_ARTIFACT_HASH_MISMATCH"
    _validate_structural_traceback(
        payload,
        expected_reason=TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH,
        expected_mode=ClassificationMode.PROVEN_INTEGRITY,
        expected_disposition=TerminalDisposition.INTEGRITY_FAILURE,
        expected_message="A committed artifact hash did not match.",
    )


def test_exception_chain_cycle_is_bounded_and_marked() -> None:
    error = RuntimeError("cycle")
    error.__cause__ = error
    payload = json.loads(sanitized_structural_traceback(error, repo_root=ROOT))

    assert payload["exception_chain_truncated"] is True
    assert len(payload["exception_chain"]) == 1
    assert payload["exception_chain"][0]["relationship"] == "outer"
