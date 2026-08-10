from __future__ import annotations

import pytest

from lifetwin.experiments.calendar_long_horizon_v019_environment import (
    V024EnvironmentError,
)
from lifetwin.experiments.calendar_long_horizon_v019_partition import (
    V024PartitionCapabilityError,
    V024PartitionContractError,
    V024WholeBundleContractError,
)
from lifetwin.experiments.calendar_long_horizon_v019_numeric_contract import (
    V024NumericContractError,
)
from lifetwin.experiments.calendar_long_horizon_v019_terminal import (
    ClassificationMode,
    TerminalDisposition,
    TerminalReason,
    V019TerminationError,
    classify_terminal_exception,
)


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (
            V024WholeBundleContractError("fixture"),
            TerminalReason.INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH,
        ),
        (
            V024PartitionContractError("fixture"),
            TerminalReason.INTEGRITY_PARTITION_CONTRACT_MISMATCH,
        ),
        (
            V024PartitionCapabilityError("fixture"),
            TerminalReason.INTEGRITY_PARTITION_CAPABILITY_MISMATCH,
        ),
    ),
)
def test_known_contract_errors_are_typed_integrity_not_unknown(error, reason) -> None:
    classified = classify_terminal_exception(error)
    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert classified.reason is reason


def test_genuinely_unregistered_exception_retains_unknown_default() -> None:
    classified = classify_terminal_exception(RuntimeError("fixture"))
    assert classified.disposition is TerminalDisposition.UNKNOWN
    assert classified.mode is ClassificationMode.UNKNOWN_DEFAULT
    assert classified.reason is TerminalReason.UNKNOWN_PRE_PREDICTION_EXCEPTION


def test_numeric_output_contract_error_is_typed_integrity_not_unknown() -> None:
    classified = classify_terminal_exception(V024NumericContractError("fixture"))
    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert (
        classified.reason is TerminalReason.INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH
    )


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (
            V024EnvironmentError("fixture"),
            TerminalReason.INTEGRITY_ENVIRONMENT_MISMATCH,
        ),
        (
            V019TerminationError("fixture"),
            TerminalReason.INTEGRITY_ARTIFACT_HASH_MISMATCH,
        ),
    ),
)
def test_environment_and_terminal_errors_are_typed(error, reason) -> None:
    classified = classify_terminal_exception(error)
    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert classified.reason is reason
