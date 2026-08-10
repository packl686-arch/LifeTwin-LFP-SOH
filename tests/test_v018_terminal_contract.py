from __future__ import annotations

import pytest

from lifetwin.experiments.calendar_long_horizon_v018_partition import (
    V023PartitionCapabilityError,
    V023PartitionContractError,
    V023WholeBundleContractError,
)
from lifetwin.experiments.calendar_long_horizon_v018_numeric_contract import (
    V023NumericContractError,
)
from lifetwin.experiments.calendar_long_horizon_v018_terminal import (
    ClassificationMode,
    TerminalDisposition,
    TerminalReason,
    classify_terminal_exception,
)


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (
            V023WholeBundleContractError("fixture"),
            TerminalReason.INTEGRITY_WHOLE_BUNDLE_CONTRACT_MISMATCH,
        ),
        (
            V023PartitionContractError("fixture"),
            TerminalReason.INTEGRITY_PARTITION_CONTRACT_MISMATCH,
        ),
        (
            V023PartitionCapabilityError("fixture"),
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
    classified = classify_terminal_exception(V023NumericContractError("fixture"))
    assert classified.disposition is TerminalDisposition.INTEGRITY_FAILURE
    assert classified.mode is ClassificationMode.PROVEN_INTEGRITY
    assert (
        classified.reason
        is TerminalReason.INTEGRITY_NUMERIC_OUTPUT_CONTRACT_MISMATCH
    )
