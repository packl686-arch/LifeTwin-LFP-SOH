"""Validation gates for evidence protocols."""

from lifetwin.validation.long_term_protocol import (
    IndependentLongTermProtocolValidationError,
    validate_independent_long_term_protocol,
)

__all__ = [
    "IndependentLongTermProtocolValidationError",
    "validate_independent_long_term_protocol",
]
