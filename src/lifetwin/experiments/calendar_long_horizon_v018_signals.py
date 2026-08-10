"""Lightweight nominal exception signals shared by V2.3 stages.

This module intentionally imports no model, data, filesystem, generator, or
scoring code. Keeping the class identity here lets the terminal publisher use
an exact nominal type check instead of trusting forgeable module/name strings.
"""

from __future__ import annotations

from typing import Sequence


class V023CalibrationTerminalInconclusive(RuntimeError):
    """A typed calibration condition that stops before prediction."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        offending_row_indices: Sequence[int] = (),
    ) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise TypeError("reason_code must be a nonempty string")
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in offending_row_indices
        ):
            raise TypeError("offending_row_indices must be nonnegative integers")
        super().__init__(message)
        self.reason_code = reason_code
        self.offending_row_indices = tuple(offending_row_indices)


__all__ = ["V023CalibrationTerminalInconclusive"]
