"""V2.4 identity boundary around the inherited prefix-only optimizer.

The optimizer's numerical declarations are frozen in V2 and contain no truth.
Only its table identity is hard-coded.  This adapter validates V2.4 tables,
maps identity in memory for the inherited computation, then validates and
returns V2.4 tables.  No V2-identified table is exposed to the caller.
"""

from __future__ import annotations

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_fit import V015FitResult
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonicalize_frame,
)
from lifetwin.experiments.calendar_long_horizon_v015_prediction import (
    V015PredictionError,
    fit_structure_library_formal,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID as V2_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v019_protocol import (
    V024_PROTOCOL_ID,
)


class V024FitError(RuntimeError):
    """Raised when the inherited fit cannot cross the V2.4 identity boundary."""


def _canonical(
    frame: pd.DataFrame,
    *,
    filename: str,
    contract: FrozenArtifactContract,
) -> pd.DataFrame:
    try:
        return canonicalize_frame(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=True,
        )
    except V015ArtifactError as exc:
        raise V024FitError(str(exc)) from exc


def _translate(
    frame: pd.DataFrame,
    *,
    source: str,
    destination: str,
) -> pd.DataFrame:
    if "protocol_id" not in frame.columns:
        raise V024FitError("A fit table lacks protocol_id")
    if set(frame["protocol_id"].astype(str)) != {source}:
        raise V024FitError("A fit table contains an unexpected protocol identity")
    translated = frame.copy(deep=False)
    translated["protocol_id"] = destination
    return translated


def _prepare_inputs(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    contract: FrozenArtifactContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if contract.protocol_id != V024_PROTOCOL_ID:
        raise V024FitError("The artifact contract is not V2.4")
    prefix = _canonical(
        prefix_pack,
        filename="prefix_pack.csv",
        contract=contract,
    )
    coordinates = _canonical(
        forecast_coordinates,
        filename="forecast_coordinates.csv",
        contract=contract,
    )
    return (
        _translate(
            prefix,
            source=V024_PROTOCOL_ID,
            destination=V2_PROTOCOL_ID,
        ),
        _translate(
            coordinates,
            source=V024_PROTOCOL_ID,
            destination=V2_PROTOCOL_ID,
        ),
    )


def _finalize(
    result: V015FitResult,
    *,
    contract: FrozenArtifactContract,
) -> V015FitResult:
    diagnostics = _translate(
        result.member_fit_diagnostics,
        source=V2_PROTOCOL_ID,
        destination=V024_PROTOCOL_ID,
    )
    forecasts = _translate(
        result.member_forecast_bundle,
        source=V2_PROTOCOL_ID,
        destination=V024_PROTOCOL_ID,
    )
    return V015FitResult(
        member_fit_diagnostics=_canonical(
            diagnostics,
            filename="member_fit_diagnostics.csv",
            contract=contract,
        ),
        member_forecast_bundle=_canonical(
            forecasts,
            filename="member_forecast_bundle.csv",
            contract=contract,
        ),
    )


def _fit_structure_library_formal_from_verified_frames_v024(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    contract: FrozenArtifactContract,
) -> V015FitResult:
    """Internal six-worker core; a formal caller must first verify provenance."""

    prefix, coordinates = _prepare_inputs(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        contract=contract,
    )
    try:
        result = fit_structure_library_formal(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
        )
    except V015PredictionError as exc:
        raise V024FitError("The inherited formal structure fit failed") from exc
    return _finalize(result, contract=contract)


__all__ = [
    "V024FitError",
]
