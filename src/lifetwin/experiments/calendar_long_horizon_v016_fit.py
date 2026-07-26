"""V2.1 identity boundary around the inherited prefix-only optimizer.

The optimizer's numerical declarations are frozen in V2 and contain no truth.
Only its table identity is hard-coded.  This adapter validates V2.1 tables,
maps identity in memory for the inherited computation, then validates and
returns V2.1 tables.  No V2-identified table is exposed to the caller.
"""

from __future__ import annotations

import pandas as pd

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    V015FitError,
    V015FitResult,
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    V015ArtifactError,
    canonicalize_frame,
)
from lifetwin.experiments.calendar_long_horizon_v015_prediction import (
    V015PredictionError,
    fit_structure_library_formal,
    fit_structure_library_parallel,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID as V2_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_PROTOCOL_ID,
)


class V021FitError(RuntimeError):
    """Raised when the inherited fit cannot cross the V2.1 identity boundary."""


def _canonical(
    frame: pd.DataFrame,
    *,
    filename: str,
    contract: FrozenArtifactContract,
    formal: bool,
) -> pd.DataFrame:
    try:
        return canonicalize_frame(
            frame,
            contract.csv_schema(filename),
            contract,
            formal=formal,
        )
    except V015ArtifactError as exc:
        raise V021FitError(str(exc)) from exc


def _translate(
    frame: pd.DataFrame,
    *,
    source: str,
    destination: str,
) -> pd.DataFrame:
    if "protocol_id" not in frame.columns:
        raise V021FitError("A fit table lacks protocol_id")
    if set(frame["protocol_id"].astype(str)) != {source}:
        raise V021FitError("A fit table contains an unexpected protocol identity")
    translated = frame.copy(deep=False)
    translated["protocol_id"] = destination
    return translated


def _prepare_inputs(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    contract: FrozenArtifactContract,
    formal: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if contract.protocol_id != V021_PROTOCOL_ID:
        raise V021FitError("The artifact contract is not V2.1")
    prefix = _canonical(
        prefix_pack,
        filename="prefix_pack.csv",
        contract=contract,
        formal=formal,
    )
    coordinates = _canonical(
        forecast_coordinates,
        filename="forecast_coordinates.csv",
        contract=contract,
        formal=formal,
    )
    return (
        _translate(
            prefix,
            source=V021_PROTOCOL_ID,
            destination=V2_PROTOCOL_ID,
        ),
        _translate(
            coordinates,
            source=V021_PROTOCOL_ID,
            destination=V2_PROTOCOL_ID,
        ),
    )


def _finalize(
    result: V015FitResult,
    *,
    contract: FrozenArtifactContract,
    formal: bool,
) -> V015FitResult:
    diagnostics = _translate(
        result.member_fit_diagnostics,
        source=V2_PROTOCOL_ID,
        destination=V021_PROTOCOL_ID,
    )
    forecasts = _translate(
        result.member_forecast_bundle,
        source=V2_PROTOCOL_ID,
        destination=V021_PROTOCOL_ID,
    )
    return V015FitResult(
        member_fit_diagnostics=_canonical(
            diagnostics,
            filename="member_fit_diagnostics.csv",
            contract=contract,
            formal=formal,
        ),
        member_forecast_bundle=_canonical(
            forecasts,
            filename="member_forecast_bundle.csv",
            contract=contract,
            formal=formal,
        ),
    )


def fit_structure_library_v021(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    contract: FrozenArtifactContract,
) -> V015FitResult:
    """Fit a hand fixture with the inherited optimizer and V2.1 identity."""

    prefix, coordinates = _prepare_inputs(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        contract=contract,
        formal=False,
    )
    try:
        result = fit_structure_library(prefix, coordinates)
    except V015FitError as exc:
        raise V021FitError("The inherited structure optimizer failed") from exc
    return _finalize(result, contract=contract, formal=False)


def fit_structure_library_parallel_v021(
    *,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    contract: FrozenArtifactContract,
    worker_count: int,
) -> V015FitResult:
    """Fit a hand fixture in workers while keeping V2 identity internal."""

    prefix, coordinates = _prepare_inputs(
        prefix_pack=prefix_pack,
        forecast_coordinates=forecast_coordinates,
        contract=contract,
        formal=False,
    )
    try:
        result = fit_structure_library_parallel(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            worker_count=worker_count,
        )
    except V015PredictionError as exc:
        raise V021FitError("The inherited parallel structure fit failed") from exc
    return _finalize(result, contract=contract, formal=False)


def _fit_structure_library_formal_from_verified_frames_v021(
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
        formal=True,
    )
    try:
        result = fit_structure_library_formal(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
        )
    except V015PredictionError as exc:
        raise V021FitError("The inherited formal structure fit failed") from exc
    return _finalize(result, contract=contract, formal=True)


__all__ = [
    "V021FitError",
    "fit_structure_library_parallel_v021",
    "fit_structure_library_v021",
]
