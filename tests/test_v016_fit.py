from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v016_fit import (
    V021FitError,
    __all__ as FIT_PUBLIC_SURFACE,
    fit_structure_library_parallel_v021,
    fit_structure_library_v021,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    DEFAULT_V021_AMENDMENT_PATH,
    V021_PROTOCOL_ID,
    load_v021_design,
)


def _contract() -> FrozenArtifactContract:
    design = load_v021_design()
    return replace(
        load_artifact_contract(),
        protocol_id=V021_PROTOCOL_ID,
        config_path=DEFAULT_V021_AMENDMENT_PATH,
        config_byte_sha256=design.config_byte_sha256,
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = _contract()
    prefix = pd.DataFrame(
        [
            {
                "protocol_id": V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "fit-fixture",
                "prefix_day": day,
                "observed_retention_pct": 100.0 - 0.9 * math.sqrt(day / 365.25),
            }
            for day in PREFIX_DAYS
        ],
        columns=contract.csv_schema("prefix_pack.csv").columns,
    )
    coordinates = pd.DataFrame(
        [
            {
                "protocol_id": V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "fit-fixture",
                "forecast_day": day,
            }
            for day in FORECAST_DAYS
        ],
        columns=contract.csv_schema("forecast_coordinates.csv").columns,
    )
    return prefix, coordinates


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(protocol_id="<PROTOCOL>").reset_index(drop=True)


def test_v021_fit_is_numerically_identical_and_does_not_mutate_inputs() -> None:
    prefix, coordinates = _inputs()
    original_prefix = prefix.copy(deep=True)
    original_coordinates = coordinates.copy(deep=True)
    adapted = fit_structure_library_v021(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        contract=_contract(),
    )
    inherited = fit_structure_library(
        prefix.assign(protocol_id=FROZEN_PROTOCOL_ID),
        coordinates.assign(protocol_id=FROZEN_PROTOCOL_ID),
    )

    pd.testing.assert_frame_equal(prefix, original_prefix)
    pd.testing.assert_frame_equal(coordinates, original_coordinates)
    pd.testing.assert_frame_equal(
        _normalized(adapted.member_fit_diagnostics),
        _normalized(inherited.member_fit_diagnostics),
    )
    pd.testing.assert_frame_equal(
        _normalized(adapted.member_forecast_bundle),
        _normalized(inherited.member_forecast_bundle),
    )
    assert set(adapted.member_fit_diagnostics["protocol_id"]) == {V021_PROTOCOL_ID}
    assert set(adapted.member_forecast_bundle["protocol_id"]) == {V021_PROTOCOL_ID}


def test_v021_one_worker_fit_is_byte_stable_at_identity_boundary() -> None:
    prefix, coordinates = _inputs()
    serial = fit_structure_library_v021(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        contract=_contract(),
    )
    parallel = fit_structure_library_parallel_v021(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        contract=_contract(),
        worker_count=1,
    )
    pd.testing.assert_frame_equal(
        serial.member_fit_diagnostics,
        parallel.member_fit_diagnostics,
    )
    pd.testing.assert_frame_equal(
        serial.member_forecast_bundle,
        parallel.member_forecast_bundle,
    )


def test_v021_fit_rejects_a_v2_contract() -> None:
    prefix, coordinates = _inputs()
    with pytest.raises(V021FitError, match="not V2.1"):
        fit_structure_library_v021(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            contract=load_artifact_contract(),
        )


def test_v021_fit_surface_is_label_free() -> None:
    forbidden = ("truth", "target", "label", "sealed", "score", "family_id")
    for function in (
        fit_structure_library_v021,
        fit_structure_library_parallel_v021,
    ):
        parameters = inspect.signature(function).parameters
        assert not any(
            token in name.lower() for name in parameters for token in forbidden
        )
    assert "fit_structure_library_formal_v021" not in FIT_PUBLIC_SURFACE
