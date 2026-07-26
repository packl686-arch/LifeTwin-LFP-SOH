from __future__ import annotations

import pandas as pd
import pytest

from lifetwin.experiments import (
    calendar_long_horizon_v016_prediction_capsule as capsule,
)


def test_formal_member_forecasts_require_all_86_variants_per_cluster() -> None:
    records: list[dict[str, object]] = []
    cluster_index = 0
    for partition, member_count in capsule.PARTITION_MEMBER_COUNTS.items():
        for local_index in range(member_count):
            model_id, variant_id = capsule.FROZEN_VARIANT_KEYS[
                cluster_index % len(capsule.FROZEN_VARIANT_KEYS)
            ]
            cluster_id = f"{partition}-{local_index:04d}"
            records.extend(
                {
                    "protocol_id": capsule.V021_PROTOCOL_ID,
                    "partition": partition,
                    "cluster_id": cluster_id,
                    "model_id": model_id,
                    "variant_id": variant_id,
                    "forecast_day": day,
                    "raw_forecast_retention_pct": 90.0,
                    "canonical_prefix_content_sha256": "0" * 64,
                }
                for day in capsule.FORECAST_DAYS
            )
            cluster_index += 1

    malformed = pd.DataFrame.from_records(
        records,
        columns=capsule._SCHEMAS["member_forecast_bundle.csv"].columns,
    )
    assert len(malformed) == 5_950 * len(capsule.FORECAST_DAYS)
    assert malformed.loc[:, ["model_id", "variant_id"]].drop_duplicates().shape[0] == 86

    with pytest.raises(
        capsule.V021PredictionCapsuleError,
        match="per-cluster exact variant registry changed",
    ):
        capsule.canonicalize_frame(
            malformed,
            "member_forecast_bundle.csv",
            formal=True,
        )


def test_joint_input_alignment_rejects_diagnostic_forecast_coordinate_drift() -> None:
    model_a, variant_a = capsule.FROZEN_VARIANT_KEYS[0]
    model_b, variant_b = capsule.FROZEN_VARIANT_KEYS[1]
    identity = {"partition": "calibration", "cluster_id": "cluster-0001"}
    frames = {
        "prefix_pack.csv": pd.DataFrame([identity]),
        "forecast_coordinates.csv": pd.DataFrame(
            [{**identity, "forecast_day": capsule.FORECAST_DAYS[0]}]
        ),
        "operating_pack.csv": pd.DataFrame([identity]),
        "member_fit_diagnostics.csv": pd.DataFrame(
            [
                {
                    **identity,
                    "model_id": model_a,
                    "variant_id": variant_a,
                }
            ]
        ),
        "member_forecast_bundle.csv": pd.DataFrame(
            [
                {
                    **identity,
                    "model_id": model_b,
                    "variant_id": variant_b,
                    "forecast_day": capsule.FORECAST_DAYS[0],
                }
            ]
        ),
    }

    with pytest.raises(
        capsule.V021PredictionCapsuleError,
        match="variant coordinates differ",
    ):
        capsule._validate_prediction_input_alignment(frames)
