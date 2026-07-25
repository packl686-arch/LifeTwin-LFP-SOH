"""Frozen artifact and firewall primitives for the V0.15 synthetic protocol.

This module deliberately contains no truth generator, model fitting, or scoring
logic.  It only enforces the byte contracts that surround those later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments.calendar_long_horizon_synthetic import (
    canonical_csv_bytes as _v1_canonical_csv_bytes,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
    FROZEN_CONFIG_CANONICAL_SHA256,
)


FROZEN_PROTOCOL_ID = "synthetic_long_horizon_identifiability_v2"
FROZEN_SCHEMA_VERSION = "2.0.0"
DEFAULT_V2_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2.json"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TRUTH_COMMITMENT_ENTRY_KEYS = frozenset(
    {"path", "row_count", "byte_count", "byte_sha256"}
)
_PREDICTION_HASH_FILES = MappingProxyType(
    {
        "model_state_byte_sha256": "model_state.json",
        "prefix_pack_byte_sha256": "prefix_pack.csv",
        "forecast_coordinates_byte_sha256": "forecast_coordinates.csv",
        "operating_pack_byte_sha256": "operating_pack.csv",
        "member_fit_diagnostics_byte_sha256": "member_fit_diagnostics.csv",
        "member_forecast_bundle_byte_sha256": "member_forecast_bundle.csv",
        "prediction_bundle_byte_sha256": "prediction_bundle.csv",
        "risk_bundle_byte_sha256": "risk_bundle.csv",
        "decision_bundle_byte_sha256": "decision_bundle.csv",
    }
)
_PREDICTION_CSV_FILENAMES = tuple(
    filename
    for filename in _PREDICTION_HASH_FILES.values()
    if filename.endswith(".csv")
)
_STRING_KEY_COLUMNS = frozenset(
    {
        "partition",
        "pair_partition",
        "cluster_id",
        "pair_id",
        "model_id",
        "variant_id",
        "score_id",
        "arm",
    }
)
_NUMERIC_KEY_COLUMNS = frozenset({"prefix_day", "forecast_day"})
_STRICT_BOOLEAN_COLUMNS = frozenset(
    {"credible_variant", "all_features_finite", "hard_eligible", "issued"}
)
_TRUTH_INVARIANT_COLUMNS = (
    "truth_family",
    "truth_parameters_json",
    "gamma",
)


class V015ArtifactError(ValueError):
    """Raised when a V0.15 artifact violates its frozen contract."""


@dataclass(frozen=True)
class CsvArtifactSchema:
    """One expanded CSV contract from the frozen protocol."""

    filename: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    required_rows: int | None = None
    required_score_ids: tuple[str, ...] = ()
    required_arms: tuple[str, ...] = ()
    expected_partition: str | None = None


@dataclass(frozen=True)
class FrozenArtifactContract:
    """Parsed, immutable subset of the V2 artifact protocol."""

    protocol_id: str
    schema_version: str
    config_path: Path
    config_byte_sha256: str
    csv_schemas: Mapping[str, CsvArtifactSchema]
    json_key_allowlists: Mapping[str, frozenset[str]]
    exposure_keys: frozenset[str]
    partitions: tuple[str, ...]
    partition_member_counts: Mapping[str, int]
    prefix_days: tuple[float, ...]
    forecast_days: tuple[float, ...]
    truth_filenames: tuple[str, ...]
    matched_pair_filenames: tuple[str, ...]

    @property
    def sealed_filenames(self) -> tuple[str, ...]:
        return self.truth_filenames + self.matched_pair_filenames

    def csv_schema(self, filename: str) -> CsvArtifactSchema:
        try:
            return self.csv_schemas[filename]
        except KeyError as exc:
            raise V015ArtifactError(
                f"{filename!r} is not a frozen V2 CSV artifact"
            ) from exc

    def json_keys(self, filename: str) -> frozenset[str]:
        try:
            return self.json_key_allowlists[filename]
        except KeyError as exc:
            raise V015ArtifactError(
                f"{filename!r} is not a frozen V2 JSON artifact"
            ) from exc


@dataclass(frozen=True)
class ArtifactMetadata:
    path: str
    row_count: int
    byte_count: int
    byte_sha256: str


@dataclass(frozen=True)
class PredictorContentHashes:
    random_policy: str
    arm_a: str
    arm_b: str
    placebo: str


@dataclass(frozen=True)
class PredictorContentPayloads:
    """Canonical ID-free predictor bytes underlying every content hash."""

    random_policy: bytes
    arm_a: bytes
    arm_b: bytes
    placebo: bytes


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V015ArtifactError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V015ArtifactError(f"{context} is not strict UTF-8 JSON") from exc


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V015ArtifactError(f"{context} must be a JSON object")
    return value


def _require_string_array(value: Any, *, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise V015ArtifactError(f"{context} must be a nonempty string array")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise V015ArtifactError(f"{context} contains duplicates")
    return result


def _positive_int_or_none(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise V015ArtifactError(f"{context} must be a positive integer")
    return value


def _expanded_schema(
    filename: str,
    raw_schema: Mapping[str, Any],
    *,
    required_rows: int | None = None,
    expected_partition: str | None = None,
) -> CsvArtifactSchema:
    columns = _require_string_array(
        raw_schema.get("columns"), context=f"{filename}.columns"
    )
    key = _require_string_array(raw_schema.get("key"), context=f"{filename}.key")
    if not set(key).issubset(columns):
        raise V015ArtifactError(f"{filename}.key is not a subset of its columns")
    frozen_rows = _positive_int_or_none(
        raw_schema.get("required_rows"), context=f"{filename}.required_rows"
    )
    if required_rows is not None and frozen_rows is not None:
        raise V015ArtifactError(f"{filename} has two frozen row-count declarations")
    return CsvArtifactSchema(
        filename=filename,
        columns=columns,
        key=key,
        required_rows=required_rows if required_rows is not None else frozen_rows,
        required_score_ids=tuple(raw_schema.get("required_score_ids", ())),
        required_arms=tuple(raw_schema.get("required_arms", ())),
        expected_partition=expected_partition,
    )


def load_artifact_contract(
    config_path: str | Path = DEFAULT_V2_CONFIG_PATH,
) -> FrozenArtifactContract:
    """Load artifact schemas directly from the committed V2 protocol."""

    path = Path(config_path).resolve()
    raw = path.read_bytes()
    root = _require_mapping(_load_json_bytes(raw, context=path.name), context="config")
    byte_sha256 = hashlib.sha256(raw).hexdigest()
    if byte_sha256 != FROZEN_CONFIG_BYTE_SHA256:
        raise V015ArtifactError("Frozen V2 config byte hash changed")
    canonical = json.dumps(
        root,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != FROZEN_CONFIG_CANONICAL_SHA256:
        raise V015ArtifactError("Frozen V2 config canonical hash changed")
    if root.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise V015ArtifactError("V2 protocol ID changed")
    if root.get("schema_version") != FROZEN_SCHEMA_VERSION:
        raise V015ArtifactError("V2 schema version changed")

    firewall = _require_mapping(
        root.get("firewall_and_artifacts"), context="firewall_and_artifacts"
    )
    schemas = _require_mapping(
        firewall.get("artifact_schemas"), context="artifact_schemas"
    )
    design = _require_mapping(
        root.get("design_partitions"), context="design_partitions"
    )
    time_grid = _require_mapping(root.get("time_grid"), context="time_grid")

    csv_schemas: dict[str, CsvArtifactSchema] = {}
    generic_names = {
        "global_csv_rule",
        "truth_csv_family",
        "matched_pair_csvs",
        "json_exact_key_allowlists",
        "exposure_log.jsonl",
    }
    for filename, raw_schema in schemas.items():
        if filename in generic_names:
            continue
        if filename.endswith(".csv"):
            csv_schemas[filename] = _expanded_schema(
                filename, _require_mapping(raw_schema, context=filename)
            )

    truth_schema = _require_mapping(
        schemas.get("truth_csv_family"), context="truth_csv_family"
    )
    truth_rows = _require_mapping(
        truth_schema.get("exact_file_rows"), context="truth_csv_family.exact_file_rows"
    )
    truth_filenames: list[str] = []
    for filename, row_count in truth_rows.items():
        if not isinstance(filename, str) or not filename.endswith("_truth.csv"):
            raise V015ArtifactError("Frozen truth filename is invalid")
        partition = {
            "intrinsic_matched_truth.csv": "intrinsic_matched_pairs",
            "stress_plan_matched_truth.csv": "stress_plan_matched_pairs",
        }.get(filename, filename.removesuffix("_truth.csv"))
        csv_schemas[filename] = _expanded_schema(
            filename,
            truth_schema,
            required_rows=_positive_int_or_none(
                row_count, context=f"truth rows for {filename}"
            ),
            expected_partition=partition,
        )
        truth_filenames.append(filename)

    matched_schema = _require_mapping(
        schemas.get("matched_pair_csvs"), context="matched_pair_csvs"
    )
    matched_rows = _positive_int_or_none(
        matched_schema.get("required_rows_each"),
        context="matched_pair_csvs.required_rows_each",
    )
    matched_filenames = (
        "intrinsic_matched_pairs.csv",
        "stress_plan_matched_pairs.csv",
    )
    for filename in matched_filenames:
        csv_schemas[filename] = _expanded_schema(
            filename,
            matched_schema,
            required_rows=matched_rows,
            expected_partition=filename.removesuffix(".csv"),
        )

    allowlists_raw = _require_mapping(
        schemas.get("json_exact_key_allowlists"),
        context="json_exact_key_allowlists",
    )
    allowlists = {
        filename: frozenset(
            _require_string_array(keys, context=f"allowlist for {filename}")
        )
        for filename, keys in allowlists_raw.items()
    }
    exposure = _require_mapping(
        schemas.get("exposure_log.jsonl"), context="exposure_log.jsonl"
    )
    exposure_keys = frozenset(
        _require_string_array(
            exposure.get("keys_per_line"), context="exposure_log.keys_per_line"
        )
    )

    partitions = tuple(
        name
        for name, value in design.items()
        if isinstance(value, Mapping)
        and ("total_clusters" in value or "member_count" in value)
    )
    if not partitions:
        raise V015ArtifactError("No V2 data partitions were found")
    partition_member_counts: dict[str, int] = {}
    for partition in partitions:
        definition = _require_mapping(
            design[partition], context=f"design_partitions.{partition}"
        )
        raw_count = definition.get("total_clusters", definition.get("member_count"))
        count = _positive_int_or_none(
            raw_count, context=f"member count for {partition}"
        )
        assert count is not None
        partition_member_counts[partition] = count
    prefix_days = tuple(float(item) for item in time_grid["prefix_days"])
    forecast_days = tuple(float(item) for item in time_grid["forecast_days"])

    return FrozenArtifactContract(
        protocol_id=FROZEN_PROTOCOL_ID,
        schema_version=FROZEN_SCHEMA_VERSION,
        config_path=path,
        config_byte_sha256=byte_sha256,
        csv_schemas=MappingProxyType(csv_schemas),
        json_key_allowlists=MappingProxyType(allowlists),
        exposure_keys=exposure_keys,
        partitions=partitions,
        partition_member_counts=MappingProxyType(partition_member_counts),
        prefix_days=prefix_days,
        forecast_days=forecast_days,
        truth_filenames=tuple(truth_filenames),
        matched_pair_filenames=matched_filenames,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise V015ArtifactError("Canonical JSON cannot contain NaN or infinity")
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical on-disk representation for a JSON artifact."""

    try:
        text = json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise V015ArtifactError("Payload is not finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _canonical_json_line_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise V015ArtifactError("Exposure event is not finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _atomic_create(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A preflight ``exists`` followed by ``os.replace`` can overwrite a file
    # created by a racing writer.  Formal commitments are write-once, so use
    # the final path with O_EXCL.  A short write intentionally remains visible
    # and makes every retry fail closed rather than hiding the interruption.
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _byte_count_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _validate_json_identity(
    payload: Mapping[str, Any],
    *,
    filename: str,
    contract: FrozenArtifactContract,
) -> None:
    expected = contract.json_keys(filename)
    if set(payload) != expected:
        raise V015ArtifactError(
            f"{filename} keys differ from the frozen allowlist: "
            f"observed={sorted(payload)}, expected={sorted(expected)}"
        )
    if payload.get("protocol_id") != contract.protocol_id:
        raise V015ArtifactError(f"{filename} protocol_id changed")
    if payload.get("config_sha256") != contract.config_byte_sha256:
        raise V015ArtifactError(f"{filename} config_sha256 changed")


def write_canonical_json(
    path: str | Path,
    payload: Mapping[str, Any],
    contract: FrozenArtifactContract,
) -> None:
    target = Path(path)
    _validate_json_identity(payload, filename=target.name, contract=contract)
    _atomic_create(target, canonical_json_bytes(payload))


def read_canonical_json(
    path: str | Path,
    contract: FrozenArtifactContract,
) -> dict[str, Any]:
    target = Path(path)
    raw = target.read_bytes()
    payload = _require_mapping(
        _load_json_bytes(raw, context=target.name), context=target.name
    )
    _validate_json_identity(payload, filename=target.name, contract=contract)
    if canonical_json_bytes(payload) != raw:
        raise V015ArtifactError(f"{target.name} is not canonical JSON")
    return dict(payload)


def _validate_protocol_column(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
    contract: FrozenArtifactContract,
) -> None:
    if "protocol_id" not in frame.columns:
        return
    observed = set(frame["protocol_id"].astype(str))
    if observed != {contract.protocol_id}:
        raise V015ArtifactError(f"{schema.filename} contains a non-frozen protocol_id")


def _validate_key_columns(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
) -> None:
    for column in schema.key:
        values = frame[column]
        if values.isna().any():
            raise V015ArtifactError(
                f"{schema.filename} key column {column!r} contains NA"
            )
        if column in _STRING_KEY_COLUMNS:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in values.tolist()
            ):
                raise V015ArtifactError(
                    f"{schema.filename} string key column {column!r} "
                    "contains a non-string, empty, or whitespace-only value"
                )
        elif column in _NUMERIC_KEY_COLUMNS:
            try:
                numeric = values.to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise V015ArtifactError(
                    f"{schema.filename} numeric key column {column!r} is invalid"
                ) from exc
            if not np.isfinite(numeric).all():
                raise V015ArtifactError(
                    f"{schema.filename} numeric key column {column!r} "
                    "contains a nonfinite value"
                )


def _validate_strict_boolean_columns(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
) -> None:
    for column in _STRICT_BOOLEAN_COLUMNS.intersection(frame.columns):
        values = frame[column]
        if values.isna().any() or any(
            not isinstance(value, (bool, np.bool_)) for value in values.tolist()
        ):
            raise V015ArtifactError(
                f"{schema.filename} column {column!r} must contain strict booleans"
            )


def _validate_partition_column(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
    contract: FrozenArtifactContract,
) -> None:
    column = (
        "partition"
        if "partition" in frame.columns
        else "pair_partition"
        if "pair_partition" in frame.columns
        else None
    )
    if column is None:
        return
    values = set(frame[column].astype(str))
    if schema.expected_partition is not None:
        if values != {schema.expected_partition}:
            raise V015ArtifactError(
                f"{schema.filename} must contain only partition "
                f"{schema.expected_partition!r}"
            )
    elif not values.issubset(set(contract.partitions)):
        raise V015ArtifactError(
            f"{schema.filename} contains an unknown V2 partition: {sorted(values)}"
        )


def _validate_truth_cluster_invariants(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
) -> None:
    if not set(_TRUTH_INVARIANT_COLUMNS).issubset(frame.columns):
        return
    for column in ("truth_family", "truth_parameters_json"):
        values = frame[column]
        if values.isna().any() or any(
            not isinstance(value, str) or not value.strip() for value in values.tolist()
        ):
            raise V015ArtifactError(
                f"{schema.filename} column {column!r} contains an invalid value"
            )
    try:
        gamma = frame["gamma"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise V015ArtifactError(f"{schema.filename} gamma is not numeric") from exc
    if not np.isfinite(gamma).all():
        raise V015ArtifactError(f"{schema.filename} gamma is not finite")
    grouped = frame.groupby(["partition", "cluster_id"], sort=False, dropna=False)
    for column in _TRUTH_INVARIANT_COLUMNS:
        if (grouped[column].nunique(dropna=False) != 1).any():
            raise V015ArtifactError(
                f"{schema.filename} changes {column} within a truth cluster"
            )


def _group_values_exact(
    frame: pd.DataFrame,
    *,
    value_column: str,
    expected_values: Sequence[Any],
    context: str,
) -> None:
    expected = set(expected_values)
    for _, group in frame.groupby(["partition", "cluster_id"], sort=False):
        observed = set(group[value_column])
        if observed != expected or len(group) != len(expected):
            raise V015ArtifactError(f"{context} group membership is incomplete")


def _validate_formal_cardinality(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
    contract: FrozenArtifactContract,
) -> None:
    if schema.required_rows is not None and len(frame) != schema.required_rows:
        raise V015ArtifactError(
            f"{schema.filename} row count is {len(frame)}, "
            f"expected {schema.required_rows}"
        )
    if {"partition", "cluster_id"}.issubset(frame.columns):
        clusters = frame.loc[:, ["partition", "cluster_id"]].drop_duplicates()
        reused = clusters.groupby("cluster_id", sort=False)["partition"].nunique()
        if (reused > 1).any():
            raise V015ArtifactError(
                f"{schema.filename} reuses an opaque cluster ID across partitions"
            )
        observed_counts = clusters.groupby("partition", sort=False).size().to_dict()
        if schema.expected_partition is not None:
            expected_counts = {
                schema.expected_partition: contract.partition_member_counts[
                    schema.expected_partition
                ]
            }
        else:
            expected_counts = dict(contract.partition_member_counts)
        if observed_counts != expected_counts:
            raise V015ArtifactError(
                f"{schema.filename} partition member counts changed: "
                f"observed={observed_counts}, expected={expected_counts}"
            )
    filename = schema.filename
    if filename == "prefix_pack.csv":
        _group_values_exact(
            frame,
            value_column="prefix_day",
            expected_values=contract.prefix_days,
            context=filename,
        )
    elif (
        filename
        in {
            "forecast_coordinates.csv",
            "prediction_bundle.csv",
        }
        or filename in contract.truth_filenames
    ):
        _group_values_exact(
            frame,
            value_column="forecast_day",
            expected_values=contract.forecast_days,
            context=filename,
        )
    elif filename == "risk_bundle.csv":
        _group_values_exact(
            frame,
            value_column="score_id",
            expected_values=schema.required_score_ids,
            context=filename,
        )
    elif filename == "decision_bundle.csv":
        _group_values_exact(
            frame,
            value_column="arm",
            expected_values=schema.required_arms,
            context=filename,
        )
    elif filename == "member_forecast_bundle.csv":
        for _, group in frame.groupby(
            ["partition", "cluster_id", "model_id", "variant_id"], sort=False
        ):
            if len(group) != len(contract.forecast_days) or set(
                group["forecast_day"]
            ) != set(contract.forecast_days):
                raise V015ArtifactError(
                    "member_forecast_bundle.csv has an incomplete variant"
                )


def canonicalize_frame(
    frame: pd.DataFrame,
    schema: CsvArtifactSchema,
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
) -> pd.DataFrame:
    """Validate and key-sort one frame without silently dropping columns."""

    if not isinstance(frame, pd.DataFrame):
        raise V015ArtifactError(f"{schema.filename} must be a dataframe")
    if tuple(frame.columns) != schema.columns:
        raise V015ArtifactError(
            f"{schema.filename} columns differ from the frozen allowlist: "
            f"observed={tuple(frame.columns)}, expected={schema.columns}"
        )
    if frame.empty:
        raise V015ArtifactError(f"{schema.filename} cannot be empty")
    _validate_key_columns(frame, schema=schema)
    if frame.duplicated(list(schema.key)).any():
        raise V015ArtifactError(f"{schema.filename} contains duplicate key rows")
    _validate_strict_boolean_columns(frame, schema=schema)
    _validate_protocol_column(frame, schema=schema, contract=contract)
    _validate_partition_column(frame, schema=schema, contract=contract)
    _validate_truth_cluster_invariants(frame, schema=schema)
    key_index = pd.MultiIndex.from_frame(
        frame.loc[:, list(schema.key)],
        names=list(schema.key),
    )
    if (
        key_index.is_monotonic_increasing
        and isinstance(frame.index, pd.RangeIndex)
        and frame.index.start == 0
        and frame.index.step == 1
    ):
        # Formal CSV readers already return canonical key order.  A shallow
        # copy preserves value isolation under pandas copy-on-write without
        # duplicating multi-million-row forecast blocks on every validation.
        ordered = frame.copy(deep=False)
    else:
        ordered = frame.sort_values(list(schema.key), kind="stable").reset_index(
            drop=True
        )
    if formal:
        _validate_formal_cardinality(ordered, schema=schema, contract=contract)
    return ordered


def canonical_csv_bytes(
    frame: pd.DataFrame,
    schema: CsvArtifactSchema,
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
) -> bytes:
    """Validate, key-sort, and serialize with the exact V1 CSV algorithm."""

    ordered = canonicalize_frame(frame, schema, contract, formal=formal)
    try:
        return _v1_canonical_csv_bytes(ordered, columns=schema.columns)
    except ValueError as exc:
        raise V015ArtifactError(str(exc)) from exc


def write_canonical_csv(
    path: str | Path,
    frame: pd.DataFrame,
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
) -> ArtifactMetadata:
    target = Path(path)
    schema = contract.csv_schema(target.name)
    raw = canonical_csv_bytes(frame, schema, contract, formal=formal)
    _atomic_create(target, raw)
    return ArtifactMetadata(
        path=target.name,
        row_count=len(frame),
        byte_count=len(raw),
        byte_sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_canonical_csv(
    path: str | Path,
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
) -> pd.DataFrame:
    target = Path(path)
    schema = contract.csv_schema(target.name)
    try:
        frame = pd.read_csv(
            target,
            encoding="utf-8",
            encoding_errors="strict",
            float_precision="round_trip",
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise V015ArtifactError(f"{target.name} is not strict canonical CSV") from exc
    ordered = canonicalize_frame(frame, schema, contract, formal=formal)
    try:
        observed_canonical = _v1_canonical_csv_bytes(ordered, columns=schema.columns)
    except ValueError as exc:
        raise V015ArtifactError(str(exc)) from exc
    byte_count, byte_sha256 = _byte_count_and_sha256(target)
    if (
        len(observed_canonical) != byte_count
        or hashlib.sha256(observed_canonical).hexdigest() != byte_sha256
    ):
        raise V015ArtifactError(
            f"{target.name} is not the frozen canonical CSV serialization"
        )
    return ordered


def _validate_bundle_frame_identity(
    frame: pd.DataFrame,
    *,
    schema: CsvArtifactSchema,
    contract: FrozenArtifactContract,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise V015ArtifactError(f"{schema.filename} must be a nonempty dataframe")
    if tuple(frame.columns) != schema.columns:
        raise V015ArtifactError(
            f"{schema.filename} columns differ from the frozen allowlist"
        )
    _validate_key_columns(frame, schema=schema)
    if frame.duplicated(list(schema.key)).any():
        raise V015ArtifactError(f"{schema.filename} contains duplicate key rows")
    _validate_strict_boolean_columns(frame, schema=schema)
    _validate_protocol_column(frame, schema=schema, contract=contract)
    _validate_partition_column(frame, schema=schema, contract=contract)


def _cluster_coordinates(
    frame: pd.DataFrame,
    *,
    filename: str,
) -> set[tuple[str, str]]:
    if not {"partition", "cluster_id"}.issubset(frame.columns):
        raise V015ArtifactError(f"{filename} has no cluster coordinates")
    return {
        (str(partition), str(cluster_id))
        for partition, cluster_id in frame.loc[
            :, ["partition", "cluster_id"]
        ].itertuples(index=False, name=None)
    }


def _validate_cluster_coordinate_set(
    coordinates: set[tuple[str, str]],
    *,
    contract: FrozenArtifactContract,
    formal: bool,
    context: str,
) -> None:
    id_partitions: dict[str, set[str]] = {}
    for partition, cluster_id in coordinates:
        id_partitions.setdefault(cluster_id, set()).add(partition)
    if any(len(partitions) != 1 for partitions in id_partitions.values()):
        raise V015ArtifactError(
            f"{context} reuses an opaque cluster ID across partitions"
        )
    if formal:
        observed = {
            partition: sum(
                item_partition == partition for item_partition, _ in coordinates
            )
            for partition in contract.partitions
        }
        expected = dict(contract.partition_member_counts)
        if observed != expected:
            raise V015ArtifactError(
                f"{context} partition member counts changed: "
                f"observed={observed}, expected={expected}"
            )


def _normalized_expected_variants(
    expected_variant_keys: Sequence[tuple[str, str]] | None,
) -> set[tuple[str, str]] | None:
    if expected_variant_keys is None:
        return None
    items = list(expected_variant_keys)
    if not items or any(
        not isinstance(item, (list, tuple))
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0].strip()
        or not isinstance(item[1], str)
        or not item[1].strip()
        for item in items
    ):
        raise V015ArtifactError("Expected diagnostic variant set is invalid")
    expected = {(item[0], item[1]) for item in items}
    if len(expected) != len(items):
        raise V015ArtifactError("Expected diagnostic variant set is invalid")
    return expected


def validate_prediction_artifact_bundle(
    frames: Mapping[str, pd.DataFrame],
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
    expected_variant_keys: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Validate cross-file cluster and diagnostic-variant alignment.

    Callers should pass frames returned by :func:`read_canonical_csv`.  The
    inexpensive identity checks below are repeated intentionally so a scorer
    cannot accidentally validate a stale or differently keyed dataframe.
    """

    if set(frames) != set(_PREDICTION_CSV_FILENAMES):
        raise V015ArtifactError(
            "Prediction artifact bundle file membership changed: "
            f"observed={sorted(frames)}, "
            f"expected={sorted(_PREDICTION_CSV_FILENAMES)}"
        )
    cluster_sets: dict[str, set[tuple[str, str]]] = {}
    for filename in _PREDICTION_CSV_FILENAMES:
        frame = frames[filename]
        schema = contract.csv_schema(filename)
        _validate_bundle_frame_identity(frame, schema=schema, contract=contract)
        cluster_sets[filename] = _cluster_coordinates(frame, filename=filename)

    reference_name = "prefix_pack.csv"
    reference = cluster_sets[reference_name]
    _validate_cluster_coordinate_set(
        reference,
        contract=contract,
        formal=formal,
        context="Prediction artifact bundle",
    )
    for filename, observed in cluster_sets.items():
        if observed != reference:
            raise V015ArtifactError(
                f"{filename} cluster set differs from {reference_name}: "
                f"missing={len(reference - observed)}, "
                f"unexpected={len(observed - reference)}"
            )

    diagnostics = frames["member_fit_diagnostics.csv"]
    forecasts = frames["member_forecast_bundle.csv"]
    diagnostic_coordinates = {
        tuple(row)
        for row in diagnostics.loc[
            :, ["partition", "cluster_id", "model_id", "variant_id"]
        ].itertuples(index=False, name=None)
    }
    forecast_coordinates = {
        tuple(row)
        for row in forecasts.loc[
            :, ["partition", "cluster_id", "model_id", "variant_id"]
        ].itertuples(index=False, name=None)
    }
    if diagnostic_coordinates != forecast_coordinates:
        raise V015ArtifactError(
            "Diagnostic and member-forecast variant coordinates differ"
        )

    if formal and expected_variant_keys is None:
        from lifetwin.experiments.calendar_long_horizon_v015_fit import (
            FROZEN_VARIANT_KEYS,
        )

        expected_variant_keys = FROZEN_VARIANT_KEYS
    expected = _normalized_expected_variants(expected_variant_keys)
    if expected is not None:
        for _, group in diagnostics.groupby(["partition", "cluster_id"], sort=False):
            observed = set(
                group.loc[:, ["model_id", "variant_id"]].itertuples(
                    index=False, name=None
                )
            )
            if observed != expected:
                raise V015ArtifactError(
                    "One or more clusters differ from the expected variant set"
                )


def read_prediction_artifact_bundle(
    label_free_root: str | Path,
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
    expected_variant_keys: Sequence[tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read and jointly validate every committed prediction CSV."""

    root = Path(label_free_root).resolve()
    frames = {
        filename: read_canonical_csv(
            _safe_child(root, filename), contract, formal=formal
        )
        for filename in _PREDICTION_CSV_FILENAMES
    }
    validate_prediction_artifact_bundle(
        frames,
        contract,
        formal=formal,
        expected_variant_keys=expected_variant_keys,
    )
    return frames


def validate_sealed_truth_bundle(
    frames: Mapping[str, pd.DataFrame],
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
) -> None:
    """Validate truth invariants and the two matched-pair membership bijections."""

    if set(frames) != set(contract.sealed_filenames):
        raise V015ArtifactError("Sealed artifact bundle file membership changed")
    truth_coordinates: set[tuple[str, str]] = set()
    for filename in contract.sealed_filenames:
        frame = frames[filename]
        schema = contract.csv_schema(filename)
        _validate_bundle_frame_identity(frame, schema=schema, contract=contract)
        if filename in contract.truth_filenames:
            _validate_truth_cluster_invariants(frame, schema=schema)
            coordinates = _cluster_coordinates(frame, filename=filename)
            for coordinate in coordinates:
                if coordinate in truth_coordinates:
                    raise V015ArtifactError(
                        "A truth cluster coordinate appears in multiple files"
                    )
                truth_coordinates.add(coordinate)
            if formal:
                _validate_formal_cardinality(frame, schema=schema, contract=contract)
        elif formal:
            _validate_formal_cardinality(frame, schema=schema, contract=contract)

    _validate_cluster_coordinate_set(
        truth_coordinates,
        contract=contract,
        formal=formal,
        context="Sealed truth bundle",
    )
    matched_files = (
        ("intrinsic_matched_pairs.csv", "intrinsic_matched_truth.csv"),
        ("stress_plan_matched_pairs.csv", "stress_plan_matched_truth.csv"),
    )
    for mapping_name, truth_name in matched_files:
        mapping = frames[mapping_name]
        for column in ("left_cluster_id", "right_cluster_id"):
            values = mapping[column]
            if values.isna().any() or any(
                not isinstance(value, str) or not value.strip()
                for value in values.tolist()
            ):
                raise V015ArtifactError(f"{mapping_name} contains an invalid {column}")
        left = mapping["left_cluster_id"].tolist()
        right = mapping["right_cluster_id"].tolist()
        if any(left_id == right_id for left_id, right_id in zip(left, right)):
            raise V015ArtifactError(f"{mapping_name} maps a cluster to itself")
        flattened = [*left, *right]
        if len(set(flattened)) != len(flattened):
            raise V015ArtifactError(f"{mapping_name} does not map members one-to-one")
        truth_members = {
            str(value) for value in frames[truth_name]["cluster_id"].tolist()
        }
        if set(flattened) != truth_members:
            raise V015ArtifactError(
                f"{mapping_name} does not cover every member in {truth_name}"
            )
        if formal:
            partition = contract.csv_schema(truth_name).expected_partition
            assert partition is not None
            expected_members = contract.partition_member_counts[partition]
            if len(flattened) != expected_members:
                raise V015ArtifactError(
                    f"{mapping_name} must cover exactly {expected_members} members"
                )


def _safe_child(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise V015ArtifactError(f"Unsafe artifact filename: {filename!r}")
    resolved_root = root.resolve()
    child = (resolved_root / filename).resolve()
    if child.parent != resolved_root:
        raise V015ArtifactError(f"Artifact escaped its frozen root: {filename}")
    return child


def assert_separate_truth_roots(
    label_free_root: str | Path,
    sealed_truth_root: str | Path,
) -> tuple[Path, Path]:
    """Require physically disjoint directory trees for predictors and truth."""

    label = Path(label_free_root).resolve()
    sealed = Path(sealed_truth_root).resolve()
    try:
        common = Path(os.path.commonpath([label, sealed]))
    except ValueError:
        return label, sealed
    if common in {label, sealed}:
        raise V015ArtifactError(
            "Label-free and sealed-truth roots must be disjoint directory trees"
        )
    return label, sealed


def _metadata_from_path(path: Path, *, row_count: int) -> dict[str, Any]:
    byte_count, byte_sha256 = _byte_count_and_sha256(path)
    return {
        "path": path.name,
        "row_count": int(row_count),
        "byte_count": byte_count,
        "byte_sha256": byte_sha256,
    }


def create_truth_commitments(
    *,
    sealed_truth_root: str | Path,
    commitment_path: str | Path,
    contract: FrozenArtifactContract,
    created_utc: str,
    formal: bool = True,
) -> dict[str, Any]:
    """Commit sealed truth bytes without copying truth into the predictor tree."""

    commitment_target = Path(commitment_path)
    _, sealed = assert_separate_truth_roots(commitment_target.parent, sealed_truth_root)
    records: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for filename in contract.sealed_filenames:
        path = _safe_child(sealed, filename)
        frame = read_canonical_csv(path, contract, formal=formal)
        frames[filename] = frame
        records.append(_metadata_from_path(path, row_count=len(frame)))
    validate_sealed_truth_bundle(frames, contract, formal=formal)
    payload = {
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_byte_sha256,
        "files": records,
        "created_utc": created_utc,
        "truth_values_withheld_by_physical_path": True,
    }
    write_canonical_json(commitment_target, payload, contract)
    return payload


def _validate_truth_commitment_entries(
    payload: Mapping[str, Any],
    contract: FrozenArtifactContract,
    *,
    formal: bool,
) -> tuple[Mapping[str, Any], ...]:
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise V015ArtifactError("truth_commitments.json files must be a nonempty array")
    entries: list[Mapping[str, Any]] = []
    names: list[str] = []
    for item in files:
        entry = _require_mapping(item, context="truth commitment file entry")
        if set(entry) != _TRUTH_COMMITMENT_ENTRY_KEYS:
            raise V015ArtifactError("Truth commitment file-entry keys changed")
        name = entry.get("path")
        if not isinstance(name, str) or name not in contract.sealed_filenames:
            raise V015ArtifactError("Truth commitment contains an unknown truth file")
        if (
            isinstance(entry.get("row_count"), bool)
            or not isinstance(entry.get("row_count"), int)
            or entry["row_count"] <= 0
            or isinstance(entry.get("byte_count"), bool)
            or not isinstance(entry.get("byte_count"), int)
            or entry["byte_count"] <= 0
            or not _is_sha256(entry.get("byte_sha256"))
        ):
            raise V015ArtifactError("Truth commitment metadata is invalid")
        names.append(name)
        entries.append(entry)
    if names != list(contract.sealed_filenames):
        raise V015ArtifactError("Truth commitment file order or membership changed")
    if formal:
        for entry in entries:
            expected = contract.csv_schema(str(entry["path"])).required_rows
            if entry["row_count"] != expected:
                raise V015ArtifactError("Truth commitment row count changed")
    if payload.get("truth_values_withheld_by_physical_path") is not True:
        raise V015ArtifactError("Truth commitment weakened its physical firewall")
    return tuple(entries)


def read_truth_commitments(
    path: str | Path,
    contract: FrozenArtifactContract,
    *,
    formal: bool = True,
) -> dict[str, Any]:
    payload = read_canonical_json(path, contract)
    _validate_truth_commitment_entries(payload, contract, formal=formal)
    return payload


def verify_sealed_truth_files(
    *,
    commitment_path: str | Path,
    sealed_truth_root: str | Path,
    contract: FrozenArtifactContract,
    truth_access_authorized: bool,
    formal: bool = True,
) -> tuple[ArtifactMetadata, ...]:
    """Verify truth only after the caller explicitly crosses the firewall."""

    if truth_access_authorized is not True:
        raise V015ArtifactError("Sealed truth access was not explicitly authorized")
    payload = read_truth_commitments(commitment_path, contract, formal=formal)
    entries = _validate_truth_commitment_entries(payload, contract, formal=formal)
    sealed = Path(sealed_truth_root).resolve()
    verified: list[ArtifactMetadata] = []
    frames: dict[str, pd.DataFrame] = {}
    for entry in entries:
        path = _safe_child(sealed, str(entry["path"]))
        frame = read_canonical_csv(path, contract, formal=formal)
        frames[path.name] = frame
        byte_count, byte_sha256 = _byte_count_and_sha256(path)
        if (
            byte_count != entry["byte_count"]
            or len(frame) != entry["row_count"]
            or byte_sha256 != entry["byte_sha256"]
        ):
            raise V015ArtifactError(f"Sealed truth commitment failed for {path.name}")
        verified.append(
            ArtifactMetadata(
                path=path.name,
                row_count=len(frame),
                byte_count=byte_count,
                byte_sha256=byte_sha256,
            )
        )
    validate_sealed_truth_bundle(frames, contract, formal=formal)
    return tuple(verified)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _parse_aware_utc(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str):
        raise V015ArtifactError(f"{context} must be an ISO timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise V015ArtifactError(f"{context} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise V015ArtifactError(f"{context} must include a timezone")
    return parsed


def _validate_exposure_event(
    event: Mapping[str, Any],
    contract: FrozenArtifactContract,
) -> None:
    if set(event) != contract.exposure_keys:
        raise V015ArtifactError("Exposure event keys differ from the frozen allowlist")
    if not isinstance(event.get("attempt_id"), str) or not event["attempt_id"]:
        raise V015ArtifactError("Exposure attempt_id is empty")
    _parse_aware_utc(event.get("created_utc"), context="exposure created_utc")
    if not isinstance(event.get("git_commit"), str) or not event["git_commit"]:
        raise V015ArtifactError("Exposure git_commit is empty")
    if not isinstance(event.get("git_dirty"), bool):
        raise V015ArtifactError("Exposure git_dirty must be boolean")
    if event.get("config_byte_sha256") != contract.config_byte_sha256:
        raise V015ArtifactError("Exposure config hash changed")
    for key in (
        "truth_commitments_byte_sha256",
        "prediction_commitment_byte_sha256",
    ):
        if event.get(key) is not None and not _is_sha256(event[key]):
            raise V015ArtifactError(f"Exposure {key} is invalid")
    opened = event.get("opened_truth_files")
    if (
        not isinstance(opened, list)
        or any(item not in contract.sealed_filenames for item in opened)
        or opened != sorted(set(opened))
    ):
        raise V015ArtifactError(
            "Exposure opened_truth_files must be a sorted unique sealed-file list"
        )
    for key in ("phase", "exit_status", "message"):
        if not isinstance(event.get(key), str):
            raise V015ArtifactError(f"Exposure {key} must be a string")


def read_exposure_log(
    path: str | Path,
    contract: FrozenArtifactContract,
) -> tuple[dict[str, Any], ...]:
    target = Path(path)
    if not target.exists():
        return ()
    raw = target.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise V015ArtifactError("Exposure log has a truncated final line")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True), start=1):
        payload = _require_mapping(
            _load_json_bytes(line, context=f"exposure line {index}"),
            context=f"exposure line {index}",
        )
        _validate_exposure_event(payload, contract)
        if _canonical_json_line_bytes(payload) != line:
            raise V015ArtifactError(f"Exposure line {index} is not canonical JSONL")
        events.append(dict(payload))
    by_attempt: dict[str, set[str]] = {}
    previous_time: datetime | None = None
    for event in events:
        observed_time = _parse_aware_utc(
            event["created_utc"], context="exposure created_utc"
        )
        if previous_time is not None and observed_time < previous_time:
            raise V015ArtifactError("Exposure timestamps moved backwards")
        previous_time = observed_time
        attempt_id = str(event["attempt_id"])
        prior_opened = by_attempt.setdefault(attempt_id, set())
        current_opened = set(event["opened_truth_files"])
        if not prior_opened.issubset(current_opened):
            raise V015ArtifactError("Exposure log forgot previously opened truth")
        by_attempt[attempt_id] = current_opened
    return tuple(events)


def append_exposure_event(
    path: str | Path,
    event: Mapping[str, Any],
    contract: FrozenArtifactContract,
) -> None:
    """Append one fsync'd canonical event without rewriting prior bytes."""

    target = Path(path)
    _validate_exposure_event(event, contract)
    existing = read_exposure_log(target, contract)
    if existing:
        last = existing[-1]
        if _parse_aware_utc(
            event["created_utc"], context="exposure created_utc"
        ) < _parse_aware_utc(last["created_utc"], context="exposure created_utc"):
            raise V015ArtifactError("Exposure timestamps moved backwards")
        same_attempt = [
            item for item in existing if item["attempt_id"] == event["attempt_id"]
        ]
        if same_attempt and not set(same_attempt[-1]["opened_truth_files"]).issubset(
            set(event["opened_truth_files"])
        ):
            raise V015ArtifactError("Exposure event forgot previously opened truth")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_line_bytes(event)
    with target.open("ab") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _numeric_records(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_column: str | None = None,
) -> list[list[Any]]:
    working = frame.loc[:, list(columns)]
    if sort_column is not None:
        working = working.sort_values(sort_column, kind="stable")
    records: list[list[Any]] = []
    for row in working.itertuples(index=False, name=None):
        clean: list[Any] = []
        for value in row:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise V015ArtifactError("Predictor content contains a nonfinite value")
            clean.append(value)
        records.append(clean)
    return records


def predictor_content_payloads(
    prefix_rows: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    operating_row: Mapping[str, Any] | pd.Series,
    *,
    enforce_frozen_counts: bool = True,
) -> PredictorContentPayloads:
    """Serialize only prediction-time content, excluding every identity field."""

    prefix_columns = ("prefix_day", "observed_retention_pct")
    forecast_columns = ("forecast_day",)
    missing_prefix = set(prefix_columns).difference(prefix_rows.columns)
    missing_forecast = set(forecast_columns).difference(forecast_rows.columns)
    if missing_prefix or missing_forecast:
        raise V015ArtifactError("Predictor content columns are incomplete")
    if prefix_rows["prefix_day"].duplicated().any():
        raise V015ArtifactError("Predictor prefix days are duplicated")
    if forecast_rows["forecast_day"].duplicated().any():
        raise V015ArtifactError("Predictor forecast days are duplicated")
    if enforce_frozen_counts and (len(prefix_rows) != 12 or len(forecast_rows) != 8):
        raise V015ArtifactError("Predictor content does not have frozen 12/8 counts")

    real_fields = (
        "past_mean_temperature_c",
        "past_mean_soc_fraction",
        "past_mean_dod_fraction",
        "past_efc_per_year",
        "planned_mean_temperature_c",
        "planned_mean_soc_fraction",
        "planned_mean_dod_fraction",
        "planned_efc_per_year",
    )
    placebo_fields = tuple(f"placebo_control_{index}" for index in range(1, 9))
    operating = dict(operating_row)
    if set(real_fields + placebo_fields).difference(operating):
        raise V015ArtifactError("Predictor operating content is incomplete")

    base = {
        "prefix": _numeric_records(
            prefix_rows, prefix_columns, sort_column="prefix_day"
        ),
        "forecast": _numeric_records(
            forecast_rows, forecast_columns, sort_column="forecast_day"
        ),
    }
    arm_a = canonical_json_bytes(base)
    arm_b_payload = {
        **base,
        "real_operating": [_json_ready(operating[name]) for name in real_fields],
    }
    placebo_payload = {
        **base,
        "placebo_operating": [_json_ready(operating[name]) for name in placebo_fields],
    }
    arm_b = canonical_json_bytes(arm_b_payload)
    placebo = canonical_json_bytes(placebo_payload)
    return PredictorContentPayloads(
        random_policy=arm_a,
        arm_a=arm_a,
        arm_b=arm_b,
        placebo=placebo,
    )


def predictor_content_hashes(
    prefix_rows: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    operating_row: Mapping[str, Any] | pd.Series,
    *,
    enforce_frozen_counts: bool = True,
) -> PredictorContentHashes:
    """Hash only prediction-time content, excluding every identity field."""

    payloads = predictor_content_payloads(
        prefix_rows,
        forecast_rows,
        operating_row,
        enforce_frozen_counts=enforce_frozen_counts,
    )
    return PredictorContentHashes(
        random_policy=hashlib.sha256(payloads.random_policy).hexdigest(),
        arm_a=hashlib.sha256(payloads.arm_a).hexdigest(),
        arm_b=hashlib.sha256(payloads.arm_b).hexdigest(),
        placebo=hashlib.sha256(payloads.placebo).hexdigest(),
    )


def _sha256_path(path: Path) -> str:
    return _byte_count_and_sha256(path)[1]


def _validate_frozen_state_artifacts(
    root: Path,
    contract: FrozenArtifactContract,
    *,
    formal: bool,
) -> None:
    """Validate state semantics and manifest bindings before formal reveal."""

    model_path = _safe_child(root, "model_state.json")
    read_canonical_json(model_path, contract)
    if not formal:
        return
    try:
        from lifetwin.experiments.calendar_long_horizon_v015_training import (
            V015TrainingError,
            deserialize_model_state_json,
            validate_calibration_manifest,
            validate_training_manifest,
            verify_calibration_manifest_state_hashes,
            verify_training_manifest_state_hashes,
        )

        decoded = deserialize_model_state_json(model_path.read_bytes())
        training_manifest = read_canonical_json(
            _safe_child(root, "training_manifest.json"), contract
        )
        calibration_manifest = read_canonical_json(
            _safe_child(root, "calibration_manifest.json"), contract
        )
        validate_training_manifest(training_manifest)
        validate_calibration_manifest(calibration_manifest)
        verify_training_manifest_state_hashes(
            training_manifest,
            center_state=decoded.training_state.center,
            risk_state=decoded.training_state.risk,
        )
        verify_calibration_manifest_state_hashes(
            calibration_manifest,
            calibration_state=decoded.training_state.calibration,
        )
        if (
            dict(training_manifest["center_development_input_hashes"])
            != decoded.input_byte_hashes["center_development"]
            or dict(training_manifest["risk_development_input_hashes"])
            != decoded.input_byte_hashes["risk_development"]
            or dict(calibration_manifest["calibration_input_hashes"])
            != decoded.input_byte_hashes["calibration"]
        ):
            raise V015TrainingError(
                "State manifests do not bind the model-state input hashes"
            )
    except (OSError, KeyError, V015TrainingError) as exc:
        raise V015ArtifactError(
            "Formal frozen state artifacts failed semantic validation"
        ) from exc


def create_prediction_commitment(
    *,
    label_free_root: str | Path,
    commitment_path: str | Path,
    contract: FrozenArtifactContract,
    created_utc: str,
    formal: bool = True,
    expected_variant_keys: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Commit every predictor artifact without accepting any truth path."""

    root = Path(label_free_root).resolve()
    target = Path(commitment_path)
    if target.resolve().parent != root:
        raise V015ArtifactError(
            "prediction_commitment.json must live in the label-free root"
        )
    _validate_frozen_state_artifacts(root, contract, formal=formal)
    frames = read_prediction_artifact_bundle(
        root,
        contract,
        formal=formal,
        expected_variant_keys=expected_variant_keys,
    )
    metadata: dict[str, ArtifactMetadata] = {}
    for filename in _PREDICTION_HASH_FILES.values():
        path = _safe_child(root, filename)
        byte_count, byte_sha256 = _byte_count_and_sha256(path)
        metadata[filename] = ArtifactMetadata(
            path=filename,
            row_count=len(frames[filename]) if filename.endswith(".csv") else 1,
            byte_count=byte_count,
            byte_sha256=byte_sha256,
        )
    payload: dict[str, Any] = {
        "protocol_id": contract.protocol_id,
        "config_sha256": contract.config_byte_sha256,
        "row_counts": {
            filename: item.row_count
            for filename, item in metadata.items()
            if filename.endswith(".csv")
        },
        "created_utc": created_utc,
        "sealed_truth_opened_before_commitment": False,
    }
    for key, filename in _PREDICTION_HASH_FILES.items():
        payload[key] = metadata[filename].byte_sha256
    write_canonical_json(target, payload, contract)
    return payload


def verify_prediction_commitment(
    *,
    commitment_path: str | Path,
    label_free_root: str | Path,
    contract: FrozenArtifactContract,
    formal: bool = True,
    expected_variant_keys: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify all predictor bytes before any caller may request truth access."""

    root = Path(label_free_root).resolve()
    target = Path(commitment_path).resolve()
    if target.parent != root:
        raise V015ArtifactError(
            "prediction_commitment.json must live in the label-free root"
        )
    payload = read_canonical_json(target, contract)
    if payload.get("sealed_truth_opened_before_commitment") is not False:
        raise V015ArtifactError("Prediction commitment reports premature truth access")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise V015ArtifactError("Prediction commitment row_counts must be an object")
    expected_row_keys = {
        filename
        for filename in _PREDICTION_HASH_FILES.values()
        if filename.endswith(".csv")
    }
    if set(row_counts) != expected_row_keys:
        raise V015ArtifactError("Prediction commitment row-count keys changed")

    _validate_frozen_state_artifacts(root, contract, formal=formal)
    frames = read_prediction_artifact_bundle(
        root,
        contract,
        formal=formal,
        expected_variant_keys=expected_variant_keys,
    )
    for key, filename in _PREDICTION_HASH_FILES.items():
        path = _safe_child(root, filename)
        if filename.endswith(".csv"):
            if row_counts[filename] != len(frames[filename]):
                raise V015ArtifactError(f"{filename} committed row count changed")
        if not _is_sha256(payload.get(key)) or payload[key] != _sha256_path(path):
            raise V015ArtifactError(f"{filename} bytes differ from their commitment")
    return payload
