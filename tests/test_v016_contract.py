from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from lifetwin.experiments import calendar_long_horizon_v016_contract as contract
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    DEFAULT_V2_CONFIG_PATH,
    FrozenArtifactContract,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    ValidatedV015Protocol,
    load_frozen_protocol_config,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    V021_EXPECTED_SEED_ROOTS,
    V021_PROTOCOL_ID,
    load_v021_design,
)


def _base_contracts() -> tuple[ValidatedV015Protocol, FrozenArtifactContract]:
    return (
        load_frozen_protocol_config(DEFAULT_V2_CONFIG_PATH),
        load_artifact_contract(DEFAULT_V2_CONFIG_PATH),
    )


def _rng_state_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_loader_validates_amendment_and_returns_actual_legacy_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = contract.load_v021_design

    def counted(path: str | Path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(contract, "load_v021_design", counted)
    view = contract.load_v021_contract_view()

    assert calls == 1
    assert type(view.protocol) is ValidatedV015Protocol
    assert type(view.artifacts) is FrozenArtifactContract
    assert view.protocol.protocol_id == V021_PROTOCOL_ID
    assert view.artifacts.protocol_id == V021_PROTOCOL_ID


def test_only_declared_protocol_and_artifact_fields_change() -> None:
    base_protocol, base_artifacts = _base_contracts()
    design = load_v021_design()
    view = contract.adapt_v021_contract_view(
        design=design,
        base_protocol=base_protocol,
        base_artifacts=base_artifacts,
    )

    protocol_differences = {
        field.name
        for field in fields(base_protocol)
        if getattr(base_protocol, field.name) != getattr(view.protocol, field.name)
    }
    artifact_differences = {
        field.name
        for field in fields(base_artifacts)
        if getattr(base_artifacts, field.name) != getattr(view.artifacts, field.name)
    }
    assert protocol_differences == {
        "protocol_id",
        "config_sha256",
        "seed_roots",
        "config_json",
    }
    assert artifact_differences == {
        "protocol_id",
        "config_path",
        "config_byte_sha256",
    }

    base_config = base_protocol.config()
    adapted_config = view.protocol.config()
    assert adapted_config["protocol_id"] == V021_PROTOCOL_ID
    assert adapted_config["design_partitions"]["seed_roots"] == dict(
        V021_EXPECTED_SEED_ROOTS
    )
    adapted_config["protocol_id"] = base_config["protocol_id"]
    adapted_config["design_partitions"]["seed_roots"] = base_config[
        "design_partitions"
    ]["seed_roots"]
    assert adapted_config == base_config


def test_amendment_byte_hash_binds_both_adapted_contracts() -> None:
    view = contract.load_v021_contract_view()
    raw = view.artifacts.config_path.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()

    assert view.artifacts.config_path == view.artifacts.config_path.resolve()
    assert view.protocol.config_sha256 == expected
    assert view.artifacts.config_byte_sha256 == expected


def test_fresh_roots_are_exact_unique_and_disjoint_from_v2() -> None:
    base_protocol, _ = _base_contracts()
    view = contract.load_v021_contract_view()
    observed = view.protocol.seed_roots

    assert observed == tuple(V021_EXPECTED_SEED_ROOTS.items())
    assert len(observed) == 13
    assert len({root for _, root in observed}) == 13
    assert not {root for _, root in observed}.intersection(
        root for _, root in base_protocol.seed_roots
    )


def test_adaptation_does_not_modify_inputs_and_outputs_are_immutable() -> None:
    base_protocol, base_artifacts = _base_contracts()
    original_protocol = base_protocol
    original_artifacts = base_artifacts
    original_protocol_config = base_protocol.config()
    original_artifact_path = base_artifacts.config_path
    original_artifact_hash = base_artifacts.config_byte_sha256

    view = contract.adapt_v021_contract_view(
        design=load_v021_design(),
        base_protocol=base_protocol,
        base_artifacts=base_artifacts,
    )

    assert base_protocol == original_protocol
    assert base_artifacts == original_artifacts
    assert base_protocol.config() == original_protocol_config
    assert base_artifacts.config_path == original_artifact_path
    assert base_artifacts.config_byte_sha256 == original_artifact_hash
    with pytest.raises(FrozenInstanceError):
        view.protocol.protocol_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        view.artifacts.csv_schemas["new.csv"] = object()  # type: ignore[index]


def test_strong_invariants_reject_base_grid_or_seed_registry_drift() -> None:
    base_protocol, base_artifacts = _base_contracts()
    design = load_v021_design()

    drifted_grid = replace(
        base_protocol,
        prefix_days=(*base_protocol.prefix_days[:-1], 731.0),
    )
    with pytest.raises(contract.V021ContractError, match="prefix grids disagree"):
        contract.adapt_v021_contract_view(
            design=design,
            base_protocol=drifted_grid,
            base_artifacts=base_artifacts,
        )

    drifted_roots = replace(
        design,
        seed_roots={
            **design.seed_roots,
            "placebo_covariate": next(iter(design.seed_roots.values())),
        },
    )
    with pytest.raises(contract.V021ContractError, match="fresh seed roots drifted"):
        contract.adapt_v021_contract_view(
            design=drifted_roots,
            base_protocol=base_protocol,
            base_artifacts=base_artifacts,
        )


def test_import_and_load_do_not_consume_legacy_numpy_rng_state() -> None:
    before = np.random.get_state()
    contract.load_v021_contract_view()
    after = np.random.get_state()
    assert _rng_state_equal(before, after)


def test_module_exposes_no_generation_or_outcome_capability() -> None:
    source_path = Path(contract.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden_import_fragments = {
        "generation",
        "pipeline",
        "prediction",
        "scoring",
        "training",
        "random",
    }
    assert not any(
        fragment in module
        for module in imported_modules
        for fragment in forbidden_import_fragments
    )

    forbidden_public_fragments = {
        "derive",
        "generate",
        "fit",
        "predict",
        "score",
        "truth",
        "rng",
        "seed",
    }
    assert not any(
        fragment in name.lower()
        for name in contract.__all__
        for fragment in forbidden_public_fragments
    )
