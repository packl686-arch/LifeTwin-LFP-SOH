from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v016_io as io
from lifetwin.experiments import calendar_long_horizon_v016_prediction as prediction
from lifetwin.experiments import (
    calendar_long_horizon_v016_prediction_capsule as prediction_capsule,
)
from lifetwin.experiments import (
    calendar_long_horizon_v016_prediction_environment as prediction_environment,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import V015FitResult
from lifetwin.experiments import calendar_long_horizon_v015_io as v015_io
from lifetwin.experiments.calendar_long_horizon_v016_contract import (
    load_v021_contract_view,
)
from lifetwin.experiments.calendar_long_horizon_v016_provenance import (
    V021ValidatedModelStateEnvelope,
)


def _frozen_view():
    return replace(
        load_v021_contract_view(),
        design_status="implementation_frozen",
    )


def test_core_formal_surfaces_have_no_path_truth_or_raw_frame_parameters() -> None:
    assert tuple(
        inspect.signature(prediction.run_formal_prediction_v021).parameters
    ) == ("label_free_bundle", "model_state_envelope")
    assert tuple(
        inspect.signature(prediction.fit_verified_generation_bundle_v021).parameters
    ) == ("bundle",)
    assert tuple(
        inspect.signature(prediction.commit_verified_fit_result_v021).parameters
    ) == ("bundle", "fit_result", "created_utc")
    forbidden = {"truth", "sealed", "score", "path", "root", "frame", "seed"}
    for function in (
        prediction.run_formal_prediction_v021,
        prediction.fit_verified_generation_bundle_v021,
        prediction.commit_verified_fit_result_v021,
    ):
        assert not any(
            token in name
            for name in inspect.signature(function).parameters
            for token in forbidden
        )
    assert tuple(
        inspect.signature(prediction.run_isolated_prediction_process_v021).parameters
    ) == ("label_free_root", "attempt_id", "repo_root")


def test_prediction_import_closure_has_no_generation_or_collision_module() -> None:
    script = """
import json
import sys
import lifetwin.experiments.calendar_long_horizon_v016_prediction
forbidden = [
    name for name in sys.modules
    if any(token in name for token in (
        "calendar_long_horizon_v016_collision",
        "calendar_long_horizon_v016_generation",
        "calendar_long_horizon_v016_actual_ledger_io",
    ))
]
print(json.dumps(forbidden))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        cwd=Path(__file__).resolve().parents[1],
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize(
    "model",
    [
        object(),
        object.__new__(V021ValidatedModelStateEnvelope),
    ],
)
def test_formal_prediction_rejects_bare_and_validated_only_state(
    model: object,
) -> None:
    forged_bundle = object.__new__(io.V021CommittedLabelFreeBundle)
    object.__setattr__(forged_bundle, "_seal", io._SEAL)
    with pytest.raises(prediction.V021PredictionError, match="rejected"):
        prediction.run_formal_prediction_v021(
            label_free_bundle=forged_bundle,
            model_state_envelope=model,  # type: ignore[arg-type]
        )


def test_verified_fit_result_cannot_be_constructed_or_cross_bound() -> None:
    with pytest.raises(TypeError, match="issued only"):
        prediction.V021VerifiedFitResult(
            _seal=object(),
            bundle=object(),  # type: ignore[arg-type]
            result=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(prediction.V021PredictionError, match="exact fit result"):
        prediction.commit_verified_fit_result_v021(
            bundle=object(),  # type: ignore[arg-type]
            fit_result=object(),  # type: ignore[arg-type]
            created_utc="2026-07-26T00:00:00Z",
        )


def test_verified_fit_result_detects_private_frame_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = object.__new__(io.V021FreshGenerationBundle)
    object.__setattr__(bundle, "_contract_view", _frozen_view())
    monkeypatch.setattr(
        v015_io,
        "canonical_csv_bytes",
        lambda frame, *args, **kwargs: str(int(frame.iloc[0, 0])).encode("ascii"),
    )
    token = prediction.V021VerifiedFitResult(
        _seal=prediction._FIT_SEAL,
        bundle=bundle,
        result=V015FitResult(
            member_fit_diagnostics=pd.DataFrame({"value": [1]}),
            member_forecast_bundle=pd.DataFrame({"value": [1]}),
        ),
    )
    dict(token._frames)["member_fit_diagnostics.csv"].iloc[0, 0] = 2
    with pytest.raises(prediction.V021PredictionError, match="changed"):
        prediction.commit_verified_fit_result_v021(
            bundle=bundle,
            fit_result=token,
            created_utc="2026-07-26T00:00:00Z",
        )


def test_isolated_adapter_binds_attempt_to_attested_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        prediction_environment,
        "verify_prediction_environment",
        lambda repo_root: SimpleNamespace(
            git_commit="a" * 40,
            config_byte_sha256="b" * 64,
        ),
    )
    bundle = object()
    observed: dict[str, object] = {}

    def load_bundle(**kwargs):
        observed.update(kwargs)
        return bundle

    monkeypatch.setattr(
        prediction_capsule,
        "load_prediction_bundle",
        load_bundle,
    )
    output = SimpleNamespace(
        prediction_bundle=pd.DataFrame(),
        primary_risk_bundle=pd.DataFrame(),
        decision_bundle=pd.DataFrame(),
    )
    monkeypatch.setattr(
        prediction_capsule,
        "run_prediction_bundle",
        lambda value, formal: output,
    )
    monkeypatch.setattr(
        prediction_capsule,
        "write_prediction_outputs",
        lambda value, result: (),
    )
    result = prediction.run_isolated_prediction_process_v021(
        label_free_root=tmp_path,
        attempt_id="v021-fixture",
        repo_root=tmp_path,
    )
    assert result.artifacts == ()
    assert observed == {
        "label_free_root": tmp_path,
        "attempt_id": "v021-fixture",
        "expected_config_sha256": "b" * 64,
        "expected_git_commit": "a" * 40,
    }


def test_prediction_commitment_verifier_enters_full_chain_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def reject_chain(**kwargs):
        raise io.V021IOError("full-chain sentinel")

    monkeypatch.setattr(io, "_load_committed_bundle", reject_chain)
    with pytest.raises(io.V021IOError, match="full-chain sentinel"):
        io.verify_prediction_commitment_v021(
            label_free_root=tmp_path,
            attempt_id="v021-fixture",
            contract_view=_frozen_view(),
            require_ledger_committed=True,
        )


def test_prediction_input_extractor_never_returns_a_root_capability() -> None:
    annotation = inspect.signature(io._extract_prediction_inputs_v021).return_annotation
    assert "Path" not in str(annotation)
