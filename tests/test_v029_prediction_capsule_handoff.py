from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v019_prediction as prediction
from lifetwin.experiments import (
    calendar_long_horizon_v019_prediction_capsule as capsule,
)


def test_isolated_loader_binds_authenticated_protocol_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_id = "fixture_protocol"
    config_sha256 = "a" * 64
    git_commit = "b" * 40
    artifact_raw = b"{}"
    frame_raw = b"fixture-frame"
    for filename in capsule._PRE_PREDICTION_FILES:
        raw = frame_raw if filename.endswith(".csv") else artifact_raw
        (tmp_path / filename).write_bytes(raw)

    monkeypatch.setattr(
        capsule,
        "_load_prediction_progress",
        lambda *args, **kwargs: (SimpleNamespace(), artifact_raw),
    )
    monkeypatch.setattr(capsule, "_strict_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        capsule,
        "_verify_truth_commitment",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        capsule,
        "read_canonical_csv",
        lambda *args, **kwargs: pd.DataFrame({"fixture": [1]}),
    )
    monkeypatch.setattr(
        capsule,
        "canonical_csv_bytes",
        lambda *args, **kwargs: frame_raw,
    )
    for name in (
        "_require_phase_hash",
        "_validate_prediction_input_alignment",
        "_verify_file_commitment",
        "_identity_json",
        "_verify_training_chain_semantics",
        "_verify_model_input_hashes",
    ):
        monkeypatch.setattr(capsule, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        capsule,
        "_decode_mask_commitment",
        lambda *args, **kwargs: object(),
    )
    prediction_state = object()
    monkeypatch.setattr(
        capsule,
        "decode_prediction_state",
        lambda *args, **kwargs: SimpleNamespace(
            protocol_id=protocol_id,
            state=prediction_state,
            input_byte_hashes={},
            model_state_byte_sha256=hashlib.sha256(artifact_raw).hexdigest(),
        ),
    )

    observed: dict[str, object] = {}

    pipeline_result = object()

    def run_bundle(
        bundle: capsule.PredictionBundle,
        *,
        formal: bool,
    ) -> object:
        assert formal is True
        observed.update(
            protocol_id=bundle._protocol_id,
            root=bundle._root,
            state=bundle._state,
        )
        return pipeline_result

    monkeypatch.setattr(capsule, "run_prediction_bundle", run_bundle)
    monkeypatch.setattr(
        capsule,
        "write_prediction_outputs",
        lambda bundle, *, result: (),
    )

    result = prediction.run_isolated_prediction_process_v024(
        label_free_root=tmp_path,
        attempt_id="fixture-attempt",
        repo_root=tmp_path,
        _environment_verifier=lambda root: SimpleNamespace(
            protocol_id=protocol_id,
            config_byte_sha256=config_sha256,
            git_commit=git_commit,
        ),
    )

    assert result.artifacts == ()
    assert observed == {
        "protocol_id": protocol_id,
        "root": tmp_path.resolve(),
        "state": prediction_state,
    }
