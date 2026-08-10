"""Truth-incapable V2.3 formal fit and prediction entry points.

Public functions accept only sealed capabilities issued by V2.3 IO.  Their
signatures deliberately contain no label-root, sealed-truth, score-root, raw
state, or caller-reported hash parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lifetwin.experiments.calendar_long_horizon_v015_fit import (
        V015FitResult,
    )
    from lifetwin.experiments.calendar_long_horizon_v015_io import (
        ArtifactMetadata,
    )
    from lifetwin.experiments.calendar_long_horizon_v018_io import (
        V023CommittedLabelFreeBundle,
        V023FreshGenerationBundle,
    )
    from lifetwin.experiments.calendar_long_horizon_v018_provenance import (
        V023CommittedModelStateEnvelope,
    )


class V023PredictionError(RuntimeError):
    """Raised when a sealed V2.3 fit or prediction capability is invalid."""


_FIT_SEAL = object()


class V023VerifiedFitResult:
    """Opaque formal-fit output tied to one exact fresh-generation bundle."""

    __slots__ = ("_bundle", "_frame_hashes", "_frames", "_seal")

    def __init__(
        self,
        *,
        _seal: object,
        bundle: V023FreshGenerationBundle,
        result: V015FitResult,
    ) -> None:
        from lifetwin.experiments.calendar_long_horizon_v015_fit import (  # noqa: PLC0415
            V015FitResult,
        )
        from lifetwin.experiments.calendar_long_horizon_v015_io import (  # noqa: PLC0415
            V015ArtifactError,
            canonical_csv_bytes,
        )

        if _seal is not _FIT_SEAL or type(self) is not V023VerifiedFitResult:
            raise TypeError(
                "Verified formal fit results are issued only by V2.3 prediction"
            )
        if type(result) is not V015FitResult:
            raise TypeError("Formal fit returned an unexpected result type")
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_bundle", bundle)
        frames = (
            (
                "member_fit_diagnostics.csv",
                result.member_fit_diagnostics.copy(deep=True),
            ),
            (
                "member_forecast_bundle.csv",
                result.member_forecast_bundle.copy(deep=True),
            ),
        )
        try:
            hashes = tuple(
                (
                    filename,
                    hashlib.sha256(
                        canonical_csv_bytes(
                            frame,
                            bundle._contract_view.artifacts.csv_schema(filename),
                            bundle._contract_view.artifacts,
                            formal=True,
                        )
                    ).hexdigest(),
                )
                for filename, frame in frames
            )
        except V015ArtifactError as exc:
            raise V023PredictionError(
                "Formal fit result is not a canonical V2.3 artifact"
            ) from exc
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_frame_hashes", hashes)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Verified formal fit results are immutable")


@dataclass(frozen=True, slots=True)
class V023PredictionWriteResult:
    """Metadata for three exclusively written prediction artifacts."""

    artifacts: tuple[ArtifactMetadata, ...]


def fit_verified_generation_bundle_v023(
    bundle: V023FreshGenerationBundle,
) -> V023VerifiedFitResult:
    """Run the six-worker formal optimizer only on verified fresh generation."""

    from lifetwin.experiments.calendar_long_horizon_v018_fit import (  # noqa: PLC0415
        V023FitError,
        _fit_structure_library_formal_from_verified_frames_v023,
    )
    from lifetwin.experiments.calendar_long_horizon_v018_io import (  # noqa: PLC0415
        V023IOError,
        _extract_fresh_generation_frames_for_formal_fit_v023,
    )

    try:
        prefix, coordinates, contract = (
            _extract_fresh_generation_frames_for_formal_fit_v023(bundle)
        )
        result = _fit_structure_library_formal_from_verified_frames_v023(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            contract=contract,
        )
    except (V023FitError, V023IOError) as exc:
        raise V023PredictionError(
            "The sealed fresh-generation fit capability was rejected"
        ) from exc
    return V023VerifiedFitResult(
        _seal=_FIT_SEAL,
        bundle=bundle,
        result=result,
    )


def commit_verified_fit_result_v023(
    *,
    bundle: V023FreshGenerationBundle,
    fit_result: V023VerifiedFitResult,
    created_utc: str,
) -> str:
    """Exclusively persist a fit produced from the same sealed generation."""

    from lifetwin.experiments.calendar_long_horizon_v015_io import (  # noqa: PLC0415
        V015ArtifactError,
        canonical_csv_bytes,
    )
    from lifetwin.experiments.calendar_long_horizon_v018_io import (  # noqa: PLC0415
        V023IOError,
        _write_verified_fit_outputs_and_commitment_v023,
    )

    if (
        type(fit_result) is not V023VerifiedFitResult
        or fit_result._seal is not _FIT_SEAL
        or fit_result._bundle is not bundle
    ):
        raise V023PredictionError(
            "An exact fit result from this generation bundle is required"
        )
    try:
        observed = tuple(
            (
                filename,
                hashlib.sha256(
                    canonical_csv_bytes(
                        frame,
                        bundle._contract_view.artifacts.csv_schema(filename),
                        bundle._contract_view.artifacts,
                        formal=True,
                    )
                ).hexdigest(),
            )
            for filename, frame in fit_result._frames
        )
    except V015ArtifactError as exc:
        raise V023PredictionError("The sealed formal fit result changed") from exc
    if observed != fit_result._frame_hashes:
        raise V023PredictionError("The sealed formal fit result changed")
    try:
        return _write_verified_fit_outputs_and_commitment_v023(
            bundle,
            frames={name: frame.copy(deep=True) for name, frame in fit_result._frames},
            created_utc=created_utc,
        )
    except V023IOError as exc:
        raise V023PredictionError(
            "The verified formal fit could not be committed"
        ) from exc


def run_formal_prediction_v023(
    *,
    label_free_bundle: V023CommittedLabelFreeBundle,
    model_state_envelope: V023CommittedModelStateEnvelope,
) -> V023PredictionWriteResult:
    """Compute and exclusively write outputs without any truth/path capability."""

    from lifetwin.experiments.calendar_long_horizon_v018_io import (  # noqa: PLC0415
        V023IOError,
        _extract_prediction_inputs_v023,
        _write_prediction_outputs_v023,
    )
    from lifetwin.experiments.calendar_long_horizon_v018_pipeline import (  # noqa: PLC0415
        V023PipelineError,
        recompute_label_free_pipeline_v023,
    )

    try:
        frames, contract = _extract_prediction_inputs_v023(
            label_free_bundle,
            model_state=model_state_envelope,
        )
        output = recompute_label_free_pipeline_v023(
            prefix_pack=frames["prefix_pack.csv"],
            forecast_coordinates=frames["forecast_coordinates.csv"],
            operating_pack=frames["operating_pack.csv"],
            member_fit_diagnostics=frames["member_fit_diagnostics.csv"],
            member_forecast_bundle=frames["member_forecast_bundle.csv"],
            model_state_envelope=model_state_envelope,
            contract=contract,
        )
        artifacts = _write_prediction_outputs_v023(
            label_free_bundle,
            frames={
                "prediction_bundle.csv": output.prediction_bundle,
                "risk_bundle.csv": output.primary_risk_bundle,
                "decision_bundle.csv": output.decision_bundle,
            },
        )
    except (V023IOError, V023PipelineError) as exc:
        raise V023PredictionError(
            "Formal V2.3 prediction capability was rejected"
        ) from exc
    return V023PredictionWriteResult(artifacts=artifacts)


def run_isolated_prediction_process_v023(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    repo_root: str | Path,
) -> V023PredictionWriteResult:
    """Attest one isolated process, load its attempt, and predict."""

    from lifetwin.experiments.calendar_long_horizon_v018_prediction_capsule import (  # noqa: PLC0415
        V023PredictionCapsuleError,
        load_prediction_bundle,
        run_prediction_bundle,
        write_prediction_outputs,
    )
    from lifetwin.experiments.calendar_long_horizon_v018_prediction_environment import (  # noqa: PLC0415
        V023PredictionEnvironmentError,
        verify_prediction_environment,
    )

    try:
        environment = verify_prediction_environment(repo_root)
        bundle = load_prediction_bundle(
            label_free_root=label_free_root,
            attempt_id=attempt_id,
            expected_config_sha256=environment.config_byte_sha256,
            expected_git_commit=environment.git_commit,
        )
        output = run_prediction_bundle(bundle, formal=True)
        artifacts = write_prediction_outputs(
            bundle,
            result=output,
        )
    except (
        V023PredictionCapsuleError,
        V023PredictionEnvironmentError,
    ) as exc:
        raise V023PredictionError(
            "The isolated prediction environment or attempt was rejected"
        ) from exc
    return V023PredictionWriteResult(artifacts=artifacts)


__all__ = [
    "V023PredictionError",
    "V023PredictionWriteResult",
    "V023VerifiedFitResult",
    "commit_verified_fit_result_v023",
    "fit_verified_generation_bundle_v023",
    "run_isolated_prediction_process_v023",
    "run_formal_prediction_v023",
]
