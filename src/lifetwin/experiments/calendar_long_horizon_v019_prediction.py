"""Truth-incapable V2.4 formal fit and prediction entry points.

Public functions accept only sealed capabilities issued by V2.4 IO.  Their
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
    from lifetwin.experiments.calendar_long_horizon_v019_io import (
        V024CommittedLabelFreeBundle,
        V024FreshGenerationBundle,
    )
    from lifetwin.experiments.calendar_long_horizon_v019_provenance import (
        V024CommittedModelStateEnvelope,
    )


class V024PredictionError(RuntimeError):
    """Raised when a sealed V2.4 fit or prediction capability is invalid."""


_FIT_SEAL = object()


class V024VerifiedFitResult:
    """Opaque formal-fit output tied to one exact fresh-generation bundle."""

    __slots__ = ("_bundle", "_frame_hashes", "_frames", "_seal")

    def __init__(
        self,
        *,
        _seal: object,
        bundle: V024FreshGenerationBundle,
        result: V015FitResult,
    ) -> None:
        from lifetwin.experiments.calendar_long_horizon_v015_fit import (  # noqa: PLC0415
            V015FitResult,
        )
        from lifetwin.experiments.calendar_long_horizon_v015_io import (  # noqa: PLC0415
            V015ArtifactError,
            canonical_csv_bytes,
        )

        if _seal is not _FIT_SEAL or type(self) is not V024VerifiedFitResult:
            raise TypeError(
                "Verified formal fit results are issued only by V2.4 prediction"
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
            raise V024PredictionError(
                "Formal fit result is not a canonical V2.4 artifact"
            ) from exc
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_frame_hashes", hashes)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Verified formal fit results are immutable")


@dataclass(frozen=True, slots=True)
class V024PredictionWriteResult:
    """Metadata for three exclusively written prediction artifacts."""

    artifacts: tuple[ArtifactMetadata, ...]


def fit_verified_generation_bundle_v024(
    bundle: V024FreshGenerationBundle,
) -> V024VerifiedFitResult:
    """Run the six-worker formal optimizer only on verified fresh generation."""

    from lifetwin.experiments.calendar_long_horizon_v019_fit import (  # noqa: PLC0415
        V024FitError,
        _fit_structure_library_formal_from_verified_frames_v024,
    )
    from lifetwin.experiments.calendar_long_horizon_v019_io import (  # noqa: PLC0415
        V024IOError,
        _extract_fresh_generation_frames_for_formal_fit_v024,
    )

    try:
        prefix, coordinates, contract = (
            _extract_fresh_generation_frames_for_formal_fit_v024(bundle)
        )
        result = _fit_structure_library_formal_from_verified_frames_v024(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            contract=contract,
        )
    except (V024FitError, V024IOError) as exc:
        raise V024PredictionError(
            "The sealed fresh-generation fit capability was rejected"
        ) from exc
    return V024VerifiedFitResult(
        _seal=_FIT_SEAL,
        bundle=bundle,
        result=result,
    )


def write_verified_fit_result_v024(
    *,
    bundle: V024FreshGenerationBundle,
    fit_result: V024VerifiedFitResult,
) -> None:
    """Persist and freshly read back a fit produced from this generation."""

    from lifetwin.experiments.calendar_long_horizon_v015_io import (  # noqa: PLC0415
        V015ArtifactError,
        canonical_csv_bytes,
    )
    from lifetwin.experiments.calendar_long_horizon_v019_io import (  # noqa: PLC0415
        V024IOError,
        _write_verified_fit_outputs_v024,
    )

    if (
        type(fit_result) is not V024VerifiedFitResult
        or fit_result._seal is not _FIT_SEAL
        or fit_result._bundle is not bundle
    ):
        raise V024PredictionError(
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
        raise V024PredictionError("The sealed formal fit result changed") from exc
    if observed != fit_result._frame_hashes:
        raise V024PredictionError("The sealed formal fit result changed")
    try:
        _write_verified_fit_outputs_v024(
            bundle,
            frames={name: frame.copy(deep=True) for name, frame in fit_result._frames},
        )
    except V024IOError as exc:
        raise V024PredictionError(
            "The verified formal fit could not be persisted"
        ) from exc


def commit_validated_fit_result_v024(
    *,
    bundle: V024FreshGenerationBundle,
    whole_bundle: object,
    created_utc: str,
) -> str:
    """Create the sole fit commitment only from a valid whole-bundle capability."""

    from lifetwin.experiments.calendar_long_horizon_v019_io import (  # noqa: PLC0415
        V024IOError,
        _create_validated_fit_commitment_v024,
    )
    from lifetwin.experiments.calendar_long_horizon_v019_partition import (  # noqa: PLC0415
        WholeBundleValidated,
    )

    if type(whole_bundle) is not WholeBundleValidated:
        raise V024PredictionError("An exact whole-bundle capability is required")
    if (
        whole_bundle._contract_hash
        != bundle._contract_view.artifacts.config_byte_sha256
    ):
        raise V024PredictionError("The whole-bundle capability contract changed")
    try:
        return _create_validated_fit_commitment_v024(
            bundle,
            validated_source_hashes=whole_bundle.source_hashes,
            created_utc=created_utc,
        )
    except V024IOError as exc:
        raise V024PredictionError(
            "The validated formal fit could not be committed"
        ) from exc


def run_formal_prediction_v024(
    *,
    label_free_bundle: V024CommittedLabelFreeBundle,
    model_state_envelope: V024CommittedModelStateEnvelope,
) -> V024PredictionWriteResult:
    """Compute and exclusively write outputs without any truth/path capability."""

    from lifetwin.experiments.calendar_long_horizon_v019_io import (  # noqa: PLC0415
        V024IOError,
        _extract_prediction_inputs_v024,
        _write_prediction_outputs_v024,
    )
    from lifetwin.experiments.calendar_long_horizon_v019_pipeline import (  # noqa: PLC0415
        V024PipelineError,
        recompute_label_free_pipeline_v024,
    )

    try:
        frames, contract = _extract_prediction_inputs_v024(
            label_free_bundle,
            model_state=model_state_envelope,
        )
        output = recompute_label_free_pipeline_v024(
            prefix_pack=frames["prefix_pack.csv"],
            forecast_coordinates=frames["forecast_coordinates.csv"],
            operating_pack=frames["operating_pack.csv"],
            member_fit_diagnostics=frames["member_fit_diagnostics.csv"],
            member_forecast_bundle=frames["member_forecast_bundle.csv"],
            model_state_envelope=model_state_envelope,
            contract=contract,
        )
        artifacts = _write_prediction_outputs_v024(
            label_free_bundle,
            frames={
                "prediction_bundle.csv": output.prediction_bundle,
                "risk_bundle.csv": output.primary_risk_bundle,
                "decision_bundle.csv": output.decision_bundle,
            },
        )
    except (V024IOError, V024PipelineError) as exc:
        raise V024PredictionError(
            "Formal V2.4 prediction capability was rejected"
        ) from exc
    return V024PredictionWriteResult(artifacts=artifacts)


def run_isolated_prediction_process_v024(
    *,
    label_free_root: str | Path,
    attempt_id: str,
    repo_root: str | Path,
) -> V024PredictionWriteResult:
    """Attest one isolated process, load its attempt, and predict."""

    from lifetwin.experiments.calendar_long_horizon_v019_prediction_capsule import (  # noqa: PLC0415
        V024PredictionCapsuleError,
        load_prediction_bundle,
        run_prediction_bundle,
        write_prediction_outputs,
    )
    from lifetwin.experiments.calendar_long_horizon_v019_prediction_environment import (  # noqa: PLC0415
        V024PredictionEnvironmentError,
        verify_prediction_environment,
    )

    try:
        environment = verify_prediction_environment(repo_root)
        bundle = load_prediction_bundle(
            label_free_root=label_free_root,
            attempt_id=attempt_id,
            expected_protocol_id=environment.protocol_id,
            expected_config_sha256=environment.config_byte_sha256,
            expected_git_commit=environment.git_commit,
        )
        output = run_prediction_bundle(bundle, formal=True)
        artifacts = write_prediction_outputs(
            bundle,
            result=output,
        )
    except (
        V024PredictionCapsuleError,
        V024PredictionEnvironmentError,
    ) as exc:
        raise V024PredictionError(
            "The isolated prediction environment or attempt was rejected"
        ) from exc
    return V024PredictionWriteResult(artifacts=artifacts)


__all__ = [
    "V024PredictionError",
    "V024PredictionWriteResult",
    "V024VerifiedFitResult",
    "commit_validated_fit_result_v024",
    "fit_verified_generation_bundle_v024",
    "run_isolated_prediction_process_v024",
    "run_formal_prediction_v024",
    "write_verified_fit_result_v024",
]
