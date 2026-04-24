"""Reducer slice for backend-neutral append-only ledger entries."""

from __future__ import annotations

from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.hdf5_dataset import Hdf5DatasetReport
from aic_signal_harness.ledger import (
    LedgerDatasetSummary,
    LedgerEntry,
    LedgerMetric,
    LedgerMetricGoal,
)
from aic_signal_harness.manifest import RunManifest
from aic_signal_harness.reducers.hdf5_dataset import Hdf5DatasetReduction
from aic_signal_harness.schemas import ArtifactRef, utc_now_iso


def build_ledger_entry(
    manifest: RunManifest,
    *,
    manifest_artifact: ArtifactRef | Mapping[str, Any],
    dataset_reduction: Hdf5DatasetReduction | None = None,
    promotion: Mapping[str, Any] | None = None,
    recorded_at_utc: str | None = None,
    metric_name: str = "evaluation.score.total",
    metric_goal: LedgerMetricGoal | str = LedgerMetricGoal.max,
) -> LedgerEntry:
    """Build one typed ledger entry from a neutral run manifest."""

    if not isinstance(manifest, RunManifest):
        raise HarnessIOError("build_ledger_entry requires a RunManifest")

    typed_manifest_artifact = (
        manifest_artifact
        if isinstance(manifest_artifact, ArtifactRef)
        else ArtifactRef.from_dict(manifest_artifact)
    )
    if not isinstance(typed_manifest_artifact, ArtifactRef):
        raise HarnessIOError("manifest_artifact must be an ArtifactRef")

    dataset_summary = None if dataset_reduction is None else _dataset_summary_from_reduction(
        manifest,
        dataset_reduction,
    )
    metric_value = None if manifest.score is None else manifest.score.total
    return LedgerEntry(
        recorded_at_utc=utc_now_iso() if recorded_at_utc is None else recorded_at_utc,
        run_id=manifest.run_id,
        status=manifest.status,
        backend_kind=manifest.backend.backend_kind,
        manifest=typed_manifest_artifact,
        metric=LedgerMetric(
            name=metric_name,
            goal=metric_goal,
            value=metric_value,
        ),
        dataset=dataset_summary,
        experiment_id=manifest.experiment_id,
        notes=manifest.notes,
        promotion={} if promotion is None else promotion,
    )


def _dataset_summary_from_reduction(
    manifest: RunManifest,
    reduction: Hdf5DatasetReduction,
) -> LedgerDatasetSummary:
    if not isinstance(reduction, Hdf5DatasetReduction):
        raise HarnessIOError("dataset_reduction must be an Hdf5DatasetReduction")
    dataset_artifacts = tuple(
        artifact for artifact in manifest.artifacts if artifact.kind == "hdf5_dataset"
    )
    if len(dataset_artifacts) > 1:
        raise HarnessIOError("run manifest already has duplicate hdf5_dataset artifact state")

    artifact = reduction.artifact
    if dataset_artifacts:
        artifact = _merge_hdf5_artifact(dataset_artifacts[0], reduction.artifact)

    report = reduction.report
    if not isinstance(report, Hdf5DatasetReport):
        raise HarnessIOError("dataset_reduction.report must be an Hdf5DatasetReport")
    if report.episode_count is None or report.step_count is None:
        raise HarnessIOError(
            "dataset_reduction.report must set episode_count and step_count for ledger summary"
        )
    return LedgerDatasetSummary(
        artifact=artifact,
        episode_count=report.episode_count,
        step_count=report.step_count,
    )


def _merge_hdf5_artifact(existing: ArtifactRef, incoming: ArtifactRef) -> ArtifactRef:
    if existing.kind != incoming.kind:
        raise HarnessIOError("run manifest already has conflicting hdf5_dataset artifact state: kind")
    return ArtifactRef(
        kind=existing.kind,
        path=_merge_optional_path(existing.path, incoming.path),
        uri=_merge_optional_field("uri", existing.uri, incoming.uri),
        sha256=_merge_optional_field("sha256", existing.sha256, incoming.sha256),
        provenance=_merge_json_mapping(
            "provenance",
            existing.provenance,
            incoming.provenance,
        ),
    )


def _merge_optional_path(existing: str | None, incoming: str | None) -> str | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if existing == incoming:
        return existing
    raise HarnessIOError("run manifest already has conflicting hdf5_dataset artifact state: path")


def _merge_optional_field(
    field_name: str,
    existing: str | None,
    incoming: str | None,
) -> str | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if existing == incoming:
        return existing
    raise HarnessIOError(
        f"run manifest already has conflicting hdf5_dataset artifact state: {field_name}"
    )


def _merge_json_mapping(
    field_name: str,
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    merged = {key: _thaw_json_value(value) for key, value in existing.items()}
    for key, incoming_value in incoming.items():
        thawed_incoming = _thaw_json_value(incoming_value)
        if key not in merged:
            merged[key] = thawed_incoming
            continue
        merged[key] = _merge_json_value(
            field_name,
            key,
            merged[key],
            thawed_incoming,
        )
    return merged


def _merge_json_value(
    field_name: str,
    key_path: str,
    existing: Any,
    incoming: Any,
) -> Any:
    if isinstance(existing, Mapping) and isinstance(incoming, Mapping):
        return _merge_json_mapping(
            f"{field_name}.{key_path}",
            existing,
            incoming,
        )
    if existing == incoming:
        return existing
    raise HarnessIOError(
        "run manifest already has conflicting hdf5_dataset artifact state: "
        f"{field_name}.{key_path}"
    )


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json_value(item) for item in value]
    return value
