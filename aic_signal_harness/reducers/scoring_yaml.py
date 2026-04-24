"""Reducer slice for official ``scoring.yaml`` artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, sha256_file
from aic_signal_harness.manifest import RunManifest
from aic_signal_harness.schemas import ArtifactRef, SchemaValidationError
from aic_signal_harness.scoring import (
    ScoreReport,
    parse_scoring_yaml,
    parse_scoring_yaml_bytes,
    read_scoring_yaml_snapshot,
)


@dataclass(frozen=True)
class ScoringYamlReduction:
    """Typed reduction binding official score output to its canonical artifact."""

    artifact: ArtifactRef
    score: ScoreReport

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.artifact, ArtifactRef):
            errors.append("scoring reduction artifact must be an ArtifactRef")
        if not isinstance(self.score, ScoreReport):
            errors.append("scoring reduction score must be a ScoreReport")
        if errors:
            raise HarnessIOError("; ".join(errors))

        if self.artifact.kind != "scoring_yaml":
            errors.append("scoring reduction artifact.kind must be 'scoring_yaml'")
        if self.artifact.sha256 is None:
            errors.append("scoring reduction artifact.sha256 must be set")
        if self.artifact.path is None:
            errors.append("scoring reduction artifact.path must point to a local scoring.yaml snapshot")
        expected_report_sha256 = _report_sha256(self.score.to_dict())
        report_sha256 = self.artifact.provenance.get("report_sha256")
        if report_sha256 != expected_report_sha256:
            errors.append("scoring reduction artifact.provenance.report_sha256 must match score")
        source_report = self.artifact.provenance.get("source_report")
        if source_report is None:
            errors.append("scoring reduction artifact.provenance.source_report must be set")
        elif not isinstance(source_report, Mapping):
            errors.append("scoring reduction artifact.provenance.source_report must be a mapping")
        else:
            try:
                provenance_report = ScoreReport.from_dict(source_report)
                if provenance_report.to_dict() != self.score.to_dict():
                    errors.append(
                        "scoring reduction artifact.provenance.source_report must match score"
                    )
            except HarnessIOError as exc:
                errors.append(
                    "scoring reduction artifact.provenance.source_report must parse as a ScoreReport: "
                    + str(exc)
                )
        if not _artifact_matches_score_source(self.artifact, self.score.source):
            errors.append(
                "scoring reduction artifact path or uri must match score.source"
            )
        local_path: Path | None = None
        try:
            local_path = local_artifact_path(
                path=self.artifact.path,
                uri=self.artifact.uri,
                field_name="scoring reduction artifact",
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if local_path is not None and self.artifact.sha256 is not None:
            try:
                if sha256_file(local_path) != self.artifact.sha256:
                    errors.append("scoring reduction artifact.sha256 must match source file")
            except HarnessIOError as exc:
                errors.append(str(exc))
            else:
                try:
                    parsed_score = parse_scoring_yaml(local_path)
                    source_aliases = tuple(
                        source
                        for source in (
                            parsed_score.source,
                            self.score.source,
                            self.artifact.path,
                            self.artifact.uri,
                            str(local_path.resolve(strict=False)),
                        )
                        if source is not None
                    )
                    if not parsed_score.equivalent_to(
                        self.score,
                        source_aliases=source_aliases,
                    ):
                        errors.append(
                            "scoring reduction score must match scoring_yaml source artifact bytes"
                        )
                except HarnessIOError as exc:
                    errors.append(f"scoring reduction artifact scoring.yaml must parse: {exc}")

        if errors:
            raise HarnessIOError("; ".join(errors))


def reduce_scoring_yaml(
    path: str | Path,
    *,
    parsed_at_utc: str | None = None,
    uri: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ScoringYamlReduction:
    """Reduce official ``scoring.yaml`` into a typed score and artifact pair."""

    scoring_bytes, artifact_path = read_scoring_yaml_snapshot(path)
    local_artifact_path(
        path=artifact_path,
        uri=uri,
        field_name="scoring reduction artifact",
    )
    score = parse_scoring_yaml_bytes(
        scoring_bytes,
        source=artifact_path,
        parsed_at_utc=parsed_at_utc,
    )
    artifact_provenance = _scoring_report_provenance(provenance, score)
    try:
        artifact = ArtifactRef(
            kind="scoring_yaml",
            path=artifact_path,
            uri=uri,
            sha256=hashlib.sha256(scoring_bytes).hexdigest(),
            provenance=artifact_provenance,
        )
    except SchemaValidationError as exc:
        raise HarnessIOError(str(exc)) from exc
    return ScoringYamlReduction(
        artifact=artifact,
        score=score,
    )


def _report_sha256(mapping: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scoring_report_provenance(
    provenance: Mapping[str, Any] | None,
    score: ScoreReport,
) -> dict[str, Any]:
    merged = {} if provenance is None else dict(provenance)
    score_payload = score.to_dict()
    expected_report_sha256 = _report_sha256(score_payload)
    existing_report_sha256 = merged.get("report_sha256")
    if existing_report_sha256 is not None and existing_report_sha256 != expected_report_sha256:
        raise HarnessIOError("scoring reduction provenance.report_sha256 must match reducer score")
    existing_source_report = merged.get("source_report")
    if existing_source_report is not None:
        if not isinstance(existing_source_report, Mapping):
            raise HarnessIOError("scoring reduction provenance.source_report must be a mapping")
        try:
            provenance_report = ScoreReport.from_dict(existing_source_report)
        except HarnessIOError as exc:
            raise HarnessIOError(
                "scoring reduction provenance.source_report must parse as a ScoreReport: "
                + str(exc)
            ) from exc
        if provenance_report.to_dict() != score_payload:
            raise HarnessIOError("scoring reduction provenance.source_report must match reducer score")
    merged["report_sha256"] = expected_report_sha256
    merged["source_report"] = score_payload
    return merged


def attach_scoring_yaml_reduction(
    manifest: RunManifest,
    reduction: ScoringYamlReduction,
    *,
    updated_at_utc: str | None = None,
) -> RunManifest:
    """Attach a scoring reduction to a manifest without silent score replacement."""

    if not isinstance(manifest, RunManifest):
        raise HarnessIOError("attach_scoring_yaml_reduction requires a RunManifest")
    if not isinstance(reduction, ScoringYamlReduction):
        raise HarnessIOError(
            "attach_scoring_yaml_reduction requires a ScoringYamlReduction"
        )

    scoring_artifact_indexes = tuple(
        index
        for index, artifact in enumerate(manifest.artifacts)
        if artifact.kind == "scoring_yaml"
    )
    scoring_artifacts = tuple(
        manifest.artifacts[index] for index in scoring_artifact_indexes
    )
    if len(scoring_artifacts) > 1:
        raise HarnessIOError("run manifest already has duplicate scoring_yaml artifact state")

    artifacts = manifest.artifacts
    merged_artifact = reduction.artifact
    if scoring_artifacts:
        merged_artifact = _merge_scoring_yaml_artifact(
            scoring_artifacts[0],
            reduction.artifact,
        )
        if merged_artifact != scoring_artifacts[0]:
            artifact_list = list(manifest.artifacts)
            artifact_list[scoring_artifact_indexes[0]] = merged_artifact
            artifacts = tuple(artifact_list)
    else:
        artifacts = manifest.artifacts + (reduction.artifact,)

    source_aliases = _artifact_source_aliases(merged_artifact)
    for artifact in scoring_artifacts:
        source_aliases = source_aliases | _artifact_source_aliases(artifact)
    source_aliases = source_aliases | _artifact_source_aliases(reduction.artifact)

    if manifest.score is not None and not manifest.score.equivalent_to(
        reduction.score,
        source_aliases=source_aliases,
    ):
        raise HarnessIOError("run manifest already has conflicting score report state")

    score = (
        reduction.score
        if manifest.score is None
        else _merge_equivalent_score_reports(manifest.score, reduction.score)
    )
    next_updated_at_utc = (
        manifest.updated_at_utc if updated_at_utc is None else updated_at_utc
    )

    if (
        artifacts == manifest.artifacts
        and score == manifest.score
        and next_updated_at_utc == manifest.updated_at_utc
    ):
        return manifest

    return RunManifest(
        run_id=manifest.run_id,
        status=manifest.status,
        backend=manifest.backend,
        created_at_utc=manifest.created_at_utc,
        updated_at_utc=next_updated_at_utc,
        artifacts=artifacts,
        score=score,
        experiment_id=manifest.experiment_id,
        hypothesis=manifest.hypothesis,
        notes=manifest.notes,
    )


def _artifact_source_aliases(artifact: ArtifactRef) -> frozenset[str]:
    aliases: set[str] = set()
    for source in (artifact.path, artifact.uri):
        if source is None:
            continue
        aliases.add(source)
        normalized = _normalize_source_path(source)
        if normalized is not None:
            aliases.add(normalized)
    return frozenset(aliases)


def _artifact_matches_score_source(artifact: ArtifactRef, source: str) -> bool:
    return source in _artifact_source_aliases(artifact)


def _merge_scoring_yaml_artifact(
    existing: ArtifactRef,
    incoming: ArtifactRef,
) -> ArtifactRef:
    if existing.kind != incoming.kind:
        raise HarnessIOError(
            "run manifest already has conflicting scoring_yaml artifact state: kind"
        )

    try:
        return ArtifactRef(
            kind=existing.kind,
            path=_merge_optional_artifact_field("path", existing.path, incoming.path),
            uri=_merge_optional_artifact_field("uri", existing.uri, incoming.uri),
            sha256=_merge_optional_artifact_field("sha256", existing.sha256, incoming.sha256),
            provenance=_merge_json_mapping(
                "provenance",
                existing.provenance,
                incoming.provenance,
            ),
        )
    except SchemaValidationError as exc:
        raise HarnessIOError(str(exc)) from exc


def _merge_optional_artifact_field(
    field_name: str,
    existing: str | None,
    incoming: str | None,
) -> str | None:
    if field_name == "path":
        return _merge_optional_path(existing, incoming)
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if existing == incoming:
        return existing
    raise HarnessIOError(
        f"run manifest already has conflicting scoring_yaml artifact state: {field_name}"
    )


def _merge_optional_path(existing: str | None, incoming: str | None) -> str | None:
    normalized_existing = _normalize_source_path(existing)
    normalized_incoming = _normalize_source_path(incoming)
    if normalized_existing is None:
        return normalized_incoming
    if normalized_incoming is None:
        return normalized_existing
    if normalized_existing == normalized_incoming:
        return normalized_existing
    raise HarnessIOError("run manifest already has conflicting scoring_yaml artifact state: path")


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
        "run manifest already has conflicting scoring_yaml artifact state: "
        f"{field_name}.{key_path}"
    )


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _normalize_source_path(source: str | None) -> str | None:
    if source is None:
        return None
    if urlparse(source).scheme:
        return source
    return str(Path(source).resolve(strict=False))


def _merge_equivalent_score_reports(existing: ScoreReport, incoming: ScoreReport) -> ScoreReport:
    parsed_at_utc = existing.parsed_at_utc or incoming.parsed_at_utc
    merged = ScoreReport(
        source=incoming.source,
        total=incoming.total,
        trials=incoming.trials,
        parsed_at_utc=parsed_at_utc,
    )
    return existing if merged == existing else merged
