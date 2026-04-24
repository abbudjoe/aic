"""Run manifest primitives for the backend-neutral AIC signal harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping, TypeVar

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import (
    SCHEMA_VERSION,
    ArtifactRef,
    PolicyBackendSpec,
    SchemaValidationError,
    utc_now_iso,
)
from aic_signal_harness.scoring import ScoreReport


_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "backend",
        "artifacts",
        "score",
        "experiment_id",
        "hypothesis",
        "notes",
    }
)
_URI_SOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class _StrEnum(str, Enum):
    @classmethod
    def parse(cls: type["_EnumT"], value: str | "_EnumT") -> "_EnumT":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise SchemaValidationError(f"{cls.__name__} must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise SchemaValidationError(
                f"unknown {cls.__name__} {value!r}; expected one of: {allowed}"
            ) from exc


_EnumT = TypeVar("_EnumT", bound=_StrEnum)


class RunStatus(_StrEnum):
    planned = "planned"
    running = "running"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"
    promoted = "promoted"


@dataclass(frozen=True)
class RunManifest:
    """Narrow manifest for one observed or planned harness run."""

    run_id: str
    status: RunStatus
    backend: PolicyBackendSpec
    created_at_utc: str = field(default_factory=utc_now_iso)
    updated_at_utc: str = field(default_factory=utc_now_iso)
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    score: ScoreReport | None = None
    experiment_id: str | None = None
    hypothesis: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "run_id"))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(self, "status", RunStatus.parse(self.status))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if not isinstance(self.backend, PolicyBackendSpec):
            try:
                object.__setattr__(self, "backend", PolicyBackendSpec.from_dict(self.backend))
            except SchemaValidationError as exc:
                errors.extend(exc.errors)

        for timestamp_field in ("created_at_utc", "updated_at_utc"):
            if not isinstance(getattr(self, timestamp_field), str) or not getattr(
                self, timestamp_field
            ).strip():
                errors.append(f"{timestamp_field} must be a nonempty string")

        try:
            artifacts = _as_sequence(self.artifacts, "artifacts")
            object.__setattr__(
                self,
                "artifacts",
                tuple(
                    artifact
                    if isinstance(artifact, ArtifactRef)
                    else ArtifactRef.from_dict(artifact)
                    for artifact in artifacts
                ),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)

        if self.score is not None and not isinstance(self.score, ScoreReport):
            try:
                object.__setattr__(self, "score", ScoreReport.from_dict(self.score))
            except HarnessIOError as exc:
                errors.append(str(exc))

        for optional_field in ("experiment_id", "hypothesis"):
            try:
                object.__setattr__(
                    self,
                    optional_field,
                    _optional_nonempty_text(getattr(self, optional_field), optional_field),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)

        try:
            notes = _as_sequence(self.notes, "notes")
            object.__setattr__(
                self,
                "notes",
                tuple(_require_nonempty_text(note, "note") for note in notes),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)

        scoring_artifacts: tuple[ArtifactRef, ...] = ()
        if not errors:
            scoring_artifacts = tuple(
                artifact for artifact in self.artifacts if artifact.kind == "scoring_yaml"
            )
            if len(scoring_artifacts) > 1:
                errors.append("run manifest must not include duplicate scoring_yaml artifacts")
            for artifact in scoring_artifacts:
                if _is_relative_local_source(artifact.path):
                    errors.append(
                        "scoring_yaml artifact.path must be an absolute local path when set"
                    )

        if isinstance(self.score, ScoreReport) and _is_relative_local_source(self.score.source):
            errors.append("score.source must be an absolute local path or uri")

        if not errors and self.score is not None:
            if not _has_matching_scoring_artifact(scoring_artifacts, self.score.source):
                errors.append(
                    "manifests with score must include a scoring_yaml artifact "
                    "whose path or uri matches score.source"
                )
            elif scoring_artifacts[0].sha256 is None:
                errors.append(
                    "manifests with score must bind score.source to a "
                    "scoring_yaml artifact with sha256"
                )

        if not errors and self.status in (RunStatus.completed, RunStatus.promoted):
            if not self.artifacts:
                errors.append(f"{self.status.value} manifest must include at least one artifact")
            if self.score is None:
                errors.append(f"{self.status.value} manifest must include a score report")

        if errors:
            raise SchemaValidationError(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "backend": self.backend.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "score": self.score.to_dict() if self.score is not None else None,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("run manifest must be a mapping")
        _reject_unknown_keys(value, _RUN_MANIFEST_KEYS, "run manifest")
        return cls(
            schema_version=value.get("schema_version"),
            run_id=value.get("run_id"),
            status=value.get("status"),
            created_at_utc=value.get("created_at_utc", ""),
            updated_at_utc=value.get("updated_at_utc", ""),
            backend=PolicyBackendSpec.from_dict(value.get("backend")),
            artifacts=value.get("artifacts", ()),
            score=value.get("score"),
            experiment_id=value.get("experiment_id"),
            hypothesis=value.get("hypothesis"),
            notes=value.get("notes", ()),
        )


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise SchemaValidationError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a nonempty string")
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise SchemaValidationError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise SchemaValidationError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _has_matching_scoring_artifact(artifacts: tuple[ArtifactRef, ...], score_source: str) -> bool:
    return any(
        artifact.kind == "scoring_yaml"
        and (
            _source_identities_match(artifact.path, score_source)
            or _source_identities_match(artifact.uri, score_source)
        )
        for artifact in artifacts
    )


def _source_identities_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return _normalize_source_identity(left) == _normalize_source_identity(right)


def _is_relative_local_source(source: str | None) -> bool:
    if source is None:
        return False
    return not _is_uri_source(source) and not _is_absolute_local_source(source)


def _normalize_source_identity(source: str) -> str:
    if _is_uri_source(source):
        return source
    if _is_windows_absolute_local_source(source) and not Path(source).is_absolute():
        return PureWindowsPath(source).as_posix()
    return str(Path(source).resolve(strict=False))


def _is_uri_source(source: str) -> bool:
    return bool(_URI_SOURCE_RE.match(source))


def _is_absolute_local_source(source: str) -> bool:
    return Path(source).is_absolute() or _is_windows_absolute_local_source(source)


def _is_windows_absolute_local_source(source: str) -> bool:
    return PureWindowsPath(source).is_absolute()
