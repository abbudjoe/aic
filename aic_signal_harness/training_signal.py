"""Training signal contracts derived from canonical AIC episode traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, sha256_file
from aic_signal_harness.schemas import (
    ArtifactRef,
    LeakageClass,
    SchemaValidationError,
)

TRAINING_SIGNAL_SCHEMA_VERSION = 2


_TRAINING_SIGNAL_KEYS = frozenset(
    {
        "signal_id",
        "kind",
        "target",
        "weight",
        "source",
        "source_event_indices",
        "extraction_method",
        "leakage_class",
        "evidence",
        "trial_id",
        "offline_only",
        "runtime_allowed",
        "consumable_by_policy_runtime",
    }
)
_TRAINING_SIGNAL_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "generated_at_utc",
        "source_trace",
        "source_reward_report",
        "source_failure_report",
        "signals",
        "notes",
    }
)


class _StrEnum(str, Enum):
    @classmethod
    def parse(cls: type["_EnumT"], value: Any) -> "_EnumT":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise HarnessIOError(f"{cls.__name__} must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise HarnessIOError(
                f"unknown {cls.__name__} {value!r}; expected one of: {allowed}"
            ) from exc


_EnumT = TypeVar("_EnumT", bound=_StrEnum)


class TrainingSignalKind(_StrEnum):
    """Deterministic signal categories that can feed later training jobs."""

    behavior_clone_action = "behavior_clone_action"
    safety_guard_avoidance = "safety_guard_avoidance"
    official_score_term = "official_score_term"
    reward_term = "reward_term"
    failure_label = "failure_label"
    dataset_episode_summary = "dataset_episode_summary"


@dataclass(frozen=True)
class TrainingSignal:
    """One trainable or diagnostic signal extracted from a canonical trace."""

    signal_id: str
    kind: TrainingSignalKind
    target: str
    weight: float
    source: str
    source_event_indices: tuple[int, ...]
    extraction_method: str
    leakage_class: LeakageClass
    evidence: Mapping[str, Any] = field(default_factory=dict)
    trial_id: str | None = None
    offline_only: bool = True
    runtime_allowed: bool = False
    consumable_by_policy_runtime: bool = False

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "signal_id",
                _require_nonempty_text(self.signal_id, "training signal.signal_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "kind", TrainingSignalKind.parse(self.kind))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "target",
                _require_nonempty_text(self.target, "training signal.target"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "weight",
                _require_finite_float(self.weight, "training signal.weight"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "source", _require_nonempty_text(self.source, "training signal.source"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "source_event_indices",
                _source_event_indices(
                    self.source_event_indices,
                    "training signal.source_event_indices",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "extraction_method",
                _require_nonempty_text(
                    self.extraction_method,
                    "training signal.extraction_method",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "leakage_class", LeakageClass.parse(self.leakage_class))
        except SchemaValidationError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trial_id",
                _optional_nonempty_text(self.trial_id, "training signal.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "evidence",
                _copy_json_mapping(self.evidence, "training signal.evidence"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.offline_only is not True:
            errors.append("training signal.offline_only must be true")
        if self.runtime_allowed is not False:
            errors.append("training signal.runtime_allowed must be false")
        if self.consumable_by_policy_runtime is not False:
            errors.append("training signal.consumable_by_policy_runtime must be false")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "signal_id": self.signal_id,
            "kind": self.kind.value,
            "target": self.target,
            "weight": self.weight,
            "source": self.source,
            "source_event_indices": list(self.source_event_indices),
            "extraction_method": self.extraction_method,
            "leakage_class": self.leakage_class.value,
            "evidence": _thaw_json(self.evidence),
            "offline_only": self.offline_only,
            "runtime_allowed": self.runtime_allowed,
            "consumable_by_policy_runtime": self.consumable_by_policy_runtime,
        }
        if self.trial_id is not None:
            value["trial_id"] = self.trial_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingSignal":
        if not isinstance(value, Mapping):
            raise HarnessIOError("training signal must be a mapping")
        _reject_unknown_keys(value, _TRAINING_SIGNAL_KEYS, "training signal")
        missing_runtime_fields = sorted(
            field_name
            for field_name in (
                "consumable_by_policy_runtime",
                "offline_only",
                "runtime_allowed",
            )
            if field_name not in value
        )
        if missing_runtime_fields:
            raise HarnessIOError(
                "training signal missing required runtime boundary fields: "
                + ", ".join(missing_runtime_fields)
            )
        return cls(
            signal_id=cast(Any, value.get("signal_id")),
            kind=cast(Any, value.get("kind")),
            target=cast(Any, value.get("target")),
            weight=cast(Any, value.get("weight")),
            source=cast(Any, value.get("source")),
            source_event_indices=cast(Any, value.get("source_event_indices")),
            extraction_method=cast(Any, value.get("extraction_method")),
            leakage_class=cast(Any, value.get("leakage_class")),
            evidence=value.get("evidence", {}),
            trial_id=value.get("trial_id"),
            offline_only=cast(bool, value.get("offline_only")),
            runtime_allowed=cast(bool, value.get("runtime_allowed")),
            consumable_by_policy_runtime=cast(
                bool,
                value.get("consumable_by_policy_runtime"),
            ),
        )


@dataclass(frozen=True)
class TrainingSignalReport:
    """Deterministic signal extraction report for one canonical episode trace."""

    run_id: str
    generated_at_utc: str
    source_trace: ArtifactRef
    source_reward_report: ArtifactRef | None = None
    source_failure_report: ArtifactRef | None = None
    signals: tuple[TrainingSignal, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = TRAINING_SIGNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != TRAINING_SIGNAL_SCHEMA_VERSION:
            errors.append(f"schema_version must be {TRAINING_SIGNAL_SCHEMA_VERSION}")
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "training signal report.run_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "generated_at_utc",
                _require_nonempty_text(self.generated_at_utc, "training signal report.generated_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not isinstance(self.source_trace, ArtifactRef):
            try:
                object.__setattr__(self, "source_trace", ArtifactRef.from_dict(self.source_trace))
            except SchemaValidationError as exc:
                errors.append(str(exc))
        if not errors:
            errors.extend(_source_trace_errors(self.source_trace, self.run_id))
        for field_name, expected_kind in (
            ("source_reward_report", "reward_report"),
            ("source_failure_report", "failure_report"),
        ):
            artifact = getattr(self, field_name)
            if artifact is not None and not isinstance(artifact, ArtifactRef):
                try:
                    object.__setattr__(self, field_name, ArtifactRef.from_dict(artifact))
                except SchemaValidationError as exc:
                    errors.append(str(exc))
            artifact = getattr(self, field_name)
            if isinstance(artifact, ArtifactRef):
                errors.extend(
                    _source_report_errors(
                        artifact,
                        self.run_id,
                        expected_kind=expected_kind,
                        field_name=f"training signal report.{field_name}",
                    )
                )
        try:
            object.__setattr__(
                self,
                "signals",
                tuple(
                    signal if isinstance(signal, TrainingSignal) else TrainingSignal.from_dict(signal)
                    for signal in _as_sequence(self.signals, "training signal report.signals")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                tuple(
                    _require_nonempty_text(note, "training signal report.note")
                    for note in _as_sequence(self.notes, "training signal report.notes")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            signal_ids = tuple(signal.signal_id for signal in self.signals)
            duplicate_signal_ids = _duplicate_text(signal_ids)
            if duplicate_signal_ids:
                errors.append(
                    "training signal report.signals must not contain duplicate signal_id: "
                    + ", ".join(duplicate_signal_ids)
                )
            if (
                any(signal.kind is TrainingSignalKind.reward_term for signal in self.signals)
                and self.source_reward_report is None
            ):
                errors.append(
                    "training signal report.source_reward_report is required when reward_term signals are present"
                )
            if (
                any(signal.kind is TrainingSignalKind.failure_label for signal in self.signals)
                and self.source_failure_report is None
            ):
                errors.append(
                    "training signal report.source_failure_report is required when failure_label signals are present"
                )
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "source_trace": self.source_trace.to_dict(),
            "signals": [signal.to_dict() for signal in self.signals],
            "notes": list(self.notes),
        }
        if self.source_reward_report is not None:
            value["source_reward_report"] = self.source_reward_report.to_dict()
        if self.source_failure_report is not None:
            value["source_failure_report"] = self.source_failure_report.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingSignalReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("training signal report must be a mapping")
        _reject_unknown_keys(value, _TRAINING_SIGNAL_REPORT_KEYS, "training signal report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_trace=cast(Any, value.get("source_trace")),
            source_reward_report=cast(Any, value.get("source_reward_report")),
            source_failure_report=cast(Any, value.get("source_failure_report")),
            signals=cast(Any, value.get("signals", ())),
            notes=cast(Any, value.get("notes", ())),
        )


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_finite_float(value: Any, field_name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _source_event_indices(value: Any, field_name: str) -> tuple[int, ...]:
    items = _as_sequence(value, field_name)
    if not items:
        raise HarnessIOError(f"{field_name} must not be empty")
    indices: list[int] = []
    for item in items:
        if type(item) is not int or item < 0:
            raise HarnessIOError(f"{field_name} must contain nonnegative integers")
        indices.append(item)
    if len(set(indices)) != len(indices):
        raise HarnessIOError(f"{field_name} must not contain duplicates")
    if tuple(indices) != tuple(sorted(indices)):
        raise HarnessIOError(f"{field_name} must be sorted")
    return tuple(indices)


def _copy_json_mapping(value: Any, field_name: str) -> MappingProxyType[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        if key in copied:
            raise HarnessIOError(f"{field_name} has duplicate key after normalization: {key}")
        copied[key] = _copy_json_value(item, f"{field_name}.{key}")
    return MappingProxyType(copied)


def _copy_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return _copy_json_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_copy_json_value(item, f"{field_name}[]") for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise HarnessIOError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _source_trace_errors(source_trace: ArtifactRef, run_id: str) -> list[str]:
    errors: list[str] = []
    if source_trace.kind != "episode_trace":
        errors.append("training signal report.source_trace.kind must be 'episode_trace'")
    if source_trace.sha256 is None:
        errors.append("training signal report.source_trace.sha256 must be set")
    if source_trace.provenance.get("run_id") != run_id:
        errors.append("training signal report.source_trace provenance run_id must match report.run_id")
    if "producer" not in source_trace.provenance:
        errors.append("training signal report.source_trace must set provenance.producer")
    if "derivation" not in source_trace.provenance:
        errors.append("training signal report.source_trace must set provenance.derivation")
    return errors


def _source_report_errors(
    source_report: ArtifactRef,
    run_id: str,
    *,
    expected_kind: str,
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if source_report.kind != expected_kind:
        errors.append(f"{field_name}.kind must be {expected_kind!r}")
    if source_report.sha256 is None:
        errors.append(f"{field_name}.sha256 must be set")
    if source_report.provenance.get("run_id") != run_id:
        errors.append(f"{field_name} provenance run_id must match report.run_id")
    if "producer" not in source_report.provenance:
        errors.append(f"{field_name} must set provenance.producer")
    if "derivation" not in source_report.provenance:
        errors.append(f"{field_name} must set provenance.derivation")
    try:
        report_path = local_artifact_path(
            path=source_report.path,
            uri=source_report.uri,
            field_name=field_name,
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        report_path = None
    if report_path is None:
        errors.append(f"{field_name} must be a local byte-verifiable report artifact")
    elif not report_path.exists():
        errors.append(f"{field_name}.path must exist")
    elif not report_path.is_file():
        errors.append(f"{field_name}.path must point to a JSON report file")
    elif source_report.sha256 is not None and sha256_file(report_path) != source_report.sha256:
        errors.append(f"{field_name}.sha256 must match path")
    return errors


def _duplicate_text(values: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(value for value, count in counts.items() if count > 1)
