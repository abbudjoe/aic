"""Training signal contracts derived from canonical AIC episode traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import (
    ArtifactRef,
    LeakageClass,
    SCHEMA_VERSION,
    SchemaValidationError,
)


_TRAINING_SIGNAL_KEYS = frozenset(
    {
        "kind",
        "target",
        "weight",
        "source",
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


@dataclass(frozen=True)
class TrainingSignal:
    """One trainable or diagnostic signal extracted from a canonical trace."""

    kind: TrainingSignalKind
    target: str
    weight: float
    source: str
    leakage_class: LeakageClass
    evidence: Mapping[str, Any] = field(default_factory=dict)
    trial_id: str | None = None
    offline_only: bool = True
    runtime_allowed: bool = False
    consumable_by_policy_runtime: bool = False

    def __post_init__(self) -> None:
        errors: list[str] = []
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
            "kind": self.kind.value,
            "target": self.target,
            "weight": self.weight,
            "source": self.source,
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
            kind=cast(Any, value.get("kind")),
            target=cast(Any, value.get("target")),
            weight=cast(Any, value.get("weight")),
            source=cast(Any, value.get("source")),
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
    signals: tuple[TrainingSignal, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
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
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "source_trace": self.source_trace.to_dict(),
            "signals": [signal.to_dict() for signal in self.signals],
            "notes": list(self.notes),
        }

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
