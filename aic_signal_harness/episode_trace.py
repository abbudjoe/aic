"""Canonical episode trace contracts for AIC experiment signal."""

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
from aic_signal_harness.scoring import TrialScore


_TIMELINE_EVENT_KEYS = frozenset(
    {
        "event_index",
        "event_kind",
        "elapsed_sec",
        "source",
        "leakage_class",
        "payload",
        "trial_id",
        "emitted_at_utc",
    }
)
_TRIAL_TRACE_KEYS = frozenset(
    {
        "trial_id",
        "start_elapsed_sec",
        "end_elapsed_sec",
        "event_count",
        "action_event_count",
        "nonzero_action_event_count",
        "safety_guard_event_count",
        "error_event_count",
        "score",
        "events",
    }
)
_EPISODE_TRACE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "generated_at_utc",
        "source_artifacts",
        "trials",
        "run_events",
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


class TimelineEventKind(_StrEnum):
    """Normalized event families inside a canonical episode trace."""

    policy_event = "policy_event"
    action_event = "action_event"
    safety_guard = "safety_guard"
    error = "error"
    official_score = "official_score"
    reward_term = "reward_term"
    failure_label = "failure_label"


@dataclass(frozen=True)
class TimelineEvent:
    """One normalized event in a run or trial timeline."""

    event_index: int
    event_kind: TimelineEventKind
    elapsed_sec: float
    source: str
    leakage_class: LeakageClass
    payload: Mapping[str, Any] = field(default_factory=dict)
    trial_id: str | None = None
    emitted_at_utc: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "event_index",
                _require_nonnegative_int(self.event_index, "timeline event.event_index"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "event_kind",
                TimelineEventKind.parse(self.event_kind),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "elapsed_sec",
                _require_nonnegative_float(self.elapsed_sec, "timeline event.elapsed_sec"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "source", _require_nonempty_text(self.source, "timeline event.source"))
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
                _optional_nonempty_text(self.trial_id, "timeline event.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "emitted_at_utc",
                _optional_nonempty_text(self.emitted_at_utc, "timeline event.emitted_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "payload",
                _copy_json_mapping(self.payload, "timeline event.payload"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event_index": self.event_index,
            "event_kind": self.event_kind.value,
            "elapsed_sec": self.elapsed_sec,
            "source": self.source,
            "leakage_class": self.leakage_class.value,
            "payload": _thaw_json(self.payload),
        }
        if self.trial_id is not None:
            value["trial_id"] = self.trial_id
        if self.emitted_at_utc is not None:
            value["emitted_at_utc"] = self.emitted_at_utc
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimelineEvent":
        if not isinstance(value, Mapping):
            raise HarnessIOError("timeline event must be a mapping")
        _reject_unknown_keys(value, _TIMELINE_EVENT_KEYS, "timeline event")
        return cls(
            event_index=cast(Any, value.get("event_index")),
            event_kind=cast(Any, value.get("event_kind")),
            elapsed_sec=cast(Any, value.get("elapsed_sec")),
            source=cast(Any, value.get("source")),
            leakage_class=cast(Any, value.get("leakage_class")),
            payload=value.get("payload", {}),
            trial_id=value.get("trial_id"),
            emitted_at_utc=value.get("emitted_at_utc"),
        )


@dataclass(frozen=True)
class TrialTrace:
    """Canonical trace for one official or policy-observed trial."""

    trial_id: str
    start_elapsed_sec: float
    end_elapsed_sec: float
    event_count: int
    action_event_count: int = 0
    nonzero_action_event_count: int = 0
    safety_guard_event_count: int = 0
    error_event_count: int = 0
    score: TrialScore | None = None
    events: tuple[TimelineEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "trial_id", _require_nonempty_text(self.trial_id, "trial trace.trial_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("start_elapsed_sec", "end_elapsed_sec"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_float(
                        getattr(self, field_name),
                        f"trial trace.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "event_count",
            "action_event_count",
            "nonzero_action_event_count",
            "safety_guard_event_count",
            "error_event_count",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_int(
                        getattr(self, field_name),
                        f"trial trace.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "events",
                tuple(
                    event if isinstance(event, TimelineEvent) else TimelineEvent.from_dict(event)
                    for event in _as_sequence(self.events, "trial trace.events")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.score is not None and not isinstance(self.score, TrialScore):
            try:
                object.__setattr__(self, "score", TrialScore.from_dict(self.score))
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not errors:
            events = cast(tuple[TimelineEvent, ...], self.events)
            mismatched = tuple(event.trial_id for event in events if event.trial_id != self.trial_id)
            if mismatched:
                errors.append("trial trace events must all match trial_id")
            if self.event_count != len(events):
                errors.append("trial trace.event_count must match events")
            if self.start_elapsed_sec > self.end_elapsed_sec:
                errors.append("trial trace start_elapsed_sec must be <= end_elapsed_sec")
            if events:
                elapsed_values = tuple(event.elapsed_sec for event in events)
                if min(elapsed_values) < self.start_elapsed_sec or max(elapsed_values) > self.end_elapsed_sec:
                    errors.append("trial trace event elapsed_sec values must be within trial bounds")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "start_elapsed_sec": self.start_elapsed_sec,
            "end_elapsed_sec": self.end_elapsed_sec,
            "event_count": self.event_count,
            "action_event_count": self.action_event_count,
            "nonzero_action_event_count": self.nonzero_action_event_count,
            "safety_guard_event_count": self.safety_guard_event_count,
            "error_event_count": self.error_event_count,
            "score": None if self.score is None else self.score.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialTrace":
        if not isinstance(value, Mapping):
            raise HarnessIOError("trial trace must be a mapping")
        _reject_unknown_keys(value, _TRIAL_TRACE_KEYS, "trial trace")
        return cls(
            trial_id=cast(Any, value.get("trial_id")),
            start_elapsed_sec=cast(Any, value.get("start_elapsed_sec")),
            end_elapsed_sec=cast(Any, value.get("end_elapsed_sec")),
            event_count=cast(Any, value.get("event_count")),
            action_event_count=cast(Any, value.get("action_event_count", 0)),
            nonzero_action_event_count=cast(Any, value.get("nonzero_action_event_count", 0)),
            safety_guard_event_count=cast(Any, value.get("safety_guard_event_count", 0)),
            error_event_count=cast(Any, value.get("error_event_count", 0)),
            score=value.get("score"),
            events=cast(Any, value.get("events", ())),
        )


@dataclass(frozen=True)
class EpisodeTrace:
    """One run's canonical trace over policy events, scores, labels, and rewards."""

    run_id: str
    generated_at_utc: str
    source_artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    trials: tuple[TrialTrace, ...] = field(default_factory=tuple)
    run_events: tuple[TimelineEvent, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "episode trace.run_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "generated_at_utc",
                _require_nonempty_text(self.generated_at_utc, "episode trace.generated_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "source_artifacts",
                tuple(
                    artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
                    for artifact in _as_sequence(self.source_artifacts, "episode trace.source_artifacts")
                ),
            )
        except (HarnessIOError, SchemaValidationError) as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trials",
                tuple(
                    trial if isinstance(trial, TrialTrace) else TrialTrace.from_dict(trial)
                    for trial in _as_sequence(self.trials, "episode trace.trials")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "run_events",
                tuple(
                    event if isinstance(event, TimelineEvent) else TimelineEvent.from_dict(event)
                    for event in _as_sequence(self.run_events, "episode trace.run_events")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                tuple(
                    _require_nonempty_text(note, "episode trace.note")
                    for note in _as_sequence(self.notes, "episode trace.notes")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            trial_ids = tuple(trial.trial_id for trial in self.trials)
            if len(set(trial_ids)) != len(trial_ids):
                errors.append("episode trace trials must have unique trial_id values")
            if not self.trials:
                errors.append("episode trace must contain at least one trial")
            if any(event.trial_id is not None for event in self.run_events):
                errors.append("episode trace run_events must not set trial_id")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "source_artifacts": [artifact.to_dict() for artifact in self.source_artifacts],
            "trials": [trial.to_dict() for trial in self.trials],
            "run_events": [event.to_dict() for event in self.run_events],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeTrace":
        if not isinstance(value, Mapping):
            raise HarnessIOError("episode trace must be a mapping")
        _reject_unknown_keys(value, _EPISODE_TRACE_KEYS, "episode trace")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_artifacts=cast(Any, value.get("source_artifacts", ())),
            trials=cast(Any, value.get("trials", ())),
            run_events=cast(Any, value.get("run_events", ())),
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


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise HarnessIOError(f"{field_name} must be a nonnegative integer")
    return value


def _require_nonnegative_float(value: Any, field_name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise HarnessIOError(f"{field_name} must be a nonnegative finite number")
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
    return MappingProxyType({str(key): _copy_json_value(item, f"{field_name}.{key}") for key, item in value.items()})


def _copy_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _copy_json_value(item, f"{field_name}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_copy_json_value(item, f"{field_name}[]") for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise HarnessIOError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
