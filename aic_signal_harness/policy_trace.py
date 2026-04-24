"""Typed policy trace JSONL contracts for the AIC signal harness."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION, LeakageClass, SchemaValidationError


_POLICY_TRACE_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "trial_id",
        "official_trial_id",
        "event_index",
        "event_type",
        "elapsed_sec",
        "emitted_at_utc",
        "source",
        "leakage_class",
        "payload",
    }
)
_POLICY_TRACE_TRIAL_REPORT_KEYS = frozenset(
    {
        "trial_id",
        "event_count",
        "first_event_index",
        "last_event_index",
        "start_elapsed_sec",
        "end_elapsed_sec",
        "action_event_count",
        "nonzero_action_event_count",
        "safety_guard_event_count",
        "error_event_count",
        "event_type_counts",
    }
)
_POLICY_TRACE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "run_id",
        "reduced_at_utc",
        "event_count",
        "start_elapsed_sec",
        "end_elapsed_sec",
        "event_type_counts",
        "leakage_classes",
        "trials",
    }
)
_ACTION_EVENT_TYPES = frozenset({"action_selected", "action_published"})
_OFFICIAL_TRIAL_ID_RE = re.compile(r"^trial_[1-9][0-9]*$")


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


class PolicyTraceEventType(_StrEnum):
    """Policy-side event kinds accepted by the trace reducer."""

    task_started = "task_started"
    observation = "observation"
    action_selected = "action_selected"
    action_published = "action_published"
    feedback = "feedback"
    planner_event = "planner_event"
    safety_guard = "safety_guard"
    task_finished = "task_finished"
    error = "error"


@dataclass(frozen=True)
class PolicyTraceEvent:
    """One backend-emitted policy trace event from JSONL."""

    run_id: str
    trial_id: str
    event_index: int
    event_type: PolicyTraceEventType
    elapsed_sec: float
    emitted_at_utc: str
    source: str
    leakage_class: LeakageClass
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    official_trial_id: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("run_id", "trial_id", "emitted_at_utc", "source"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(getattr(self, field_name), f"policy trace event.{field_name}"),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "official_trial_id",
                _optional_official_trial_id(
                    self.official_trial_id,
                    "policy trace event.official_trial_id",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "event_index",
                _require_nonnegative_int(self.event_index, "policy trace event.event_index"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "event_type",
                PolicyTraceEventType.parse(self.event_type),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "elapsed_sec",
                _require_nonnegative_float(self.elapsed_sec, "policy trace event.elapsed_sec"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "leakage_class",
                LeakageClass.parse(self.leakage_class),
            )
        except SchemaValidationError as exc:
            errors.append(str(exc))
        try:
            payload = _copy_json_mapping(self.payload, "policy trace event.payload")
            object.__setattr__(self, "payload", payload)
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors and self.event_type.value in _ACTION_EVENT_TYPES:
            errors.extend(_action_payload_errors(self.payload, "policy trace action payload"))
            if self.leakage_class is not LeakageClass.legal_policy_action_output:
                errors.append(
                    "policy trace action events must use legal_policy_action_output leakage_class"
                )
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "event_index": self.event_index,
            "event_type": self.event_type.value,
            "elapsed_sec": self.elapsed_sec,
            "emitted_at_utc": self.emitted_at_utc,
            "source": self.source,
            "leakage_class": self.leakage_class.value,
            "payload": _thaw_json(self.payload),
        }
        if self.official_trial_id is not None:
            value["official_trial_id"] = self.official_trial_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyTraceEvent":
        if not isinstance(value, Mapping):
            raise HarnessIOError("policy trace event must be a mapping")
        _reject_unknown_keys(value, _POLICY_TRACE_EVENT_KEYS, "policy trace event")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            trial_id=cast(Any, value.get("trial_id")),
            event_index=cast(Any, value.get("event_index")),
            event_type=cast(Any, value.get("event_type")),
            elapsed_sec=cast(Any, value.get("elapsed_sec")),
            emitted_at_utc=cast(Any, value.get("emitted_at_utc")),
            source=cast(Any, value.get("source")),
            leakage_class=cast(Any, value.get("leakage_class")),
            payload=value.get("payload", {}),
            official_trial_id=value.get("official_trial_id"),
        )


@dataclass(frozen=True)
class PolicyTraceTrialReport:
    """Per-trial summary derived from policy trace events."""

    trial_id: str
    event_count: int
    first_event_index: int
    last_event_index: int
    start_elapsed_sec: float
    end_elapsed_sec: float
    action_event_count: int = 0
    nonzero_action_event_count: int = 0
    safety_guard_event_count: int = 0
    error_event_count: int = 0
    event_type_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "trial_id",
                _require_nonempty_text(self.trial_id, "policy trace trial.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in (
            "event_count",
            "first_event_index",
            "last_event_index",
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
                        f"policy trace trial.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in ("start_elapsed_sec", "end_elapsed_sec"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_float(
                        getattr(self, field_name),
                        f"policy trace trial.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            counts = _copy_count_mapping(
                self.event_type_counts,
                "policy trace trial.event_type_counts",
            )
            object.__setattr__(self, "event_type_counts", counts)
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            if self.event_count <= 0:
                errors.append("policy trace trial.event_count must be positive")
            if self.last_event_index < self.first_event_index:
                errors.append("policy trace trial.last_event_index must be >= first_event_index")
            if self.end_elapsed_sec < self.start_elapsed_sec:
                errors.append("policy trace trial.end_elapsed_sec must be >= start_elapsed_sec")
            if sum(self.event_type_counts.values()) != self.event_count:
                errors.append("policy trace trial.event_type_counts must sum to event_count")
            if self.nonzero_action_event_count > self.action_event_count:
                errors.append("nonzero_action_event_count must not exceed action_event_count")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "event_count": self.event_count,
            "first_event_index": self.first_event_index,
            "last_event_index": self.last_event_index,
            "start_elapsed_sec": self.start_elapsed_sec,
            "end_elapsed_sec": self.end_elapsed_sec,
            "action_event_count": self.action_event_count,
            "nonzero_action_event_count": self.nonzero_action_event_count,
            "safety_guard_event_count": self.safety_guard_event_count,
            "error_event_count": self.error_event_count,
            "event_type_counts": dict(self.event_type_counts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyTraceTrialReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("policy trace trial report must be a mapping")
        _reject_unknown_keys(value, _POLICY_TRACE_TRIAL_REPORT_KEYS, "policy trace trial report")
        return cls(
            trial_id=cast(Any, value.get("trial_id")),
            event_count=cast(Any, value.get("event_count")),
            first_event_index=cast(Any, value.get("first_event_index")),
            last_event_index=cast(Any, value.get("last_event_index")),
            start_elapsed_sec=cast(Any, value.get("start_elapsed_sec")),
            end_elapsed_sec=cast(Any, value.get("end_elapsed_sec")),
            action_event_count=value.get("action_event_count", 0),
            nonzero_action_event_count=value.get("nonzero_action_event_count", 0),
            safety_guard_event_count=value.get("safety_guard_event_count", 0),
            error_event_count=value.get("error_event_count", 0),
            event_type_counts=value.get("event_type_counts", {}),
        )


@dataclass(frozen=True)
class PolicyTraceReport:
    """Run-level summary derived from backend-emitted policy trace JSONL."""

    source: str
    run_id: str
    reduced_at_utc: str
    event_count: int
    start_elapsed_sec: float
    end_elapsed_sec: float
    event_type_counts: Mapping[str, int]
    leakage_classes: tuple[LeakageClass, ...]
    trials: tuple[PolicyTraceTrialReport, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("source", "run_id", "reduced_at_utc"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(getattr(self, field_name), f"policy trace report.{field_name}"),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "event_count",
                _require_nonnegative_int(self.event_count, "policy trace report.event_count"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("start_elapsed_sec", "end_elapsed_sec"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_float(
                        getattr(self, field_name),
                        f"policy trace report.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "event_type_counts",
                _copy_count_mapping(
                    self.event_type_counts,
                    "policy trace report.event_type_counts",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "leakage_classes",
                _as_leakage_tuple(self.leakage_classes, "policy trace report.leakage_classes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trials",
                tuple(
                    trial
                    if isinstance(trial, PolicyTraceTrialReport)
                    else PolicyTraceTrialReport.from_dict(trial)
                    for trial in _as_sequence(self.trials, "policy trace report.trials")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            if self.event_count <= 0:
                errors.append("policy trace report.event_count must be positive")
            if not self.trials:
                errors.append("policy trace report.trials must not be empty")
            if self.end_elapsed_sec < self.start_elapsed_sec:
                errors.append("policy trace report.end_elapsed_sec must be >= start_elapsed_sec")
            if sum(self.event_type_counts.values()) != self.event_count:
                errors.append("policy trace report.event_type_counts must sum to event_count")
            if sum(trial.event_count for trial in self.trials) != self.event_count:
                errors.append("policy trace report trial event counts must sum to event_count")
            duplicate_trials = _duplicates(trial.trial_id for trial in self.trials)
            if duplicate_trials:
                errors.append("policy trace report duplicate trial_id: " + ", ".join(duplicate_trials))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "run_id": self.run_id,
            "reduced_at_utc": self.reduced_at_utc,
            "event_count": self.event_count,
            "start_elapsed_sec": self.start_elapsed_sec,
            "end_elapsed_sec": self.end_elapsed_sec,
            "event_type_counts": dict(self.event_type_counts),
            "leakage_classes": [item.value for item in self.leakage_classes],
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyTraceReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("policy trace report must be a mapping")
        _reject_unknown_keys(value, _POLICY_TRACE_REPORT_KEYS, "policy trace report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            source=cast(Any, value.get("source")),
            run_id=cast(Any, value.get("run_id")),
            reduced_at_utc=cast(Any, value.get("reduced_at_utc")),
            event_count=cast(Any, value.get("event_count")),
            start_elapsed_sec=cast(Any, value.get("start_elapsed_sec")),
            end_elapsed_sec=cast(Any, value.get("end_elapsed_sec")),
            event_type_counts=value.get("event_type_counts", {}),
            leakage_classes=cast(Any, value.get("leakage_classes", ())),
            trials=value.get("trials", ()),
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


def _optional_official_trial_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    trial_id = _require_nonempty_text(value, field_name)
    if not _OFFICIAL_TRIAL_ID_RE.match(trial_id):
        raise HarnessIOError(f"{field_name} must be an official trial_* id")
    return trial_id


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


def _as_vec3(value: Any, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a 3-number list or tuple")
    if len(value) != 3:
        raise HarnessIOError(f"{field_name} must contain exactly 3 numbers")
    return (
        _require_finite_float(value[0], f"{field_name}[0]"),
        _require_finite_float(value[1], f"{field_name}[1]"),
        _require_finite_float(value[2], f"{field_name}[2]"),
    )


def _require_finite_float(value: Any, field_name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _as_leakage_tuple(value: Any, field_name: str) -> tuple[LeakageClass, ...]:
    items = _as_sequence(value, field_name)
    if not items:
        raise HarnessIOError(f"{field_name} must not be empty")
    parsed: list[LeakageClass] = []
    for item in items:
        try:
            parsed.append(LeakageClass.parse(item))
        except SchemaValidationError as exc:
            raise HarnessIOError(str(exc)) from exc
    return tuple(parsed)


def _copy_count_mapping(value: Any, field_name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        if type(item) is not int or item < 0:
            raise HarnessIOError(f"{field_name}.{key} must be a nonnegative integer")
        copied[key] = item
    if not copied:
        raise HarnessIOError(f"{field_name} must not be empty")
    return MappingProxyType(copied)


def _copy_json_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        copied[key] = _copy_json_value(item, f"{field_name}.{key}")
    return MappingProxyType(copied)


def _copy_json_value(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessIOError(f"{field_name} must be a finite JSON number")
        return value
    if isinstance(value, Mapping):
        return _copy_json_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_copy_json_value(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    raise HarnessIOError(f"{field_name} is not JSON serializable")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _action_payload_errors(payload: Mapping[str, Any], field_name: str) -> list[str]:
    errors: list[str] = []
    for key in ("linear", "angular"):
        if key not in payload:
            errors.append(f"{field_name}.{key} is required")
            continue
        try:
            _as_vec3(payload[key], f"{field_name}.{key}")
        except HarnessIOError as exc:
            errors.append(str(exc))
    frame_id = payload.get("frame_id")
    if frame_id is not None and (not isinstance(frame_id, str) or not frame_id.strip()):
        errors.append(f"{field_name}.frame_id must be a nonempty string when set")
    return errors


def action_payload_is_nonzero(payload: Mapping[str, Any]) -> bool:
    """Return whether an already-validated action payload commands motion."""

    linear = _as_vec3(payload["linear"], "policy trace action payload.linear")
    angular = _as_vec3(payload["angular"], "policy trace action payload.angular")
    return any(abs(value) > 1e-12 for value in (*linear, *angular))


def _duplicates(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
