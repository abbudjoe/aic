"""Typed failure labels for offline AIC experiment analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION, LeakageClass, SchemaValidationError


_FAILURE_LABEL_KEYS = frozenset(
    {
        "kind",
        "severity",
        "summary",
        "source",
        "leakage_class",
        "trial_id",
        "evidence",
    }
)
_FAILURE_REPORT_KEYS = frozenset(
    {"schema_version", "run_id", "generated_at_utc", "labels", "notes"}
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


class FailureKind(_StrEnum):
    """Evidence-backed failure categories used by the Gate 2 loop."""

    inconclusive_missing_evidence = "inconclusive_missing_evidence"
    no_partial_or_full_insertion = "no_partial_or_full_insertion"
    off_limit_contact = "off_limit_contact"
    contact_guard_stop_recommended = "contact_guard_stop_recommended"
    below_promotion_margin = "below_promotion_margin"
    score_regression = "score_regression"


class FailureSeverity(_StrEnum):
    info = "info"
    warning = "warning"
    blocker = "blocker"


@dataclass(frozen=True)
class FailureLabel:
    """One typed failure label tied to concrete evidence."""

    kind: FailureKind
    severity: FailureSeverity
    summary: str
    source: str
    leakage_class: LeakageClass
    trial_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "kind", FailureKind.parse(self.kind))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "severity", FailureSeverity.parse(self.severity))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "summary",
                _require_nonempty_text(self.summary, "failure.summary"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "source",
                _require_nonempty_text(self.source, "failure.source"),
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
            object.__setattr__(
                self,
                "trial_id",
                _optional_nonempty_text(self.trial_id, "failure.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "evidence",
                _copy_json_mapping(self.evidence, "failure.evidence"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "summary": self.summary,
            "source": self.source,
            "leakage_class": self.leakage_class.value,
        }
        if self.trial_id is not None:
            value["trial_id"] = self.trial_id
        if self.evidence:
            value["evidence"] = _thaw_json(self.evidence)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureLabel":
        if not isinstance(value, Mapping):
            raise HarnessIOError("failure label must be a mapping")
        _reject_unknown_keys(value, _FAILURE_LABEL_KEYS, "failure label")
        return cls(
            kind=cast(Any, value.get("kind")),
            severity=cast(Any, value.get("severity")),
            summary=cast(Any, value.get("summary")),
            source=cast(Any, value.get("source")),
            leakage_class=cast(Any, value.get("leakage_class")),
            trial_id=value.get("trial_id"),
            evidence=value.get("evidence", {}),
        )


@dataclass(frozen=True)
class FailureReport:
    """Failure labels derived from typed AIC experiment evidence."""

    run_id: str
    generated_at_utc: str
    labels: tuple[FailureLabel, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "run_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "generated_at_utc",
                _require_nonempty_text(self.generated_at_utc, "generated_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "labels",
                tuple(
                    label if isinstance(label, FailureLabel) else FailureLabel.from_dict(label)
                    for label in _as_sequence(self.labels, "failure labels")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                tuple(
                    _require_nonempty_text(note, "failure note")
                    for note in _as_sequence(self.notes, "failure notes")
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
            "labels": [label.to_dict() for label in self.labels],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("failure report must be a mapping")
        _reject_unknown_keys(value, _FAILURE_REPORT_KEYS, "failure report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            labels=value.get("labels", ()),
            notes=value.get("notes", ()),
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


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _copy_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied, errors = _jsonable_copy(value, field_name)
    if errors:
        raise HarnessIOError("; ".join(errors))
    return _freeze_json(copied)


def _jsonable_copy(value: Any, field_name: str) -> tuple[Any, list[str]]:
    if value is None or isinstance(value, (str, bool, int)):
        return value, []
    if isinstance(value, float):
        if not math.isfinite(value):
            return value, [f"{field_name} must be a finite JSON number"]
        return value, []
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        errors: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{field_name} keys must be nonempty strings")
                continue
            copied_item, item_errors = _jsonable_copy(item, f"{field_name}.{key}")
            copied[key] = copied_item
            errors.extend(item_errors)
        return copied, errors
    if isinstance(value, (list, tuple)):
        copied_list: list[Any] = []
        errors = []
        for index, item in enumerate(value):
            copied_item, item_errors = _jsonable_copy(item, f"{field_name}[{index}]")
            copied_list.append(copied_item)
            errors.extend(item_errors)
        return copied_list, errors
    return value, [f"{field_name} must be JSON serializable"]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
    return value
