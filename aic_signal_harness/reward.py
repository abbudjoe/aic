"""Typed reward reports for offline AIC experiment analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION, LeakageClass, SchemaValidationError


_REWARD_TERM_KEYS = frozenset(
    {
        "name",
        "value",
        "signal_kind",
        "leakage_class",
        "source",
        "trial_id",
        "provenance",
    }
)
_REWARD_REPORT_KEYS = frozenset(
    {"schema_version", "run_id", "generated_at_utc", "terms", "notes"}
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


class RewardSignalKind(_StrEnum):
    """Whether a reward term is official score truth or offline diagnosis."""

    official_score = "official_score"
    diagnostic = "diagnostic"
    shaped_reward = "shaped_reward"


@dataclass(frozen=True)
class RewardTerm:
    """One scalar reward or diagnostic term tied to explicit provenance."""

    name: str
    value: float
    signal_kind: RewardSignalKind
    leakage_class: LeakageClass
    source: str
    trial_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "name", _require_nonempty_text(self.name, "reward.name"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "value", _require_finite_float(self.value, "reward.value"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "signal_kind",
                RewardSignalKind.parse(self.signal_kind),
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
                "source",
                _require_nonempty_text(self.source, "reward.source"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trial_id",
                _optional_nonempty_text(self.trial_id, "reward.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "provenance",
                _copy_json_mapping(self.provenance, "reward.provenance"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "signal_kind": self.signal_kind.value,
            "leakage_class": self.leakage_class.value,
            "source": self.source,
        }
        if self.trial_id is not None:
            value["trial_id"] = self.trial_id
        if self.provenance:
            value["provenance"] = _thaw_json(self.provenance)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RewardTerm":
        if not isinstance(value, Mapping):
            raise HarnessIOError("reward term must be a mapping")
        _reject_unknown_keys(value, _REWARD_TERM_KEYS, "reward term")
        return cls(
            name=cast(Any, value.get("name")),
            value=cast(Any, value.get("value")),
            signal_kind=cast(Any, value.get("signal_kind")),
            leakage_class=cast(Any, value.get("leakage_class")),
            source=cast(Any, value.get("source")),
            trial_id=value.get("trial_id"),
            provenance=value.get("provenance", {}),
        )


@dataclass(frozen=True)
class RewardReport:
    """Reward terms derived from typed AIC experiment evidence."""

    run_id: str
    generated_at_utc: str
    terms: tuple[RewardTerm, ...] = field(default_factory=tuple)
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
                "terms",
                tuple(
                    term if isinstance(term, RewardTerm) else RewardTerm.from_dict(term)
                    for term in _as_sequence(self.terms, "reward terms")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                tuple(
                    _require_nonempty_text(note, "reward note")
                    for note in _as_sequence(self.notes, "reward notes")
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
            "terms": [term.to_dict() for term in self.terms],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RewardReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("reward report must be a mapping")
        _reject_unknown_keys(value, _REWARD_REPORT_KEYS, "reward report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            terms=value.get("terms", ()),
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
