"""Typed next-experiment reports for the AIC signal harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION


_NEXT_EXPERIMENT_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "kind",
        "priority",
        "action",
        "title",
        "rationale",
        "evidence_labels",
        "evidence_terms",
        "evidence_manifest_fields",
        "acceptance_checks",
        "blocked_by_labels",
        "runtime_notes",
        "leakage_notes",
    }
)
_NEXT_EXPERIMENT_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "gate_id",
        "generated_at_utc",
        "decision",
        "objective",
        "candidates",
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


class NextExperimentDecision(_StrEnum):
    iterate = "iterate"
    block = "block"
    promote = "promote"
    stop = "stop"


class NextExperimentCandidateKind(_StrEnum):
    evidence_repair = "evidence_repair"
    reject_control_primitive = "reject_control_primitive"
    safety_contract = "safety_contract"
    controller_development = "controller_development"
    offline_acceptance_gate = "offline_acceptance_gate"
    failure_analysis = "failure_analysis"


class NextExperimentPriority(_StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class NextExperimentAction(_StrEnum):
    analyze = "analyze"
    block = "block"
    build = "build"
    keep = "keep"
    reject = "reject"
    require = "require"


@dataclass(frozen=True)
class NextExperimentCandidate:
    """One possible next action, tied to machine-readable evidence."""

    candidate_id: str
    kind: NextExperimentCandidateKind
    priority: NextExperimentPriority
    action: NextExperimentAction
    title: str
    rationale: str
    evidence_labels: tuple[str, ...] = field(default_factory=tuple)
    evidence_terms: tuple[str, ...] = field(default_factory=tuple)
    evidence_manifest_fields: tuple[str, ...] = field(default_factory=tuple)
    acceptance_checks: tuple[str, ...] = field(default_factory=tuple)
    blocked_by_labels: tuple[str, ...] = field(default_factory=tuple)
    runtime_notes: str = ""
    leakage_notes: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("candidate_id", "title", "rationale"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"next_experiment.candidate.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "kind",
                NextExperimentCandidateKind.parse(self.kind),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "priority",
                NextExperimentPriority.parse(self.priority),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "action",
                NextExperimentAction.parse(self.action),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in (
            "evidence_labels",
            "evidence_terms",
            "evidence_manifest_fields",
            "acceptance_checks",
            "blocked_by_labels",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _as_text_sequence(
                        getattr(self, field_name),
                        f"next_experiment.candidate.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in ("runtime_notes", "leakage_notes"):
            if not isinstance(getattr(self, field_name), str):
                errors.append(f"next_experiment.candidate.{field_name} must be a string")
        if (
            not errors
            and not self.evidence_labels
            and not self.evidence_terms
            and not self.evidence_manifest_fields
        ):
            errors.append(
                "next_experiment candidates must cite evidence_labels, evidence_terms, or evidence_manifest_fields"
            )
        if not errors and self.action is NextExperimentAction.block and not self.blocked_by_labels:
            errors.append("blocking candidates must set blocked_by_labels")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "priority": self.priority.value,
            "action": self.action.value,
            "title": self.title,
            "rationale": self.rationale,
            "evidence_labels": list(self.evidence_labels),
            "evidence_terms": list(self.evidence_terms),
            "evidence_manifest_fields": list(self.evidence_manifest_fields),
            "acceptance_checks": list(self.acceptance_checks),
            "blocked_by_labels": list(self.blocked_by_labels),
            "runtime_notes": self.runtime_notes,
            "leakage_notes": self.leakage_notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextExperimentCandidate":
        if not isinstance(value, Mapping):
            raise HarnessIOError("next experiment candidate must be a mapping")
        _reject_unknown_keys(value, _NEXT_EXPERIMENT_CANDIDATE_KEYS, "next experiment candidate")
        return cls(
            candidate_id=cast(Any, value.get("candidate_id")),
            kind=cast(Any, value.get("kind")),
            priority=cast(Any, value.get("priority")),
            action=cast(Any, value.get("action")),
            title=cast(Any, value.get("title")),
            rationale=cast(Any, value.get("rationale")),
            evidence_labels=value.get("evidence_labels", ()),
            evidence_terms=value.get("evidence_terms", ()),
            evidence_manifest_fields=value.get("evidence_manifest_fields", ()),
            acceptance_checks=value.get("acceptance_checks", ()),
            blocked_by_labels=value.get("blocked_by_labels", ()),
            runtime_notes=value.get("runtime_notes", ""),
            leakage_notes=value.get("leakage_notes", ""),
        )


@dataclass(frozen=True)
class NextExperimentReport:
    """A deterministic recommendation report, not an autonomous action."""

    run_id: str
    gate_id: str
    generated_at_utc: str
    decision: NextExperimentDecision
    candidates: tuple[NextExperimentCandidate, ...]
    objective: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("run_id", "gate_id", "generated_at_utc"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(getattr(self, field_name), field_name),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "decision",
                NextExperimentDecision.parse(self.decision),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            candidates = tuple(
                candidate
                if isinstance(candidate, NextExperimentCandidate)
                else NextExperimentCandidate.from_dict(candidate)
                for candidate in _as_sequence(self.candidates, "next_experiment.candidates")
            )
            if not candidates:
                errors.append("next_experiment.candidates must not be empty")
            duplicate_ids = _duplicates(candidate.candidate_id for candidate in candidates)
            if duplicate_ids:
                errors.append(
                    "next_experiment.candidates must not contain duplicate candidate_id: "
                    + ", ".join(duplicate_ids)
                )
            object.__setattr__(self, "candidates", candidates)
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "objective",
                _optional_nonempty_text(self.objective, "objective"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "next_experiment.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "gate_id": self.gate_id,
            "generated_at_utc": self.generated_at_utc,
            "decision": self.decision.value,
            "objective": self.objective,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextExperimentReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("next experiment report must be a mapping")
        _reject_unknown_keys(value, _NEXT_EXPERIMENT_REPORT_KEYS, "next experiment report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            gate_id=cast(Any, value.get("gate_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            decision=cast(Any, value.get("decision")),
            objective=value.get("objective"),
            candidates=value.get("candidates", ()),
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


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    items = _as_sequence(value, field_name)
    return tuple(_require_nonempty_text(item, field_name) for item in items)


def _duplicates(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
