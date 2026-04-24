"""Executable next-experiment plan contracts for offline candidate handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file
from aic_signal_harness.next_experiment import (
    NextExperimentAction,
    NextExperimentCandidateKind,
    NextExperimentPriority,
    NextExperimentReport,
)
from aic_signal_harness.manifest import RunManifest
from aic_signal_harness.schemas import (
    ArtifactRef,
    ExperimentSpec,
    LeakageClass,
    SCHEMA_VERSION,
    SchemaValidationError,
)


_PLAN_CANDIDATE_KEYS = frozenset(
    {
        "plan_id",
        "candidate_id",
        "candidate_kind",
        "candidate_action",
        "priority",
        "status",
        "execution_mode",
        "title",
        "rationale",
        "evidence_labels",
        "evidence_terms",
        "evidence_manifest_fields",
        "acceptance_checks",
        "blocked_by_labels",
        "experiment",
        "offline_only",
        "launchable",
        "autonomous_launch_allowed",
        "leakage_class",
        "notes",
    }
)
_PLAN_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "gate_id",
        "generated_at_utc",
        "source_next_experiment",
        "source_manifest",
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


class NextExperimentPlanStatus(_StrEnum):
    """Consumption state for a derived candidate plan."""

    blocked = "blocked"
    launchable = "launchable"
    requires_implementation = "requires_implementation"
    no_launch = "no_launch"


class NextExperimentExecutionMode(_StrEnum):
    """Narrow offline execution modes understood by future runners."""

    evidence_repair = "evidence_repair"
    failure_analysis = "failure_analysis"
    offline_gate = "offline_gate"
    policy_development = "policy_development"
    promotion_review = "promotion_review"
    rejection_record = "rejection_record"
    safety_contract = "safety_contract"


@dataclass(frozen=True)
class NextExperimentPlanCandidate:
    """One executable or explicitly non-executable plan derived from a recommendation."""

    plan_id: str
    candidate_id: str
    candidate_kind: NextExperimentCandidateKind
    candidate_action: NextExperimentAction
    priority: NextExperimentPriority
    status: NextExperimentPlanStatus
    execution_mode: NextExperimentExecutionMode
    title: str
    rationale: str
    evidence_labels: tuple[str, ...] = field(default_factory=tuple)
    evidence_terms: tuple[str, ...] = field(default_factory=tuple)
    evidence_manifest_fields: tuple[str, ...] = field(default_factory=tuple)
    acceptance_checks: tuple[str, ...] = field(default_factory=tuple)
    blocked_by_labels: tuple[str, ...] = field(default_factory=tuple)
    experiment: ExperimentSpec | None = None
    offline_only: bool = True
    launchable: bool = False
    autonomous_launch_allowed: bool = False
    leakage_class: LeakageClass = LeakageClass.post_hoc_label
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("plan_id", "candidate_id", "title", "rationale"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"next_experiment_plan.candidate.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name, enum_type in (
            ("candidate_kind", NextExperimentCandidateKind),
            ("candidate_action", NextExperimentAction),
            ("priority", NextExperimentPriority),
            ("status", NextExperimentPlanStatus),
            ("execution_mode", NextExperimentExecutionMode),
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    enum_type.parse(getattr(self, field_name)),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "evidence_labels",
            "evidence_terms",
            "evidence_manifest_fields",
            "acceptance_checks",
            "blocked_by_labels",
            "notes",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _as_text_sequence(
                        getattr(self, field_name),
                        f"next_experiment_plan.candidate.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if self.experiment is not None and not isinstance(self.experiment, ExperimentSpec):
            try:
                object.__setattr__(self, "experiment", ExperimentSpec.from_dict(self.experiment))
            except SchemaValidationError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(self, "leakage_class", LeakageClass.parse(self.leakage_class))
        except SchemaValidationError as exc:
            errors.append(str(exc))
        if self.offline_only is not True:
            errors.append("next_experiment_plan.candidate.offline_only must be true")
        if not isinstance(self.launchable, bool):
            errors.append("next_experiment_plan.candidate.launchable must be a boolean")
        if self.autonomous_launch_allowed is not False:
            errors.append("next_experiment_plan.candidate.autonomous_launch_allowed must be false")
        if self.leakage_class is not LeakageClass.post_hoc_label:
            errors.append("next_experiment_plan.candidate.leakage_class must be post_hoc_label")
        if not errors:
            errors.extend(_candidate_plan_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind.value,
            "candidate_action": self.candidate_action.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "execution_mode": self.execution_mode.value,
            "title": self.title,
            "rationale": self.rationale,
            "evidence_labels": list(self.evidence_labels),
            "evidence_terms": list(self.evidence_terms),
            "evidence_manifest_fields": list(self.evidence_manifest_fields),
            "acceptance_checks": list(self.acceptance_checks),
            "blocked_by_labels": list(self.blocked_by_labels),
            "offline_only": self.offline_only,
            "launchable": self.launchable,
            "autonomous_launch_allowed": self.autonomous_launch_allowed,
            "leakage_class": self.leakage_class.value,
            "notes": list(self.notes),
        }
        if self.experiment is not None:
            value["experiment"] = self.experiment.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextExperimentPlanCandidate":
        if not isinstance(value, Mapping):
            raise HarnessIOError("next experiment plan candidate must be a mapping")
        _reject_unknown_keys(value, _PLAN_CANDIDATE_KEYS, "next experiment plan candidate")
        return cls(
            plan_id=cast(Any, value.get("plan_id")),
            candidate_id=cast(Any, value.get("candidate_id")),
            candidate_kind=cast(Any, value.get("candidate_kind")),
            candidate_action=cast(Any, value.get("candidate_action")),
            priority=cast(Any, value.get("priority")),
            status=cast(Any, value.get("status")),
            execution_mode=cast(Any, value.get("execution_mode")),
            title=cast(Any, value.get("title")),
            rationale=cast(Any, value.get("rationale")),
            evidence_labels=value.get("evidence_labels", ()),
            evidence_terms=value.get("evidence_terms", ()),
            evidence_manifest_fields=value.get("evidence_manifest_fields", ()),
            acceptance_checks=value.get("acceptance_checks", ()),
            blocked_by_labels=value.get("blocked_by_labels", ()),
            experiment=value.get("experiment"),
            offline_only=cast(bool, value.get("offline_only")),
            launchable=cast(bool, value.get("launchable")),
            autonomous_launch_allowed=cast(bool, value.get("autonomous_launch_allowed")),
            leakage_class=cast(Any, value.get("leakage_class")),
            notes=value.get("notes", ()),
        )


@dataclass(frozen=True)
class NextExperimentPlan:
    """Typed offline plan derived from a single ``next_experiment.json`` report."""

    run_id: str
    gate_id: str
    generated_at_utc: str
    source_next_experiment: ArtifactRef
    source_manifest: ArtifactRef
    candidates: tuple[NextExperimentPlanCandidate, ...]
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
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"next_experiment_plan.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.source_next_experiment, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "source_next_experiment",
                    ArtifactRef.from_dict(self.source_next_experiment),
                )
            except SchemaValidationError as exc:
                errors.append(str(exc))
        if not errors:
            errors.extend(
                _source_next_experiment_errors(
                    self.source_next_experiment,
                    run_id=self.run_id,
                    gate_id=self.gate_id,
                )
            )
        if not isinstance(self.source_manifest, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "source_manifest",
                    ArtifactRef.from_dict(self.source_manifest),
                )
            except SchemaValidationError as exc:
                errors.append(str(exc))
        if not errors:
            errors.extend(_source_manifest_errors(self.source_manifest, run_id=self.run_id))
        try:
            candidates = tuple(
                candidate
                if isinstance(candidate, NextExperimentPlanCandidate)
                else NextExperimentPlanCandidate.from_dict(candidate)
                for candidate in _as_sequence(self.candidates, "next_experiment_plan.candidates")
            )
            if not candidates:
                errors.append("next_experiment_plan.candidates must not be empty")
            duplicate_plan_ids = _duplicates(candidate.plan_id for candidate in candidates)
            if duplicate_plan_ids:
                errors.append(
                    "next_experiment_plan.candidates must not contain duplicate plan_id: "
                    + ", ".join(duplicate_plan_ids)
                )
            duplicate_candidate_ids = _duplicates(candidate.candidate_id for candidate in candidates)
            if duplicate_candidate_ids:
                errors.append(
                    "next_experiment_plan.candidates must not contain duplicate candidate_id: "
                    + ", ".join(duplicate_candidate_ids)
                )
            object.__setattr__(self, "candidates", candidates)
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "next_experiment_plan.notes"),
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
            "source_next_experiment": self.source_next_experiment.to_dict(),
            "source_manifest": self.source_manifest.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextExperimentPlan":
        if not isinstance(value, Mapping):
            raise HarnessIOError("next experiment plan must be a mapping")
        _reject_unknown_keys(value, _PLAN_REPORT_KEYS, "next experiment plan")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            gate_id=cast(Any, value.get("gate_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_next_experiment=cast(Any, value.get("source_next_experiment")),
            source_manifest=cast(Any, value.get("source_manifest")),
            candidates=cast(Any, value.get("candidates", ())),
            notes=cast(Any, value.get("notes", ())),
        )


def next_experiment_plan_id(
    run_id: str,
    gate_id: str,
    candidate_id: str,
    *,
    source_next_experiment_sha256: str | None,
    source_manifest_sha256: str | None,
) -> str:
    payload = {
        "run_id": run_id,
        "gate_id": gate_id,
        "candidate_id": candidate_id,
        "source_next_experiment_sha256": source_next_experiment_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "nxplan_" + hashlib.sha256(encoded).hexdigest()[:20]


def _candidate_plan_consistency_errors(candidate: NextExperimentPlanCandidate) -> list[str]:
    errors: list[str] = []
    if candidate.status is NextExperimentPlanStatus.blocked and not candidate.blocked_by_labels:
        errors.append("blocked next experiment plan candidates must set blocked_by_labels")
    if candidate.status is NextExperimentPlanStatus.blocked and candidate.experiment is not None:
        errors.append("blocked next experiment plan candidates must not include experiment")
    if candidate.status is NextExperimentPlanStatus.launchable:
        if not candidate.launchable:
            errors.append("launchable next experiment plan candidates must set launchable=true")
        if candidate.experiment is None:
            errors.append("launchable next experiment plan candidates must include experiment")
        if not candidate.acceptance_checks:
            errors.append("launchable next experiment plan candidates must set acceptance_checks")
    else:
        if candidate.launchable:
            errors.append("non-launchable next experiment plan candidates must set launchable=false")
    if candidate.status is NextExperimentPlanStatus.requires_implementation:
        if candidate.experiment is not None:
            errors.append("requires_implementation candidates must not include experiment")
        if not candidate.acceptance_checks:
            errors.append("requires_implementation candidates must set acceptance_checks")
    if candidate.status is NextExperimentPlanStatus.no_launch and candidate.experiment is not None:
        errors.append("no_launch next experiment plan candidates must not include experiment")
    return errors


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


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    items = _as_sequence(value, field_name)
    return tuple(_require_nonempty_text(item, field_name) for item in items)


def _copy_json_mapping(value: Any, field_name: str) -> MappingProxyType[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        copied[key] = _copy_json_value(item, f"{field_name}.{key}")
    return MappingProxyType(copied)


def _copy_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return _copy_json_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_copy_json_value(item, f"{field_name}[]") for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    raise HarnessIOError(f"{field_name} must be JSON-compatible")


def _source_next_experiment_errors(
    source_next_experiment: ArtifactRef,
    *,
    run_id: str,
    gate_id: str,
) -> list[str]:
    errors: list[str] = []
    if source_next_experiment.kind != "next_experiment_report":
        errors.append("next_experiment_plan.source_next_experiment.kind must be 'next_experiment_report'")
    if source_next_experiment.sha256 is None:
        errors.append("next_experiment_plan.source_next_experiment.sha256 must be set")
    if source_next_experiment.provenance.get("run_id") != run_id:
        errors.append("next_experiment_plan.source_next_experiment provenance run_id must match plan.run_id")
    if source_next_experiment.provenance.get("gate_id") != gate_id:
        errors.append("next_experiment_plan.source_next_experiment provenance gate_id must match plan.gate_id")
    if "producer" not in source_next_experiment.provenance:
        errors.append("next_experiment_plan.source_next_experiment must set provenance.producer")
    if "derivation" not in source_next_experiment.provenance:
        errors.append("next_experiment_plan.source_next_experiment must set provenance.derivation")
    try:
        report_path = local_artifact_path(
            path=source_next_experiment.path,
            uri=source_next_experiment.uri,
            field_name="source_next_experiment",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        report_path = None
    if report_path is None:
        errors.append("next_experiment_plan.source_next_experiment must be local byte-verifiable")
    elif not report_path.exists():
        errors.append("next_experiment_plan.source_next_experiment.path must exist")
    elif not report_path.is_file():
        errors.append("next_experiment_plan.source_next_experiment.path must point to a JSON report file")
    elif source_next_experiment.sha256 is not None:
        if sha256_file(report_path) != source_next_experiment.sha256:
            errors.append("next_experiment_plan.source_next_experiment.sha256 must match path")
        else:
            try:
                source_report = NextExperimentReport.from_dict(read_json(report_path))
            except HarnessIOError as exc:
                errors.append(f"next_experiment_plan.source_next_experiment must parse: {exc}")
            else:
                if source_report.run_id != run_id:
                    errors.append("source_next_experiment.run_id must match plan.run_id")
                if source_report.gate_id != gate_id:
                    errors.append("source_next_experiment.gate_id must match plan.gate_id")
    return errors


def _source_manifest_errors(source_manifest: ArtifactRef, *, run_id: str) -> list[str]:
    errors: list[str] = []
    if source_manifest.kind != "run_manifest":
        errors.append("next_experiment_plan.source_manifest.kind must be 'run_manifest'")
    if source_manifest.sha256 is None:
        errors.append("next_experiment_plan.source_manifest.sha256 must be set")
    if source_manifest.provenance.get("run_id") != run_id:
        errors.append("next_experiment_plan.source_manifest provenance run_id must match plan.run_id")
    if "producer" not in source_manifest.provenance:
        errors.append("next_experiment_plan.source_manifest must set provenance.producer")
    try:
        manifest_path = local_artifact_path(
            path=source_manifest.path,
            uri=source_manifest.uri,
            field_name="source_manifest",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        manifest_path = None
    if manifest_path is None:
        errors.append("next_experiment_plan.source_manifest must be local byte-verifiable")
    elif not manifest_path.exists():
        errors.append("next_experiment_plan.source_manifest.path must exist")
    elif not manifest_path.is_file():
        errors.append("next_experiment_plan.source_manifest.path must point to a JSON manifest file")
    elif source_manifest.sha256 is not None:
        if sha256_file(manifest_path) != source_manifest.sha256:
            errors.append("next_experiment_plan.source_manifest.sha256 must match path")
        else:
            try:
                manifest = RunManifest.from_dict(read_json(manifest_path))
            except (HarnessIOError, SchemaValidationError) as exc:
                errors.append(f"next_experiment_plan.source_manifest must parse: {exc}")
            else:
                if manifest.run_id != run_id:
                    errors.append("source_manifest.run_id must match plan.run_id")
    return errors


def _duplicates(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
