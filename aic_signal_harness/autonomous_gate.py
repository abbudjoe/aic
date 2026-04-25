"""Offline autonomous gate contracts for trained policy eval preconditions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.candidate_registry import (
    CANDIDATE_POLICY_REGISTRY_KIND,
    CandidatePolicyRecord,
    read_candidate_policy_registry_artifact,
)
from aic_signal_harness.policy_training import (
    PolicyTrainingExecutionReport,
    PolicyTrainingExecutionStatus,
    PolicyTrainingRunReport,
)
from aic_signal_harness.reducers.policy_training import read_training_dataset_report_artifact
from aic_signal_harness.schemas import ArtifactRef, LeakageClass, SCHEMA_VERSION, SchemaValidationError


AUTONOMOUS_GATE_DECISION_KIND = "autonomous_training_gate_decision"
POLICY_TRAINING_REPORT_KIND = "policy_training_report"

_GATE_CONFIG_KEYS = frozenset(
    {
        "gate_id",
        "min_training_examples",
        "max_train_validation_loss_gap",
        "require_controlled_execution",
        "rejected_training_leakage_classes",
        "notes",
    }
)
_GATE_FAILURE_KEYS = frozenset({"kind", "summary", "evidence"})
_GATE_DECISION_KEYS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "gate_id",
        "verdict",
        "eval_preconditions_met",
        "autonomous_launch_allowed",
        "candidate_registry",
        "selected_candidate_id",
        "policy_training_report",
        "train_run_id",
        "source_training_dataset_report",
        "policy_artifact",
        "config",
        "failures",
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


class AutonomousGateVerdict(_StrEnum):
    """Training gate verdict before a live eval may be launched."""

    passed = "passed"
    blocked = "blocked"


class AutonomousGateFailureKind(_StrEnum):
    """Typed blocker categories for the training/eval preflight gate."""

    candidate_binding_mismatch = "candidate_binding_mismatch"
    checkpoint_stale = "checkpoint_stale"
    dataset_quality = "dataset_quality"
    execution_not_controlled = "execution_not_controlled"
    leakage_violation = "leakage_violation"
    overfitting_risk = "overfitting_risk"
    runtime_contract_violation = "runtime_contract_violation"
    timeout_risk = "timeout_risk"
    training_report_invalid = "training_report_invalid"


@dataclass(frozen=True)
class AutonomousGateConfig:
    """Deterministic thresholds for training evidence before live eval."""

    gate_id: str
    min_training_examples: int = 1
    max_train_validation_loss_gap: float | None = None
    require_controlled_execution: bool = True
    rejected_training_leakage_classes: tuple[LeakageClass, ...] = (
        LeakageClass.privileged_eval_signal,
        LeakageClass.post_hoc_label,
        LeakageClass.unknown,
    )
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "gate_id", _require_nonempty_text(self.gate_id, "gate config.gate_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "min_training_examples",
                _require_positive_int(
                    self.min_training_examples,
                    "gate config.min_training_examples",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.max_train_validation_loss_gap is not None:
            try:
                object.__setattr__(
                    self,
                    "max_train_validation_loss_gap",
                    _require_nonnegative_finite_float(
                        self.max_train_validation_loss_gap,
                        "gate config.max_train_validation_loss_gap",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if type(self.require_controlled_execution) is not bool:
            errors.append("gate config.require_controlled_execution must be a boolean")
        if self.require_controlled_execution is not True:
            errors.append("gate config.require_controlled_execution must be true")
        try:
            object.__setattr__(
                self,
                "rejected_training_leakage_classes",
                _leakage_tuple(
                    self.rejected_training_leakage_classes,
                    "gate config.rejected_training_leakage_classes",
                    allow_empty=True,
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "notes", _as_text_sequence(self.notes, "gate config.notes"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "gate_id": self.gate_id,
            "min_training_examples": self.min_training_examples,
            "require_controlled_execution": self.require_controlled_execution,
            "rejected_training_leakage_classes": [
                leakage.value for leakage in self.rejected_training_leakage_classes
            ],
            "notes": list(self.notes),
        }
        if self.max_train_validation_loss_gap is not None:
            value["max_train_validation_loss_gap"] = self.max_train_validation_loss_gap
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AutonomousGateConfig":
        if not isinstance(value, Mapping):
            raise HarnessIOError("gate config must be a mapping")
        _reject_unknown_keys(value, _GATE_CONFIG_KEYS, "gate config")
        _require_keys(value, {"gate_id"}, "gate config")
        return cls(
            gate_id=cast(Any, value.get("gate_id")),
            min_training_examples=cast(Any, value.get("min_training_examples", 1)),
            max_train_validation_loss_gap=cast(
                Any,
                value.get("max_train_validation_loss_gap"),
            ),
            require_controlled_execution=cast(
                Any,
                value.get("require_controlled_execution", True),
            ),
            rejected_training_leakage_classes=cast(
                Any,
                value.get(
                    "rejected_training_leakage_classes",
                    (
                        LeakageClass.privileged_eval_signal.value,
                        LeakageClass.post_hoc_label.value,
                        LeakageClass.unknown.value,
                    ),
                ),
            ),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class AutonomousGateFailure:
    """One blocking reason produced by the autonomous training gate."""

    kind: AutonomousGateFailureKind
    summary: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "kind", AutonomousGateFailureKind.parse(self.kind))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "summary",
                _require_nonempty_text(self.summary, "gate failure.summary"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "evidence",
                _copy_json_mapping(self.evidence, "gate failure.evidence"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "summary": self.summary,
            "evidence": _thaw_json(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AutonomousGateFailure":
        if not isinstance(value, Mapping):
            raise HarnessIOError("gate failure must be a mapping")
        _reject_unknown_keys(value, _GATE_FAILURE_KEYS, "gate failure")
        _require_keys(value, _GATE_FAILURE_KEYS - {"evidence"}, "gate failure")
        return cls(
            kind=cast(Any, value.get("kind")),
            summary=cast(Any, value.get("summary")),
            evidence=cast(Any, value.get("evidence", {})),
        )


@dataclass(frozen=True)
class AutonomousGateDecision:
    """Pure evidence gate output for one selected candidate and training run."""

    generated_at_utc: str
    gate_id: str
    verdict: AutonomousGateVerdict
    eval_preconditions_met: bool
    autonomous_launch_allowed: bool
    candidate_registry: ArtifactRef
    selected_candidate_id: str
    policy_training_report: ArtifactRef
    config: AutonomousGateConfig
    failures: tuple[AutonomousGateFailure, ...] = field(default_factory=tuple)
    train_run_id: str | None = None
    source_training_dataset_report: ArtifactRef | None = None
    policy_artifact: ArtifactRef | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("generated_at_utc", "gate_id", "selected_candidate_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"gate decision.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(self, "verdict", AutonomousGateVerdict.parse(self.verdict))
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("eval_preconditions_met", "autonomous_launch_allowed"):
            if type(getattr(self, field_name)) is not bool:
                errors.append(f"gate decision.{field_name} must be a boolean")
        for field_name in ("candidate_registry", "policy_training_report"):
            if not isinstance(getattr(self, field_name), ArtifactRef):
                try:
                    object.__setattr__(
                        self,
                        field_name,
                        ArtifactRef.from_dict(getattr(self, field_name)),
                    )
                except SchemaValidationError as exc:
                    errors.extend(exc.errors)
        if not isinstance(self.config, AutonomousGateConfig):
            try:
                object.__setattr__(self, "config", AutonomousGateConfig.from_dict(self.config))
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            failures = tuple(
                failure
                if isinstance(failure, AutonomousGateFailure)
                else AutonomousGateFailure.from_dict(failure)
                for failure in _as_sequence(self.failures, "gate decision.failures")
            )
            object.__setattr__(self, "failures", failures)
        except HarnessIOError as exc:
            errors.append(str(exc))
            failures = tuple()
        for field_name in ("source_training_dataset_report", "policy_artifact"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ArtifactRef):
                try:
                    object.__setattr__(self, field_name, ArtifactRef.from_dict(value))
                except SchemaValidationError as exc:
                    errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "train_run_id",
                _optional_nonempty_text(self.train_run_id, "gate decision.train_run_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "notes", _as_text_sequence(self.notes, "gate decision.notes"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_decision_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "gate_id": self.gate_id,
            "verdict": self.verdict.value,
            "eval_preconditions_met": self.eval_preconditions_met,
            "autonomous_launch_allowed": self.autonomous_launch_allowed,
            "candidate_registry": self.candidate_registry.to_dict(),
            "selected_candidate_id": self.selected_candidate_id,
            "policy_training_report": self.policy_training_report.to_dict(),
            "config": self.config.to_dict(),
            "failures": [failure.to_dict() for failure in self.failures],
            "notes": list(self.notes),
        }
        if self.train_run_id is not None:
            value["train_run_id"] = self.train_run_id
        if self.source_training_dataset_report is not None:
            value["source_training_dataset_report"] = self.source_training_dataset_report.to_dict()
        if self.policy_artifact is not None:
            value["policy_artifact"] = self.policy_artifact.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AutonomousGateDecision":
        if not isinstance(value, Mapping):
            raise HarnessIOError("gate decision must be a mapping")
        _reject_unknown_keys(value, _GATE_DECISION_KEYS, "gate decision")
        _require_keys(
            value,
            _GATE_DECISION_KEYS
            - {
                "failures",
                "notes",
                "policy_artifact",
                "source_training_dataset_report",
                "train_run_id",
            },
            "gate decision",
        )
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            gate_id=cast(Any, value.get("gate_id")),
            verdict=cast(Any, value.get("verdict")),
            eval_preconditions_met=cast(Any, value.get("eval_preconditions_met")),
            autonomous_launch_allowed=cast(Any, value.get("autonomous_launch_allowed")),
            candidate_registry=cast(Any, value.get("candidate_registry")),
            selected_candidate_id=cast(Any, value.get("selected_candidate_id")),
            policy_training_report=cast(Any, value.get("policy_training_report")),
            train_run_id=cast(Any, value.get("train_run_id")),
            source_training_dataset_report=cast(
                Any,
                value.get("source_training_dataset_report"),
            ),
            policy_artifact=cast(Any, value.get("policy_artifact")),
            config=cast(Any, value.get("config")),
            failures=cast(Any, value.get("failures", ())),
            notes=cast(Any, value.get("notes", ())),
        )


def evaluate_autonomous_training_gate(
    *,
    candidate_registry: ArtifactRef | Mapping[str, Any],
    policy_training_report: ArtifactRef | Mapping[str, Any],
    generated_at_utc: str,
    config: AutonomousGateConfig | Mapping[str, Any],
    notes: tuple[str, ...] = (),
) -> AutonomousGateDecision:
    """Evaluate offline training evidence before any live eval launch."""

    typed_config = config if isinstance(config, AutonomousGateConfig) else AutonomousGateConfig.from_dict(config)
    registry_artifact = (
        candidate_registry
        if isinstance(candidate_registry, ArtifactRef)
        else ArtifactRef.from_dict(candidate_registry)
    )
    training_artifact = (
        policy_training_report
        if isinstance(policy_training_report, ArtifactRef)
        else ArtifactRef.from_dict(policy_training_report)
    )
    registry = read_candidate_policy_registry_artifact(registry_artifact)
    candidate = registry.selected_candidate
    if candidate is None:
        raise HarnessIOError("candidate registry must select a candidate")
    selected_candidate_id = registry.selected_candidate_id
    assert selected_candidate_id is not None

    failures: list[AutonomousGateFailure] = []
    if candidate.eval.gate_id != typed_config.gate_id:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.candidate_binding_mismatch,
                summary="Selected candidate eval gate_id does not match autonomous gate config.",
                evidence={
                    "candidate_gate_id": candidate.eval.gate_id,
                    "gate_id": typed_config.gate_id,
                },
            )
        )
    report: PolicyTrainingRunReport | None = None
    try:
        report = read_policy_training_report_artifact(training_artifact)
    except HarnessIOError as exc:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.training_report_invalid,
                summary="Policy training report artifact did not validate.",
                evidence={"error": str(exc)},
            )
        )

    if report is not None:
        failures.extend(_candidate_binding_failures(candidate, report))
        failures.extend(_dataset_gate_failures(candidate, report, typed_config))
        failures.extend(_execution_gate_failures(candidate, report, typed_config))
        failures.extend(_runtime_gate_failures(report))
        failures.extend(_metric_gate_failures(report, typed_config))

    verdict = AutonomousGateVerdict.blocked if failures else AutonomousGateVerdict.passed
    return AutonomousGateDecision(
        generated_at_utc=generated_at_utc,
        gate_id=typed_config.gate_id,
        verdict=verdict,
        eval_preconditions_met=not failures,
        autonomous_launch_allowed=False,
        candidate_registry=registry_artifact,
        selected_candidate_id=selected_candidate_id,
        policy_training_report=training_artifact,
        train_run_id=None if report is None else report.run_id,
        source_training_dataset_report=(
            None if report is None else report.source_training_dataset_report
        ),
        policy_artifact=None if report is None else report.policy_artifact,
        config=typed_config,
        failures=tuple(failures),
        notes=notes
        + (
            "Gate certifies offline eval preconditions only; autonomous launch remains disabled until the orchestrator contract enables it.",
        ),
    )


def read_policy_training_report_artifact(
    artifact: ArtifactRef | Mapping[str, Any],
) -> PolicyTrainingRunReport:
    """Read and validate a local policy-training report artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != POLICY_TRAINING_REPORT_KIND:
        errors.append(f"policy training report artifact.kind must be {POLICY_TRAINING_REPORT_KIND!r}")
    if typed_artifact.sha256 is None:
        errors.append("policy training report artifact.sha256 must be set")
    for provenance_key in ("producer", "derivation"):
        if provenance_key not in typed_artifact.provenance:
            errors.append(f"policy training report artifact provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"policy training report artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    try:
        report_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="policy training report artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        report_path = None
    if report_path is None:
        errors.append("policy training report artifact must be local byte-verifiable")
    elif not report_path.exists():
        errors.append("policy training report artifact.path must exist")
    elif not report_path.is_file():
        errors.append("policy training report artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(report_path) != typed_artifact.sha256:
        errors.append("policy training report artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert report_path is not None
    report = PolicyTrainingRunReport.from_dict(read_json(report_path))
    if typed_artifact.provenance.get("run_id") not in (None, report.run_id):
        raise HarnessIOError("policy training report artifact provenance run_id must match report.run_id")
    return report


def write_autonomous_gate_decision(
    path: str | Path,
    decision: AutonomousGateDecision | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ArtifactRef:
    """Write an autonomous gate decision and return its artifact reference."""

    typed_decision = (
        decision if isinstance(decision, AutonomousGateDecision) else AutonomousGateDecision.from_dict(decision)
    )
    output_path = Path(path).expanduser().resolve(strict=False)
    write_json(output_path, typed_decision.to_dict(), overwrite=overwrite)
    return ArtifactRef(
        kind=AUTONOMOUS_GATE_DECISION_KIND,
        path=str(output_path),
        sha256=sha256_file(output_path),
        provenance={
            "producer": "aic_signal_harness.autonomous_gate",
            "derivation": "write_autonomous_gate_decision",
            "gate_id": typed_decision.gate_id,
            "verdict": typed_decision.verdict.value,
            "selected_candidate_id": typed_decision.selected_candidate_id,
        },
    )


def read_autonomous_gate_decision_artifact(
    artifact: ArtifactRef | Mapping[str, Any],
) -> AutonomousGateDecision:
    """Read and validate a local autonomous-gate decision artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != AUTONOMOUS_GATE_DECISION_KIND:
        errors.append(f"autonomous gate decision artifact.kind must be {AUTONOMOUS_GATE_DECISION_KIND!r}")
    if typed_artifact.sha256 is None:
        errors.append("autonomous gate decision artifact.sha256 must be set")
    for provenance_key in ("producer", "derivation", "gate_id", "verdict", "selected_candidate_id"):
        if provenance_key not in typed_artifact.provenance:
            errors.append(f"autonomous gate decision artifact provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"autonomous gate decision artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if typed_artifact.provenance.get("producer") != "aic_signal_harness.autonomous_gate":
        errors.append(
            "autonomous gate decision artifact provenance producer must be "
            "'aic_signal_harness.autonomous_gate'"
        )
    if typed_artifact.provenance.get("derivation") != "write_autonomous_gate_decision":
        errors.append(
            "autonomous gate decision artifact provenance derivation must be "
            "'write_autonomous_gate_decision'"
        )
    try:
        decision_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="autonomous gate decision artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        decision_path = None
    if decision_path is None:
        errors.append("autonomous gate decision artifact must be local byte-verifiable")
    elif not decision_path.exists():
        errors.append("autonomous gate decision artifact.path must exist")
    elif not decision_path.is_file():
        errors.append("autonomous gate decision artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(decision_path) != typed_artifact.sha256:
        errors.append("autonomous gate decision artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert decision_path is not None
    decision = AutonomousGateDecision.from_dict(read_json(decision_path))
    if typed_artifact.provenance.get("gate_id") != decision.gate_id:
        raise HarnessIOError("autonomous gate decision artifact provenance gate_id must match")
    if typed_artifact.provenance.get("verdict") != decision.verdict.value:
        raise HarnessIOError("autonomous gate decision artifact provenance verdict must match")
    if typed_artifact.provenance.get("selected_candidate_id") != decision.selected_candidate_id:
        raise HarnessIOError(
            "autonomous gate decision artifact provenance selected_candidate_id must match"
        )
    return decision


def _candidate_binding_failures(
    candidate: CandidatePolicyRecord,
    report: PolicyTrainingRunReport,
) -> list[AutonomousGateFailure]:
    failures: list[AutonomousGateFailure] = []
    if report.source_training_dataset_report.sha256 != candidate.source_training_dataset_report.sha256:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.candidate_binding_mismatch,
                summary="Training report source dataset does not match the selected candidate.",
                evidence={
                    "candidate_dataset_sha256": candidate.source_training_dataset_report.sha256,
                    "training_report_dataset_sha256": report.source_training_dataset_report.sha256,
                },
            )
        )
    trainer_mismatches = _trainer_mismatch_fields(candidate, report)
    if trainer_mismatches:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.candidate_binding_mismatch,
                summary="Training report trainer does not match the selected candidate trainer.",
                evidence={"fields": trainer_mismatches},
            )
        )
    if report.candidate_policy.backend_kind is not candidate.policy_template.backend_kind:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.candidate_binding_mismatch,
                summary="Trained policy backend does not match the selected candidate template.",
                evidence={
                    "candidate_backend": candidate.policy_template.backend_kind.value,
                    "training_backend": report.candidate_policy.backend_kind.value,
                },
            )
        )
    if report.candidate_policy.config.get("planner_mode") != candidate.eval.planner_mode:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.candidate_binding_mismatch,
                summary="Trained policy planner_mode does not match candidate eval planner_mode.",
                evidence={
                    "candidate_planner_mode": candidate.eval.planner_mode,
                    "training_planner_mode": report.candidate_policy.config.get("planner_mode"),
                },
            )
        )
    if report.policy_artifact.kind != candidate.checkpoint.artifact_kind:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.checkpoint_stale,
                summary="Policy artifact kind does not match candidate checkpoint expectation.",
                evidence={
                    "expected_kind": candidate.checkpoint.artifact_kind,
                    "actual_kind": report.policy_artifact.kind,
                },
            )
        )
    checkpoint_path = local_artifact_path(
        path=report.policy_artifact.path,
        uri=report.policy_artifact.uri,
        field_name="autonomous gate policy_artifact",
    )
    if checkpoint_path is not None and checkpoint_path.name != candidate.checkpoint.output_name:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.checkpoint_stale,
                summary="Policy checkpoint file name does not match candidate checkpoint expectation.",
                evidence={
                    "expected_output_name": candidate.checkpoint.output_name,
                    "actual_output_name": checkpoint_path.name,
                },
            )
        )
    if (
        candidate.checkpoint.required_sha256 is not None
        and report.policy_artifact.sha256 != candidate.checkpoint.required_sha256
    ):
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.checkpoint_stale,
                summary="Policy checkpoint sha256 does not match candidate checkpoint expectation.",
                evidence={
                    "expected_sha256": candidate.checkpoint.required_sha256,
                    "actual_sha256": report.policy_artifact.sha256,
                },
            )
        )
    return failures


def _dataset_gate_failures(
    candidate: CandidatePolicyRecord,
    report: PolicyTrainingRunReport,
    config: AutonomousGateConfig,
) -> list[AutonomousGateFailure]:
    failures: list[AutonomousGateFailure] = []
    dataset = read_training_dataset_report_artifact(report.source_training_dataset_report)
    if dataset.example_count < config.min_training_examples:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.dataset_quality,
                summary="Training dataset has too few examples for the configured gate.",
                evidence={
                    "example_count": dataset.example_count,
                    "min_training_examples": config.min_training_examples,
                },
            )
        )
    present_leakage = {
        LeakageClass.parse(leakage_class)
        for leakage_class, count in dataset.signals_by_leakage_class.items()
        if count > 0
    }
    rejected = sorted(
        leakage.value
        for leakage in present_leakage
        if leakage in set(config.rejected_training_leakage_classes)
    )
    outside_candidate = sorted(
        leakage.value
        for leakage in present_leakage
        if leakage not in set(candidate.safety.allowed_training_leakage_classes)
    )
    if rejected or outside_candidate:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.leakage_violation,
                summary="Training dataset leakage classes are not allowed for this candidate gate.",
                evidence={
                    "rejected_training_leakage_classes": rejected,
                    "outside_candidate_allowlist": outside_candidate,
                },
            )
        )
    return failures


def _execution_gate_failures(
    candidate: CandidatePolicyRecord,
    report: PolicyTrainingRunReport,
    config: AutonomousGateConfig,
) -> list[AutonomousGateFailure]:
    failures: list[AutonomousGateFailure] = []
    if report.policy_training_execution_report is None:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.execution_not_controlled,
                summary="Training report lacks a controlled policy_training_execution_report.",
                evidence={"require_controlled_execution": config.require_controlled_execution},
            )
        )
        return failures
    execution = _read_execution_report_artifact(report.policy_training_execution_report)
    if execution.status is not PolicyTrainingExecutionStatus.completed or execution.ok is not True:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.execution_not_controlled,
                summary="Policy training execution report is not completed and ok.",
                evidence={"status": execution.status.value, "ok": execution.ok},
            )
        )
    if execution.timeout_seconds > candidate.safety.max_training_runtime_seconds:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.timeout_risk,
                summary="Policy training timeout exceeds the candidate safety contract.",
                evidence={
                    "timeout_seconds": execution.timeout_seconds,
                    "max_training_runtime_seconds": candidate.safety.max_training_runtime_seconds,
                },
            )
        )
    return failures


def _runtime_gate_failures(report: PolicyTrainingRunReport) -> list[AutonomousGateFailure]:
    failures: list[AutonomousGateFailure] = []
    if report.offline_only is not True or report.runtime_allowed is not False:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.runtime_contract_violation,
                summary="Policy training report must remain offline-only and not runtime-allowed.",
                evidence={
                    "offline_only": report.offline_only,
                    "runtime_allowed": report.runtime_allowed,
                },
            )
        )
    if report.autonomous_launch_allowed is not False:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.runtime_contract_violation,
                summary="Policy training report may not grant autonomous launch permission.",
                evidence={"autonomous_launch_allowed": report.autonomous_launch_allowed},
            )
        )
    if (
        report.candidate_policy.runtime_boundary is not None
        and report.candidate_policy.runtime_boundary.uses_online_language_model_control is not False
    ):
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.runtime_contract_violation,
                summary="Trained live policy must not use online language-model control.",
                evidence={"uses_online_language_model_control": True},
            )
        )
    return failures


def _metric_gate_failures(
    report: PolicyTrainingRunReport,
    config: AutonomousGateConfig,
) -> list[AutonomousGateFailure]:
    failures: list[AutonomousGateFailure] = []
    if config.max_train_validation_loss_gap is None:
        return failures
    train_loss = report.metrics.get("train.loss")
    validation_loss = report.metrics.get("validation.loss")
    if train_loss is None or validation_loss is None:
        missing_metrics = sorted(
            metric_name
            for metric_name, metric_value in (
                ("train.loss", train_loss),
                ("validation.loss", validation_loss),
            )
            if metric_value is None
        )
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.overfitting_risk,
                summary="Configured overfitting gate is missing required metrics.",
                evidence={
                    "missing_metrics": missing_metrics,
                    "max_train_validation_loss_gap": config.max_train_validation_loss_gap,
                },
            )
        )
        return failures
    gap = validation_loss - train_loss
    if gap > config.max_train_validation_loss_gap:
        failures.append(
            AutonomousGateFailure(
                kind=AutonomousGateFailureKind.overfitting_risk,
                summary="Validation loss gap exceeds the configured overfitting gate.",
                evidence={
                    "train.loss": train_loss,
                    "validation.loss": validation_loss,
                    "gap": gap,
                    "max_train_validation_loss_gap": config.max_train_validation_loss_gap,
                },
            )
        )
    return failures


def _read_execution_report_artifact(artifact: ArtifactRef) -> PolicyTrainingExecutionReport:
    path = local_artifact_path(
        path=artifact.path,
        uri=artifact.uri,
        field_name="autonomous gate policy_training_execution_report",
    )
    if path is None:
        raise HarnessIOError("policy_training_execution_report must be local byte-verifiable")
    if not path.exists():
        raise HarnessIOError("policy_training_execution_report.path must exist")
    if not path.is_file():
        raise HarnessIOError("policy_training_execution_report.path must be a file")
    if artifact.sha256 is None:
        raise HarnessIOError("policy_training_execution_report.sha256 must be set")
    if sha256_file(path) != artifact.sha256:
        raise HarnessIOError("policy_training_execution_report.sha256 must match path")
    return PolicyTrainingExecutionReport.from_dict(read_json(path))


def _trainer_mismatch_fields(
    candidate: CandidatePolicyRecord,
    report: PolicyTrainingRunReport,
) -> tuple[str, ...]:
    candidate_trainer = candidate.trainer
    report_trainer = report.trainer
    mismatches: list[str] = []
    for field_name in (
        "trainer_id",
        "trainer_name",
        "backend_kind",
        "command",
        "config",
        "environment",
        "container_image",
        "seed",
        "offline_only",
        "runtime_allowed",
        "uses_online_language_model_control",
    ):
        if getattr(candidate_trainer, field_name) != getattr(report_trainer, field_name):
            mismatches.append(field_name)
    if (
        candidate_trainer.working_directory is not None
        and candidate_trainer.working_directory != report_trainer.working_directory
    ):
        mismatches.append("working_directory")
    return tuple(mismatches)


def _decision_consistency_errors(decision: AutonomousGateDecision) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _artifact_shape_errors(
            decision.candidate_registry,
            expected_kind=CANDIDATE_POLICY_REGISTRY_KIND,
            field_name="gate decision.candidate_registry",
        )
    )
    errors.extend(
        _artifact_shape_errors(
            decision.policy_training_report,
            expected_kind=POLICY_TRAINING_REPORT_KIND,
            field_name="gate decision.policy_training_report",
        )
    )
    if decision.source_training_dataset_report is not None:
        errors.extend(
            _artifact_shape_errors(
                decision.source_training_dataset_report,
                expected_kind="training_dataset_report",
                field_name="gate decision.source_training_dataset_report",
            )
        )
    if decision.policy_artifact is not None:
        errors.extend(
            _artifact_shape_errors(
                decision.policy_artifact,
                expected_kind="policy_checkpoint",
                field_name="gate decision.policy_artifact",
            )
        )
    try:
        registry = read_candidate_policy_registry_artifact(decision.candidate_registry)
    except HarnessIOError as exc:
        errors.append("gate decision.candidate_registry must validate: " + str(exc))
        registry = None
    if registry is not None and registry.selected_candidate_id != decision.selected_candidate_id:
        errors.append("gate decision.selected_candidate_id must match candidate_registry")
    if registry is not None:
        candidate = registry.selected_candidate
        if candidate is None:
            errors.append("gate decision.candidate_registry must select a candidate")
        elif candidate.eval.gate_id != decision.config.gate_id and (
            decision.verdict is AutonomousGateVerdict.passed
            or not any(
                failure.kind is AutonomousGateFailureKind.candidate_binding_mismatch
                for failure in decision.failures
            )
        ):
            errors.append("gate decision candidate eval.gate_id must match config.gate_id")
    if decision.gate_id != decision.config.gate_id:
        errors.append("gate decision.gate_id must match config.gate_id")
    if decision.autonomous_launch_allowed is not False:
        errors.append("gate decision.autonomous_launch_allowed must remain false before orchestrator")
    if decision.verdict is AutonomousGateVerdict.passed:
        if not decision.eval_preconditions_met:
            errors.append("passed gate decisions must set eval_preconditions_met=true")
        if decision.failures:
            errors.append("passed gate decisions must not include failures")
        if decision.train_run_id is None:
            errors.append("passed gate decisions must set train_run_id")
        if decision.source_training_dataset_report is None:
            errors.append("passed gate decisions must set source_training_dataset_report")
        if decision.policy_artifact is None:
            errors.append("passed gate decisions must set policy_artifact")
        try:
            report = read_policy_training_report_artifact(decision.policy_training_report)
        except HarnessIOError as exc:
            errors.append("passed gate decision.policy_training_report must validate: " + str(exc))
        else:
            if decision.train_run_id != report.run_id:
                errors.append("passed gate decision.train_run_id must match policy_training_report")
            if decision.source_training_dataset_report != report.source_training_dataset_report:
                errors.append(
                    "passed gate decision.source_training_dataset_report must match policy_training_report"
                )
            if decision.policy_artifact != report.policy_artifact:
                errors.append("passed gate decision.policy_artifact must match policy_training_report")
    if decision.verdict is AutonomousGateVerdict.blocked:
        if decision.eval_preconditions_met:
            errors.append("blocked gate decisions must set eval_preconditions_met=false")
        if not decision.failures:
            errors.append("blocked gate decisions must include at least one failure")
    return errors


def _artifact_shape_errors(
    artifact: ArtifactRef,
    *,
    expected_kind: str,
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if artifact.kind != expected_kind:
        errors.append(f"{field_name}.kind must be {expected_kind!r}")
    if artifact.sha256 is None:
        errors.append(f"{field_name}.sha256 must be set")
    for provenance_key in ("producer", "derivation"):
        if provenance_key not in artifact.provenance:
            errors.append(f"{field_name} provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    artifact.provenance.get(provenance_key),
                    f"{field_name} provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    try:
        path = local_artifact_path(path=artifact.path, uri=artifact.uri, field_name=field_name)
    except HarnessIOError as exc:
        errors.append(str(exc))
        path = None
    if path is None:
        errors.append(f"{field_name} must be local byte-verifiable")
    elif not path.exists():
        errors.append(f"{field_name}.path must exist")
    elif not path.is_file():
        errors.append(f"{field_name}.path must be a file")
    elif artifact.sha256 is not None and sha256_file(path) != artifact.sha256:
        errors.append(f"{field_name}.sha256 must match path")
    return errors


def _require_keys(value: Mapping[str, Any], required_keys: set[str] | frozenset[str], field_name: str) -> None:
    missing = sorted(key for key in required_keys if key not in value)
    if missing:
        raise HarnessIOError(f"{field_name} missing required fields: {', '.join(missing)}")


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
    return value.strip()


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_positive_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise HarnessIOError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessIOError(f"{field_name} must be a finite number")
    number = float(value)
    if number < 0.0:
        raise HarnessIOError(f"{field_name} must be >= 0")
    return number


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    items = _as_sequence(value, field_name)
    return tuple(_require_nonempty_text(item, f"{field_name}[{index}]") for index, item in enumerate(items))


def _leakage_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[LeakageClass, ...]:
    values = _as_sequence(value, field_name)
    if not values and not allow_empty:
        raise HarnessIOError(f"{field_name} must not be empty")
    parsed: list[LeakageClass] = []
    for item in values:
        try:
            parsed.append(LeakageClass.parse(item))
        except SchemaValidationError as exc:
            raise HarnessIOError(str(exc)) from exc
    duplicates = sorted({item.value for item in parsed if parsed.count(item) > 1})
    if duplicates:
        raise HarnessIOError(f"{field_name} must not contain duplicates: {', '.join(duplicates)}")
    return tuple(parsed)


def _copy_json_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
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
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise HarnessIOError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
