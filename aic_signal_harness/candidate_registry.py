"""Candidate policy registry contracts for the AIC outer loop."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.constants import DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH
from aic_signal_harness.policy_training import TrainerInvocation
from aic_signal_harness.policy_training_runner import (
    CandidatePolicyTemplate,
)
from aic_signal_harness.next_experiment_plan import NextExperimentPlan
from aic_signal_harness.reducers.policy_training import read_training_dataset_report_artifact
from aic_signal_harness.schemas import (
    ArtifactRef,
    LeakageClass,
    SCHEMA_VERSION,
    SchemaValidationError,
)


CANDIDATE_POLICY_REGISTRY_KIND = "candidate_policy_registry"

_CHECKPOINT_EXPECTATION_KEYS = frozenset(
    {
        "artifact_kind",
        "output_name",
        "runtime_container_path",
        "required_sha256",
        "notes",
    }
)
_EVAL_SPEC_KEYS = frozenset(
    {
        "mode",
        "gate_id",
        "planner_mode",
        "min_improvement",
        "bootstrap_promotion",
        "eligible_for_submission",
        "notes",
    }
)
_SAFETY_CONTRACT_KEYS = frozenset(
    {
        "allowed_training_leakage_classes",
        "max_training_runtime_seconds",
        "max_eval_runtime_seconds",
        "require_policy_trace",
        "autonomous_launch_allowed",
        "notes",
    }
)
_CANDIDATE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "fingerprint_sha256",
        "generated_at_utc",
        "source_training_dataset_report",
        "trainer",
        "policy_template",
        "checkpoint",
        "eval",
        "safety",
        "source_next_experiment_plan",
        "source_plan_candidate_id",
        "notes",
    }
)
_REGISTRY_KEYS = frozenset(
    {
        "schema_version",
        "registry_id",
        "generated_at_utc",
        "candidates",
        "selected_candidate_id",
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


class CandidateEvalMode(_StrEnum):
    """Execution target expected for a candidate policy."""

    local_live_eval = "local_live_eval"
    gcp_live_eval = "gcp_live_eval"
    dry_run = "dry_run"


@dataclass(frozen=True)
class CandidateCheckpointExpectation:
    """Expected checkpoint artifact shape before training produces bytes."""

    artifact_kind: str = "policy_checkpoint"
    output_name: str = "policy.ckpt"
    runtime_container_path: str = DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH
    required_sha256: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.artifact_kind != "policy_checkpoint":
            errors.append("candidate checkpoint artifact_kind must be 'policy_checkpoint'")
        try:
            object.__setattr__(
                self,
                "output_name",
                _require_output_name(self.output_name, "candidate checkpoint.output_name"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "runtime_container_path",
                _require_absolute_container_path(
                    self.runtime_container_path,
                    "candidate checkpoint.runtime_container_path",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.required_sha256 is not None:
            try:
                object.__setattr__(
                    self,
                    "required_sha256",
                    _require_sha256(self.required_sha256, "candidate checkpoint.required_sha256"),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "candidate checkpoint.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "artifact_kind": self.artifact_kind,
            "output_name": self.output_name,
            "runtime_container_path": self.runtime_container_path,
            "notes": list(self.notes),
        }
        if self.required_sha256 is not None:
            value["required_sha256"] = self.required_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateCheckpointExpectation":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate checkpoint must be a mapping")
        _reject_unknown_keys(value, _CHECKPOINT_EXPECTATION_KEYS, "candidate checkpoint")
        return cls(
            artifact_kind=cast(str, value.get("artifact_kind", "policy_checkpoint")),
            output_name=cast(str, value.get("output_name", "policy.ckpt")),
            runtime_container_path=cast(
                str,
                value.get(
                    "runtime_container_path",
                    DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH,
                ),
            ),
            required_sha256=cast(str | None, value.get("required_sha256")),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class CandidateEvalSpec:
    """Expected eval-mode knobs for one candidate."""

    mode: CandidateEvalMode
    gate_id: str
    planner_mode: str
    min_improvement: float = 0.0
    bootstrap_promotion: bool = False
    eligible_for_submission: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "mode", CandidateEvalMode.parse(self.mode))
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("gate_id", "planner_mode"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"candidate eval.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "min_improvement",
                _require_nonnegative_finite_float(
                    self.min_improvement,
                    "candidate eval.min_improvement",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("bootstrap_promotion", "eligible_for_submission"):
            if type(getattr(self, field_name)) is not bool:
                errors.append(f"candidate eval.{field_name} must be a boolean")
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "candidate eval.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "gate_id": self.gate_id,
            "planner_mode": self.planner_mode,
            "min_improvement": self.min_improvement,
            "bootstrap_promotion": self.bootstrap_promotion,
            "eligible_for_submission": self.eligible_for_submission,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateEvalSpec":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate eval must be a mapping")
        _reject_unknown_keys(value, _EVAL_SPEC_KEYS, "candidate eval")
        _require_keys(
            value,
            _EVAL_SPEC_KEYS
            - {
                "bootstrap_promotion",
                "eligible_for_submission",
                "min_improvement",
                "notes",
            },
            "candidate eval",
        )
        return cls(
            mode=cast(Any, value.get("mode")),
            gate_id=cast(Any, value.get("gate_id")),
            planner_mode=cast(Any, value.get("planner_mode")),
            min_improvement=cast(Any, value.get("min_improvement", 0.0)),
            bootstrap_promotion=cast(bool, value.get("bootstrap_promotion", False)),
            eligible_for_submission=cast(bool, value.get("eligible_for_submission", False)),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class CandidateSafetyContract:
    """Training and eval safety constraints for a candidate."""

    allowed_training_leakage_classes: tuple[LeakageClass, ...] = (
        LeakageClass.legal_policy_input,
        LeakageClass.legal_policy_action_output,
    )
    max_training_runtime_seconds: float = 3600.0
    max_eval_runtime_seconds: float = 1800.0
    require_policy_trace: bool = True
    autonomous_launch_allowed: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "allowed_training_leakage_classes",
                _leakage_tuple(
                    self.allowed_training_leakage_classes,
                    "candidate safety.allowed_training_leakage_classes",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("max_training_runtime_seconds", "max_eval_runtime_seconds"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_positive_finite_float(
                        getattr(self, field_name),
                        f"candidate safety.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in ("require_policy_trace", "autonomous_launch_allowed"):
            if type(getattr(self, field_name)) is not bool:
                errors.append(f"candidate safety.{field_name} must be a boolean")
        if self.autonomous_launch_allowed is not False:
            errors.append("candidate safety.autonomous_launch_allowed must be false before orchestrator")
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "candidate safety.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_training_leakage_classes": [
                leakage.value for leakage in self.allowed_training_leakage_classes
            ],
            "max_training_runtime_seconds": self.max_training_runtime_seconds,
            "max_eval_runtime_seconds": self.max_eval_runtime_seconds,
            "require_policy_trace": self.require_policy_trace,
            "autonomous_launch_allowed": self.autonomous_launch_allowed,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateSafetyContract":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate safety must be a mapping")
        _reject_unknown_keys(value, _SAFETY_CONTRACT_KEYS, "candidate safety")
        return cls(
            allowed_training_leakage_classes=cast(
                Any,
                value.get(
                    "allowed_training_leakage_classes",
                    (
                        LeakageClass.legal_policy_input.value,
                        LeakageClass.legal_policy_action_output.value,
                    ),
                ),
            ),
            max_training_runtime_seconds=cast(
                Any,
                value.get("max_training_runtime_seconds", 3600.0),
            ),
            max_eval_runtime_seconds=cast(Any, value.get("max_eval_runtime_seconds", 1800.0)),
            require_policy_trace=cast(bool, value.get("require_policy_trace", True)),
            autonomous_launch_allowed=cast(bool, value.get("autonomous_launch_allowed", False)),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class CandidatePolicyRecord:
    """One executable candidate identity before a checkpoint is trained."""

    generated_at_utc: str
    source_training_dataset_report: ArtifactRef
    trainer: TrainerInvocation
    policy_template: CandidatePolicyTemplate
    eval: CandidateEvalSpec
    candidate_id: str | None = None
    fingerprint_sha256: str | None = None
    checkpoint: CandidateCheckpointExpectation = field(default_factory=CandidateCheckpointExpectation)
    safety: CandidateSafetyContract = field(default_factory=CandidateSafetyContract)
    source_next_experiment_plan: ArtifactRef | None = None
    source_plan_candidate_id: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "generated_at_utc",
                _require_nonempty_text(self.generated_at_utc, "candidate.generated_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not isinstance(self.source_training_dataset_report, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "source_training_dataset_report",
                    ArtifactRef.from_dict(self.source_training_dataset_report),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        if not isinstance(self.trainer, TrainerInvocation):
            try:
                object.__setattr__(self, "trainer", TrainerInvocation.from_dict(self.trainer))
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.policy_template, CandidatePolicyTemplate):
            try:
                object.__setattr__(
                    self,
                    "policy_template",
                    CandidatePolicyTemplate.from_dict(self.policy_template),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.checkpoint, CandidateCheckpointExpectation):
            try:
                object.__setattr__(
                    self,
                    "checkpoint",
                    CandidateCheckpointExpectation.from_dict(self.checkpoint),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.eval, CandidateEvalSpec):
            try:
                object.__setattr__(self, "eval", CandidateEvalSpec.from_dict(self.eval))
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.safety, CandidateSafetyContract):
            try:
                object.__setattr__(
                    self,
                    "safety",
                    CandidateSafetyContract.from_dict(self.safety),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if self.source_next_experiment_plan is not None and not isinstance(
            self.source_next_experiment_plan,
            ArtifactRef,
        ):
            try:
                object.__setattr__(
                    self,
                    "source_next_experiment_plan",
                    ArtifactRef.from_dict(self.source_next_experiment_plan),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "source_plan_candidate_id",
                _optional_nonempty_text(
                    self.source_plan_candidate_id,
                    "candidate.source_plan_candidate_id",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "notes", _as_text_sequence(self.notes, "candidate.notes"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_candidate_record_consistency_errors(self))
            fingerprint = candidate_policy_fingerprint_sha256(self)
            expected_id = candidate_policy_record_id_from_fingerprint(fingerprint)
            if self.fingerprint_sha256 is None:
                object.__setattr__(self, "fingerprint_sha256", fingerprint)
            elif self.fingerprint_sha256 != fingerprint:
                errors.append("candidate.fingerprint_sha256 must match candidate content")
            if self.candidate_id is None:
                object.__setattr__(self, "candidate_id", expected_id)
            elif self.candidate_id != expected_id:
                errors.append("candidate.candidate_id must be deterministic from fingerprint")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "fingerprint_sha256": self.fingerprint_sha256,
            "generated_at_utc": self.generated_at_utc,
            "source_training_dataset_report": self.source_training_dataset_report.to_dict(),
            "trainer": self.trainer.to_dict(),
            "policy_template": self.policy_template.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "eval": self.eval.to_dict(),
            "safety": self.safety.to_dict(),
            "notes": list(self.notes),
        }
        if self.source_next_experiment_plan is not None:
            value["source_next_experiment_plan"] = self.source_next_experiment_plan.to_dict()
        if self.source_plan_candidate_id is not None:
            value["source_plan_candidate_id"] = self.source_plan_candidate_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidatePolicyRecord":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate policy record must be a mapping")
        _reject_unknown_keys(value, _CANDIDATE_RECORD_KEYS, "candidate policy record")
        _require_keys(
            value,
            _CANDIDATE_RECORD_KEYS
            - {
                "candidate_id",
                "checkpoint",
                "fingerprint_sha256",
                "notes",
                "source_next_experiment_plan",
                "source_plan_candidate_id",
            },
            "candidate policy record",
        )
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            candidate_id=cast(str | None, value.get("candidate_id")),
            fingerprint_sha256=cast(str | None, value.get("fingerprint_sha256")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_training_dataset_report=cast(Any, value.get("source_training_dataset_report")),
            trainer=cast(Any, value.get("trainer")),
            policy_template=cast(Any, value.get("policy_template")),
            checkpoint=cast(Any, value.get("checkpoint", {})),
            eval=cast(Any, value.get("eval")),
            safety=cast(Any, value.get("safety", {})),
            source_next_experiment_plan=cast(
                Any,
                value.get("source_next_experiment_plan"),
            ),
            source_plan_candidate_id=cast(str | None, value.get("source_plan_candidate_id")),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class CandidatePolicyRegistry:
    """A deterministic registry of candidate policies for one outer-loop choice point."""

    registry_id: str
    generated_at_utc: str
    candidates: tuple[CandidatePolicyRecord, ...]
    selected_candidate_id: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("registry_id", "generated_at_utc"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"candidate registry.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            candidates = tuple(
                candidate
                if isinstance(candidate, CandidatePolicyRecord)
                else CandidatePolicyRecord.from_dict(candidate)
                for candidate in _as_sequence(self.candidates, "candidate registry.candidates")
            )
            object.__setattr__(self, "candidates", candidates)
        except HarnessIOError as exc:
            errors.append(str(exc))
            candidates = tuple()
        try:
            object.__setattr__(
                self,
                "selected_candidate_id",
                _optional_nonempty_text(
                    self.selected_candidate_id,
                    "candidate registry.selected_candidate_id",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "candidate registry.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_registry_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "generated_at_utc": self.generated_at_utc,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "notes": list(self.notes),
        }
        if self.selected_candidate_id is not None:
            value["selected_candidate_id"] = self.selected_candidate_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidatePolicyRegistry":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate policy registry must be a mapping")
        _reject_unknown_keys(value, _REGISTRY_KEYS, "candidate policy registry")
        _require_keys(
            value,
            _REGISTRY_KEYS - {"notes", "selected_candidate_id"},
            "candidate policy registry",
        )
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            registry_id=cast(Any, value.get("registry_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            candidates=cast(Any, value.get("candidates")),
            selected_candidate_id=cast(str | None, value.get("selected_candidate_id")),
            notes=cast(Any, value.get("notes", ())),
        )

    @property
    def selected_candidate(self) -> CandidatePolicyRecord | None:
        if self.selected_candidate_id is None:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == self.selected_candidate_id:
                return candidate
        raise HarnessIOError("candidate registry.selected_candidate_id must exist")


def candidate_policy_fingerprint_sha256(candidate: CandidatePolicyRecord) -> str:
    """Return a stable fingerprint for candidate identity fields."""

    payload = {
        "source_training_dataset_report": candidate.source_training_dataset_report.to_dict(),
        "trainer": candidate.trainer.to_dict(),
        "policy_template": candidate.policy_template.to_dict(),
        "checkpoint": candidate.checkpoint.to_dict(),
        "eval": candidate.eval.to_dict(),
        "safety": candidate.safety.to_dict(),
        "source_next_experiment_plan": (
            None
            if candidate.source_next_experiment_plan is None
            else candidate.source_next_experiment_plan.to_dict()
        ),
        "source_plan_candidate_id": candidate.source_plan_candidate_id,
    }
    return _stable_json_sha256(payload)


def candidate_policy_record_id_from_fingerprint(fingerprint_sha256: str) -> str:
    return "cand_" + _require_sha256(fingerprint_sha256, "candidate fingerprint")[:20]


def write_candidate_policy_registry(
    path: str | Path,
    registry: CandidatePolicyRegistry | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ArtifactRef:
    """Write a registry and return its local artifact reference."""

    typed_registry = (
        registry if isinstance(registry, CandidatePolicyRegistry) else CandidatePolicyRegistry.from_dict(registry)
    )
    if typed_registry.selected_candidate_id is None:
        raise HarnessIOError("candidate policy registry artifact requires selected_candidate_id")
    output_path = Path(path).expanduser().resolve(strict=False)
    write_json(output_path, typed_registry.to_dict(), overwrite=overwrite)
    return ArtifactRef(
        kind=CANDIDATE_POLICY_REGISTRY_KIND,
        path=str(output_path),
        sha256=sha256_file(output_path),
        provenance={
            "producer": "aic_signal_harness.candidate_registry",
            "derivation": "write_candidate_policy_registry",
            "registry_id": typed_registry.registry_id,
            "selected_candidate_id": typed_registry.selected_candidate_id,
        },
    )


def read_candidate_policy_registry_artifact(artifact: ArtifactRef | Mapping[str, Any]) -> CandidatePolicyRegistry:
    """Read and validate a local candidate-policy registry artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != CANDIDATE_POLICY_REGISTRY_KIND:
        errors.append(
            f"candidate policy registry artifact.kind must be {CANDIDATE_POLICY_REGISTRY_KIND!r}"
        )
    if typed_artifact.sha256 is None:
        errors.append("candidate policy registry artifact.sha256 must be set")
    for provenance_key in ("producer", "derivation", "registry_id", "selected_candidate_id"):
        if provenance_key not in typed_artifact.provenance:
            errors.append(
                f"candidate policy registry artifact provenance must set {provenance_key}"
            )
    for provenance_key in ("producer", "derivation", "registry_id", "selected_candidate_id"):
        if provenance_key in typed_artifact.provenance:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"candidate policy registry artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    try:
        registry_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="candidate policy registry artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        registry_path = None
    if registry_path is None:
        errors.append("candidate policy registry artifact must be local byte-verifiable")
    elif not registry_path.exists():
        errors.append("candidate policy registry artifact.path must exist")
    elif not registry_path.is_file():
        errors.append("candidate policy registry artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(registry_path) != typed_artifact.sha256:
        errors.append("candidate policy registry artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert registry_path is not None
    registry = CandidatePolicyRegistry.from_dict(read_json(registry_path))
    if typed_artifact.provenance.get("registry_id") != registry.registry_id:
        raise HarnessIOError("candidate policy registry artifact provenance registry_id must match")
    if typed_artifact.provenance.get("selected_candidate_id") != registry.selected_candidate_id:
        raise HarnessIOError(
            "candidate policy registry artifact provenance selected_candidate_id must match"
        )
    return registry


def _candidate_record_consistency_errors(candidate: CandidatePolicyRecord) -> list[str]:
    errors: list[str] = []
    try:
        source_report = read_training_dataset_report_artifact(candidate.source_training_dataset_report)
    except HarnessIOError as exc:
        errors.append("candidate.source_training_dataset_report must validate: " + str(exc))
        source_report = None
    if source_report is not None and source_report.example_count < 1:
        errors.append("candidate.source_training_dataset_report must contain at least one example")
    if source_report is not None:
        disallowed_leakage = sorted(
            leakage_class
            for leakage_class, count in source_report.signals_by_leakage_class.items()
            if count > 0
            and LeakageClass.parse(leakage_class)
            not in candidate.safety.allowed_training_leakage_classes
        )
        if disallowed_leakage:
            errors.append(
                "candidate.source_training_dataset_report contains leakage classes outside "
                "candidate safety.allowed_training_leakage_classes: "
                + ", ".join(disallowed_leakage)
            )
    if candidate.trainer.backend_kind is not candidate.policy_template.backend_kind:
        errors.append("candidate trainer.backend_kind must match policy_template.backend_kind")
    if candidate.trainer.offline_only is not True:
        errors.append("candidate trainer.offline_only must be true")
    if candidate.trainer.runtime_allowed is not False:
        errors.append("candidate trainer.runtime_allowed must be false")
    if candidate.trainer.uses_online_language_model_control is not False:
        errors.append("candidate trainer must not use online language-model control")
    if not set(candidate.safety.allowed_training_leakage_classes).issubset(
        {
            LeakageClass.legal_policy_input,
            LeakageClass.legal_policy_action_output,
        }
    ):
        errors.append("candidate safety allowed_training_leakage_classes must be legal offline inputs")
    if candidate.eval.planner_mode != candidate.policy_template.config.get("planner_mode"):
        errors.append("candidate eval.planner_mode must match policy_template.config.planner_mode")
    if (
        candidate.eval.mode
        in (CandidateEvalMode.local_live_eval, CandidateEvalMode.gcp_live_eval)
        and candidate.safety.require_policy_trace is not True
    ):
        errors.append("candidate safety.require_policy_trace must be true for live eval candidates")
    if candidate.source_next_experiment_plan is None:
        if candidate.source_plan_candidate_id is not None:
            errors.append(
                "candidate.source_plan_candidate_id requires source_next_experiment_plan"
            )
    else:
        errors.extend(
            _source_next_experiment_plan_errors(
                candidate.source_next_experiment_plan,
                source_plan_candidate_id=candidate.source_plan_candidate_id,
            )
        )
    return errors


def _source_next_experiment_plan_errors(
    artifact: ArtifactRef,
    *,
    source_plan_candidate_id: str | None,
) -> list[str]:
    errors: list[str] = []
    if artifact.kind != "next_experiment_plan":
        errors.append("candidate.source_next_experiment_plan.kind must be 'next_experiment_plan'")
    if artifact.sha256 is None:
        errors.append("candidate.source_next_experiment_plan.sha256 must be set")
    for provenance_key in ("producer", "derivation"):
        if provenance_key not in artifact.provenance:
            errors.append(
                f"candidate.source_next_experiment_plan provenance must set {provenance_key}"
            )
        else:
            try:
                _require_nonempty_text(
                    artifact.provenance.get(provenance_key),
                    f"candidate.source_next_experiment_plan provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if source_plan_candidate_id is None:
        errors.append("candidate.source_plan_candidate_id must be set with source_next_experiment_plan")
    try:
        plan_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="candidate.source_next_experiment_plan",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        plan_path = None
    if plan_path is None:
        errors.append("candidate.source_next_experiment_plan must be local byte-verifiable")
    elif not plan_path.exists():
        errors.append("candidate.source_next_experiment_plan.path must exist")
    elif not plan_path.is_file():
        errors.append("candidate.source_next_experiment_plan.path must be a file")
    elif artifact.sha256 is not None and sha256_file(plan_path) != artifact.sha256:
        errors.append("candidate.source_next_experiment_plan.sha256 must match path")
    elif source_plan_candidate_id is not None:
        try:
            plan = NextExperimentPlan.from_dict(read_json(plan_path))
        except HarnessIOError as exc:
            errors.append(f"candidate.source_next_experiment_plan must parse: {exc}")
        else:
            plan_candidate_ids = {candidate.candidate_id for candidate in plan.candidates}
            if source_plan_candidate_id not in plan_candidate_ids:
                errors.append(
                    "candidate.source_plan_candidate_id must match a next experiment plan candidate"
                )
    return errors


def _registry_consistency_errors(registry: CandidatePolicyRegistry) -> list[str]:
    errors: list[str] = []
    if not registry.candidates:
        errors.append("candidate registry.candidates must contain at least one candidate")
    ids = tuple(candidate.candidate_id for candidate in registry.candidates)
    duplicate_ids = _duplicates(id_value for id_value in ids if id_value is not None)
    if duplicate_ids:
        errors.append(
            "candidate registry.candidates must not contain duplicate candidate_id: "
            + ", ".join(duplicate_ids)
        )
    if registry.selected_candidate_id is not None and registry.selected_candidate_id not in ids:
        errors.append("candidate registry.selected_candidate_id must reference a candidate")
    return errors


def _leakage_tuple(value: Any, field_name: str) -> tuple[LeakageClass, ...]:
    values = _as_sequence(value, field_name)
    if not values:
        raise HarnessIOError(f"{field_name} must not be empty")
    parsed: list[LeakageClass] = []
    for item in values:
        try:
            parsed.append(LeakageClass.parse(item))
        except SchemaValidationError as exc:
            raise HarnessIOError(str(exc)) from exc
    duplicate_values = _duplicates(item.value for item in parsed)
    if duplicate_values:
        raise HarnessIOError(f"{field_name} must not contain duplicates: {', '.join(duplicate_values)}")
    return tuple(parsed)


def _require_absolute_container_path(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name)
    path = PurePosixPath(text)
    if not path.is_absolute():
        raise HarnessIOError(f"{field_name} must be an absolute container path")
    if any(part in (".", "..") for part in path.parts):
        raise HarnessIOError(f"{field_name} must not contain '.' or '..' segments")
    return text


def _require_output_name(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name)
    path = PurePosixPath(text)
    if path.is_absolute() or len(path.parts) != 1 or path.name != text:
        raise HarnessIOError(f"{field_name} must be a single file name")
    if text in (".", "..") or "/" in text:
        raise HarnessIOError(f"{field_name} must not contain path separators")
    return text


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_sha256(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise HarnessIOError(f"{field_name} must be a lowercase sha256 hex digest")
    return text


def _require_positive_finite_float(value: Any, field_name: str) -> float:
    number = _require_finite_float(value, field_name)
    if number <= 0.0:
        raise HarnessIOError(f"{field_name} must be > 0")
    return number


def _require_nonnegative_finite_float(value: Any, field_name: str) -> float:
    number = _require_finite_float(value, field_name)
    if number < 0.0:
        raise HarnessIOError(f"{field_name} must be >= 0")
    return number


def _require_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or type(value) not in (float, int):
        raise HarnessIOError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise HarnessIOError(f"{field_name} must be finite")
    return number


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    values = _as_sequence(value, field_name)
    return tuple(_require_nonempty_text(item, field_name) for item in values)


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_keys(
    value: Mapping[str, Any],
    required_keys: frozenset[str],
    field_name: str,
) -> None:
    missing_keys = sorted(repr(key) for key in required_keys if key not in value)
    if missing_keys:
        raise HarnessIOError(f"{field_name} is missing required fields: {', '.join(missing_keys)}")


def _duplicates(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicate_seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        text = str(value)
        if text in seen and text not in duplicate_seen:
            duplicates.append(text)
            duplicate_seen.add(text)
        seen.add(text)
    return tuple(duplicates)


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _thaw_json(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
    return value
