"""LEWM trainer adapter plans for the AIC outer loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file, write_json
from aic_signal_harness.candidate_registry import (
    CANDIDATE_POLICY_REGISTRY_KIND,
    CandidateCheckpointExpectation,
    CandidatePolicyRecord,
    read_candidate_policy_registry_artifact,
)
from aic_signal_harness.dataset_materialization import (
    TRAINING_DATASET_MATERIALIZATION_KIND,
    TRAINER_DATASET_JSONL_KIND,
    TrainingDatasetMaterializationFormat,
    TrainingDatasetMaterializationReport,
    read_training_dataset_materialization_report_artifact,
)
from aic_signal_harness.policy_training import TrainerInvocation
from aic_signal_harness.policy_training_runner import CandidatePolicyTemplate
from aic_signal_harness.schemas import ArtifactRef, BackendKind, SCHEMA_VERSION, SchemaValidationError


LEWM_TRAINER_ADAPTER_PLAN_KIND = "lewm_trainer_adapter_plan"

_PRODUCER = "aic_signal_harness.lewm_trainer_adapter"
_WRITE_PLAN_DERIVATION = "write_lewm_trainer_adapter_plan"
_DEFAULT_MATERIALIZATION_REPORT_ENV_KEY = "AIC_LEWM_MATERIALIZATION_REPORT"
_DEFAULT_MATERIALIZED_DATASET_ENV_KEY = "AIC_LEWM_MATERIALIZED_DATASET"
_DEFAULT_OUTPUT_CHECKPOINT_ENV_KEY = "AIC_LEWM_OUTPUT_CHECKPOINT"

_ADAPTER_CONFIG_KEYS = frozenset(
    {
        "plan_id",
        "generated_at_utc",
        "train_run_id",
        "launch_target",
        "timeout_seconds",
        "materialization_report_env_key",
        "materialized_dataset_env_key",
        "output_checkpoint_env_key",
        "notes",
    }
)
_ADAPTER_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "generated_at_utc",
        "train_run_id",
        "launch_target",
        "timeout_seconds",
        "candidate_registry",
        "selected_candidate_id",
        "dataset_materialization_report",
        "source_training_dataset_report",
        "materialized_dataset",
        "trainer",
        "policy_template",
        "checkpoint",
        "artifact_environment_contract",
        "offline_only",
        "runtime_allowed",
        "autonomous_launch_allowed",
        "ok",
        "errors",
        "notes",
    }
)
_ARTIFACT_ENV_CONTRACT_ROLES = MappingProxyType(
    {
        _DEFAULT_MATERIALIZATION_REPORT_ENV_KEY: "dataset_materialization_report",
        _DEFAULT_MATERIALIZED_DATASET_ENV_KEY: "materialized_dataset",
        _DEFAULT_OUTPUT_CHECKPOINT_ENV_KEY: "policy_checkpoint_output",
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


class LewmTrainerLaunchTarget(_StrEnum):
    """Training execution target selected later by the launch layer."""

    local_command = "local_command"
    gcp_vm = "gcp_vm"


@dataclass(frozen=True)
class LewmTrainerAdapterConfig:
    """Operator knobs for one LEWM trainer adapter plan."""

    plan_id: str
    generated_at_utc: str
    train_run_id: str
    launch_target: LewmTrainerLaunchTarget = LewmTrainerLaunchTarget.local_command
    timeout_seconds: float = 3600.0
    materialization_report_env_key: str = _DEFAULT_MATERIALIZATION_REPORT_ENV_KEY
    materialized_dataset_env_key: str = _DEFAULT_MATERIALIZED_DATASET_ENV_KEY
    output_checkpoint_env_key: str = _DEFAULT_OUTPUT_CHECKPOINT_ENV_KEY
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("plan_id", "generated_at_utc", "train_run_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"lewm trainer adapter config.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "launch_target",
                LewmTrainerLaunchTarget.parse(self.launch_target),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "timeout_seconds",
                _require_positive_finite_float(
                    self.timeout_seconds,
                    "lewm trainer adapter config.timeout_seconds",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in (
            "materialization_report_env_key",
            "materialized_dataset_env_key",
            "output_checkpoint_env_key",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_env_key(
                        getattr(self, field_name),
                        f"lewm trainer adapter config.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "lewm trainer adapter config.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            env_keys = (
                self.materialization_report_env_key,
                self.materialized_dataset_env_key,
                self.output_checkpoint_env_key,
            )
            if len(set(env_keys)) != len(env_keys):
                errors.append("lewm trainer adapter config artifact env keys must be distinct")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "generated_at_utc": self.generated_at_utc,
            "train_run_id": self.train_run_id,
            "launch_target": self.launch_target.value,
            "timeout_seconds": self.timeout_seconds,
            "materialization_report_env_key": self.materialization_report_env_key,
            "materialized_dataset_env_key": self.materialized_dataset_env_key,
            "output_checkpoint_env_key": self.output_checkpoint_env_key,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LewmTrainerAdapterConfig":
        if not isinstance(value, Mapping):
            raise HarnessIOError("lewm trainer adapter config must be a mapping")
        _reject_unknown_keys(value, _ADAPTER_CONFIG_KEYS, "lewm trainer adapter config")
        _require_keys(
            value,
            _ADAPTER_CONFIG_KEYS
            - {
                "launch_target",
                "materialization_report_env_key",
                "materialized_dataset_env_key",
                "notes",
                "output_checkpoint_env_key",
                "timeout_seconds",
            },
            "lewm trainer adapter config",
        )
        return cls(
            plan_id=cast(Any, value.get("plan_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            train_run_id=cast(Any, value.get("train_run_id")),
            launch_target=cast(Any, value.get("launch_target", LewmTrainerLaunchTarget.local_command.value)),
            timeout_seconds=cast(Any, value.get("timeout_seconds", 3600.0)),
            materialization_report_env_key=cast(
                Any,
                value.get(
                    "materialization_report_env_key",
                    _DEFAULT_MATERIALIZATION_REPORT_ENV_KEY,
                ),
            ),
            materialized_dataset_env_key=cast(
                Any,
                value.get(
                    "materialized_dataset_env_key",
                    _DEFAULT_MATERIALIZED_DATASET_ENV_KEY,
                ),
            ),
            output_checkpoint_env_key=cast(
                Any,
                value.get("output_checkpoint_env_key", _DEFAULT_OUTPUT_CHECKPOINT_ENV_KEY),
            ),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class LewmTrainerAdapterPlan:
    """A reusable, non-launching LEWM training handoff for the outer loop."""

    plan_id: str
    generated_at_utc: str
    train_run_id: str
    launch_target: LewmTrainerLaunchTarget
    timeout_seconds: float
    candidate_registry: ArtifactRef
    selected_candidate_id: str
    dataset_materialization_report: ArtifactRef
    source_training_dataset_report: ArtifactRef
    materialized_dataset: ArtifactRef
    trainer: TrainerInvocation
    policy_template: CandidatePolicyTemplate
    checkpoint: CandidateCheckpointExpectation
    artifact_environment_contract: Mapping[str, str]
    offline_only: bool = True
    runtime_allowed: bool = False
    autonomous_launch_allowed: bool = False
    ok: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("plan_id", "generated_at_utc", "train_run_id", "selected_candidate_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"lewm trainer adapter plan.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "launch_target",
                LewmTrainerLaunchTarget.parse(self.launch_target),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "timeout_seconds",
                _require_positive_finite_float(
                    self.timeout_seconds,
                    "lewm trainer adapter plan.timeout_seconds",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in (
            "candidate_registry",
            "dataset_materialization_report",
            "source_training_dataset_report",
            "materialized_dataset",
        ):
            if not isinstance(getattr(self, field_name), ArtifactRef):
                try:
                    object.__setattr__(
                        self,
                        field_name,
                        ArtifactRef.from_dict(getattr(self, field_name)),
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
        try:
            object.__setattr__(
                self,
                "artifact_environment_contract",
                _artifact_environment_contract(
                    self.artifact_environment_contract,
                    "lewm trainer adapter plan.artifact_environment_contract",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("offline_only", "runtime_allowed", "autonomous_launch_allowed", "ok"):
            if type(getattr(self, field_name)) is not bool:
                errors.append(f"lewm trainer adapter plan.{field_name} must be a boolean")
        try:
            object.__setattr__(
                self,
                "errors",
                _as_text_sequence(self.errors, "lewm trainer adapter plan.errors"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "lewm trainer adapter plan.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_adapter_plan_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "generated_at_utc": self.generated_at_utc,
            "train_run_id": self.train_run_id,
            "launch_target": self.launch_target.value,
            "timeout_seconds": self.timeout_seconds,
            "candidate_registry": self.candidate_registry.to_dict(),
            "selected_candidate_id": self.selected_candidate_id,
            "dataset_materialization_report": self.dataset_materialization_report.to_dict(),
            "source_training_dataset_report": self.source_training_dataset_report.to_dict(),
            "materialized_dataset": self.materialized_dataset.to_dict(),
            "trainer": self.trainer.to_dict(),
            "policy_template": self.policy_template.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "artifact_environment_contract": dict(self.artifact_environment_contract),
            "offline_only": self.offline_only,
            "runtime_allowed": self.runtime_allowed,
            "autonomous_launch_allowed": self.autonomous_launch_allowed,
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LewmTrainerAdapterPlan":
        if not isinstance(value, Mapping):
            raise HarnessIOError("lewm trainer adapter plan must be a mapping")
        _reject_unknown_keys(value, _ADAPTER_PLAN_KEYS, "lewm trainer adapter plan")
        _require_keys(value, _ADAPTER_PLAN_KEYS - {"notes"}, "lewm trainer adapter plan")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            plan_id=cast(Any, value.get("plan_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            train_run_id=cast(Any, value.get("train_run_id")),
            launch_target=cast(Any, value.get("launch_target")),
            timeout_seconds=cast(Any, value.get("timeout_seconds")),
            candidate_registry=cast(Any, value.get("candidate_registry")),
            selected_candidate_id=cast(Any, value.get("selected_candidate_id")),
            dataset_materialization_report=cast(Any, value.get("dataset_materialization_report")),
            source_training_dataset_report=cast(Any, value.get("source_training_dataset_report")),
            materialized_dataset=cast(Any, value.get("materialized_dataset")),
            trainer=cast(Any, value.get("trainer")),
            policy_template=cast(Any, value.get("policy_template")),
            checkpoint=cast(Any, value.get("checkpoint")),
            artifact_environment_contract=cast(Any, value.get("artifact_environment_contract")),
            offline_only=cast(Any, value.get("offline_only")),
            runtime_allowed=cast(Any, value.get("runtime_allowed")),
            autonomous_launch_allowed=cast(Any, value.get("autonomous_launch_allowed")),
            ok=cast(Any, value.get("ok")),
            errors=cast(Any, value.get("errors")),
            notes=cast(Any, value.get("notes", ())),
        )


def build_lewm_trainer_adapter_plan(
    *,
    candidate_registry: ArtifactRef | Mapping[str, Any],
    dataset_materialization_report: ArtifactRef | Mapping[str, Any],
    config: LewmTrainerAdapterConfig | Mapping[str, Any],
) -> LewmTrainerAdapterPlan:
    """Bind one selected LEWM candidate to materialized training bytes."""

    registry_artifact = (
        candidate_registry
        if isinstance(candidate_registry, ArtifactRef)
        else ArtifactRef.from_dict(candidate_registry)
    )
    materialization_artifact = (
        dataset_materialization_report
        if isinstance(dataset_materialization_report, ArtifactRef)
        else ArtifactRef.from_dict(dataset_materialization_report)
    )
    typed_config = config if isinstance(config, LewmTrainerAdapterConfig) else LewmTrainerAdapterConfig.from_dict(config)
    registry = read_candidate_policy_registry_artifact(registry_artifact)
    selected = registry.selected_candidate
    if selected is None:
        raise HarnessIOError("lewm trainer adapter requires a selected candidate")
    materialization = read_training_dataset_materialization_report_artifact(materialization_artifact)
    return LewmTrainerAdapterPlan(
        plan_id=typed_config.plan_id,
        generated_at_utc=typed_config.generated_at_utc,
        train_run_id=typed_config.train_run_id,
        launch_target=typed_config.launch_target,
        timeout_seconds=typed_config.timeout_seconds,
        candidate_registry=registry_artifact,
        selected_candidate_id=selected.candidate_id or "",
        dataset_materialization_report=materialization_artifact,
        source_training_dataset_report=materialization.source_training_dataset_report,
        materialized_dataset=materialization.materialized_dataset,
        trainer=selected.trainer,
        policy_template=selected.policy_template,
        checkpoint=selected.checkpoint,
        artifact_environment_contract={
            typed_config.materialization_report_env_key: "dataset_materialization_report",
            typed_config.materialized_dataset_env_key: "materialized_dataset",
            typed_config.output_checkpoint_env_key: "policy_checkpoint_output",
        },
        offline_only=True,
        runtime_allowed=False,
        autonomous_launch_allowed=False,
        ok=True,
        errors=(),
        notes=typed_config.notes
        + (
            "LEWM trainer adapter plan binds candidate identity to materialized trainer input; it does not launch training.",
        ),
    )


def write_lewm_trainer_adapter_plan(
    path: str | Path,
    plan: LewmTrainerAdapterPlan | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ArtifactRef:
    """Write a LEWM trainer adapter plan and return its artifact reference."""

    typed_plan = plan if isinstance(plan, LewmTrainerAdapterPlan) else LewmTrainerAdapterPlan.from_dict(plan)
    output_path = Path(path).expanduser().resolve(strict=False)
    write_json(output_path, typed_plan.to_dict(), overwrite=overwrite)
    return ArtifactRef(
        kind=LEWM_TRAINER_ADAPTER_PLAN_KIND,
        path=str(output_path),
        sha256=sha256_file(output_path),
        provenance={
            "producer": _PRODUCER,
            "derivation": _WRITE_PLAN_DERIVATION,
            "plan_id": typed_plan.plan_id,
            "train_run_id": typed_plan.train_run_id,
            "selected_candidate_id": typed_plan.selected_candidate_id,
            "materialized_dataset_sha256": typed_plan.materialized_dataset.sha256,
        },
    )


def read_lewm_trainer_adapter_plan_artifact(
    artifact: ArtifactRef | Mapping[str, Any],
) -> LewmTrainerAdapterPlan:
    """Read and validate a local LEWM trainer adapter plan artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != LEWM_TRAINER_ADAPTER_PLAN_KIND:
        errors.append(f"lewm trainer adapter plan artifact.kind must be {LEWM_TRAINER_ADAPTER_PLAN_KIND!r}")
    if typed_artifact.sha256 is None:
        errors.append("lewm trainer adapter plan artifact.sha256 must be set")
    for provenance_key in (
        "producer",
        "derivation",
        "plan_id",
        "train_run_id",
        "selected_candidate_id",
        "materialized_dataset_sha256",
    ):
        if provenance_key not in typed_artifact.provenance:
            errors.append(f"lewm trainer adapter plan artifact provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"lewm trainer adapter plan artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if typed_artifact.provenance.get("producer") != _PRODUCER:
        errors.append(f"lewm trainer adapter plan artifact provenance producer must be {_PRODUCER!r}")
    if typed_artifact.provenance.get("derivation") != _WRITE_PLAN_DERIVATION:
        errors.append(
            "lewm trainer adapter plan artifact provenance derivation must be "
            f"{_WRITE_PLAN_DERIVATION!r}"
        )
    try:
        plan_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="lewm trainer adapter plan artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        plan_path = None
    if plan_path is None:
        errors.append("lewm trainer adapter plan artifact must be local byte-verifiable")
    elif not plan_path.exists():
        errors.append("lewm trainer adapter plan artifact.path must exist")
    elif not plan_path.is_file():
        errors.append("lewm trainer adapter plan artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(plan_path) != typed_artifact.sha256:
        errors.append("lewm trainer adapter plan artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert plan_path is not None
    plan = LewmTrainerAdapterPlan.from_dict(read_json(plan_path))
    if typed_artifact.provenance.get("plan_id") != plan.plan_id:
        raise HarnessIOError("lewm trainer adapter plan artifact provenance plan_id must match")
    if typed_artifact.provenance.get("train_run_id") != plan.train_run_id:
        raise HarnessIOError("lewm trainer adapter plan artifact provenance train_run_id must match")
    if typed_artifact.provenance.get("selected_candidate_id") != plan.selected_candidate_id:
        raise HarnessIOError(
            "lewm trainer adapter plan artifact provenance selected_candidate_id must match"
        )
    if typed_artifact.provenance.get("materialized_dataset_sha256") != plan.materialized_dataset.sha256:
        raise HarnessIOError(
            "lewm trainer adapter plan artifact provenance materialized_dataset_sha256 must match"
        )
    return plan


def _adapter_plan_consistency_errors(plan: LewmTrainerAdapterPlan) -> list[str]:
    errors: list[str] = []
    if plan.offline_only is not True:
        errors.append("lewm trainer adapter plan.offline_only must be true")
    if plan.runtime_allowed is not False:
        errors.append("lewm trainer adapter plan.runtime_allowed must be false")
    if plan.autonomous_launch_allowed is not False:
        errors.append("lewm trainer adapter plan.autonomous_launch_allowed must be false")
    if plan.ok is not True:
        errors.append("lewm trainer adapter plan.ok must be true")
    if plan.errors:
        errors.append("lewm trainer adapter plan.errors must be empty when ok=true")
    candidate: CandidatePolicyRecord | None
    try:
        registry = read_candidate_policy_registry_artifact(plan.candidate_registry)
        candidate = registry.selected_candidate
    except HarnessIOError as exc:
        errors.append("lewm trainer adapter plan.candidate_registry must validate: " + str(exc))
        candidate = None
    if candidate is not None:
        errors.extend(_selected_candidate_errors(plan, candidate))
    try:
        materialization = read_training_dataset_materialization_report_artifact(
            plan.dataset_materialization_report
        )
    except HarnessIOError as exc:
        errors.append(
            "lewm trainer adapter plan.dataset_materialization_report must validate: "
            + str(exc)
        )
        materialization = None
    if materialization is not None:
        errors.extend(_materialization_binding_errors(plan, materialization))
    return errors


def _selected_candidate_errors(
    plan: LewmTrainerAdapterPlan,
    candidate: CandidatePolicyRecord,
) -> list[str]:
    errors: list[str] = []
    if candidate.candidate_id != plan.selected_candidate_id:
        errors.append("lewm trainer adapter plan.selected_candidate_id must match registry selection")
    if candidate.trainer != plan.trainer:
        errors.append("lewm trainer adapter plan.trainer must match selected candidate")
    if candidate.policy_template != plan.policy_template:
        errors.append("lewm trainer adapter plan.policy_template must match selected candidate")
    if candidate.checkpoint != plan.checkpoint:
        errors.append("lewm trainer adapter plan.checkpoint must match selected candidate")
    if candidate.source_training_dataset_report != plan.source_training_dataset_report:
        errors.append(
            "lewm trainer adapter plan.source_training_dataset_report must match selected candidate"
        )
    if candidate.trainer.backend_kind is not BackendKind.lewm_world_model:
        errors.append("lewm trainer adapter requires trainer.backend_kind lewm_world_model")
    if candidate.policy_template.backend_kind is not BackendKind.lewm_world_model:
        errors.append("lewm trainer adapter requires policy_template.backend_kind lewm_world_model")
    if candidate.trainer.offline_only is not True or candidate.trainer.runtime_allowed is not False:
        errors.append("lewm trainer adapter selected trainer must be offline-only")
    if candidate.trainer.uses_online_language_model_control is not False:
        errors.append("lewm trainer adapter selected trainer must not use online language-model control")
    forbidden_command_terms = _forbidden_command_terms(candidate.trainer.command)
    if forbidden_command_terms:
        errors.append(
            "lewm trainer adapter selected trainer.command must not reference runtime "
            "launch surfaces or hidden artifact locations: "
            + ", ".join(forbidden_command_terms)
        )
    env_collisions = sorted(
        key
        for key in plan.artifact_environment_contract
        if key in candidate.trainer.environment
    )
    if env_collisions:
        errors.append(
            "lewm trainer adapter artifact_environment_contract keys must not overlap "
            "trainer.environment: "
            + ", ".join(env_collisions)
        )
    planner_mode = candidate.policy_template.config.get("planner_mode")
    if not isinstance(planner_mode, str) or not planner_mode.strip():
        errors.append("lewm trainer adapter policy_template.config.planner_mode must be set")
    return errors


def _materialization_binding_errors(
    plan: LewmTrainerAdapterPlan,
    materialization: TrainingDatasetMaterializationReport,
) -> list[str]:
    errors: list[str] = []
    if materialization.format is not TrainingDatasetMaterializationFormat.jsonl:
        errors.append("lewm trainer adapter currently requires jsonl materialization format")
    if materialization.materialized_dataset.kind != TRAINER_DATASET_JSONL_KIND:
        errors.append(
            f"lewm trainer adapter materialized_dataset.kind must be {TRAINER_DATASET_JSONL_KIND!r}"
        )
    if materialization.source_training_dataset_report != plan.source_training_dataset_report:
        errors.append(
            "lewm trainer adapter source_training_dataset_report must match materialization report"
        )
    if materialization.materialized_dataset != plan.materialized_dataset:
        errors.append("lewm trainer adapter materialized_dataset must match materialization report")
    if materialization.offline_only is not True or materialization.runtime_allowed is not False:
        errors.append("lewm trainer adapter materialization must be offline-only")
    if materialization.consumable_by_policy_runtime is not False:
        errors.append("lewm trainer adapter materialization must not be consumable by policy runtime")
    if materialization.example_count < 1:
        errors.append("lewm trainer adapter materialization must contain at least one example")
    dataset_path = local_artifact_path(
        path=materialization.materialized_dataset.path,
        uri=materialization.materialized_dataset.uri,
        field_name="lewm trainer adapter materialized_dataset",
    )
    if dataset_path is None:
        errors.append("lewm trainer adapter materialized_dataset must be local byte-verifiable")
    return errors


def _artifact_environment_contract(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, role in value.items():
        env_key = _require_env_key(key, f"{field_name}.key")
        role_text = _require_nonempty_text(role, f"{field_name}.{env_key}")
        normalized[env_key] = role_text
    expected_roles = set(_ARTIFACT_ENV_CONTRACT_ROLES.values())
    actual_roles = set(normalized.values())
    if actual_roles != expected_roles:
        raise HarnessIOError(
            f"{field_name} roles must be exactly: " + ", ".join(sorted(expected_roles))
        )
    if len(normalized) != len(expected_roles):
        raise HarnessIOError(f"{field_name} env keys must be distinct")
    return dict(sorted(normalized.items()))


def _require_env_key(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name)
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(character not in allowed for character in text):
        raise HarnessIOError(f"{field_name} must be an uppercase environment variable name")
    if text[0].isdigit():
        raise HarnessIOError(f"{field_name} must not start with a digit")
    return text


def _forbidden_command_terms(command: tuple[str, ...]) -> tuple[str, ...]:
    forbidden: list[str] = []
    runtime_terms = (
        "aic_controller",
        "aic_evaluator",
        "aic_model",
        "gazebo",
        "roslaunch",
        "run_learned_eval",
        "scoring.yaml",
        "vm_eval",
    )
    artifact_flags = {
        "--checkpoint",
        "--dataset",
        "--model",
        "--policy",
        "--weights",
    }
    for index, token in enumerate(command):
        normalized = token.strip().lower()
        if not normalized:
            continue
        token_name = Path(normalized.split("=", 1)[0]).name
        if any(term == token_name or token_name.startswith(term + ".") for term in runtime_terms):
            forbidden.append(token)
        elif "=" in normalized and any(term in normalized.split("=", 1)[1] for term in runtime_terms):
            forbidden.append(token)
        elif _looks_like_artifact_location(normalized):
            forbidden.append(token)
        elif normalized in artifact_flags:
            forbidden.append(token)
            if index + 1 < len(command):
                forbidden.append(command[index + 1])
    return tuple(sorted(set(forbidden)))


def _looks_like_artifact_location(value: str) -> bool:
    parsed = urlparse(value)
    return (
        bool(parsed.scheme and (parsed.netloc or parsed.scheme == "file"))
        or value.endswith((".ckpt", ".h5", ".hdf5", ".jsonl", ".pt", ".pth", ".safetensors"))
        or any(
            marker in value
            for marker in (
                "--checkpoint=",
                "--dataset=",
                "--model=",
                "--policy=",
                "--weights=",
            )
        )
    )


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _require_positive_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or type(value) not in (float, int):
        raise HarnessIOError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise HarnessIOError(f"{field_name} must be finite")
    if number <= 0.0:
        raise HarnessIOError(f"{field_name} must be > 0")
    return number


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(
        _require_nonempty_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


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
