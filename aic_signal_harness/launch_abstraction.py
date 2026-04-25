"""Cloud/local launch-plan contracts for the AIC outer loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeVar, cast
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file, write_json
from aic_signal_harness.autonomous_gate import (
    AutonomousGateVerdict,
    read_autonomous_gate_decision_artifact,
)
from aic_signal_harness.candidate_registry import (
    CandidateEvalMode,
    read_candidate_policy_registry_artifact,
)
from aic_signal_harness.lewm_trainer_adapter import (
    LewmTrainerAdapterPlan,
    LewmTrainerLaunchTarget,
    read_lewm_trainer_adapter_plan_artifact,
)
from aic_signal_harness.schemas import ArtifactRef, SCHEMA_VERSION, SchemaValidationError


OUTER_LOOP_LAUNCH_PLAN_KIND = "outer_loop_launch_plan"

_PRODUCER = "aic_signal_harness.launch_abstraction"
_WRITE_PLAN_DERIVATION = "write_outer_loop_launch_plan"

_LAUNCH_CONFIG_KEYS = frozenset(
    {
        "plan_id",
        "generated_at_utc",
        "launch_target",
        "eval_run_id",
        "environment",
        "notes",
    }
)
_LAUNCH_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "generated_at_utc",
        "operation",
        "launch_target",
        "source_artifacts",
        "command",
        "environment",
        "expected_outputs",
        "timeout_seconds",
        "requires_manual_authorization",
        "autonomous_launch_allowed",
        "ok",
        "errors",
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


class OuterLoopLaunchOperation(_StrEnum):
    """Outer-loop work item represented by a launch plan."""

    policy_training = "policy_training"
    live_eval = "live_eval"


class OuterLoopLaunchTarget(_StrEnum):
    """Execution substrate selected by the outer loop."""

    local = "local"
    gcp_vm = "gcp_vm"


@dataclass(frozen=True)
class OuterLoopLaunchConfig:
    """Shared knobs for a non-launching cloud/local plan."""

    plan_id: str
    generated_at_utc: str
    launch_target: OuterLoopLaunchTarget = OuterLoopLaunchTarget.local
    eval_run_id: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("plan_id", "generated_at_utc"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"launch config.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "launch_target",
                OuterLoopLaunchTarget.parse(self.launch_target),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "eval_run_id",
                _optional_nonempty_text(self.eval_run_id, "launch config.eval_run_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "environment",
                _environment_mapping(self.environment, "launch config.environment"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "launch config.notes", allow_empty=True),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "plan_id": self.plan_id,
            "generated_at_utc": self.generated_at_utc,
            "launch_target": self.launch_target.value,
            "environment": dict(self.environment),
            "notes": list(self.notes),
        }
        if self.eval_run_id is not None:
            value["eval_run_id"] = self.eval_run_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OuterLoopLaunchConfig":
        if not isinstance(value, Mapping):
            raise HarnessIOError("launch config must be a mapping")
        _reject_unknown_keys(value, _LAUNCH_CONFIG_KEYS, "launch config")
        _require_keys(value, _LAUNCH_CONFIG_KEYS - {"environment", "eval_run_id", "launch_target", "notes"}, "launch config")
        return cls(
            plan_id=cast(Any, value.get("plan_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            launch_target=cast(Any, value.get("launch_target", OuterLoopLaunchTarget.local.value)),
            eval_run_id=cast(Any, value.get("eval_run_id")),
            environment=cast(Any, value.get("environment", {})),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class OuterLoopLaunchPlan:
    """A durable launch contract that does not execute by itself."""

    plan_id: str
    generated_at_utc: str
    operation: OuterLoopLaunchOperation
    launch_target: OuterLoopLaunchTarget
    source_artifacts: Mapping[str, ArtifactRef]
    command: tuple[str, ...]
    environment: Mapping[str, str]
    expected_outputs: Mapping[str, str]
    timeout_seconds: float
    requires_manual_authorization: bool = True
    autonomous_launch_allowed: bool = False
    ok: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("plan_id", "generated_at_utc"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"launch plan.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "operation",
                OuterLoopLaunchOperation.parse(self.operation),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "launch_target",
                OuterLoopLaunchTarget.parse(self.launch_target),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "source_artifacts",
                _artifact_mapping(self.source_artifacts, "launch plan.source_artifacts"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "command",
                _as_text_sequence(self.command, "launch plan.command", allow_empty=False),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "environment",
                _environment_mapping(self.environment, "launch plan.environment"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "expected_outputs",
                _output_path_mapping(self.expected_outputs, "launch plan.expected_outputs"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "timeout_seconds",
                _require_positive_finite_float(
                    self.timeout_seconds,
                    "launch plan.timeout_seconds",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("requires_manual_authorization", "autonomous_launch_allowed", "ok"):
            if type(getattr(self, field_name)) is not bool:
                errors.append(f"launch plan.{field_name} must be a boolean")
        try:
            object.__setattr__(
                self,
                "errors",
                _as_text_sequence(self.errors, "launch plan.errors", allow_empty=True),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "launch plan.notes", allow_empty=True),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_launch_plan_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "generated_at_utc": self.generated_at_utc,
            "operation": self.operation.value,
            "launch_target": self.launch_target.value,
            "source_artifacts": {
                role: artifact.to_dict()
                for role, artifact in self.source_artifacts.items()
            },
            "command": list(self.command),
            "environment": dict(self.environment),
            "expected_outputs": dict(self.expected_outputs),
            "timeout_seconds": self.timeout_seconds,
            "requires_manual_authorization": self.requires_manual_authorization,
            "autonomous_launch_allowed": self.autonomous_launch_allowed,
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OuterLoopLaunchPlan":
        if not isinstance(value, Mapping):
            raise HarnessIOError("launch plan must be a mapping")
        _reject_unknown_keys(value, _LAUNCH_PLAN_KEYS, "launch plan")
        _require_keys(value, _LAUNCH_PLAN_KEYS - {"notes"}, "launch plan")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            plan_id=cast(Any, value.get("plan_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            operation=cast(Any, value.get("operation")),
            launch_target=cast(Any, value.get("launch_target")),
            source_artifacts=cast(Any, value.get("source_artifacts")),
            command=cast(Any, value.get("command")),
            environment=cast(Any, value.get("environment")),
            expected_outputs=cast(Any, value.get("expected_outputs")),
            timeout_seconds=cast(Any, value.get("timeout_seconds")),
            requires_manual_authorization=cast(Any, value.get("requires_manual_authorization")),
            autonomous_launch_allowed=cast(Any, value.get("autonomous_launch_allowed")),
            ok=cast(Any, value.get("ok")),
            errors=cast(Any, value.get("errors")),
            notes=cast(Any, value.get("notes", ())),
        )


def build_training_launch_plan_from_lewm_adapter(
    *,
    lewm_trainer_adapter_plan: ArtifactRef | Mapping[str, Any],
    output_root: str | Path,
    config: OuterLoopLaunchConfig | Mapping[str, Any],
) -> OuterLoopLaunchPlan:
    """Create a non-launching training plan from a LEWM adapter plan."""

    adapter_artifact = (
        lewm_trainer_adapter_plan
        if isinstance(lewm_trainer_adapter_plan, ArtifactRef)
        else ArtifactRef.from_dict(lewm_trainer_adapter_plan)
    )
    typed_config = config if isinstance(config, OuterLoopLaunchConfig) else OuterLoopLaunchConfig.from_dict(config)
    adapter_plan = read_lewm_trainer_adapter_plan_artifact(adapter_artifact)
    output_dir = Path(output_root).expanduser().resolve(strict=False)
    expected_outputs = _training_expected_outputs(output_dir, adapter_plan)
    env = dict(typed_config.environment)
    env.update(
        _artifact_environment_values(
            adapter_plan=adapter_plan,
            expected_outputs=expected_outputs,
        )
    )
    env["AIC_TRAIN_RUN_ID"] = adapter_plan.train_run_id
    env["AIC_OUTER_LOOP_LAUNCH_PLAN_ID"] = typed_config.plan_id
    return OuterLoopLaunchPlan(
        plan_id=typed_config.plan_id,
        generated_at_utc=typed_config.generated_at_utc,
        operation=OuterLoopLaunchOperation.policy_training,
        launch_target=_launch_target_from_lewm(adapter_plan.launch_target, typed_config.launch_target),
        source_artifacts={
            "lewm_trainer_adapter_plan": adapter_artifact,
            "candidate_registry": adapter_plan.candidate_registry,
            "dataset_materialization_report": adapter_plan.dataset_materialization_report,
            "source_training_dataset_report": adapter_plan.source_training_dataset_report,
            "materialized_dataset": adapter_plan.materialized_dataset,
        },
        command=adapter_plan.trainer.command,
        environment=env,
        expected_outputs=expected_outputs,
        timeout_seconds=adapter_plan.timeout_seconds,
        requires_manual_authorization=True,
        autonomous_launch_allowed=False,
        ok=True,
        errors=(),
        notes=typed_config.notes
        + (
            "Training launch plan is evidence only; the orchestrator must explicitly execute it later.",
        ),
    )


def build_live_eval_launch_plan_from_gate(
    *,
    autonomous_gate_decision: ArtifactRef | Mapping[str, Any],
    result_root: str | Path,
    harness_root: str | Path,
    config: OuterLoopLaunchConfig | Mapping[str, Any],
) -> OuterLoopLaunchPlan:
    """Create a non-launching live-eval plan from a passed autonomous gate."""

    gate_artifact = (
        autonomous_gate_decision
        if isinstance(autonomous_gate_decision, ArtifactRef)
        else ArtifactRef.from_dict(autonomous_gate_decision)
    )
    typed_config = config if isinstance(config, OuterLoopLaunchConfig) else OuterLoopLaunchConfig.from_dict(config)
    gate = read_autonomous_gate_decision_artifact(gate_artifact)
    if gate.verdict is not AutonomousGateVerdict.passed or gate.eval_preconditions_met is not True:
        raise HarnessIOError("live eval launch plan requires a passed autonomous gate")
    if gate.train_run_id is None or gate.source_training_dataset_report is None or gate.policy_artifact is None:
        raise HarnessIOError("live eval launch plan requires completed training evidence")
    registry = read_candidate_policy_registry_artifact(gate.candidate_registry)
    candidate = registry.selected_candidate
    if candidate is None:
        raise HarnessIOError("live eval launch plan requires a selected candidate registry")
    expected_target = _launch_target_from_candidate_eval(candidate.eval.mode)
    if typed_config.launch_target is not expected_target:
        raise HarnessIOError(
            "launch config.launch_target must match selected candidate eval.mode"
        )
    eval_run_id = typed_config.eval_run_id or f"eval-{gate.train_run_id}"
    result_dir = Path(result_root).expanduser().resolve(strict=False)
    harness_dir = Path(harness_root).expanduser().resolve(strict=False)
    expected_outputs = _live_eval_expected_outputs(result_dir, harness_dir)
    env = dict(typed_config.environment)
    if (
        typed_config.launch_target is OuterLoopLaunchTarget.gcp_vm
        and "AIC_RUNTIME_POLICY_CHECKPOINT_PATH" in env
    ):
        raise HarnessIOError(
            "GCP live eval launch plans must not set AIC_RUNTIME_POLICY_CHECKPOINT_PATH"
        )
    policy_training_report_path = _artifact_path(gate.policy_training_report)
    policy_checkpoint_path = _artifact_path(gate.policy_artifact)
    env.update(
        {
            "AIC_EVAL_RUN_ID": eval_run_id,
            "AIC_EVAL_RESULT_ROOT": str(result_dir),
            "AIC_HARNESS_ROOT": str(harness_dir),
            "AIC_HARNESS_GATE_ID": gate.gate_id,
            "AIC_POLICY_TRAINING_REPORT_PATH": policy_training_report_path,
            "AIC_OUTER_LOOP_LAUNCH_PLAN_ID": typed_config.plan_id,
        }
    )
    if typed_config.launch_target is OuterLoopLaunchTarget.local:
        env["AIC_RUNTIME_POLICY_CHECKPOINT_PATH"] = policy_checkpoint_path
    command = _default_eval_command(typed_config.launch_target)
    return OuterLoopLaunchPlan(
        plan_id=typed_config.plan_id,
        generated_at_utc=typed_config.generated_at_utc,
        operation=OuterLoopLaunchOperation.live_eval,
        launch_target=typed_config.launch_target,
        source_artifacts={
            "autonomous_gate_decision": gate_artifact,
            "candidate_registry": gate.candidate_registry,
            "policy_training_report": gate.policy_training_report,
            "source_training_dataset_report": gate.source_training_dataset_report,
            "policy_artifact": gate.policy_artifact,
        },
        command=command,
        environment=env,
        expected_outputs=expected_outputs,
        timeout_seconds=_eval_timeout_seconds(env),
        requires_manual_authorization=True,
        autonomous_launch_allowed=False,
        ok=True,
        errors=(),
        notes=typed_config.notes
        + (
            "Live-eval launch plan is evidence only; the orchestrator must explicitly execute it later.",
        ),
    )


def write_outer_loop_launch_plan(
    path: str | Path,
    plan: OuterLoopLaunchPlan | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ArtifactRef:
    """Write a launch plan and return its artifact reference."""

    typed_plan = plan if isinstance(plan, OuterLoopLaunchPlan) else OuterLoopLaunchPlan.from_dict(plan)
    output_path = Path(path).expanduser().resolve(strict=False)
    _reject_output_aliases(output_path, typed_plan.source_artifacts, "outer loop launch plan.output_path")
    write_json(output_path, typed_plan.to_dict(), overwrite=overwrite)
    return ArtifactRef(
        kind=OUTER_LOOP_LAUNCH_PLAN_KIND,
        path=str(output_path),
        sha256=sha256_file(output_path),
        provenance={
            "producer": _PRODUCER,
            "derivation": _WRITE_PLAN_DERIVATION,
            "plan_id": typed_plan.plan_id,
            "operation": typed_plan.operation.value,
            "launch_target": typed_plan.launch_target.value,
        },
    )


def read_outer_loop_launch_plan_artifact(
    artifact: ArtifactRef | Mapping[str, Any],
) -> OuterLoopLaunchPlan:
    """Read and validate a local outer-loop launch plan artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != OUTER_LOOP_LAUNCH_PLAN_KIND:
        errors.append(f"outer loop launch plan artifact.kind must be {OUTER_LOOP_LAUNCH_PLAN_KIND!r}")
    if typed_artifact.sha256 is None:
        errors.append("outer loop launch plan artifact.sha256 must be set")
    for provenance_key in ("producer", "derivation", "plan_id", "operation", "launch_target"):
        if provenance_key not in typed_artifact.provenance:
            errors.append(f"outer loop launch plan artifact provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"outer loop launch plan artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if typed_artifact.provenance.get("producer") != _PRODUCER:
        errors.append(f"outer loop launch plan artifact provenance producer must be {_PRODUCER!r}")
    if typed_artifact.provenance.get("derivation") != _WRITE_PLAN_DERIVATION:
        errors.append(
            "outer loop launch plan artifact provenance derivation must be "
            f"{_WRITE_PLAN_DERIVATION!r}"
        )
    try:
        plan_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="outer loop launch plan artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        plan_path = None
    if plan_path is None:
        errors.append("outer loop launch plan artifact must be local byte-verifiable")
    elif not plan_path.exists():
        errors.append("outer loop launch plan artifact.path must exist")
    elif not plan_path.is_file():
        errors.append("outer loop launch plan artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(plan_path) != typed_artifact.sha256:
        errors.append("outer loop launch plan artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert plan_path is not None
    plan = OuterLoopLaunchPlan.from_dict(read_json(plan_path))
    if typed_artifact.provenance.get("plan_id") != plan.plan_id:
        raise HarnessIOError("outer loop launch plan artifact provenance plan_id must match")
    if typed_artifact.provenance.get("operation") != plan.operation.value:
        raise HarnessIOError("outer loop launch plan artifact provenance operation must match")
    if typed_artifact.provenance.get("launch_target") != plan.launch_target.value:
        raise HarnessIOError("outer loop launch plan artifact provenance launch_target must match")
    return plan


def _launch_plan_consistency_errors(plan: OuterLoopLaunchPlan) -> list[str]:
    errors: list[str] = []
    if plan.requires_manual_authorization is not True:
        errors.append("launch plan.requires_manual_authorization must be true before orchestrator")
    if plan.autonomous_launch_allowed is not False:
        errors.append("launch plan.autonomous_launch_allowed must be false before orchestrator")
    if plan.ok is not True:
        errors.append("launch plan.ok must be true")
    if plan.errors:
        errors.append("launch plan.errors must be empty when ok=true")
    if not plan.source_artifacts:
        errors.append("launch plan.source_artifacts must not be empty")
    source_paths: dict[str, Path] = {}
    for role, artifact in plan.source_artifacts.items():
        try:
            source_paths[role] = _local_existing_artifact_file(artifact, f"launch plan.source_artifacts.{role}")
        except HarnessIOError as exc:
            errors.append(str(exc))
    output_paths = {role: Path(path) for role, path in plan.expected_outputs.items()}
    duplicate_outputs = _duplicate_resolved_paths(output_paths)
    if duplicate_outputs:
        errors.append(
            "launch plan.expected_outputs must not alias each other: "
            + ", ".join(duplicate_outputs)
        )
    for output_role, output_path in output_paths.items():
        for source_role, source_path in source_paths.items():
            if _paths_alias(output_path, source_path):
                errors.append(
                    f"launch plan.expected_outputs.{output_role} must not alias source artifact {source_role}"
                )
    allowed_locations = {str(path) for path in output_paths.values()}
    allowed_locations.update(str(path) for path in source_paths.values())
    hidden_env = _hidden_environment_locations(plan.environment, allowed_locations)
    if hidden_env:
        errors.append(
            "launch plan.environment must not hide artifact locations outside declared "
            "source_artifacts or expected_outputs: "
            + ", ".join(hidden_env)
        )
    errors.extend(_owned_command_binding_errors(plan))
    return errors


def _owned_command_binding_errors(plan: OuterLoopLaunchPlan) -> list[str]:
    errors: list[str] = []
    if plan.operation is OuterLoopLaunchOperation.policy_training:
        adapter_artifact = plan.source_artifacts.get("lewm_trainer_adapter_plan")
        if adapter_artifact is None:
            return ["policy_training launch plan requires source_artifacts.lewm_trainer_adapter_plan"]
        try:
            adapter_plan = read_lewm_trainer_adapter_plan_artifact(adapter_artifact)
        except HarnessIOError as exc:
            return ["policy_training launch plan lewm_trainer_adapter_plan must validate: " + str(exc)]
        expected_target = _launch_target_from_lewm(
            adapter_plan.launch_target,
            plan.launch_target,
        )
        if plan.launch_target is not expected_target:
            errors.append(
                "policy_training launch plan.launch_target must match lewm trainer adapter launch_target"
            )
        if plan.command != adapter_plan.trainer.command:
            errors.append("policy_training launch plan.command must match lewm trainer adapter trainer.command")
    elif plan.operation is OuterLoopLaunchOperation.live_eval:
        gate_artifact = plan.source_artifacts.get("autonomous_gate_decision")
        if gate_artifact is None:
            return ["live_eval launch plan requires source_artifacts.autonomous_gate_decision"]
        try:
            gate = read_autonomous_gate_decision_artifact(gate_artifact)
            registry = read_candidate_policy_registry_artifact(gate.candidate_registry)
            candidate = registry.selected_candidate
        except HarnessIOError as exc:
            return ["live_eval launch plan autonomous gate/candidate registry must validate: " + str(exc)]
        if candidate is None:
            errors.append("live_eval launch plan requires selected candidate registry")
        else:
            expected_target = _launch_target_from_candidate_eval(candidate.eval.mode)
            if plan.launch_target is not expected_target:
                errors.append(
                    "live_eval launch plan.launch_target must match selected candidate eval.mode"
                )
        expected_command = _default_eval_command(plan.launch_target)
        if plan.command != expected_command:
            errors.append("live_eval launch plan.command must match launch_target default command")
    return errors


def _training_expected_outputs(
    output_root: Path,
    adapter_plan: LewmTrainerAdapterPlan,
) -> dict[str, str]:
    output_dir = output_root.expanduser().resolve(strict=False)
    return {
        "policy_checkpoint": str(output_dir / adapter_plan.checkpoint.output_name),
        "stdout": str(output_dir / "policy_training_stdout.log"),
        "stderr": str(output_dir / "policy_training_stderr.log"),
        "execution_report": str(output_dir / "policy_training_execution_report.json"),
        "policy_training_report": str(output_dir / "policy_training_report.json"),
    }


def _live_eval_expected_outputs(result_root: Path, harness_root: Path) -> dict[str, str]:
    result_dir = result_root.expanduser().resolve(strict=False)
    harness_dir = harness_root.expanduser().resolve(strict=False)
    return {
        "result_root": str(result_dir),
        "harness_root": str(harness_dir),
        "scoring_yaml": str(result_dir / "eval" / "scoring.yaml"),
        "policy_trace": str(harness_dir / "policy_trace.jsonl"),
        "run_manifest": str(harness_dir / "run_manifest.json"),
        "ledger_entry": str(harness_dir / "ledger_entry.json"),
        "promotion_report": str(harness_dir / "promotion_report.json"),
        "live_eval_summary": str(harness_dir / "live_eval_summary.json"),
    }


def _artifact_environment_values(
    *,
    adapter_plan: LewmTrainerAdapterPlan,
    expected_outputs: Mapping[str, str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for env_key, role in adapter_plan.artifact_environment_contract.items():
        if role == "dataset_materialization_report":
            values[env_key] = _artifact_path(adapter_plan.dataset_materialization_report)
        elif role == "materialized_dataset":
            values[env_key] = _artifact_path(adapter_plan.materialized_dataset)
        elif role == "policy_checkpoint_output":
            values[env_key] = expected_outputs["policy_checkpoint"]
        else:
            raise HarnessIOError(f"unknown lewm trainer artifact environment role: {role}")
    return values


def _launch_target_from_lewm(
    adapter_target: LewmTrainerLaunchTarget,
    config_target: OuterLoopLaunchTarget,
) -> OuterLoopLaunchTarget:
    expected = (
        OuterLoopLaunchTarget.gcp_vm
        if adapter_target is LewmTrainerLaunchTarget.gcp_vm
        else OuterLoopLaunchTarget.local
    )
    if config_target is not expected:
        raise HarnessIOError(
            "launch config.launch_target must match lewm trainer adapter launch_target"
        )
    return expected


def _default_eval_command(target: OuterLoopLaunchTarget) -> tuple[str, ...]:
    if target is OuterLoopLaunchTarget.gcp_vm:
        return ("aic_lewm_policy/cloud/gcp/run_learned_eval.sh",)
    return ("aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh",)


def _launch_target_from_candidate_eval(mode: CandidateEvalMode) -> OuterLoopLaunchTarget:
    if mode is CandidateEvalMode.gcp_live_eval:
        return OuterLoopLaunchTarget.gcp_vm
    if mode is CandidateEvalMode.local_live_eval:
        return OuterLoopLaunchTarget.local
    raise HarnessIOError("live eval launch plan requires selected candidate live eval mode")


def _eval_timeout_seconds(environment: Mapping[str, str]) -> float:
    raw = environment.get("AIC_EVAL_TIMEOUT_SEC", "1800")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise HarnessIOError("AIC_EVAL_TIMEOUT_SEC must be a number") from exc
    return _require_positive_finite_float(timeout, "AIC_EVAL_TIMEOUT_SEC")


def _artifact_path(artifact: ArtifactRef) -> str:
    return str(_local_existing_artifact_file(artifact, "artifact"))


def _artifact_mapping(value: Any, field_name: str) -> Mapping[str, ArtifactRef]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    normalized: dict[str, ArtifactRef] = {}
    for role, artifact in value.items():
        role_text = _require_role(role, f"{field_name}.role")
        if not isinstance(artifact, ArtifactRef):
            try:
                artifact = ArtifactRef.from_dict(artifact)
            except SchemaValidationError as exc:
                raise HarnessIOError(str(exc)) from exc
        if role_text in normalized:
            raise HarnessIOError(f"{field_name} must not contain duplicate role {role_text}")
        normalized[role_text] = artifact
    return dict(sorted(normalized.items()))


def _output_path_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    if not value:
        raise HarnessIOError(f"{field_name} must not be empty")
    normalized: dict[str, str] = {}
    for role, path in value.items():
        role_text = _require_role(role, f"{field_name}.role")
        path_text = _require_nonempty_text(path, f"{field_name}.{role_text}")
        output_path = Path(path_text).expanduser()
        if not output_path.is_absolute():
            raise HarnessIOError(f"{field_name}.{role_text} must be an absolute path")
        normalized[role_text] = str(output_path.resolve(strict=False))
    return dict(sorted(normalized.items()))


def _environment_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        env_key = _require_env_key(key, f"{field_name}.key")
        env_value = _require_nonempty_text(item, f"{field_name}.{env_key}")
        normalized[env_key] = env_value
    return dict(sorted(normalized.items()))


def _local_existing_artifact_file(artifact: ArtifactRef, field_name: str) -> Path:
    path = local_artifact_path(path=artifact.path, uri=artifact.uri, field_name=field_name)
    if path is None:
        raise HarnessIOError(f"{field_name} must be local byte-verifiable")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise HarnessIOError(f"{field_name}.path must exist") from exc
    except OSError as exc:
        raise HarnessIOError(f"{field_name}.path cannot be resolved: {exc}") from exc
    if not resolved.is_file():
        raise HarnessIOError(f"{field_name}.path must be a file")
    if artifact.sha256 is None:
        raise HarnessIOError(f"{field_name}.sha256 must be set")
    if sha256_file(resolved) != artifact.sha256:
        raise HarnessIOError(f"{field_name}.sha256 must match path")
    return resolved


def _hidden_environment_locations(
    environment: Mapping[str, str],
    allowed_locations: set[str],
) -> tuple[str, ...]:
    hidden: list[str] = []
    for key, value in environment.items():
        if _looks_like_location(value):
            path = _normalize_location(value)
            if path not in allowed_locations:
                hidden.append(key)
    return tuple(sorted(hidden))


def _normalize_location(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        local_path = local_artifact_path(path=None, uri=value, field_name="environment")
        assert local_path is not None
        return str(local_path.expanduser().resolve(strict=False))
    if not parsed.scheme:
        return str(Path(value).expanduser().resolve(strict=False))
    return value


def _looks_like_location(value: str) -> bool:
    normalized = value.strip()
    parsed = urlparse(normalized)
    return (
        bool(parsed.scheme and (parsed.netloc or parsed.scheme == "file"))
        or normalized.startswith(("/", "./", "../", "~"))
        or normalized.endswith((".json", ".jsonl", ".yaml", ".yml", ".ckpt", ".h5", ".hdf5", ".log"))
    )


def _reject_output_aliases(
    output_path: Path,
    source_artifacts: Mapping[str, ArtifactRef],
    field_name: str,
) -> None:
    for role, artifact in source_artifacts.items():
        source_path = _local_existing_artifact_file(artifact, f"{field_name}.{role}")
        if _paths_alias(output_path, source_path):
            raise HarnessIOError(f"{field_name} must not alias source artifact {role}")


def _paths_alias(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve(strict=False)
    right_resolved = right.expanduser().resolve(strict=False)
    if left_resolved == right_resolved:
        return True
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            return False
    return False


def _duplicate_resolved_paths(paths: Mapping[str, Path]) -> tuple[str, ...]:
    by_resolved: dict[str, str] = {}
    duplicates: list[str] = []
    for role, path in paths.items():
        resolved = str(path.expanduser().resolve(strict=False))
        if resolved in by_resolved:
            duplicates.append(f"{by_resolved[resolved]}={role}")
        else:
            by_resolved[resolved] = role
    return tuple(duplicates)


def _require_role(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name)
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    if any(character not in allowed for character in text):
        raise HarnessIOError(f"{field_name} must be a lowercase snake_case role")
    if text[0].isdigit():
        raise HarnessIOError(f"{field_name} must not start with a digit")
    return text


def _require_env_key(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name)
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(character not in allowed for character in text):
        raise HarnessIOError(f"{field_name} must be an uppercase environment variable name")
    if text[0].isdigit():
        raise HarnessIOError(f"{field_name} must not start with a digit")
    return text


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_positive_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or type(value) not in (float, int):
        raise HarnessIOError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise HarnessIOError(f"{field_name} must be finite")
    if number <= 0.0:
        raise HarnessIOError(f"{field_name} must be > 0")
    return number


def _as_text_sequence(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if not value and not allow_empty:
        raise HarnessIOError(f"{field_name} must not be empty")
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
