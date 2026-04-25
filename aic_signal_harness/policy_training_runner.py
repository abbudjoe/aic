"""Bounded offline policy-training command runner."""

from __future__ import annotations

import math
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast
from urllib.parse import urlparse

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    sha256_file,
    write_json,
)
from aic_signal_harness.policy_training import (
    PolicyTrainingExecutionReport,
    PolicyTrainingExecutionStatus,
    PolicyTrainingRunReport,
    TrainerInvocation,
    trainer_invocation_fingerprint_sha256,
)
from aic_signal_harness.reducers.policy_training import (
    policy_checkpoint_artifact,
    read_training_dataset_report_artifact,
    record_policy_training_run,
)
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    LeakageClass,
    PolicyBackendSpec,
    RuntimeBoundaryProof,
    RuntimeRole,
    SchemaValidationError,
    SimulatorKind,
    TrainingSourceKind,
)


AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT = "AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT"
AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT = "AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT"

_CANDIDATE_POLICY_TEMPLATE_KEYS = frozenset(
    {
        "backend_kind",
        "name",
        "training_sources",
        "simulator_sources",
        "legal_observation_contract",
        "description",
        "config",
        "provenance",
        "deterministic",
        "runtime_boundary_notes",
    }
)
_CRITICAL_CANDIDATE_POLICY_PROVENANCE_KEYS = frozenset(
    {
        "producer",
        "run_id",
        "derivation",
        "source_training_dataset_report_sha256",
        "policy_artifact_sha256",
        "trainer_fingerprint_sha256",
        "policy_training_execution_report_sha256",
        "allowed_training_leakage_classes",
        "dataset_leakage_summary",
    }
)
_RUNNER_PRODUCER = "aic_signal_harness.policy_training_runner"
_RUNNER_DERIVATION = "run_policy_training_command"
_RUNNER_ARTIFACT_ENV_KEYS = (
    AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT,
    AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT,
)
_STDOUT_ARTIFACT_KIND = "policy_training_stdout"
_STDERR_ARTIFACT_KIND = "policy_training_stderr"
_EXECUTION_REPORT_ARTIFACT_KIND = "policy_training_execution_report"
_OFFLINE_GUARD_ARTIFACT_KIND = "policy_training_offline_guard"
_OFFLINE_GUARD_ENV_KEY = "AIC_POLICY_TRAINING_OFFLINE_GUARD"
_OFFLINE_GUARD_KIND = "python_audit_no_child_processes"
_OFFLINE_GUARD_EXIT_CODE = 86
_OFFLINE_GUARD_BLOCKED_AUDIT_EVENTS = (
    "os.exec",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.putenv",
    "os.spawn",
    "os.system",
    "os.unsetenv",
    "pty.spawn",
    "subprocess.Popen",
)
_ALWAYS_REJECTED_TRAINING_LEAKAGE_CLASSES = frozenset(
    {
        LeakageClass.privileged_eval_signal,
        LeakageClass.post_hoc_label,
        LeakageClass.unknown,
    }
)
_DEFAULT_ALLOWED_TRAINING_LEAKAGE_CLASSES = (
    LeakageClass.legal_policy_input,
    LeakageClass.legal_policy_action_output,
)
_PROHIBITED_RUNTIME_COMMAND_TOKENS = (
    "aic_controller",
    "aic_evaluator",
    "aic_model",
    "gazebo",
    "roslaunch",
    "run_learned_eval",
    "scoring.yaml",
    "vm_eval",
)
_ARTIFACT_VALUE_SUFFIXES = (
    ".ckpt",
    ".h5",
    ".hdf5",
    ".mcap",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
)
_PYTHON_OFFLINE_GUARD_SOURCE = '''"""AIC offline policy-training guard."""

from __future__ import annotations

import os
import sys

_EXPECTED_GUARD = "python_audit_no_child_processes"
_BLOCKED_EVENTS = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.putenv",
        "os.spawn",
        "os.system",
        "os.unsetenv",
        "pty.spawn",
        "subprocess.Popen",
    }
)


def _aic_policy_training_audit(event: str, args: tuple[object, ...]) -> None:
    if os.environ.get("AIC_POLICY_TRAINING_OFFLINE_GUARD") != _EXPECTED_GUARD:
        _aic_policy_training_guard_exit("guard_env_mismatch")
    if event in _BLOCKED_EVENTS:
        _aic_policy_training_guard_exit(event)


def _aic_policy_training_guard_exit(event: str) -> None:
    try:
        try:
            os.write(
                2,
                (
                    "AIC offline policy-training guard blocked audit event: "
                    + event
                    + "\\n"
                ).encode("utf-8", errors="replace"),
            )
        except OSError:
            pass
        os._exit(86)
    except BaseException:
        os._exit(86)


sys.addaudithook(_aic_policy_training_audit)
'''


@dataclass(frozen=True)
class CandidatePolicyTemplate:
    """Policy backend fields that become complete after a checkpoint exists."""

    backend_kind: BackendKind
    name: str
    training_sources: tuple[TrainingSourceKind, ...]
    simulator_sources: tuple[SimulatorKind, ...]
    legal_observation_contract: str
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    deterministic: bool = True
    runtime_boundary_notes: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "backend_kind", BackendKind.parse(self.backend_kind))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "training_sources",
                _enum_tuple(
                    self.training_sources,
                    TrainingSourceKind,
                    "candidate policy template.training_sources",
                    allow_empty=False,
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "simulator_sources",
                _enum_tuple(
                    self.simulator_sources,
                    SimulatorKind,
                    "candidate policy template.simulator_sources",
                    allow_empty=False,
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("name", "legal_observation_contract"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"candidate policy template.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.description, str):
            errors.append("candidate policy template.description must be a string")
        if type(self.deterministic) is not bool:
            errors.append("candidate policy template.deterministic must be a boolean")
        if not isinstance(self.runtime_boundary_notes, str):
            errors.append("candidate policy template.runtime_boundary_notes must be a string")
        try:
            object.__setattr__(
                self,
                "config",
                _copy_json_mapping(self.config, "candidate policy template.config"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            provenance = _copy_json_mapping(
                self.provenance,
                "candidate policy template.provenance",
            )
            critical_overrides = sorted(
                key for key in provenance if key in _CRITICAL_CANDIDATE_POLICY_PROVENANCE_KEYS
            )
            if critical_overrides:
                errors.append(
                    "candidate policy template.provenance must not override critical fields: "
                    + ", ".join(critical_overrides)
                )
            object.__setattr__(self, "provenance", provenance)
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_candidate_policy_template_backend_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "name": self.name,
            "training_sources": [source.value for source in self.training_sources],
            "simulator_sources": [source.value for source in self.simulator_sources],
            "legal_observation_contract": self.legal_observation_contract,
            "description": self.description,
            "config": _thaw_json(self.config),
            "provenance": _thaw_json(self.provenance),
            "deterministic": self.deterministic,
            "runtime_boundary_notes": self.runtime_boundary_notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidatePolicyTemplate":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate policy template must be a mapping")
        _reject_unknown_keys(
            value,
            _CANDIDATE_POLICY_TEMPLATE_KEYS,
            "candidate policy template",
        )
        _require_keys(
            value,
            _CANDIDATE_POLICY_TEMPLATE_KEYS
            - {
                "config",
                "description",
                "deterministic",
                "provenance",
                "runtime_boundary_notes",
            },
            "candidate policy template",
        )
        return cls(
            backend_kind=cast(Any, value.get("backend_kind")),
            name=cast(Any, value.get("name")),
            training_sources=cast(Any, value.get("training_sources")),
            simulator_sources=cast(Any, value.get("simulator_sources")),
            legal_observation_contract=cast(Any, value.get("legal_observation_contract")),
            description=cast(str, value.get("description", "")),
            config=cast(Any, value.get("config", {})),
            provenance=cast(Any, value.get("provenance", {})),
            deterministic=cast(bool, value.get("deterministic", True)),
            runtime_boundary_notes=cast(str, value.get("runtime_boundary_notes", "")),
        )

    def to_policy_backend(
        self,
        *,
        run_id: str,
        source_training_dataset_report: ArtifactRef,
        trainer: TrainerInvocation,
        policy_artifact: ArtifactRef,
        policy_training_execution_report_sha256: str,
        allowed_training_leakage_classes: tuple[LeakageClass, ...],
        dataset_leakage_summary: Mapping[str, int],
    ) -> PolicyBackendSpec:
        trainer_fingerprint = trainer_invocation_fingerprint_sha256(trainer)
        provenance = dict(self.provenance)
        provenance.update(
            {
                "producer": _RUNNER_PRODUCER,
                "run_id": run_id,
                "derivation": _RUNNER_DERIVATION,
                "source_training_dataset_report_sha256": source_training_dataset_report.sha256,
                "policy_artifact_sha256": policy_artifact.sha256,
                "trainer_fingerprint_sha256": trainer_fingerprint,
                "policy_training_execution_report_sha256": policy_training_execution_report_sha256,
                "allowed_training_leakage_classes": [
                    leakage.value for leakage in allowed_training_leakage_classes
                ],
                "dataset_leakage_summary": dict(dataset_leakage_summary),
            }
        )
        try:
            return PolicyBackendSpec(
                backend_kind=self.backend_kind,
                name=self.name,
                runtime_role=RuntimeRole.live_policy,
                training_sources=self.training_sources,
                simulator_sources=self.simulator_sources,
                runtime_allowed=True,
                leakage_class=LeakageClass.legal_policy_input,
                runtime_boundary=RuntimeBoundaryProof(
                    deterministic=self.deterministic,
                    uses_online_language_model_control=False,
                    legal_observation_contract=self.legal_observation_contract,
                    policy_artifact=policy_artifact,
                    notes=self.runtime_boundary_notes,
                ),
                description=self.description,
                config=self.config,
                provenance=provenance,
            )
        except SchemaValidationError as exc:
            raise HarnessIOError("candidate policy template is invalid: " + str(exc)) from exc


@dataclass(frozen=True)
class PolicyTrainingCommandResult:
    """Result of a bounded offline trainer command and finalized provenance."""

    report: PolicyTrainingRunReport
    stdout_path: str
    stderr_path: str
    report_path: str | None
    artifact_environment: Mapping[str, str]
    execution_report: "PolicyTrainingExecutionReport"
    execution_report_artifact: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.report, PolicyTrainingRunReport):
            raise HarnessIOError("policy training command result.report must be a PolicyTrainingRunReport")
        object.__setattr__(
            self,
            "stdout_path",
            str(_require_path(self.stdout_path, "policy training command result.stdout_path")),
        )
        object.__setattr__(
            self,
            "stderr_path",
            str(_require_path(self.stderr_path, "policy training command result.stderr_path")),
        )
        if self.report_path is not None:
            object.__setattr__(
                self,
                "report_path",
                str(_require_path(self.report_path, "policy training command result.report_path")),
            )
        object.__setattr__(
            self,
            "artifact_environment",
            _environment_mapping(
                self.artifact_environment,
                "policy training command result.artifact_environment",
            ),
        )
        if not isinstance(self.execution_report, PolicyTrainingExecutionReport):
            raise HarnessIOError(
                "policy training command result.execution_report must be a PolicyTrainingExecutionReport"
            )
        if not isinstance(self.execution_report_artifact, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "execution_report_artifact",
                    ArtifactRef.from_dict(self.execution_report_artifact),
                )
            except SchemaValidationError as exc:
                raise HarnessIOError(str(exc)) from exc


def run_policy_training_command(
    *,
    run_id: str,
    generated_at_utc: str,
    source_training_dataset_report: ArtifactRef,
    trainer: TrainerInvocation | Mapping[str, Any],
    candidate_policy_template: CandidatePolicyTemplate | Mapping[str, Any],
    policy_checkpoint_path: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    execution_report_path: str | Path,
    timeout_seconds: float,
    report_path: str | Path | None = None,
    metrics: Mapping[str, float] | None = None,
    notes: tuple[str, ...] = (),
    allowed_training_leakage_classes: tuple[LeakageClass, ...] = _DEFAULT_ALLOWED_TRAINING_LEAKAGE_CLASSES,
    overwrite: bool = False,
) -> PolicyTrainingCommandResult:
    """Run one offline trainer command and emit provenance only after success."""

    typed_trainer = trainer if isinstance(trainer, TrainerInvocation) else TrainerInvocation.from_dict(trainer)
    typed_template = (
        candidate_policy_template
        if isinstance(candidate_policy_template, CandidatePolicyTemplate)
        else CandidatePolicyTemplate.from_dict(candidate_policy_template)
    )
    _require_nonempty_text(run_id, "policy training runner.run_id")
    _require_nonempty_text(generated_at_utc, "policy training runner.generated_at_utc")
    timeout = _require_timeout(timeout_seconds)
    source_report = read_training_dataset_report_artifact(source_training_dataset_report)
    allowed_leakage = _allowed_training_leakage_classes(allowed_training_leakage_classes)
    _validate_training_dataset_leakage(source_report, allowed_leakage)
    dataset_leakage_summary = _dataset_leakage_summary(source_report.signals_by_leakage_class)
    source_report_path = local_artifact_path(
        path=source_training_dataset_report.path,
        uri=source_training_dataset_report.uri,
        field_name="policy training runner.source_training_dataset_report",
    )
    assert source_report_path is not None
    source_report_path = source_report_path.expanduser().resolve(strict=True)
    checkpoint_path = _prepare_output_path(
        policy_checkpoint_path,
        "policy training runner.policy_checkpoint_path",
        overwrite=overwrite,
        unlink_existing=overwrite,
    )
    stdout_log_path = _prepare_output_path(
        stdout_path,
        "policy training runner.stdout_path",
        overwrite=overwrite,
        unlink_existing=False,
    )
    stderr_log_path = _prepare_output_path(
        stderr_path,
        "policy training runner.stderr_path",
        overwrite=overwrite,
        unlink_existing=False,
    )
    execution_output_path = _prepare_output_path(
        execution_report_path,
        "policy training runner.execution_report_path",
        overwrite=overwrite,
        unlink_existing=False,
    )
    report_output_path = (
        _prepare_output_path(
            report_path,
            "policy training runner.report_path",
            overwrite=overwrite,
            unlink_existing=False,
        )
        if report_path is not None
        else None
    )
    _reject_duplicate_output_paths(
        {
            "policy_checkpoint_path": checkpoint_path,
            "stdout_path": stdout_log_path,
            "stderr_path": stderr_log_path,
            "execution_report_path": execution_output_path,
            **({"report_path": report_output_path} if report_output_path is not None else {}),
        }
    )
    cwd = _trainer_working_directory(typed_trainer)
    typed_trainer = _trainer_with_effective_working_directory(typed_trainer, cwd)
    _require_python_offline_guardable_command(typed_trainer)
    offline_guard_path = _write_python_offline_guard(
        execution_output_path.parent,
        run_id=run_id,
        overwrite=overwrite,
    )
    offline_guard_artifact = _guard_artifact(offline_guard_path, run_id=run_id)
    artifact_environment = {
        AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT: str(source_report_path),
        AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT: str(checkpoint_path),
    }
    env = _subprocess_environment(
        typed_trainer,
        artifact_environment,
        guard_dir=offline_guard_path.parent,
    )
    _require_executable_resolution(typed_trainer, env)
    _reject_prohibited_command(typed_trainer)
    _reject_hidden_command_artifacts(typed_trainer)
    trainer_environment_sha256 = _stable_json_sha256(dict(typed_trainer.environment))

    try:
        completed = subprocess.run(
            typed_trainer.command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        _write_process_output(stdout_log_path, exc.stdout)
        _write_process_output(stderr_log_path, exc.stderr)
        _write_execution_report(
            execution_output_path,
            _execution_report(
                run_id=run_id,
                generated_at_utc=generated_at_utc,
                status=PolicyTrainingExecutionStatus.timed_out,
                trainer=typed_trainer,
                cwd=cwd,
                timeout_seconds=timeout,
                returncode=None,
                stdout_path=stdout_log_path,
                stderr_path=stderr_log_path,
                offline_guard_artifact=offline_guard_artifact,
                checkpoint_path=checkpoint_path,
                artifact_environment=artifact_environment,
                trainer_environment_sha256=trainer_environment_sha256,
                allowed_training_leakage_classes=allowed_leakage,
                dataset_leakage_summary=dataset_leakage_summary,
                ok=False,
                errors=(f"policy training command timed out after {timeout:g} seconds",),
            ),
            overwrite=overwrite,
        )
        raise HarnessIOError(
            f"policy training command timed out after {timeout:g} seconds"
        ) from exc
    except OSError as exc:
        _write_process_output(stdout_log_path, b"")
        _write_process_output(stderr_log_path, str(exc).encode("utf-8"))
        _write_execution_report(
            execution_output_path,
            _execution_report(
                run_id=run_id,
                generated_at_utc=generated_at_utc,
                status=PolicyTrainingExecutionStatus.launch_failed,
                trainer=typed_trainer,
                cwd=cwd,
                timeout_seconds=timeout,
                returncode=None,
                stdout_path=stdout_log_path,
                stderr_path=stderr_log_path,
                offline_guard_artifact=offline_guard_artifact,
                checkpoint_path=checkpoint_path,
                artifact_environment=artifact_environment,
                trainer_environment_sha256=trainer_environment_sha256,
                allowed_training_leakage_classes=allowed_leakage,
                dataset_leakage_summary=dataset_leakage_summary,
                ok=False,
                errors=(f"policy training command could not be launched: {exc}",),
            ),
            overwrite=overwrite,
        )
        raise HarnessIOError(f"policy training command could not be launched: {exc}") from exc

    _write_process_output(stdout_log_path, completed.stdout)
    _write_process_output(stderr_log_path, completed.stderr)
    if _is_guard_violation(completed):
        guard_violation_error = _guard_violation_error(completed.stderr)
        _write_execution_report(
            execution_output_path,
            _execution_report(
                run_id=run_id,
                generated_at_utc=generated_at_utc,
                status=PolicyTrainingExecutionStatus.guard_violation,
                trainer=typed_trainer,
                cwd=cwd,
                timeout_seconds=timeout,
                returncode=completed.returncode,
                stdout_path=stdout_log_path,
                stderr_path=stderr_log_path,
                offline_guard_artifact=offline_guard_artifact,
                checkpoint_path=checkpoint_path,
                artifact_environment=artifact_environment,
                trainer_environment_sha256=trainer_environment_sha256,
                allowed_training_leakage_classes=allowed_leakage,
                dataset_leakage_summary=dataset_leakage_summary,
                ok=False,
                errors=(guard_violation_error,),
            ),
            overwrite=overwrite,
        )
        raise HarnessIOError(guard_violation_error)
    if completed.returncode != 0:
        _write_execution_report(
            execution_output_path,
            _execution_report(
                run_id=run_id,
                generated_at_utc=generated_at_utc,
                status=PolicyTrainingExecutionStatus.failed,
                trainer=typed_trainer,
                cwd=cwd,
                timeout_seconds=timeout,
                returncode=completed.returncode,
                stdout_path=stdout_log_path,
                stderr_path=stderr_log_path,
                offline_guard_artifact=offline_guard_artifact,
                checkpoint_path=checkpoint_path,
                artifact_environment=artifact_environment,
                trainer_environment_sha256=trainer_environment_sha256,
                allowed_training_leakage_classes=allowed_leakage,
                dataset_leakage_summary=dataset_leakage_summary,
                ok=False,
                errors=(f"policy training command exited with code {completed.returncode}",),
            ),
            overwrite=overwrite,
        )
        raise HarnessIOError(f"policy training command exited with code {completed.returncode}")
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        _write_execution_report(
            execution_output_path,
            _execution_report(
                run_id=run_id,
                generated_at_utc=generated_at_utc,
                status=PolicyTrainingExecutionStatus.checkpoint_missing,
                trainer=typed_trainer,
                cwd=cwd,
                timeout_seconds=timeout,
                returncode=completed.returncode,
                stdout_path=stdout_log_path,
                stderr_path=stderr_log_path,
                offline_guard_artifact=offline_guard_artifact,
                checkpoint_path=checkpoint_path,
                artifact_environment=artifact_environment,
                trainer_environment_sha256=trainer_environment_sha256,
                allowed_training_leakage_classes=allowed_leakage,
                dataset_leakage_summary=dataset_leakage_summary,
                ok=False,
                errors=("policy checkpoint path must exist and be a file after exit 0",),
            ),
            overwrite=overwrite,
        )
        raise HarnessIOError("policy checkpoint path must exist and be a file after exit 0")
    execution_report = _execution_report(
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        status=PolicyTrainingExecutionStatus.completed,
        trainer=typed_trainer,
        cwd=cwd,
        timeout_seconds=timeout,
        returncode=completed.returncode,
        stdout_path=stdout_log_path,
        stderr_path=stderr_log_path,
        offline_guard_artifact=offline_guard_artifact,
        checkpoint_path=checkpoint_path,
        artifact_environment=artifact_environment,
        trainer_environment_sha256=trainer_environment_sha256,
        allowed_training_leakage_classes=allowed_leakage,
        dataset_leakage_summary=dataset_leakage_summary,
        ok=True,
        errors=(),
    )
    _write_execution_report(execution_output_path, execution_report, overwrite=overwrite)
    execution_report_artifact = _execution_report_artifact(
        execution_output_path,
        run_id=run_id,
    )
    execution_report_sha256 = execution_report_artifact.sha256
    assert execution_report_sha256 is not None
    policy_artifact = policy_checkpoint_artifact(
        checkpoint_path,
        run_id=run_id,
        source_training_dataset_report=source_training_dataset_report,
        trainer=typed_trainer,
        producer=_RUNNER_PRODUCER,
        extra_provenance={
            "runner_derivation": _RUNNER_DERIVATION,
            "source_dataset_run_id": source_report.run_id,
            "artifact_environment_keys": list(_RUNNER_ARTIFACT_ENV_KEYS),
            "policy_training_execution_report_sha256": execution_report_sha256,
            "allowed_training_leakage_classes": [
                leakage.value for leakage in allowed_leakage
            ],
            "dataset_leakage_summary": dataset_leakage_summary,
        },
    )
    candidate_policy = typed_template.to_policy_backend(
        run_id=run_id,
        source_training_dataset_report=source_training_dataset_report,
        trainer=typed_trainer,
        policy_artifact=policy_artifact,
        policy_training_execution_report_sha256=execution_report_sha256,
        allowed_training_leakage_classes=allowed_leakage,
        dataset_leakage_summary=dataset_leakage_summary,
    )
    report = record_policy_training_run(
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        source_training_dataset_report=source_training_dataset_report,
        trainer=typed_trainer,
        policy_artifact=policy_artifact,
        candidate_policy=candidate_policy,
        policy_training_execution_report=execution_report_artifact,
        metrics=metrics,
        notes=notes
        + (
            "Policy checkpoint was created by a bounded offline policy-training runner command.",
        ),
    )
    if report_output_path is not None:
        write_json(report_output_path, report.to_dict(), overwrite=overwrite)
    return PolicyTrainingCommandResult(
        report=report,
        stdout_path=str(stdout_log_path),
        stderr_path=str(stderr_log_path),
        report_path=str(report_output_path) if report_output_path is not None else None,
        artifact_environment=artifact_environment,
        execution_report=execution_report,
        execution_report_artifact=execution_report_artifact,
    )


def _execution_report(
    *,
    run_id: str,
    generated_at_utc: str,
    status: PolicyTrainingExecutionStatus,
    trainer: TrainerInvocation,
    cwd: Path,
    timeout_seconds: float,
    returncode: int | None,
    stdout_path: Path,
    stderr_path: Path,
    offline_guard_artifact: ArtifactRef,
    checkpoint_path: Path,
    artifact_environment: Mapping[str, str],
    trainer_environment_sha256: str,
    allowed_training_leakage_classes: tuple[LeakageClass, ...],
    dataset_leakage_summary: Mapping[str, int],
    ok: bool,
    errors: tuple[str, ...],
) -> PolicyTrainingExecutionReport:
    return PolicyTrainingExecutionReport(
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        status=status,
        command=trainer.command,
        working_directory=str(cwd),
        timeout_seconds=timeout_seconds,
        returncode=returncode,
        stdout=_log_artifact(stdout_path, kind=_STDOUT_ARTIFACT_KIND, run_id=run_id),
        stderr=_log_artifact(stderr_path, kind=_STDERR_ARTIFACT_KIND, run_id=run_id),
        offline_execution_guard=offline_guard_artifact,
        expected_policy_checkpoint_path=str(checkpoint_path),
        artifact_environment=artifact_environment,
        trainer_environment_sha256=trainer_environment_sha256,
        allowed_training_leakage_classes=allowed_training_leakage_classes,
        dataset_leakage_summary=dataset_leakage_summary,
        ok=ok,
        errors=errors,
    )


def _write_execution_report(
    path: Path,
    report: PolicyTrainingExecutionReport,
    *,
    overwrite: bool,
) -> None:
    write_json(path, report.to_dict(), overwrite=overwrite)


def _log_artifact(path: Path, *, kind: str, run_id: str) -> ArtifactRef:
    if not path.exists():
        raise HarnessIOError(f"{kind} path must exist before artifact binding")
    if not path.is_file():
        raise HarnessIOError(f"{kind} path must be a file")
    return ArtifactRef(
        kind=kind,
        path=str(path),
        sha256=sha256_file(path),
        provenance={
            "producer": _RUNNER_PRODUCER,
            "run_id": run_id,
            "derivation": _RUNNER_DERIVATION,
        },
    )


def _execution_report_artifact(path: Path, *, run_id: str) -> ArtifactRef:
    return ArtifactRef(
        kind=_EXECUTION_REPORT_ARTIFACT_KIND,
        path=str(path),
        sha256=sha256_file(path),
        provenance={
            "producer": _RUNNER_PRODUCER,
            "run_id": run_id,
            "derivation": _RUNNER_DERIVATION,
        },
    )


def _guard_artifact(path: Path, *, run_id: str) -> ArtifactRef:
    return ArtifactRef(
        kind=_OFFLINE_GUARD_ARTIFACT_KIND,
        path=str(path),
        sha256=sha256_file(path),
        provenance={
            "producer": _RUNNER_PRODUCER,
            "run_id": run_id,
            "derivation": _RUNNER_DERIVATION,
            "guard_kind": _OFFLINE_GUARD_KIND,
            "blocked_audit_events": list(_OFFLINE_GUARD_BLOCKED_AUDIT_EVENTS),
        },
    )


def _write_python_offline_guard(
    output_dir: Path,
    *,
    run_id: str,
    overwrite: bool,
) -> Path:
    guard_dir = output_dir / ".aic_policy_training_runner_guard" / _stable_run_slug(run_id)
    guard_path = guard_dir / "sitecustomize.py"
    try:
        guard_dir.mkdir(parents=True, exist_ok=True)
        if guard_path.exists():
            existing_source = guard_path.read_text(encoding="utf-8")
            if existing_source != _PYTHON_OFFLINE_GUARD_SOURCE:
                if not overwrite:
                    raise HarnessIOError(
                        "policy training runner offline guard already exists with different content"
                    )
                guard_path.write_text(_PYTHON_OFFLINE_GUARD_SOURCE, encoding="utf-8")
        else:
            guard_path.write_text(_PYTHON_OFFLINE_GUARD_SOURCE, encoding="utf-8")
    except OSError as exc:
        raise HarnessIOError(f"failed to write offline execution guard: {exc}") from exc
    return guard_path


def _is_guard_violation(completed: subprocess.CompletedProcess[bytes]) -> bool:
    return completed.returncode == _OFFLINE_GUARD_EXIT_CODE


def _guard_violation_error(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if "AIC offline policy-training guard blocked audit event:" in line:
            return "policy training offline guard blocked audit event: " + line.rsplit(
                ": ",
                1,
            )[-1]
    return "policy training offline guard blocked audit event"


def _stable_run_slug(run_id: str) -> str:
    encoded = run_id.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _enum_tuple(
    value: Any,
    enum_type: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if not allow_empty and not value:
        raise HarnessIOError(f"{field_name} must not be empty")
    items: list[Any] = []
    errors: list[str] = []
    for item in value:
        try:
            items.append(enum_type.parse(item))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
    if errors:
        raise HarnessIOError("; ".join(errors))
    return tuple(items)


def _allowed_training_leakage_classes(value: Any) -> tuple[LeakageClass, ...]:
    allowed = _enum_tuple(
        value,
        LeakageClass,
        "policy training runner.allowed_training_leakage_classes",
        allow_empty=False,
    )
    rejected = sorted(
        leakage.value
        for leakage in allowed
        if leakage in _ALWAYS_REJECTED_TRAINING_LEAKAGE_CLASSES
    )
    if rejected:
        raise HarnessIOError(
            "policy training runner.allowed_training_leakage_classes cannot include: "
            + ", ".join(rejected)
        )
    return cast(tuple[LeakageClass, ...], allowed)


def _validate_training_dataset_leakage(
    source_report: Any,
    allowed_training_leakage_classes: tuple[LeakageClass, ...],
) -> None:
    allowed = set(allowed_training_leakage_classes)
    disallowed: set[str] = set()
    for example in source_report.examples:
        if example.leakage_class in _ALWAYS_REJECTED_TRAINING_LEAKAGE_CLASSES:
            disallowed.add(example.leakage_class.value)
        elif example.leakage_class not in allowed:
            disallowed.add(example.leakage_class.value)
    if disallowed:
        raise HarnessIOError(
            "source training dataset contains leakage classes not allowed for "
            "policy training runner: "
            + ", ".join(sorted(disallowed))
        )


def _dataset_leakage_summary(value: Mapping[str, int]) -> dict[str, int]:
    return dict(_count_mapping(value, "policy training runner.dataset_leakage_summary"))


def _candidate_policy_template_backend_errors(
    template: CandidatePolicyTemplate,
) -> list[str]:
    dummy_artifact = ArtifactRef(
        kind="policy_checkpoint",
        path="/__aic_policy_training_template_validation__/policy.ckpt",
        sha256="0" * 64,
        provenance={
            "producer": _RUNNER_PRODUCER,
            "run_id": "template-validation",
            "derivation": _RUNNER_DERIVATION,
            "source_training_dataset_report_sha256": "0" * 64,
            "trainer_fingerprint_sha256": "0" * 64,
        },
    )
    try:
        PolicyBackendSpec(
            backend_kind=template.backend_kind,
            name=template.name,
            runtime_role=RuntimeRole.live_policy,
            training_sources=template.training_sources,
            simulator_sources=template.simulator_sources,
            runtime_allowed=True,
            leakage_class=LeakageClass.legal_policy_input,
            runtime_boundary=RuntimeBoundaryProof(
                deterministic=template.deterministic,
                uses_online_language_model_control=False,
                legal_observation_contract=template.legal_observation_contract,
                policy_artifact=dummy_artifact,
                notes=template.runtime_boundary_notes,
            ),
            description=template.description,
            config=template.config,
            provenance={
                "producer": _RUNNER_PRODUCER,
                "run_id": "template-validation",
                "derivation": _RUNNER_DERIVATION,
                "source_training_dataset_report_sha256": "0" * 64,
                "policy_artifact_sha256": "0" * 64,
                "trainer_fingerprint_sha256": "0" * 64,
            },
        )
    except SchemaValidationError as exc:
        return list(exc.errors)
    return []


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


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value


def _count_mapping(value: Any, field_name: str) -> MappingProxyType[str, int]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, int] = {}
    for key, item in value.items():
        key_text = _require_nonempty_text(key, f"{field_name} key")
        if type(item) is not int or item < 0:
            raise HarnessIOError(f"{field_name}.{key_text} must be a nonnegative integer")
        copied[key_text] = item
    return MappingProxyType(copied)


def _require_path(value: str | Path, field_name: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        raise HarnessIOError(f"{field_name} must be a nonempty path")
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise HarnessIOError(f"{field_name} cannot be resolved") from exc


def _prepare_output_path(
    value: str | Path,
    field_name: str,
    *,
    overwrite: bool,
    unlink_existing: bool,
) -> Path:
    path = _require_path(value, field_name)
    if path.exists():
        if path.is_dir():
            raise HarnessIOError(f"{field_name} must not point to a directory")
        if not overwrite:
            raise HarnessIOError(f"{field_name} already exists; pass overwrite=True to replace it")
        if unlink_existing:
            try:
                path.unlink()
            except OSError as exc:
                raise HarnessIOError(f"{field_name} could not remove existing file") from exc
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HarnessIOError(f"{field_name} parent directory could not be created") from exc
    return path


def _reject_duplicate_output_paths(paths: Mapping[str, Path]) -> None:
    seen: dict[Path, str] = {}
    for field_name, path in paths.items():
        previous = seen.get(path)
        if previous is not None:
            raise HarnessIOError(
                "policy training runner output paths must be distinct: "
                f"{previous} and {field_name}"
            )
        seen[path] = field_name


def _require_timeout(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)) or float(value) <= 0:
        raise HarnessIOError("policy training runner.timeout_seconds must be a finite positive number")
    return float(value)


def _trainer_working_directory(trainer: TrainerInvocation) -> Path:
    if trainer.working_directory is None:
        try:
            return Path.cwd().resolve(strict=True)
        except OSError as exc:
            raise HarnessIOError("policy training runner current working directory cannot be resolved") from exc
    cwd = _require_path(trainer.working_directory, "trainer invocation.working_directory")
    if not cwd.exists():
        raise HarnessIOError("trainer invocation.working_directory must exist")
    if not cwd.is_dir():
        raise HarnessIOError("trainer invocation.working_directory must be a directory")
    return cwd


def _trainer_with_effective_working_directory(
    trainer: TrainerInvocation,
    cwd: Path,
) -> TrainerInvocation:
    if trainer.working_directory == str(cwd):
        return trainer
    return TrainerInvocation(
        trainer_id=trainer.trainer_id,
        trainer_name=trainer.trainer_name,
        backend_kind=trainer.backend_kind,
        command=trainer.command,
        config=trainer.config,
        environment=trainer.environment,
        working_directory=str(cwd),
        container_image=trainer.container_image,
        seed=trainer.seed,
        offline_only=trainer.offline_only,
        runtime_allowed=trainer.runtime_allowed,
        uses_online_language_model_control=trainer.uses_online_language_model_control,
    )


def _subprocess_environment(
    trainer: TrainerInvocation,
    artifact_environment: Mapping[str, str],
    *,
    guard_dir: Path,
) -> dict[str, str]:
    env = dict(trainer.environment)
    runner_owned_keys = set(artifact_environment) | {
        _OFFLINE_GUARD_ENV_KEY,
        "PYTHONPATH",
    }
    conflicting_keys = sorted(key for key in runner_owned_keys if key in env)
    if conflicting_keys:
        raise HarnessIOError(
            "trainer invocation.environment must not define runner-owned keys: "
            + ", ".join(conflicting_keys)
        )
    env.update(artifact_environment)
    env[_OFFLINE_GUARD_ENV_KEY] = _OFFLINE_GUARD_KIND
    env["PYTHONPATH"] = str(guard_dir)
    return env


def _require_executable_resolution(trainer: TrainerInvocation, env: Mapping[str, str]) -> None:
    command = trainer.command
    if not command:
        raise HarnessIOError("trainer invocation.command must not be empty")
    executable = Path(command[0]).expanduser()
    if not executable.is_absolute() and "PATH" not in env:
        raise HarnessIOError(
            "policy training runner requires an absolute trainer command executable "
            "or explicit PATH in trainer invocation.environment"
        )


def _require_python_offline_guardable_command(trainer: TrainerInvocation) -> None:
    executable = Path(trainer.command[0]).expanduser()
    if not executable.is_absolute():
        raise HarnessIOError(
            "policy training runner requires an absolute Python trainer executable "
            "for the offline audit guard"
        )
    if not executable.exists() or not executable.is_file():
        raise HarnessIOError(
            "policy training runner Python trainer executable must exist and be a file"
        )
    current_interpreter = Path(sys.executable).expanduser().resolve(strict=True)
    if executable.resolve(strict=True) != current_interpreter:
        raise HarnessIOError(
            "policy training runner requires trainer.command[0] to be this harness "
            "Python interpreter so the offline audit guard is enforceable"
        )
    executable_name = executable.name.lower()
    if not executable_name.startswith("python"):
        raise HarnessIOError(
            "policy training runner currently supports only Python trainer commands "
            "so the offline audit guard can be enforced"
        )
    disabled_guard_flags = sorted(
        flag
        for flag in trainer.command[1:]
        if flag in {"-S", "-E", "-I"} or flag.startswith(("-S", "-E", "-I"))
    )
    if disabled_guard_flags:
        raise HarnessIOError(
            "trainer invocation.command must not disable the Python offline audit guard: "
            + ", ".join(disabled_guard_flags)
        )


def _reject_prohibited_command(trainer: TrainerInvocation) -> None:
    lowered_command = tuple(part.lower() for part in trainer.command)
    violations = sorted(
        token
        for token in _PROHIBITED_RUNTIME_COMMAND_TOKENS
        if any(token in part for part in lowered_command)
    )
    if violations:
        raise HarnessIOError(
            "policy training runner must not launch runtime/evaluation surfaces: "
            + ", ".join(violations)
        )


def _reject_hidden_command_artifacts(trainer: TrainerInvocation) -> None:
    hidden_args: list[str] = []
    for index, value in enumerate(trainer.command[1:], start=1):
        if _looks_like_artifact_location(value):
            hidden_args.append(f"command[{index}]")
    if hidden_args:
        raise HarnessIOError(
            "trainer invocation.command must not hide artifact references; use "
            "runner-owned artifact bindings instead: "
            + ", ".join(hidden_args)
        )


def _looks_like_artifact_location(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized.startswith("-"):
        option_value = _option_embedded_value(normalized)
        return option_value is not None and _looks_like_artifact_location(option_value)
    parsed = urlparse(normalized)
    return (
        bool(parsed.scheme and (parsed.netloc or parsed.scheme == "file"))
        or normalized.startswith(("/", "./", "../", "~"))
        or normalized.endswith(_ARTIFACT_VALUE_SUFFIXES)
    )


def _option_embedded_value(normalized: str) -> str | None:
    for separator in ("=", ":"):
        if separator in normalized:
            return normalized.split(separator, 1)[1].strip()
    return None


def _write_process_output(path: Path, value: bytes | str | None) -> None:
    if value is None:
        data = b""
    elif isinstance(value, bytes):
        data = value
    else:
        data = value.encode("utf-8", errors="replace")
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise HarnessIOError(f"failed to write process output {path}: {exc}") from exc


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _environment_mapping(value: Any, field_name: str) -> MappingProxyType[str, str]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, str] = {}
    for key, item in value.items():
        key_text = _require_nonempty_text(key, f"{field_name} key")
        item_text = _require_nonempty_text(item, f"{field_name}.{key_text}")
        copied[key_text] = item_text
    return MappingProxyType(copied)


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
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise HarnessIOError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
