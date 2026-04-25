"""Train-eval-promote finalization for offline-trained AIC policies.

This module binds a completed offline policy-training report to the live eval
that used the resulting checkpoint, then delegates scoring, ledger, and
promotion to the existing live eval finalizer. It observes evidence only after
the official evaluator has completed.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, cast

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.constants import DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH
from aic_signal_harness.live_eval import (
    LiveEvalFinalization,
    finalize_live_eval_run,
    runtime_env_from_mapping,
)
from aic_signal_harness.ledger import LedgerEntry, read_ledger_entries
from aic_signal_harness.manifest import RunManifest, RunStatus
from aic_signal_harness.policy_training import (
    PolicyTrainingExecutionReport,
    PolicyTrainingRunReport,
)
from aic_signal_harness.promotion import PromotionDecision
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    PolicyBackendSpec,
    RuntimeBoundaryProof,
    SCHEMA_VERSION,
    SchemaValidationError,
    utc_now_iso,
)
from aic_signal_harness.training_dataset import TrainingDatasetReport


TRAINED_POLICY_EVAL_BINDING_KIND = "trained_policy_eval_binding"
POLICY_TRAINING_REPORT_KIND = "policy_training_report"
POLICY_CHECKPOINT_KIND = "policy_checkpoint"
TRAIN_EVAL_PROMOTE_SUMMARY_NAME = "train_eval_promote_summary.json"
TRAINED_POLICY_EVAL_BINDING_NAME = "trained_policy_eval_binding.json"
_BINDING_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "train_run_id",
        "eval_run_id",
        "source_policy_training_report",
        "source_policy_training_execution_report",
        "source_policy_artifact",
        "runtime_policy_artifact",
        "runtime_image_artifact",
        "runtime_checkpoint_container_path",
        "candidate_policy",
        "eval_backend",
        "ok",
        "errors",
        "notes",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "train_run_id",
        "eval_run_id",
        "binding_report",
        "manifest",
        "ledger_entry_path",
        "promotion_path",
        "live_eval_summary_path",
        "score_total",
    }
)


@dataclass(frozen=True)
class TrainedPolicyEvalBindingReport:
    """Proof that one live eval is bound to one completed training run."""

    generated_at_utc: str
    train_run_id: str
    eval_run_id: str
    source_policy_training_report: ArtifactRef
    source_policy_training_execution_report: ArtifactRef
    source_policy_artifact: ArtifactRef
    runtime_policy_artifact: ArtifactRef
    runtime_image_artifact: ArtifactRef
    runtime_checkpoint_container_path: str
    candidate_policy: PolicyBackendSpec
    eval_backend: PolicyBackendSpec
    ok: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("generated_at_utc", "train_run_id", "eval_run_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"trained policy eval binding.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "source_policy_training_report",
            "source_policy_training_execution_report",
            "source_policy_artifact",
            "runtime_policy_artifact",
            "runtime_image_artifact",
        ):
            value = getattr(self, field_name)
            if isinstance(value, ArtifactRef):
                continue
            try:
                object.__setattr__(self, field_name, ArtifactRef.from_dict(value))
            except SchemaValidationError as exc:
                errors.append(str(exc))
        for field_name in ("candidate_policy", "eval_backend"):
            value = getattr(self, field_name)
            if isinstance(value, PolicyBackendSpec):
                continue
            try:
                object.__setattr__(self, field_name, PolicyBackendSpec.from_dict(value))
            except SchemaValidationError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "runtime_checkpoint_container_path",
                _require_absolute_container_path(
                    self.runtime_checkpoint_container_path,
                    "trained policy eval binding.runtime_checkpoint_container_path",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("errors", "notes"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _as_text_sequence(
                        getattr(self, field_name),
                        f"trained policy eval binding.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if self.ok is not True:
            errors.append("trained policy eval binding.ok must be true")
        if self.errors:
            errors.append("trained policy eval binding.errors must be empty")

        if not errors:
            typed_report = self.source_policy_training_report
            typed_execution = self.source_policy_training_execution_report
            typed_policy = self.source_policy_artifact
            typed_runtime = self.runtime_policy_artifact
            typed_image = self.runtime_image_artifact
            typed_candidate = self.candidate_policy
            typed_eval_backend = self.eval_backend
            assert isinstance(typed_report, ArtifactRef)
            assert isinstance(typed_execution, ArtifactRef)
            assert isinstance(typed_policy, ArtifactRef)
            assert isinstance(typed_runtime, ArtifactRef)
            assert isinstance(typed_image, ArtifactRef)
            assert isinstance(typed_candidate, PolicyBackendSpec)
            assert isinstance(typed_eval_backend, PolicyBackendSpec)
            training_report = _read_policy_training_report_artifact(
                typed_report,
                expected_run_id=self.train_run_id,
            )
            errors.extend(
                _binding_consistency_errors(
                    binding=self,
                    training_report=training_report,
                    source_policy_training_execution_report=typed_execution,
                    source_policy_artifact=typed_policy,
                    runtime_policy_artifact=typed_runtime,
                    runtime_image_artifact=typed_image,
                    candidate_policy=typed_candidate,
                    eval_backend=typed_eval_backend,
                )
            )
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "train_run_id": self.train_run_id,
            "eval_run_id": self.eval_run_id,
            "source_policy_training_report": self.source_policy_training_report.to_dict(),
            "source_policy_training_execution_report": (
                self.source_policy_training_execution_report.to_dict()
            ),
            "source_policy_artifact": self.source_policy_artifact.to_dict(),
            "runtime_policy_artifact": self.runtime_policy_artifact.to_dict(),
            "runtime_image_artifact": self.runtime_image_artifact.to_dict(),
            "runtime_checkpoint_container_path": self.runtime_checkpoint_container_path,
            "candidate_policy": self.candidate_policy.to_dict(),
            "eval_backend": self.eval_backend.to_dict(),
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainedPolicyEvalBindingReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("trained policy eval binding report must be a mapping")
        _reject_unknown_keys(value, _BINDING_REPORT_KEYS, "trained policy eval binding report")
        _require_keys(
            value,
            _BINDING_REPORT_KEYS - {"notes"},
            "trained policy eval binding report",
        )
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            train_run_id=cast(Any, value.get("train_run_id")),
            eval_run_id=cast(Any, value.get("eval_run_id")),
            source_policy_training_report=cast(Any, value.get("source_policy_training_report")),
            source_policy_training_execution_report=cast(
                Any,
                value.get("source_policy_training_execution_report"),
            ),
            source_policy_artifact=cast(Any, value.get("source_policy_artifact")),
            runtime_policy_artifact=cast(Any, value.get("runtime_policy_artifact")),
            runtime_image_artifact=cast(Any, value.get("runtime_image_artifact")),
            runtime_checkpoint_container_path=cast(
                Any,
                value.get("runtime_checkpoint_container_path"),
            ),
            candidate_policy=cast(Any, value.get("candidate_policy")),
            eval_backend=cast(Any, value.get("eval_backend")),
            ok=cast(Any, value.get("ok")),
            errors=cast(Any, value.get("errors")),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class TrainEvalPromoteSummary:
    """High-level summary of one completed train-eval-promote finalization."""

    generated_at_utc: str
    train_run_id: str
    eval_run_id: str
    binding_report: ArtifactRef
    manifest: ArtifactRef
    ledger_entry_path: str
    promotion_path: str | None
    live_eval_summary_path: str
    score_total: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in (
            "generated_at_utc",
            "train_run_id",
            "eval_run_id",
            "ledger_entry_path",
            "live_eval_summary_path",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"train-eval-promote summary.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "promotion_path",
                _optional_nonempty_text(
                    self.promotion_path,
                    "train-eval-promote summary.promotion_path",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("binding_report", "manifest"):
            value = getattr(self, field_name)
            if isinstance(value, ArtifactRef):
                continue
            try:
                object.__setattr__(self, field_name, ArtifactRef.from_dict(value))
            except SchemaValidationError as exc:
                errors.append(str(exc))
        if type(self.score_total) not in (float, int):
            errors.append("train-eval-promote summary.score_total must be a number")
        elif isinstance(self.score_total, bool) or not math.isfinite(float(self.score_total)):
            errors.append("train-eval-promote summary.score_total must be a finite number")
        else:
            object.__setattr__(self, "score_total", float(self.score_total))
        if not errors:
            if self.binding_report.kind != TRAINED_POLICY_EVAL_BINDING_KIND:
                errors.append(
                    "train-eval-promote summary.binding_report.kind must be "
                    f"{TRAINED_POLICY_EVAL_BINDING_KIND!r}"
                )
            if self.manifest.kind != "run_manifest":
                errors.append("train-eval-promote summary.manifest.kind must be 'run_manifest'")
            errors.extend(_summary_evidence_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "train_run_id": self.train_run_id,
            "eval_run_id": self.eval_run_id,
            "binding_report": self.binding_report.to_dict(),
            "manifest": self.manifest.to_dict(),
            "ledger_entry_path": self.ledger_entry_path,
            "promotion_path": self.promotion_path,
            "live_eval_summary_path": self.live_eval_summary_path,
            "score_total": float(self.score_total),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainEvalPromoteSummary":
        if not isinstance(value, Mapping):
            raise HarnessIOError("train-eval-promote summary must be a mapping")
        _reject_unknown_keys(value, _SUMMARY_KEYS, "train-eval-promote summary")
        _require_keys(value, _SUMMARY_KEYS, "train-eval-promote summary")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            train_run_id=cast(Any, value.get("train_run_id")),
            eval_run_id=cast(Any, value.get("eval_run_id")),
            binding_report=cast(Any, value.get("binding_report")),
            manifest=cast(Any, value.get("manifest")),
            ledger_entry_path=cast(Any, value.get("ledger_entry_path")),
            promotion_path=cast(Any, value.get("promotion_path")),
            live_eval_summary_path=cast(Any, value.get("live_eval_summary_path")),
            score_total=cast(Any, value.get("score_total")),
        )


@dataclass(frozen=True)
class TrainEvalPromoteFinalization:
    """Paths and typed outputs produced by the train-eval-promote wrapper."""

    binding_report: TrainedPolicyEvalBindingReport
    binding_artifact: ArtifactRef
    binding_path: Path
    live_eval: LiveEvalFinalization
    summary: TrainEvalPromoteSummary
    summary_path: Path


@dataclass(frozen=True)
class StagedPolicyTrainingBundle:
    """Self-contained policy-training report bundle staged for remote eval."""

    local_bundle_root: Path
    runtime_bundle_root: str
    policy_training_report_path: Path
    runtime_policy_training_report_path: str
    policy_checkpoint_path: Path
    runtime_policy_checkpoint_path: str


def finalize_trained_policy_eval(
    *,
    run_id: str,
    result_root: str | Path,
    harness_root: str | Path,
    scoring_yaml: str | Path,
    policy_training_report: str | Path | ArtifactRef | Mapping[str, Any],
    runtime_policy_checkpoint: str | Path,
    runtime_checkpoint_container_path: str = DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH,
    policy_trace: str | Path | None = None,
    ledger_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    update_baseline_path: str | Path | None = None,
    bootstrap_promotion: bool = False,
    min_improvement: float = 0.0,
    eligible_for_submission: bool = False,
    gate_id: str = "trained_policy_live_eval",
    experiment_id: str | None = None,
    hypothesis: str | None = None,
    model_image: str | None = None,
    model_image_id: str | None = None,
    backend_kind: str | None = None,
    planner_mode: str | None = None,
    runtime_env: Mapping[str, str] | None = None,
    binding_output_path: str | Path | None = None,
    summary_output_path: str | Path | None = None,
    overwrite: bool = False,
    append_ledger: bool = True,
    write_next_experiment: bool = True,
    generated_at_utc: str | None = None,
) -> TrainEvalPromoteFinalization:
    """Finalize one live eval of a completed offline-trained policy."""

    generated_at = utc_now_iso() if generated_at_utc is None else generated_at_utc
    eval_run_id = _require_nonempty_text(run_id, "train-eval-promote.run_id")
    harness_dir = Path(harness_root).expanduser().resolve(strict=False)
    binding_path = (
        harness_dir / TRAINED_POLICY_EVAL_BINDING_NAME
        if binding_output_path is None
        else Path(binding_output_path).expanduser().resolve(strict=False)
    )
    summary_path = (
        harness_dir / TRAIN_EVAL_PROMOTE_SUMMARY_NAME
        if summary_output_path is None
        else Path(summary_output_path).expanduser().resolve(strict=False)
    )
    _reject_output_collisions(binding_path, summary_path)
    _preflight_train_eval_promote_outputs(
        run_id=eval_run_id,
        binding_path=binding_path,
        summary_path=summary_path,
        harness_root=harness_dir,
        ledger_path=ledger_path,
        baseline_path=baseline_path,
        update_baseline_path=update_baseline_path,
        bootstrap_promotion=bootstrap_promotion,
        append_ledger=append_ledger,
        overwrite=overwrite,
    )

    source_training_artifact = _policy_training_report_artifact(policy_training_report)
    training_report = _read_policy_training_report_artifact(
        source_training_artifact,
        expected_run_id=None,
    )
    _validate_backend_kind_override(backend_kind, training_report.candidate_policy.backend_kind)
    if training_report.policy_training_execution_report is None:
        raise HarnessIOError(
            "train-eval-promote requires a controlled policy_training_execution_report"
        )
    runtime_env_with_checkpoint = _runtime_env_with_checkpoint(
        runtime_env,
        runtime_checkpoint_container_path=runtime_checkpoint_container_path,
        planner_mode=planner_mode,
    )
    runtime_planner_mode = _resolved_runtime_planner_mode(
        runtime_env_with_checkpoint,
        planner_mode=planner_mode,
    )
    runtime_artifact = _runtime_policy_checkpoint_artifact(
        runtime_policy_checkpoint,
        train_run_id=training_report.run_id,
        eval_run_id=eval_run_id,
        runtime_checkpoint_container_path=runtime_checkpoint_container_path,
        expected_policy_sha256=training_report.policy_artifact.sha256,
    )
    runtime_image_artifact = _runtime_image_artifact(
        model_image=model_image,
        model_image_id=model_image_id,
        eval_run_id=eval_run_id,
    )
    eval_backend = _eval_backend_from_training_report(
        training_report,
        source_policy_training_report=source_training_artifact,
        runtime_policy_artifact=runtime_artifact,
        runtime_image_artifact=runtime_image_artifact,
        eval_run_id=eval_run_id,
        runtime_checkpoint_container_path=runtime_checkpoint_container_path,
        runtime_env=runtime_env_with_checkpoint,
        planner_mode=runtime_planner_mode,
    )
    binding_report = TrainedPolicyEvalBindingReport(
        generated_at_utc=generated_at,
        train_run_id=training_report.run_id,
        eval_run_id=eval_run_id,
        source_policy_training_report=source_training_artifact,
        source_policy_training_execution_report=training_report.policy_training_execution_report,
        source_policy_artifact=training_report.policy_artifact,
        runtime_policy_artifact=runtime_artifact,
        runtime_image_artifact=runtime_image_artifact,
        runtime_checkpoint_container_path=runtime_checkpoint_container_path,
        candidate_policy=training_report.candidate_policy,
        eval_backend=eval_backend,
        notes=(
            "Generated before live eval ledger/promotion; proves the evaluated runtime checkpoint bytes match the completed training report.",
        ),
    )
    write_json(binding_path, binding_report.to_dict(), overwrite=overwrite)
    binding_artifact = ArtifactRef(
        kind=TRAINED_POLICY_EVAL_BINDING_KIND,
        path=str(binding_path),
        sha256=sha256_file(binding_path),
        provenance={
            "producer": "aic_signal_harness.train_eval_promote",
            "derivation": "finalize_trained_policy_eval",
            "train_run_id": training_report.run_id,
            "eval_run_id": eval_run_id,
            "source_policy_training_report_sha256": source_training_artifact.sha256,
            "runtime_policy_artifact_sha256": runtime_artifact.sha256,
            "runtime_image_artifact_sha256": runtime_image_artifact.sha256,
        },
    )
    extra_artifacts = (
        source_training_artifact,
        training_report.policy_training_execution_report,
        runtime_artifact,
        runtime_image_artifact,
        binding_artifact,
    )
    live_eval = finalize_live_eval_run(
        run_id=eval_run_id,
        result_root=result_root,
        harness_root=harness_dir,
        scoring_yaml=scoring_yaml,
        policy_trace=policy_trace,
        ledger_path=ledger_path,
        baseline_path=baseline_path,
        update_baseline_path=update_baseline_path,
        bootstrap_promotion=bootstrap_promotion,
        min_improvement=min_improvement,
        eligible_for_submission=eligible_for_submission,
        gate_id=gate_id,
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        model_image=model_image,
        model_image_id=model_image_id,
        backend=eval_backend,
        backend_kind=backend_kind,
        planner_mode=planner_mode,
        runtime_env=runtime_env_with_checkpoint,
        extra_manifest_artifacts=extra_artifacts,
        overwrite=overwrite,
        append_ledger=append_ledger,
        write_next_experiment=write_next_experiment,
        generated_at_utc=generated_at,
    )
    if live_eval.manifest.score is None:
        raise HarnessIOError("train-eval-promote live eval manifest must include score")
    summary = TrainEvalPromoteSummary(
        generated_at_utc=generated_at,
        train_run_id=training_report.run_id,
        eval_run_id=eval_run_id,
        binding_report=binding_artifact,
        manifest=live_eval.manifest_artifact,
        ledger_entry_path=str(live_eval.ledger_entry_path),
        promotion_path=None if live_eval.promotion_path is None else str(live_eval.promotion_path),
        live_eval_summary_path=str(live_eval.summary_path),
        score_total=live_eval.manifest.score.total,
    )
    write_json(summary_path, summary.to_dict(), overwrite=overwrite)
    return TrainEvalPromoteFinalization(
        binding_report=binding_report,
        binding_artifact=binding_artifact,
        binding_path=binding_path,
        live_eval=live_eval,
        summary=summary,
        summary_path=summary_path,
    )


def stage_policy_training_report_bundle(
    *,
    policy_training_report: str | Path,
    bundle_root: str | Path,
    runtime_bundle_root: str | Path | None = None,
    overwrite: bool = False,
) -> StagedPolicyTrainingBundle:
    """Copy a policy-training report and its byte-bound sidecars into one bundle."""

    source_report_path = Path(policy_training_report).expanduser().resolve(strict=False)
    if not source_report_path.exists():
        raise HarnessIOError(f"policy_training_report does not exist: {source_report_path}")
    if not source_report_path.is_file():
        raise HarnessIOError(f"policy_training_report must be a file: {source_report_path}")
    local_bundle_root = Path(bundle_root).expanduser().resolve(strict=False)
    runtime_root = _runtime_bundle_root(
        runtime_bundle_root,
        local_bundle_root=local_bundle_root,
    )
    source_report = PolicyTrainingRunReport.from_dict(read_json(source_report_path))
    if source_report.policy_training_execution_report is None:
        raise HarnessIOError(
            "stage-training-bundle requires a controlled policy_training_execution_report"
        )
    report_payload = copy.deepcopy(source_report.to_dict())
    dataset_payload = TrainingDatasetReport.from_dict(
        read_json(_artifact_local_path(source_report.source_training_dataset_report.to_dict()))
    ).to_dict()
    execution_payload = PolicyTrainingExecutionReport.from_dict(
        read_json(_artifact_local_path(source_report.policy_training_execution_report.to_dict()))
    ).to_dict()

    if source_report.policy_artifact.path is not None and Path(
        source_report.policy_artifact.path
    ).expanduser().resolve(strict=False) == source_report_path:
        raise HarnessIOError("policy training report must not also be the policy artifact")
    if (
        source_report.policy_training_execution_report.path is not None
        and Path(source_report.policy_training_execution_report.path)
        .expanduser()
        .resolve(strict=False)
        == source_report_path
    ):
        raise HarnessIOError(
            "policy training report must not also be the policy_training_execution_report"
        )

    local_bundle_root.mkdir(parents=True, exist_ok=True)
    staged_signal_path = local_bundle_root / "training_signal_report.json"
    staged_dataset_path = local_bundle_root / "training_dataset_report.json"
    staged_checkpoint_path = local_bundle_root / _checkpoint_bundle_name(
        source_report.policy_artifact.to_dict()
    )
    staged_stdout_path = local_bundle_root / "policy_training_stdout.log"
    staged_stderr_path = local_bundle_root / "policy_training_stderr.log"
    staged_guard_path = local_bundle_root / "policy_training_offline_guard.py"
    staged_execution_path = local_bundle_root / "policy_training_execution_report.json"
    staged_report_path = local_bundle_root / "policy_training_report.json"

    _copy_artifact_file(
        dataset_payload["source_training_signal_report"],
        staged_signal_path,
        overwrite=overwrite,
    )
    _copy_artifact_file(report_payload["policy_artifact"], staged_checkpoint_path, overwrite=overwrite)
    _copy_artifact_file(execution_payload["stdout"], staged_stdout_path, overwrite=overwrite)
    _copy_artifact_file(execution_payload["stderr"], staged_stderr_path, overwrite=overwrite)
    _copy_artifact_file(
        execution_payload["offline_execution_guard"],
        staged_guard_path,
        overwrite=overwrite,
    )

    _set_artifact_path(
        dataset_payload["source_training_signal_report"],
        _runtime_child(runtime_root, staged_signal_path.name),
    )
    write_json(staged_dataset_path, dataset_payload, overwrite=overwrite)
    staged_dataset_sha = sha256_file(staged_dataset_path)

    runtime_dataset_path = _runtime_child(runtime_root, staged_dataset_path.name)
    runtime_checkpoint_path = _runtime_child(runtime_root, staged_checkpoint_path.name)
    _set_artifact_path(execution_payload["stdout"], _runtime_child(runtime_root, staged_stdout_path.name))
    _set_artifact_path(execution_payload["stderr"], _runtime_child(runtime_root, staged_stderr_path.name))
    _set_artifact_path(
        execution_payload["offline_execution_guard"],
        _runtime_child(runtime_root, staged_guard_path.name),
    )
    execution_payload["expected_policy_checkpoint_path"] = runtime_checkpoint_path
    artifact_environment = execution_payload["artifact_environment"]
    artifact_environment["AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT"] = runtime_dataset_path
    artifact_environment["AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT"] = runtime_checkpoint_path
    write_json(staged_execution_path, execution_payload, overwrite=overwrite)
    staged_execution_sha = sha256_file(staged_execution_path)

    source_dataset_artifact = report_payload["source_training_dataset_report"]
    _set_artifact_path(source_dataset_artifact, runtime_dataset_path)
    source_dataset_artifact["sha256"] = staged_dataset_sha
    execution_artifact = report_payload["policy_training_execution_report"]
    _set_artifact_path(execution_artifact, _runtime_child(runtime_root, staged_execution_path.name))
    execution_artifact["sha256"] = staged_execution_sha
    _retarget_policy_artifact(
        report_payload["policy_artifact"],
        runtime_checkpoint_path=runtime_checkpoint_path,
        source_dataset_sha=staged_dataset_sha,
        execution_sha=staged_execution_sha,
    )
    candidate_policy = report_payload["candidate_policy"]
    candidate_policy["provenance"]["source_training_dataset_report_sha256"] = staged_dataset_sha
    candidate_policy["provenance"]["policy_training_execution_report_sha256"] = staged_execution_sha
    candidate_policy["runtime_boundary"]["policy_artifact"] = copy.deepcopy(
        report_payload["policy_artifact"]
    )
    write_json(staged_report_path, report_payload, overwrite=overwrite)

    if runtime_root == str(local_bundle_root):
        PolicyTrainingRunReport.from_dict(read_json(staged_report_path))

    return StagedPolicyTrainingBundle(
        local_bundle_root=local_bundle_root,
        runtime_bundle_root=runtime_root,
        policy_training_report_path=staged_report_path,
        runtime_policy_training_report_path=_runtime_child(runtime_root, staged_report_path.name),
        policy_checkpoint_path=staged_checkpoint_path,
        runtime_policy_checkpoint_path=runtime_checkpoint_path,
    )


def _policy_training_report_artifact(
    source: str | Path | ArtifactRef | Mapping[str, Any],
) -> ArtifactRef:
    if isinstance(source, ArtifactRef):
        return source
    if isinstance(source, Mapping):
        return ArtifactRef.from_dict(source)
    report_path = Path(source).expanduser().resolve(strict=False)
    if not report_path.exists():
        raise HarnessIOError(f"policy_training_report does not exist: {report_path}")
    if not report_path.is_file():
        raise HarnessIOError(f"policy_training_report must be a file: {report_path}")
    report = PolicyTrainingRunReport.from_dict(read_json(report_path))
    return ArtifactRef(
        kind=POLICY_TRAINING_REPORT_KIND,
        path=str(report_path),
        sha256=sha256_file(report_path),
        provenance={
            "producer": "aic_signal_harness.train_eval_promote",
            "derivation": "policy_training_report_artifact",
            "run_id": report.run_id,
        },
    )


def _runtime_bundle_root(runtime_bundle_root: str | Path | None, *, local_bundle_root: Path) -> str:
    if runtime_bundle_root is None:
        return str(local_bundle_root)
    value = _require_nonempty_text(str(runtime_bundle_root), "runtime_bundle_root")
    return _require_absolute_container_path(value, "runtime_bundle_root")


def _artifact_local_path(artifact_payload: Mapping[str, Any]) -> Path:
    artifact = ArtifactRef.from_dict(artifact_payload)
    path = local_artifact_path(
        path=artifact.path,
        uri=artifact.uri,
        field_name=f"{artifact.kind} artifact",
    )
    if path is None:
        raise HarnessIOError(f"{artifact.kind} artifact must be local byte-verifiable")
    if not path.exists():
        raise HarnessIOError(f"{artifact.kind} artifact path does not exist: {path}")
    if not path.is_file():
        raise HarnessIOError(f"{artifact.kind} artifact path must be a file: {path}")
    if artifact.sha256 is not None and sha256_file(path) != artifact.sha256:
        raise HarnessIOError(f"{artifact.kind} artifact sha256 must match path")
    return path


def _copy_artifact_file(
    artifact_payload: Mapping[str, Any],
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    source = _artifact_local_path(artifact_payload)
    if destination.exists() and not overwrite:
        raise HarnessIOError(
            f"{destination} already exists; pass overwrite=True to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _checkpoint_bundle_name(policy_artifact: Mapping[str, Any]) -> str:
    source_path = _artifact_local_path(policy_artifact)
    suffix = source_path.suffix
    return "policy_checkpoint" + (suffix if suffix else ".ckpt")


def _set_artifact_path(artifact_payload: dict[str, Any], runtime_path: str) -> None:
    artifact_payload["path"] = runtime_path
    artifact_payload.pop("uri", None)


def _retarget_policy_artifact(
    artifact_payload: dict[str, Any],
    *,
    runtime_checkpoint_path: str,
    source_dataset_sha: str,
    execution_sha: str,
) -> None:
    _set_artifact_path(artifact_payload, runtime_checkpoint_path)
    provenance = artifact_payload["provenance"]
    provenance["source_training_dataset_report_sha256"] = source_dataset_sha
    provenance["policy_training_execution_report_sha256"] = execution_sha


def _runtime_child(runtime_root: str, name: str) -> str:
    return str(PurePosixPath(runtime_root) / name)


def _read_policy_training_report_artifact(
    artifact: ArtifactRef,
    *,
    expected_run_id: str | None,
) -> PolicyTrainingRunReport:
    errors: list[str] = []
    if artifact.kind != POLICY_TRAINING_REPORT_KIND:
        errors.append(
            f"policy training report artifact.kind must be {POLICY_TRAINING_REPORT_KIND!r}"
        )
    if artifact.sha256 is None:
        errors.append("policy training report artifact.sha256 must be set")
    try:
        report_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
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
    elif artifact.sha256 is not None and sha256_file(report_path) != artifact.sha256:
        errors.append("policy training report artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert report_path is not None
    report = PolicyTrainingRunReport.from_dict(read_json(report_path))
    if expected_run_id is not None and report.run_id != expected_run_id:
        raise HarnessIOError(
            "policy training report artifact.run_id does not match expected train_run_id"
        )
    if artifact.provenance.get("run_id") not in (None, report.run_id):
        raise HarnessIOError(
            "policy training report artifact provenance run_id must match report.run_id"
        )
    return report


def _runtime_policy_checkpoint_artifact(
    path: str | Path,
    *,
    train_run_id: str,
    eval_run_id: str,
    runtime_checkpoint_container_path: str,
    expected_policy_sha256: str | None,
) -> ArtifactRef:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.exists():
        raise HarnessIOError(f"runtime_policy_checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise HarnessIOError(f"runtime_policy_checkpoint must be a file: {checkpoint_path}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if expected_policy_sha256 is None:
        raise HarnessIOError("policy training report.policy_artifact.sha256 must be set")
    if checkpoint_sha != expected_policy_sha256:
        raise HarnessIOError(
            "runtime_policy_checkpoint sha256 does not match trained policy artifact"
        )
    return ArtifactRef(
        kind=POLICY_CHECKPOINT_KIND,
        path=str(checkpoint_path),
        sha256=checkpoint_sha,
        provenance={
            "producer": "aic_signal_harness.train_eval_promote",
            "derivation": "bind_runtime_policy_checkpoint",
            "train_run_id": train_run_id,
            "eval_run_id": eval_run_id,
            "container_path": runtime_checkpoint_container_path,
            "source_policy_artifact_sha256": expected_policy_sha256,
        },
    )


def _runtime_image_artifact(
    *,
    model_image: str | None,
    model_image_id: str | None,
    eval_run_id: str,
) -> ArtifactRef:
    image = _optional_nonempty_text(model_image, "model_image") or "aic-lewm-learned:latest"
    digest = _docker_image_sha256(model_image_id)
    return ArtifactRef(
        kind="docker_image",
        uri="docker://" + image.lstrip("/"),
        sha256=digest,
        provenance={
            "producer": "aic_signal_harness.train_eval_promote",
            "derivation": "bind_runtime_policy_image",
            "eval_run_id": eval_run_id,
            "model_image": image,
            "model_image_id": _require_nonempty_text(model_image_id, "model_image_id"),
        },
    )


def _eval_backend_from_training_report(
    report: PolicyTrainingRunReport,
    *,
    source_policy_training_report: ArtifactRef,
    runtime_policy_artifact: ArtifactRef,
    runtime_image_artifact: ArtifactRef,
    eval_run_id: str,
    runtime_checkpoint_container_path: str,
    runtime_env: Mapping[str, str],
    planner_mode: str | None,
) -> PolicyBackendSpec:
    candidate = report.candidate_policy
    if candidate.runtime_boundary is None:
        raise HarnessIOError("policy training report.candidate_policy must set runtime_boundary")
    config = dict(candidate.config)
    candidate_planner_mode = config.get("planner_mode")
    if candidate_planner_mode is not None and (
        not isinstance(candidate_planner_mode, str) or not candidate_planner_mode.strip()
    ):
        raise HarnessIOError(
            "policy training report.candidate_policy.config.planner_mode must be a nonempty string"
        )
    if planner_mode is not None:
        if (
            isinstance(candidate_planner_mode, str)
            and candidate_planner_mode.strip() != planner_mode
        ):
            raise HarnessIOError(
                "policy training report.candidate_policy.config.planner_mode "
                "does not match eval runtime planner_mode"
            )
        config["planner_mode"] = planner_mode
    provenance = dict(candidate.provenance)
    provenance.update(
        {
            "producer": "aic_signal_harness.train_eval_promote",
            "derivation": "bind_trained_policy_eval_backend",
            "train_run_id": report.run_id,
            "eval_run_id": eval_run_id,
            "source_policy_training_report_sha256": source_policy_training_report.sha256,
            "source_policy_artifact_sha256": report.policy_artifact.sha256,
            "runtime_policy_artifact_sha256": runtime_policy_artifact.sha256,
            "runtime_image_artifact_sha256": runtime_image_artifact.sha256,
            "runtime_checkpoint_container_path": runtime_checkpoint_container_path,
            "docker_artifact": runtime_image_artifact.to_dict(),
            "runtime_env": dict(runtime_env),
        }
    )
    try:
        return PolicyBackendSpec(
            backend_kind=candidate.backend_kind,
            name=candidate.name,
            runtime_role=candidate.runtime_role,
            training_sources=candidate.training_sources,
            simulator_sources=candidate.simulator_sources,
            runtime_allowed=candidate.runtime_allowed,
            leakage_class=candidate.leakage_class,
            runtime_boundary=RuntimeBoundaryProof(
                deterministic=candidate.runtime_boundary.deterministic,
                uses_online_language_model_control=(
                    candidate.runtime_boundary.uses_online_language_model_control
                ),
                legal_observation_contract=(
                    candidate.runtime_boundary.legal_observation_contract
                ),
                policy_artifact=runtime_policy_artifact,
                notes=candidate.runtime_boundary.notes,
            ),
            description=candidate.description,
            config=config,
            provenance=provenance,
        )
    except SchemaValidationError as exc:
        raise HarnessIOError("trained policy eval backend is invalid: " + str(exc)) from exc


def _validate_backend_kind_override(
    backend_kind: str | BackendKind | None,
    expected_backend_kind: BackendKind,
) -> None:
    if backend_kind is None:
        return
    try:
        requested_backend_kind = BackendKind.parse(backend_kind)
    except SchemaValidationError as exc:
        raise HarnessIOError(f"backend_kind is invalid: {exc}") from exc
    if requested_backend_kind is not expected_backend_kind:
        raise HarnessIOError(
            "backend_kind override must match policy training report candidate backend_kind: "
            f"{requested_backend_kind.value!r} != {expected_backend_kind.value!r}"
        )


def _binding_consistency_errors(
    *,
    binding: TrainedPolicyEvalBindingReport,
    training_report: PolicyTrainingRunReport,
    source_policy_training_execution_report: ArtifactRef,
    source_policy_artifact: ArtifactRef,
    runtime_policy_artifact: ArtifactRef,
    runtime_image_artifact: ArtifactRef,
    candidate_policy: PolicyBackendSpec,
    eval_backend: PolicyBackendSpec,
) -> list[str]:
    errors: list[str] = []
    if training_report.run_id != binding.train_run_id:
        errors.append("trained policy eval binding.train_run_id must match training report")
    if training_report.policy_training_execution_report is None:
        errors.append("trained policy eval binding requires policy_training_execution_report")
    elif training_report.policy_training_execution_report != source_policy_training_execution_report:
        errors.append(
            "trained policy eval binding.source_policy_training_execution_report "
            "must match training report"
        )
    if training_report.policy_artifact != source_policy_artifact:
        errors.append("trained policy eval binding.source_policy_artifact must match training report")
    if training_report.candidate_policy != candidate_policy:
        errors.append("trained policy eval binding.candidate_policy must match training report")
    errors.extend(
        _local_policy_checkpoint_errors(
            source_policy_artifact,
            field_name="trained policy eval binding.source_policy_artifact",
        )
    )
    errors.extend(
        _local_policy_checkpoint_errors(
            runtime_policy_artifact,
            field_name="trained policy eval binding.runtime_policy_artifact",
        )
    )
    if runtime_image_artifact.kind != "docker_image":
        errors.append("trained policy eval binding.runtime_image_artifact.kind must be 'docker_image'")
    if runtime_image_artifact.sha256 is None:
        errors.append("trained policy eval binding.runtime_image_artifact.sha256 must be set")
    if source_policy_artifact.sha256 != runtime_policy_artifact.sha256:
        errors.append("runtime policy checkpoint sha256 must match source policy artifact")
    if runtime_policy_artifact.provenance.get("container_path") != binding.runtime_checkpoint_container_path:
        errors.append("runtime policy artifact provenance container_path must match binding")
    if runtime_policy_artifact.provenance.get("eval_run_id") != binding.eval_run_id:
        errors.append("runtime policy artifact provenance eval_run_id must match binding")
    if eval_backend.runtime_boundary is None:
        errors.append("trained policy eval backend must set runtime_boundary")
    else:
        if eval_backend.runtime_boundary.policy_artifact != runtime_policy_artifact:
            errors.append(
                "trained policy eval backend runtime_boundary.policy_artifact "
                "must match runtime_policy_artifact"
            )
    if eval_backend.provenance.get("train_run_id") != binding.train_run_id:
        errors.append("trained policy eval backend provenance train_run_id must match binding")
    if eval_backend.provenance.get("eval_run_id") != binding.eval_run_id:
        errors.append("trained policy eval backend provenance eval_run_id must match binding")
    if (
        eval_backend.provenance.get("source_policy_training_report_sha256")
        != binding.source_policy_training_report.sha256
    ):
        errors.append(
            "trained policy eval backend provenance source_policy_training_report_sha256 "
            "must match binding"
        )
    if eval_backend.provenance.get("runtime_policy_artifact_sha256") != runtime_policy_artifact.sha256:
        errors.append(
            "trained policy eval backend provenance runtime_policy_artifact_sha256 "
            "must match runtime_policy_artifact"
        )
    if eval_backend.provenance.get("runtime_image_artifact_sha256") != runtime_image_artifact.sha256:
        errors.append(
            "trained policy eval backend provenance runtime_image_artifact_sha256 "
            "must match runtime_image_artifact"
        )
    return errors


def _local_policy_checkpoint_errors(
    artifact: ArtifactRef,
    *,
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    if artifact.kind != POLICY_CHECKPOINT_KIND:
        errors.append(f"{field_name}.kind must be {POLICY_CHECKPOINT_KIND!r}")
    if artifact.sha256 is None:
        errors.append(f"{field_name}.sha256 must be set")
    try:
        artifact_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=field_name,
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        artifact_path = None
    if artifact_path is None:
        errors.append(f"{field_name} must be local byte-verifiable")
    elif not artifact_path.exists():
        errors.append(f"{field_name}.path must exist")
    elif not artifact_path.is_file():
        errors.append(f"{field_name}.path must be a file")
    elif artifact.sha256 is not None and sha256_file(artifact_path) != artifact.sha256:
        errors.append(f"{field_name}.sha256 must match path")
    return errors


def _runtime_env_with_checkpoint(
    runtime_env: Mapping[str, str] | None,
    *,
    runtime_checkpoint_container_path: str,
    planner_mode: str | None,
) -> Mapping[str, str]:
    env = runtime_env_from_mapping(runtime_env)
    env["AIC_LEWM_CHECKPOINT"] = _require_absolute_container_path(
        runtime_checkpoint_container_path,
        "runtime_checkpoint_container_path",
    )
    if planner_mode is not None:
        env["AIC_LEWM_PLANNER_MODE"] = _require_nonempty_text(
            planner_mode,
            "planner_mode",
        )
    return MappingProxyType(env)


def _resolved_runtime_planner_mode(
    runtime_env: Mapping[str, str],
    *,
    planner_mode: str | None,
) -> str | None:
    if planner_mode is not None:
        return _require_nonempty_text(planner_mode, "planner_mode")
    raw_mode = runtime_env.get("AIC_LEWM_PLANNER_MODE")
    if raw_mode is None:
        return None
    return _require_nonempty_text(raw_mode, "runtime_env.AIC_LEWM_PLANNER_MODE")


def _reject_output_collisions(binding_path: Path, summary_path: Path) -> None:
    try:
        binding_resolved = binding_path.expanduser().resolve(strict=False)
        summary_resolved = summary_path.expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise HarnessIOError("train-eval-promote output path cannot be resolved") from exc
    if binding_resolved == summary_resolved:
        raise HarnessIOError("binding_output_path and summary_output_path must be distinct")


def _preflight_train_eval_promote_outputs(
    *,
    run_id: str,
    binding_path: Path,
    summary_path: Path,
    harness_root: Path,
    ledger_path: str | Path | None,
    baseline_path: str | Path | None,
    update_baseline_path: str | Path | None,
    bootstrap_promotion: bool,
    append_ledger: bool,
    overwrite: bool,
) -> None:
    output_paths = (
        binding_path,
        summary_path,
        harness_root / "score_report.json",
        harness_root / "scoring_yaml_artifact.json",
        harness_root / "policy_trace_report.json",
        harness_root / "policy_trace_artifact.json",
        harness_root / "episode_trace.json",
        harness_root / "training_signal_report.json",
        harness_root / "run_manifest.json",
        harness_root / "promotion_report.json",
        harness_root / "reward_report.json",
        harness_root / "failure_report.json",
        harness_root / "next_experiment.json",
        harness_root / "next_experiment_plan.json",
        harness_root / "ledger_entry.json",
        harness_root / "live_eval_summary.json",
    )
    resolved_outputs = _resolved_path_map(output_paths)
    duplicate_outputs = _duplicates(resolved_outputs)
    if duplicate_outputs:
        raise HarnessIOError(
            "train-eval-promote output paths must be distinct: "
            + ", ".join(str(path) for path in duplicate_outputs)
        )
    external_paths = tuple(
        Path(path).expanduser().resolve(strict=False)
        for path in (ledger_path, baseline_path, update_baseline_path)
        if path is not None
    )
    for external_path in external_paths:
        if external_path in resolved_outputs:
            raise HarnessIOError(
                "train-eval-promote output path must not collide with ledger or baseline path: "
                f"{external_path}"
            )
    if (baseline_path is not None or bootstrap_promotion) and (
        not append_ledger or ledger_path is None
    ):
        raise HarnessIOError("promotion decisions require successful ledger append")
    if update_baseline_path is not None and (not append_ledger or ledger_path is None):
        raise HarnessIOError("accepted baseline update requires successful ledger append")
    if append_ledger and ledger_path is not None:
        if any(entry.run_id == run_id for entry in read_ledger_entries(ledger_path)):
            raise HarnessIOError(f"ledger already contains run_id={run_id}")
    if not overwrite:
        existing_outputs = [
            path for path in output_paths if path.exists()
        ]
        if existing_outputs:
            raise HarnessIOError(
                "train-eval-promote output already exists; pass overwrite=True to replace it: "
                + ", ".join(str(path) for path in existing_outputs)
            )


def _resolved_path_map(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    try:
        return tuple(path.expanduser().resolve(strict=False) for path in paths)
    except (OSError, ValueError) as exc:
        raise HarnessIOError("train-eval-promote output path cannot be resolved") from exc


def _duplicates(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for path in paths:
        if path in seen:
            duplicates.append(path)
        seen.add(path)
    return tuple(duplicates)


def _summary_evidence_errors(summary: TrainEvalPromoteSummary) -> list[str]:
    errors: list[str] = []
    binding_path = _local_verified_artifact_path(
        summary.binding_report,
        expected_kind=TRAINED_POLICY_EVAL_BINDING_KIND,
        field_name="train-eval-promote summary.binding_report",
        errors=errors,
    )
    manifest_path = _local_verified_artifact_path(
        summary.manifest,
        expected_kind="run_manifest",
        field_name="train-eval-promote summary.manifest",
        errors=errors,
    )
    if binding_path is not None:
        try:
            binding = TrainedPolicyEvalBindingReport.from_dict(read_json(binding_path))
        except HarnessIOError as exc:
            errors.append("train-eval-promote summary.binding_report must parse: " + str(exc))
        else:
            if binding.train_run_id != summary.train_run_id:
                errors.append("train-eval-promote summary.train_run_id must match binding")
            if binding.eval_run_id != summary.eval_run_id:
                errors.append("train-eval-promote summary.eval_run_id must match binding")
    if manifest_path is not None:
        try:
            manifest = RunManifest.from_dict(read_json(manifest_path))
        except Exception as exc:
            errors.append("train-eval-promote summary.manifest must parse: " + str(exc))
        else:
            if manifest.run_id != summary.eval_run_id:
                errors.append("train-eval-promote summary.manifest run_id must match eval_run_id")
            if manifest.status is not RunStatus.completed:
                errors.append("train-eval-promote summary.manifest status must be completed")
            if manifest.score is None:
                errors.append("train-eval-promote summary.manifest must include score")
            elif manifest.score.total != summary.score_total:
                errors.append("train-eval-promote summary.score_total must match manifest")
    ledger_entry = _read_summary_json_path(
        summary.ledger_entry_path,
        field_name="train-eval-promote summary.ledger_entry_path",
        errors=errors,
        parser=LedgerEntry.from_dict,
    )
    if ledger_entry is not None:
        if ledger_entry.run_id != summary.eval_run_id:
            errors.append("train-eval-promote summary ledger run_id must match eval_run_id")
        if ledger_entry.manifest.sha256 != summary.manifest.sha256:
            errors.append("train-eval-promote summary ledger manifest sha must match manifest")
        if ledger_entry.metric.value != summary.score_total:
            errors.append("train-eval-promote summary ledger metric must match score_total")
    if summary.promotion_path is not None:
        promotion = _read_summary_json_path(
            summary.promotion_path,
            field_name="train-eval-promote summary.promotion_path",
            errors=errors,
            parser=PromotionDecision.from_dict,
        )
        if promotion is not None:
            if promotion.run_id != summary.eval_run_id:
                errors.append("train-eval-promote summary promotion run_id must match eval_run_id")
            if promotion.manifest.sha256 != summary.manifest.sha256:
                errors.append(
                    "train-eval-promote summary promotion manifest sha must match manifest"
                )
    _read_summary_json_path(
        summary.live_eval_summary_path,
        field_name="train-eval-promote summary.live_eval_summary_path",
        errors=errors,
        parser=lambda value: value,
    )
    return errors


def _local_verified_artifact_path(
    artifact: ArtifactRef,
    *,
    expected_kind: str,
    field_name: str,
    errors: list[str],
) -> Path | None:
    if artifact.kind != expected_kind:
        errors.append(f"{field_name}.kind must be {expected_kind!r}")
    if artifact.sha256 is None:
        errors.append(f"{field_name}.sha256 must be set")
    try:
        path = local_artifact_path(path=artifact.path, uri=artifact.uri, field_name=field_name)
    except HarnessIOError as exc:
        errors.append(str(exc))
        return None
    if path is None:
        errors.append(f"{field_name} must be local byte-verifiable")
        return None
    if not path.exists():
        errors.append(f"{field_name}.path must exist")
        return None
    if not path.is_file():
        errors.append(f"{field_name}.path must be a file")
        return None
    if artifact.sha256 is not None and sha256_file(path) != artifact.sha256:
        errors.append(f"{field_name}.sha256 must match path")
        return None
    return path


def _docker_image_sha256(image_id: str | None) -> str:
    value = _require_nonempty_text(image_id, "model_image_id")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise HarnessIOError("model_image_id must be a sha256 Docker image id")
    return digest.lower()


def _read_summary_json_path(
    path: str,
    *,
    field_name: str,
    errors: list[str],
    parser,
) -> Any | None:
    try:
        json_path = Path(path).expanduser()
        if not json_path.exists():
            errors.append(f"{field_name} must exist")
            return None
        if not json_path.is_file():
            errors.append(f"{field_name} must be a file")
            return None
        return parser(read_json(json_path))
    except HarnessIOError as exc:
        errors.append(f"{field_name} must parse: {exc}")
    return None


def _require_absolute_container_path(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name)
    path = PurePosixPath(text)
    if not path.is_absolute():
        raise HarnessIOError(f"{field_name} must be an absolute container path")
    if any(part in (".", "..") for part in path.parts):
        raise HarnessIOError(f"{field_name} must not contain '.' or '..' segments")
    return text


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(_require_nonempty_text(item, field_name) for item in value)


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


def _cmd_finalize(args: argparse.Namespace) -> int:
    finalization = finalize_trained_policy_eval(
        run_id=args.run_id,
        result_root=args.result_root,
        harness_root=args.harness_root,
        scoring_yaml=args.scoring_yaml,
        policy_training_report=args.policy_training_report,
        runtime_policy_checkpoint=args.runtime_policy_checkpoint,
        runtime_checkpoint_container_path=args.runtime_checkpoint_container_path,
        policy_trace=args.policy_trace,
        ledger_path=args.ledger,
        baseline_path=args.baseline,
        update_baseline_path=args.update_baseline,
        bootstrap_promotion=args.bootstrap_promotion,
        min_improvement=args.min_improvement,
        eligible_for_submission=args.eligible_for_submission,
        gate_id=args.gate_id,
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        model_image=args.model_image,
        model_image_id=args.model_image_id,
        backend_kind=args.backend_kind,
        planner_mode=args.planner_mode,
        binding_output_path=args.binding_output,
        summary_output_path=args.summary_output,
        overwrite=args.overwrite,
        append_ledger=not args.no_append_ledger,
        write_next_experiment=not args.no_next_experiment,
    )
    print(f"AIC_TRAIN_EVAL_PROMOTE_BINDING_PATH={finalization.binding_path}")
    print(f"AIC_TRAIN_EVAL_PROMOTE_SUMMARY_PATH={finalization.summary_path}")
    print(f"AIC_HARNESS_MANIFEST_PATH={finalization.live_eval.manifest_path}")
    print(f"AIC_HARNESS_LEDGER_ENTRY_PATH={finalization.live_eval.ledger_entry_path}")
    if finalization.live_eval.ledger_path is not None:
        print(f"AIC_HARNESS_LEDGER_PATH={finalization.live_eval.ledger_path}")
    if finalization.live_eval.promotion_path is not None:
        print(f"AIC_HARNESS_PROMOTION_PATH={finalization.live_eval.promotion_path}")
    return 0


def _cmd_stage_training_bundle(args: argparse.Namespace) -> int:
    staged = stage_policy_training_report_bundle(
        policy_training_report=args.policy_training_report,
        bundle_root=args.bundle_root,
        runtime_bundle_root=args.runtime_bundle_root,
        overwrite=args.overwrite,
    )
    print(f"AIC_STAGED_POLICY_TRAINING_REPORT_PATH={staged.policy_training_report_path}")
    print(
        "AIC_STAGED_RUNTIME_POLICY_TRAINING_REPORT_PATH="
        f"{staged.runtime_policy_training_report_path}"
    )
    print(f"AIC_STAGED_POLICY_CHECKPOINT_PATH={staged.policy_checkpoint_path}")
    print(
        "AIC_STAGED_RUNTIME_POLICY_CHECKPOINT_PATH="
        f"{staged.runtime_policy_checkpoint_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize a trained policy eval into train-eval-promote evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize", help="Finalize one trained policy live eval")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--result-root", required=True)
    finalize.add_argument("--harness-root", required=True)
    finalize.add_argument("--scoring-yaml", required=True)
    finalize.add_argument("--policy-training-report", required=True)
    finalize.add_argument("--runtime-policy-checkpoint", required=True)
    finalize.add_argument(
        "--runtime-checkpoint-container-path",
        default=DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH,
    )
    finalize.add_argument("--policy-trace")
    finalize.add_argument("--ledger")
    finalize.add_argument("--baseline")
    finalize.add_argument("--update-baseline")
    finalize.add_argument("--bootstrap-promotion", action="store_true")
    finalize.add_argument("--min-improvement", type=float, default=0.0)
    finalize.add_argument("--eligible-for-submission", action="store_true")
    finalize.add_argument("--gate-id", default="trained_policy_live_eval")
    finalize.add_argument("--experiment-id")
    finalize.add_argument("--hypothesis")
    finalize.add_argument("--model-image")
    finalize.add_argument("--model-image-id")
    finalize.add_argument("--backend-kind")
    finalize.add_argument("--planner-mode")
    finalize.add_argument("--binding-output")
    finalize.add_argument("--summary-output")
    finalize.add_argument("--overwrite", action="store_true")
    finalize.add_argument("--no-append-ledger", action="store_true")
    finalize.add_argument("--no-next-experiment", action="store_true")
    finalize.set_defaults(func=_cmd_finalize)
    stage = subparsers.add_parser(
        "stage-training-bundle",
        help="Stage a policy-training report and byte-bound sidecars for remote eval",
    )
    stage.add_argument("--policy-training-report", required=True)
    stage.add_argument("--bundle-root", required=True)
    stage.add_argument("--runtime-bundle-root")
    stage.add_argument("--overwrite", action="store_true")
    stage.set_defaults(func=_cmd_stage_training_bundle)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HarnessIOError as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
