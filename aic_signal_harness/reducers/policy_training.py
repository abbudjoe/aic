"""Record completed offline policy training without launching trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file
from aic_signal_harness.policy_training import (
    PolicyTrainingRunReport,
    PolicyTrainingStatus,
    TrainerInvocation,
    trainer_invocation_fingerprint_sha256,
)
from aic_signal_harness.schemas import ArtifactRef, PolicyBackendSpec
from aic_signal_harness.training_dataset import TrainingDatasetReport


def record_policy_training_run(
    *,
    run_id: str,
    generated_at_utc: str,
    source_training_dataset_report: ArtifactRef,
    trainer: TrainerInvocation | Mapping[str, Any],
    policy_artifact: ArtifactRef,
    candidate_policy: PolicyBackendSpec | Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    notes: tuple[str, ...] = (),
) -> PolicyTrainingRunReport:
    """Bind a completed offline trainer invocation to its policy artifact."""

    typed_trainer = trainer if isinstance(trainer, TrainerInvocation) else TrainerInvocation.from_dict(trainer)
    typed_candidate_policy = (
        candidate_policy
        if isinstance(candidate_policy, PolicyBackendSpec)
        else PolicyBackendSpec.from_dict(candidate_policy)
    )
    source_report = _validate_source_training_dataset_report(source_training_dataset_report)
    return PolicyTrainingRunReport(
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        source_dataset_run_id=source_report.run_id,
        source_training_dataset_report=source_training_dataset_report,
        trainer=typed_trainer,
        trainer_fingerprint_sha256=trainer_invocation_fingerprint_sha256(typed_trainer),
        status=PolicyTrainingStatus.completed,
        policy_artifact=policy_artifact,
        candidate_policy=typed_candidate_policy,
        metrics={} if metrics is None else metrics,
        notes=notes
        + (
            "Recorded after offline policy training completed; this reducer does not launch training or eval.",
        ),
    )


def policy_checkpoint_artifact(
    path: str | Path,
    *,
    run_id: str,
    source_training_dataset_report: ArtifactRef,
    trainer: TrainerInvocation | Mapping[str, Any],
    uri: str | None = None,
    producer: str = "aic_signal_harness.reducers.policy_training",
    extra_provenance: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Create the policy-checkpoint artifact reference for a completed training run."""

    typed_trainer = trainer if isinstance(trainer, TrainerInvocation) else TrainerInvocation.from_dict(trainer)
    if not isinstance(producer, str) or not producer.strip():
        raise HarnessIOError("policy checkpoint producer must be a nonempty string")
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.exists():
        raise HarnessIOError("policy checkpoint path must exist")
    if not checkpoint_path.is_file():
        raise HarnessIOError("policy checkpoint path must be a file")
    provenance = {
        "producer": producer,
        "run_id": run_id,
        "derivation": "record_policy_training_run",
        "source_training_dataset_report_sha256": source_training_dataset_report.sha256,
        "trainer_fingerprint_sha256": trainer_invocation_fingerprint_sha256(typed_trainer),
    }
    if extra_provenance is not None:
        critical_keys = set(provenance)
        conflicting_keys = sorted(key for key in extra_provenance if key in critical_keys)
        if conflicting_keys:
            raise HarnessIOError(
                "policy checkpoint extra_provenance must not override critical fields: "
                + ", ".join(conflicting_keys)
            )
        provenance.update(extra_provenance)
    return ArtifactRef(
        kind="policy_checkpoint",
        path=str(checkpoint_path),
        uri=uri,
        sha256=sha256_file(checkpoint_path),
        provenance=provenance,
    )


def _validate_source_training_dataset_report(source: ArtifactRef) -> TrainingDatasetReport:
    errors: list[str] = []
    if source.kind != "training_dataset_report":
        errors.append("source_training_dataset_report.kind must be 'training_dataset_report'")
    if source.sha256 is None:
        errors.append("source_training_dataset_report.sha256 must be set")
    if not isinstance(source.provenance.get("producer"), str) or not source.provenance.get("producer", "").strip():
        errors.append("source_training_dataset_report provenance producer must be a nonempty string")
    if "derivation" not in source.provenance:
        errors.append("source_training_dataset_report must set provenance.derivation")
    try:
        source_path = local_artifact_path(
            path=source.path,
            uri=source.uri,
            field_name="source_training_dataset_report",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        source_path = None
    if source_path is None:
        errors.append("source_training_dataset_report must be local byte-verifiable")
    elif not source_path.exists():
        errors.append("source_training_dataset_report.path must exist")
    elif not source_path.is_file():
        errors.append("source_training_dataset_report.path must be a file")
    elif source.sha256 is not None:
        if sha256_file(source_path) != source.sha256:
            errors.append("source_training_dataset_report.sha256 must match path")
        else:
            try:
                source_report = TrainingDatasetReport.from_dict(read_json(source_path))
            except HarnessIOError as exc:
                errors.append("source_training_dataset_report must parse: " + str(exc))
            else:
                if not source_report.ok:
                    errors.append("source_training_dataset_report.ok must be true")
                if source_report.example_count < 1:
                    errors.append("source_training_dataset_report.example_count must be >= 1")
                if source.provenance.get("run_id") != source_report.run_id:
                    errors.append(
                        "source_training_dataset_report provenance run_id must match source report run_id"
                    )
                if errors:
                    raise HarnessIOError("; ".join(errors))
                return source_report
    if errors:
        raise HarnessIOError("; ".join(errors))
    raise HarnessIOError("source_training_dataset_report could not be validated")
