from __future__ import annotations

from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    HarnessIOError,
    LeakageClass,
    PolicyBackendSpec,
    PolicyTrainingRunReport,
    RuntimeBoundaryProof,
    RuntimeRole,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    derive_training_dataset_report,
    policy_checkpoint_artifact,
    record_policy_training_run,
    sha256_file,
    trainer_invocation_fingerprint_sha256,
    write_json,
)


def test_policy_training_run_report_round_trip_and_provenance(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    policy_artifact = _policy_artifact(tmp_path, source_dataset_artifact, trainer)
    candidate_policy = _candidate_policy("train-run-a", source_dataset_artifact, trainer, policy_artifact)

    report = record_policy_training_run(
        run_id="train-run-a",
        generated_at_utc="2026-04-24T00:01:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=trainer,
        policy_artifact=policy_artifact,
        candidate_policy=candidate_policy,
        metrics={"train.loss": 0.125},
        notes=("pytest policy training record",),
    )

    assert PolicyTrainingRunReport.from_dict(report.to_dict()) == report
    assert report.source_dataset_run_id == "source-run-a"
    assert report.trainer_fingerprint_sha256 == trainer_invocation_fingerprint_sha256(trainer)
    assert report.policy_artifact.sha256 == sha256_file(Path(report.policy_artifact.path or ""))
    assert report.candidate_policy.runtime_boundary is not None
    assert report.candidate_policy.runtime_boundary.policy_artifact == report.policy_artifact
    assert report.offline_only is True
    assert report.runtime_allowed is False
    assert report.autonomous_launch_allowed is False
    assert report.ok is True
    assert report.errors == ()


def test_policy_training_run_rejects_bad_source_dataset_artifacts(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    policy_artifact = _policy_artifact(tmp_path, source_dataset_artifact, trainer)
    candidate_policy = _candidate_policy("train-run-a", source_dataset_artifact, trainer, policy_artifact)

    uri_only_source = ArtifactRef(
        kind="training_dataset_report",
        uri="memory://pytest/training_dataset_report.json",
        sha256=source_dataset_artifact.sha256,
        provenance={
            "producer": "pytest",
            "run_id": "source-run-a",
            "derivation": "derive_training_dataset_report",
        },
    )
    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        record_policy_training_run(
            run_id="train-run-a",
            generated_at_utc="2026-04-24T00:01:00Z",
            source_training_dataset_report=uri_only_source,
            trainer=trainer,
            policy_artifact=policy_artifact,
            candidate_policy=candidate_policy,
        )

    wrong_kind_source = ArtifactRef(
        kind="training_signal_report",
        path=source_dataset_artifact.path,
        sha256=source_dataset_artifact.sha256,
        provenance={
            "producer": "pytest",
            "run_id": "source-run-a",
            "derivation": "derive_training_dataset_report",
        },
    )
    with pytest.raises(HarnessIOError, match="kind must be 'training_dataset_report'"):
        record_policy_training_run(
            run_id="train-run-a",
            generated_at_utc="2026-04-24T00:01:00Z",
            source_training_dataset_report=wrong_kind_source,
            trainer=trainer,
            policy_artifact=policy_artifact,
            candidate_policy=candidate_policy,
        )

    empty_source = _training_dataset_artifact(tmp_path, signals=(), path_name="empty_training_dataset.json")
    with pytest.raises(HarnessIOError, match="ok must be true|example_count"):
        record_policy_training_run(
            run_id="train-run-a",
            generated_at_utc="2026-04-24T00:01:00Z",
            source_training_dataset_report=empty_source,
            trainer=trainer,
            policy_artifact=policy_artifact,
            candidate_policy=candidate_policy,
        )


def test_policy_training_report_binds_policy_artifact_and_candidate_policy(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    policy_artifact = _policy_artifact(tmp_path, source_dataset_artifact, trainer)
    candidate_policy = _candidate_policy("train-run-a", source_dataset_artifact, trainer, policy_artifact)
    report = record_policy_training_run(
        run_id="train-run-a",
        generated_at_utc="2026-04-24T00:01:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=trainer,
        policy_artifact=policy_artifact,
        candidate_policy=candidate_policy,
    )

    forged_payload = report.to_dict()
    forged_payload["policy_artifact"]["provenance"]["source_training_dataset_report_sha256"] = "b" * 64
    with pytest.raises(HarnessIOError, match="source_training_dataset_report_sha256"):
        PolicyTrainingRunReport.from_dict(forged_payload)

    forged_policy_payload = report.to_dict()
    forged_policy_payload["candidate_policy"]["runtime_boundary"]["policy_artifact"]["sha256"] = "c" * 64
    with pytest.raises(HarnessIOError, match="runtime_boundary.policy_artifact"):
        PolicyTrainingRunReport.from_dict(forged_policy_payload)

    forged_candidate_provenance = report.to_dict()
    forged_candidate_provenance["candidate_policy"]["provenance"]["trainer_fingerprint_sha256"] = "d" * 64
    with pytest.raises(HarnessIOError, match="candidate_policy provenance trainer_fingerprint_sha256"):
        PolicyTrainingRunReport.from_dict(forged_candidate_provenance)

    mismatched_backend = report.to_dict()
    mismatched_backend["candidate_policy"]["backend_kind"] = "open_vla"
    with pytest.raises(HarnessIOError, match="trainer.backend_kind"):
        PolicyTrainingRunReport.from_dict(mismatched_backend)


def test_policy_training_rejects_hidden_trainer_artifacts_and_online_llm() -> None:
    with pytest.raises(HarnessIOError, match="config must not hide artifact references"):
        TrainerInvocation(
            trainer_id="lewm-trainer",
            trainer_name="LEWM trainer",
            backend_kind=BackendKind.lewm_world_model,
            command=("python", "train.py"),
            config={"checkpoint_uri": "gs://bucket/checkpoint.ckpt"},
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        )

    for config in (
        {"policy": "candidate-v1"},
        {"model": "candidate-v1"},
        {"weights": "candidate-v1"},
        {"artifacts": "candidate-v1"},
        {"checkpoints": "candidate-v1"},
        {"datasets": "candidate-v1"},
        {"models": "candidate-v1"},
        {"policies": "candidate-v1"},
        {"policy": "gs://bucket/policy.ckpt"},
        {"model": "/tmp/model.pt"},
        {"weights": "model.safetensors"},
        {"artifacts": "gs://bucket/policy.ckpt"},
        {"output": "gs://bucket/policy.ckpt"},
        {"outputs": ["gs://bucket/policy.ckpt"]},
        {"outputs": ["/tmp/policy.ckpt"]},
    ):
        with pytest.raises(HarnessIOError, match="config must not hide artifact references"):
            TrainerInvocation(
                trainer_id="lewm-trainer",
                trainer_name="LEWM trainer",
                backend_kind=BackendKind.lewm_world_model,
                command=("python", "train.py"),
                config=config,
                offline_only=True,
                runtime_allowed=False,
                uses_online_language_model_control=False,
            )

    with pytest.raises(HarnessIOError, match="environment must not hide artifact references"):
        TrainerInvocation(
            trainer_id="lewm-trainer",
            trainer_name="LEWM trainer",
            backend_kind=BackendKind.lewm_world_model,
            command=("python", "train.py"),
            environment={"AIC_DATASET_GCS_URI": "gs://bucket/dataset.h5"},
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        )

    for environment in (
        {"AIC_POLICY": "candidate-v1"},
        {"AIC_OUTPUT": "gs://bucket/policy.ckpt"},
        {"AIC_POLICY": "gs://bucket/policy.ckpt"},
        {"AIC_WEIGHTS": "/tmp/weights.safetensors"},
        {"AIC_ARTIFACTS": "gs://bucket/policy.ckpt"},
        {"AIC_ARTIFACT_URI": "gs://bucket/policy.ckpt"},
    ):
        with pytest.raises(HarnessIOError, match="environment must not hide artifact references"):
            TrainerInvocation(
                trainer_id="lewm-trainer",
                trainer_name="LEWM trainer",
                backend_kind=BackendKind.lewm_world_model,
                command=("python", "train.py"),
                environment=environment,
                offline_only=True,
                runtime_allowed=False,
                uses_online_language_model_control=False,
            )

    with pytest.raises(HarnessIOError) as duplicate_error:
        TrainerInvocation(
            trainer_id="lewm-trainer",
            trainer_name="LEWM trainer",
            backend_kind=BackendKind.lewm_world_model,
            command=("python", "train.py"),
            config={"artifacts": {"uri": "gs://bucket/policy.ckpt"}},
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        )
    assert str(duplicate_error.value).count("trainer invocation.config.artifacts.uri") == 1

    with pytest.raises(HarnessIOError, match="config must not hide artifact references"):
        TrainerInvocation(
            trainer_id="lewm-trainer",
            trainer_name="LEWM trainer",
            backend_kind=BackendKind.lewm_world_model,
            command=("python", "train.py"),
            config={"data": {"uri": "gs://bucket/dataset.h5"}},
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        )

    with pytest.raises(HarnessIOError, match="uses_online_language_model_control must be false"):
        TrainerInvocation(
            trainer_id="lewm-trainer",
            trainer_name="LEWM trainer",
            backend_kind=BackendKind.lewm_world_model,
            command=("python", "train.py"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=True,
        )


def test_policy_training_report_rejects_missing_required_fields(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    policy_artifact = _policy_artifact(tmp_path, source_dataset_artifact, trainer)
    candidate_policy = _candidate_policy("train-run-a", source_dataset_artifact, trainer, policy_artifact)
    report = record_policy_training_run(
        run_id="train-run-a",
        generated_at_utc="2026-04-24T00:01:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=trainer,
        policy_artifact=policy_artifact,
        candidate_policy=candidate_policy,
    )
    payload = report.to_dict()
    del payload["policy_artifact"]

    with pytest.raises(HarnessIOError, match="missing required fields"):
        PolicyTrainingRunReport.from_dict(payload)


def test_policy_training_rejects_empty_source_dataset_producer(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    policy_artifact = _policy_artifact(tmp_path, source_dataset_artifact, trainer)
    candidate_policy = _candidate_policy("train-run-a", source_dataset_artifact, trainer, policy_artifact)
    forged_source = ArtifactRef(
        kind=source_dataset_artifact.kind,
        path=source_dataset_artifact.path,
        sha256=source_dataset_artifact.sha256,
        provenance={
            "producer": "",
            "run_id": "source-run-a",
            "derivation": "derive_training_dataset_report",
        },
    )

    with pytest.raises(HarnessIOError, match="producer must be a nonempty string|provenance producer"):
        record_policy_training_run(
            run_id="train-run-a",
            generated_at_utc="2026-04-24T00:01:00Z",
            source_training_dataset_report=forged_source,
            trainer=trainer,
            policy_artifact=policy_artifact,
            candidate_policy=candidate_policy,
        )

    report_payload = record_policy_training_run(
        run_id="train-run-a",
        generated_at_utc="2026-04-24T00:01:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=trainer,
        policy_artifact=policy_artifact,
        candidate_policy=candidate_policy,
    ).to_dict()
    report_payload["source_training_dataset_report"]["provenance"]["producer"] = ""
    with pytest.raises(HarnessIOError, match="provenance producer"):
        PolicyTrainingRunReport.from_dict(report_payload)


def test_policy_checkpoint_artifact_rejects_provenance_overrides(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    checkpoint_path = tmp_path / "policy.ckpt"
    checkpoint_path.write_bytes(b"pytest policy checkpoint")

    with pytest.raises(HarnessIOError, match="must not override critical fields"):
        policy_checkpoint_artifact(
            checkpoint_path,
            run_id="train-run-a",
            source_training_dataset_report=source_dataset_artifact,
            trainer=trainer,
            extra_provenance={"trainer_fingerprint_sha256": "0" * 64},
        )

    with pytest.raises(HarnessIOError, match="producer must be a nonempty string"):
        policy_checkpoint_artifact(
            checkpoint_path,
            run_id="train-run-a",
            source_training_dataset_report=source_dataset_artifact,
            trainer=trainer,
            producer="",
        )


def test_policy_training_report_rejects_empty_checkpoint_producer(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    trainer = _trainer()
    policy_artifact = _policy_artifact(tmp_path, source_dataset_artifact, trainer)
    candidate_policy = _candidate_policy("train-run-a", source_dataset_artifact, trainer, policy_artifact)
    report = record_policy_training_run(
        run_id="train-run-a",
        generated_at_utc="2026-04-24T00:01:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=trainer,
        policy_artifact=policy_artifact,
        candidate_policy=candidate_policy,
    )
    payload = report.to_dict()
    payload["policy_artifact"]["provenance"]["producer"] = ""
    payload["candidate_policy"]["runtime_boundary"]["policy_artifact"]["provenance"]["producer"] = ""

    with pytest.raises(HarnessIOError, match="provenance producer"):
        PolicyTrainingRunReport.from_dict(payload)


def _trainer() -> TrainerInvocation:
    return TrainerInvocation(
        trainer_id="lewm-trainer-v1",
        trainer_name="LEWM trainer",
        backend_kind=BackendKind.lewm_world_model,
        command=("python", "train.py", "data=aic"),
        config={
            "learning_rate": 0.0001,
            "max_epochs": 1,
            "gradient_checkpointing": True,
        },
        environment={"AIC_LEWM_MAX_EPOCHS": "1"},
        container_image="ghcr.io/example/aic-lewm-train:pytest",
        seed=7,
        offline_only=True,
        runtime_allowed=False,
        uses_online_language_model_control=False,
    )


def _candidate_policy(
    run_id: str,
    source_dataset_artifact: ArtifactRef,
    trainer: TrainerInvocation,
    policy_artifact: ArtifactRef,
) -> PolicyBackendSpec:
    return PolicyBackendSpec(
        backend_kind=BackendKind.lewm_world_model,
        name="lewm-candidate",
        runtime_role=RuntimeRole.live_policy,
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        runtime_allowed=True,
        leakage_class=LeakageClass.legal_policy_input,
        runtime_boundary=RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
            policy_artifact=policy_artifact,
        ),
        description="Offline-trained LEWM policy candidate.",
        config={"planner_mode": "learned_policy"},
        provenance={
            "producer": "pytest",
            "run_id": run_id,
            "source_training_dataset_report_sha256": source_dataset_artifact.sha256,
            "policy_artifact_sha256": policy_artifact.sha256,
            "trainer_fingerprint_sha256": trainer_invocation_fingerprint_sha256(trainer),
        },
    )


def _policy_artifact(
    tmp_path: Path,
    source_dataset_artifact: ArtifactRef,
    trainer: TrainerInvocation,
) -> ArtifactRef:
    checkpoint_path = tmp_path / "policy.ckpt"
    checkpoint_path.write_bytes(b"pytest policy checkpoint")
    return policy_checkpoint_artifact(
        checkpoint_path,
        run_id="train-run-a",
        source_training_dataset_report=source_dataset_artifact,
        trainer=trainer,
        producer="pytest",
    )


def _training_dataset_artifact(
    tmp_path: Path,
    *,
    signals: tuple[TrainingSignal, ...] | None = None,
    path_name: str = "training_dataset_report.json",
) -> ArtifactRef:
    signal_report = _training_signal_report(signals=signals)
    signal_path = tmp_path / f"{path_name}.source_signals.json"
    write_json(signal_path, signal_report.to_dict())
    signal_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(signal_path),
        sha256=sha256_file(signal_path),
        provenance={
            "producer": "pytest",
            "run_id": signal_report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )
    dataset_report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=signal_artifact,
    )
    dataset_path = tmp_path / path_name
    write_json(dataset_path, dataset_report.to_dict())
    return ArtifactRef(
        kind="training_dataset_report",
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        provenance={
            "producer": "pytest",
            "run_id": dataset_report.run_id,
            "derivation": "derive_training_dataset_report",
        },
    )


def _training_signal_report(
    *,
    signals: tuple[TrainingSignal, ...] | None = None,
) -> TrainingSignalReport:
    return TrainingSignalReport(
        run_id="source-run-a",
        generated_at_utc="2026-04-24T00:00:06Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "run_id": "source-run-a",
                "derivation": "derive_episode_trace",
            },
        ),
        signals=(
            TrainingSignal(
                signal_id="tsig_action_1",
                kind=TrainingSignalKind.behavior_clone_action,
                target="policy.action",
                weight=1.0,
                source="episode_trace",
                source_event_indices=(1,),
                extraction_method="episode_trace.v1.behavior_clone_action",
                leakage_class=LeakageClass.legal_policy_action_output,
                evidence={"payload": {"linear": [0.1, 0.0, 0.0]}},
                trial_id="trial-a",
            ),
        )
        if signals is None
        else signals,
        notes=("Signals are offline-only.",),
    )
