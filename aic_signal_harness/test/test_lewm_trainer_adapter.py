from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    CandidateEvalMode,
    CandidateEvalSpec,
    CandidatePolicyRecord,
    CandidatePolicyRegistry,
    CandidatePolicyTemplate,
    HarnessIOError,
    LEWM_TRAINER_ADAPTER_PLAN_KIND,
    LeakageClass,
    LewmTrainerAdapterConfig,
    LewmTrainerAdapterPlan,
    LewmTrainerLaunchTarget,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    build_lewm_trainer_adapter_plan,
    derive_training_dataset_report,
    materialize_training_dataset_jsonl,
    read_lewm_trainer_adapter_plan_artifact,
    sha256_file,
    write_candidate_policy_registry,
    write_json,
    write_lewm_trainer_adapter_plan,
    write_training_dataset_materialization_report,
)


def test_lewm_trainer_adapter_binds_candidate_and_materialized_dataset(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)
    registry_artifact = _candidate_registry_artifact(tmp_path, dataset_artifact)

    plan = build_lewm_trainer_adapter_plan(
        candidate_registry=registry_artifact,
        dataset_materialization_report=materialization_artifact,
        config=LewmTrainerAdapterConfig(
            plan_id="lewm-plan-a",
            generated_at_utc="2026-04-25T04:00:00Z",
            train_run_id="train-lewm-a",
            launch_target=LewmTrainerLaunchTarget.local_command,
            timeout_seconds=120.0,
        ),
    )
    plan_artifact = write_lewm_trainer_adapter_plan(tmp_path / "lewm_plan.json", plan)
    reread = read_lewm_trainer_adapter_plan_artifact(plan_artifact)

    assert reread == plan
    assert plan_artifact.kind == LEWM_TRAINER_ADAPTER_PLAN_KIND
    assert plan.candidate_registry == registry_artifact
    assert plan.dataset_materialization_report == materialization_artifact
    assert plan.source_training_dataset_report == dataset_artifact
    assert plan.materialized_dataset.sha256 is not None
    assert plan.trainer.backend_kind is BackendKind.lewm_world_model
    assert plan.policy_template.config["planner_mode"] == "lewm_mpc"
    assert plan.checkpoint.output_name == "policy.ckpt"
    assert dict(plan.artifact_environment_contract) == {
        "AIC_LEWM_MATERIALIZATION_REPORT": "dataset_materialization_report",
        "AIC_LEWM_MATERIALIZED_DATASET": "materialized_dataset",
        "AIC_LEWM_OUTPUT_CHECKPOINT": "policy_checkpoint_output",
    }
    assert plan.offline_only is True
    assert plan.runtime_allowed is False
    assert plan.autonomous_launch_allowed is False


def test_lewm_trainer_adapter_rejects_non_lewm_candidate(tmp_path: Path) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)
    registry_artifact = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        backend_kind=BackendKind.open_vla,
    )

    with pytest.raises(HarnessIOError, match="lewm_world_model"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=registry_artifact,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )


def test_lewm_trainer_adapter_rejects_dataset_mismatch(tmp_path: Path) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path, name="candidate")
    other_dataset_artifact = _training_dataset_artifact(tmp_path, name="other")
    materialization_artifact = _materialization_artifact(tmp_path, other_dataset_artifact)
    registry_artifact = _candidate_registry_artifact(tmp_path, dataset_artifact)

    with pytest.raises(HarnessIOError, match="source_training_dataset_report"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=registry_artifact,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )


def test_lewm_trainer_adapter_rejects_remote_materialization_report(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)
    registry_artifact = _candidate_registry_artifact(tmp_path, dataset_artifact)

    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=registry_artifact,
            dataset_materialization_report=ArtifactRef(
                kind=materialization_artifact.kind,
                uri="gs://aic-bucket/materialization.json",
                sha256=materialization_artifact.sha256,
                provenance=materialization_artifact.provenance,
            ),
            config=_adapter_config(),
        )


def test_lewm_trainer_adapter_rejects_plan_artifact_provenance_regressions(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)
    registry_artifact = _candidate_registry_artifact(tmp_path, dataset_artifact)
    plan = build_lewm_trainer_adapter_plan(
        candidate_registry=registry_artifact,
        dataset_materialization_report=materialization_artifact,
        config=_adapter_config(),
    )
    plan_artifact = write_lewm_trainer_adapter_plan(tmp_path / "lewm_plan.json", plan)

    with pytest.raises(HarnessIOError, match="selected_candidate_id must match"):
        read_lewm_trainer_adapter_plan_artifact(
            ArtifactRef(
                kind=plan_artifact.kind,
                path=plan_artifact.path,
                sha256=plan_artifact.sha256,
                provenance={
                    **dict(plan_artifact.provenance),
                    "selected_candidate_id": "other-candidate",
                },
            )
        )

    payload = plan.to_dict()
    payload["artifact_environment_contract"] = {
        "AIC_LEWM_MATERIALIZATION_REPORT": "dataset_materialization_report",
        "AIC_LEWM_MATERIALIZED_DATASET": "materialized_dataset",
        "AIC_LEWM_OTHER_CHECKPOINT": "policy_checkpoint_output",
    }
    assert LewmTrainerAdapterPlan.from_dict(payload).artifact_environment_contract[
        "AIC_LEWM_OTHER_CHECKPOINT"
    ] == "policy_checkpoint_output"
    payload["artifact_environment_contract"]["AIC_LEWM_MATERIALIZED_DATASET"] = (
        "policy_checkpoint_output"
    )
    with pytest.raises(HarnessIOError, match="roles must be exactly|distinct"):
        LewmTrainerAdapterPlan.from_dict(payload)


def test_lewm_trainer_adapter_rejects_hidden_launch_and_artifact_commands(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)

    runtime_registry = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-runtime",
            trainer_name="LEWM trainer runtime",
            backend_kind=BackendKind.lewm_world_model,
            command=("aic_controller", "--once"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )
    with pytest.raises(HarnessIOError, match="runtime launch surfaces"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=runtime_registry,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )

    hidden_artifact_registry = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-hidden-artifact",
            trainer_name="LEWM trainer hidden artifact",
            backend_kind=BackendKind.lewm_world_model,
            command=(sys.executable, "-m", "trainer", "--checkpoint=/tmp/policy.ckpt"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )
    with pytest.raises(HarnessIOError, match="hidden artifact locations"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=hidden_artifact_registry,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )

    split_artifact_registry = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-split-artifact",
            trainer_name="LEWM trainer split artifact",
            backend_kind=BackendKind.lewm_world_model,
            command=(sys.executable, "-m", "trainer", "--dataset", "trainer.jsonl"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )
    with pytest.raises(HarnessIOError, match="hidden artifact locations"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=split_artifact_registry,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )

    generic_jsonl_registry = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-generic-jsonl",
            trainer_name="LEWM trainer generic jsonl",
            backend_kind=BackendKind.lewm_world_model,
            command=(sys.executable, "-m", "trainer", "--input", "trainer.jsonl"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )
    with pytest.raises(HarnessIOError, match="hidden artifact locations"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=generic_jsonl_registry,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )

    runtime_flag_value_registry = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-runtime-flag-value",
            trainer_name="LEWM trainer runtime flag value",
            backend_kind=BackendKind.lewm_world_model,
            command=(sys.executable, "-m", "trainer", "--entrypoint=aic_controller"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )
    with pytest.raises(HarnessIOError, match="runtime launch surfaces"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=runtime_flag_value_registry,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )

    normal_python_registry = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-normal-python",
            trainer_name="LEWM trainer normal python",
            backend_kind=BackendKind.lewm_world_model,
            command=("/tmp/aic_model_env/bin/python", "-m", "trainer"),
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )
    build_lewm_trainer_adapter_plan(
        candidate_registry=normal_python_registry,
        dataset_materialization_report=materialization_artifact,
        config=_adapter_config(),
    )


def test_lewm_trainer_adapter_rejects_artifact_env_contract_collisions(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)
    registry_artifact = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-env-collision",
            trainer_name="LEWM trainer env collision",
            backend_kind=BackendKind.lewm_world_model,
            command=(sys.executable, "-c", "print('train')"),
            environment={"AIC_LEWM_MATERIALIZATION_REPORT": "reserved"},
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
    )

    with pytest.raises(HarnessIOError, match="must not overlap trainer.environment"):
        build_lewm_trainer_adapter_plan(
            candidate_registry=registry_artifact,
            dataset_materialization_report=materialization_artifact,
            config=_adapter_config(),
        )


def _adapter_config() -> LewmTrainerAdapterConfig:
    return LewmTrainerAdapterConfig(
        plan_id="lewm-plan-a",
        generated_at_utc="2026-04-25T04:00:00Z",
        train_run_id="train-lewm-a",
    )


def _candidate_registry_artifact(
    tmp_path: Path,
    dataset_artifact: ArtifactRef,
    *,
    backend_kind: BackendKind = BackendKind.lewm_world_model,
    trainer: TrainerInvocation | None = None,
) -> ArtifactRef:
    typed_trainer = trainer or TrainerInvocation(
        trainer_id="lewm-trainer-v1",
        trainer_name="LEWM trainer",
        backend_kind=backend_kind,
        command=(sys.executable, "-c", "print('train')"),
        config={"max_epochs": 1},
        seed=7,
        offline_only=True,
        runtime_allowed=False,
        uses_online_language_model_control=False,
    )
    candidate = CandidatePolicyRecord(
        generated_at_utc="2026-04-25T03:30:00Z",
        source_training_dataset_report=dataset_artifact,
        trainer=typed_trainer,
        policy_template=CandidatePolicyTemplate(
            backend_kind=backend_kind,
            name="lewm-candidate",
            training_sources=(TrainingSourceKind.official_demo,),
            simulator_sources=(SimulatorKind.offline_replay,),
            legal_observation_contract="official aic_model observations only",
            description="LEWM candidate.",
            config={"planner_mode": "lewm_mpc"},
            deterministic=True,
        ),
        eval=CandidateEvalSpec(
            mode=CandidateEvalMode.local_live_eval,
            gate_id="trained_policy_pre_eval",
            planner_mode="lewm_mpc",
            min_improvement=0.0,
        ),
    )
    registry = CandidatePolicyRegistry(
        registry_id=f"registry-{typed_trainer.trainer_id}",
        generated_at_utc="2026-04-25T03:30:00Z",
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
    )
    return write_candidate_policy_registry(
        tmp_path / f"candidate_registry_{typed_trainer.trainer_id}.json",
        registry,
    )


def _materialization_artifact(
    tmp_path: Path,
    dataset_artifact: ArtifactRef,
) -> ArtifactRef:
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=dataset_artifact,
        output_path=tmp_path / f"{Path(dataset_artifact.path or '').stem}.jsonl",
        generated_at_utc="2026-04-25T03:00:00Z",
    )
    return write_training_dataset_materialization_report(
        tmp_path / f"{Path(dataset_artifact.path or '').stem}_materialization.json",
        report,
    )


def _training_dataset_artifact(
    tmp_path: Path,
    *,
    name: str = "source",
) -> ArtifactRef:
    signal_report = TrainingSignalReport(
        run_id=f"{name}-run-a",
        generated_at_utc="2026-04-25T02:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri=f"memory://pytest/{name}/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "derivation": "fixture",
                "run_id": f"{name}-run-a",
            },
        ),
        signals=(
            TrainingSignal(
                signal_id=f"{name}-signal-action-1",
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
        ),
    )
    signal_path = tmp_path / f"{name}_training_signal_report.json"
    write_json(signal_path, signal_report.to_dict(), overwrite=True)
    signal_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(signal_path),
        sha256=sha256_file(signal_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_training_signal_report",
            "run_id": signal_report.run_id,
        },
    )
    dataset_report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=signal_artifact,
    )
    dataset_path = tmp_path / f"{name}_training_dataset_report.json"
    write_json(dataset_path, dataset_report.to_dict(), overwrite=True)
    return ArtifactRef(
        kind="training_dataset_report",
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_training_dataset_report",
            "run_id": dataset_report.run_id,
        },
    )
