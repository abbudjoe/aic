from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    LeakageClass,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    derive_training_dataset_report,
    sha256_file,
    write_candidate_policy_registry,
    write_json,
)
from aic_signal_harness.outer_loop_state import (
    OUTER_LOOP_STATE_KIND,
    OuterLoopArtifactRole,
    OuterLoopRunState,
    OuterLoopStage,
    OuterLoopTransition,
    advance_outer_loop_state,
    initialize_outer_loop_state,
    outer_loop_transition_fingerprint_sha256,
    outer_loop_transition_id_from_fingerprint,
    read_outer_loop_state_artifact,
    write_outer_loop_state,
)


def test_outer_loop_state_round_trips_full_train_eval_promote_chain(tmp_path: Path) -> None:
    registry_artifact, selected_candidate_id, dataset_artifact = _candidate_registry_artifact(tmp_path)
    state = initialize_outer_loop_state(
        loop_id="outer-loop-a",
        candidate_registry=registry_artifact,
        created_at_utc="2026-04-25T00:00:00Z",
    )

    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.dataset_ready,
        recorded_at_utc="2026-04-25T00:01:00Z",
        reason="training dataset materialized",
        artifacts={OuterLoopArtifactRole.training_dataset_report: dataset_artifact},
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.training,
        recorded_at_utc="2026-04-25T00:02:00Z",
        reason="bounded offline trainer launched",
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.trained,
        recorded_at_utc="2026-04-25T00:03:00Z",
        reason="offline trainer produced a checkpoint",
        artifacts={
            OuterLoopArtifactRole.policy_training_execution_report: _artifact(
                tmp_path,
                "policy_training_execution_report.json",
                "policy_training_execution_report",
            ),
            OuterLoopArtifactRole.policy_training_report: _artifact(
                tmp_path,
                "policy_training_report.json",
                "policy_training_report",
            ),
            OuterLoopArtifactRole.policy_checkpoint: _artifact(
                tmp_path,
                "policy.ckpt",
                "policy_checkpoint",
                payload={"weights": [1, 2, 3]},
            ),
        },
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.eval_ready,
        recorded_at_utc="2026-04-25T00:04:00Z",
        reason="live eval launch plan recorded",
        artifacts={
            OuterLoopArtifactRole.eval_launch_plan: _artifact(
                tmp_path,
                "eval_launch_plan.json",
                "eval_launch_plan",
            )
        },
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.eval_running,
        recorded_at_utc="2026-04-25T00:05:00Z",
        reason="official evaluator run started",
        artifacts={
            OuterLoopArtifactRole.eval_run_manifest: _artifact(
                tmp_path,
                "eval_run_manifest.json",
                "run_manifest",
            )
        },
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.finalized,
        recorded_at_utc="2026-04-25T00:06:00Z",
        reason="official eval finalized into ledger evidence",
        artifacts={
            OuterLoopArtifactRole.ledger_entry: _artifact(
                tmp_path,
                "ledger_entry.json",
                "ledger_entry",
            )
        },
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.promoted,
        recorded_at_utc="2026-04-25T00:07:00Z",
        reason="promotion gate accepted the candidate",
        artifacts={
            OuterLoopArtifactRole.promotion_decision: _artifact(
                tmp_path,
                "promotion_decision.json",
                "promotion_decision",
            )
        },
    )

    decoded = OuterLoopRunState.from_dict(state.to_dict())
    artifact = write_outer_loop_state(tmp_path / "outer_loop_state.json", decoded)
    reread = read_outer_loop_state_artifact(artifact)

    assert decoded == state
    assert reread == state
    assert artifact.kind == OUTER_LOOP_STATE_KIND
    assert state.current_stage is OuterLoopStage.promoted
    assert state.selected_candidate_id == selected_candidate_id
    assert state.updated_at_utc == "2026-04-25T00:07:00Z"
    assert state.transitions[-1].transition_id == outer_loop_transition_id_from_fingerprint(
        outer_loop_transition_fingerprint_sha256(state.transitions[-1])
    )


def test_outer_loop_state_rejects_illegal_transition_and_terminal_advance(tmp_path: Path) -> None:
    registry_artifact, _, _ = _candidate_registry_artifact(tmp_path)
    state = initialize_outer_loop_state(
        loop_id="outer-loop-a",
        candidate_registry=registry_artifact,
        created_at_utc="2026-04-25T00:00:00Z",
    )

    with pytest.raises(HarnessIOError, match="planned->trained is not allowed"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.trained,
            recorded_at_utc="2026-04-25T00:01:00Z",
            reason="skip evidence",
        )

    failed = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.failed,
        recorded_at_utc="2026-04-25T00:01:00Z",
        reason="dataset materialization failed closed",
    )
    with pytest.raises(HarnessIOError, match="failed->dataset_ready is not allowed"):
        advance_outer_loop_state(
            failed,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="2026-04-25T00:02:00Z",
            reason="resurrect failed loop",
        )


def test_outer_loop_state_requires_stage_artifacts_and_replay_integrity(tmp_path: Path) -> None:
    registry_artifact, _, dataset_artifact = _candidate_registry_artifact(tmp_path)
    state = initialize_outer_loop_state(
        loop_id="outer-loop-a",
        candidate_registry=registry_artifact,
        created_at_utc="2026-04-25T00:00:00Z",
    )

    with pytest.raises(HarnessIOError, match="training_dataset_report"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="2026-04-25T00:01:00Z",
            reason="missing dataset artifact",
        )

    with pytest.raises(HarnessIOError, match="may not introduce artifact roles: policy_checkpoint"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="2026-04-25T00:01:00Z",
            reason="future artifact smuggling",
            artifacts={
                OuterLoopArtifactRole.training_dataset_report: dataset_artifact,
                OuterLoopArtifactRole.policy_checkpoint: _artifact(
                    tmp_path,
                    "too_early_policy.ckpt",
                    "policy_checkpoint",
                ),
            },
        )

    dataset_ready = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.dataset_ready,
        recorded_at_utc="2026-04-25T00:01:00Z",
        reason="dataset materialized",
        artifacts={OuterLoopArtifactRole.training_dataset_report: dataset_artifact},
    )
    payload = dataset_ready.to_dict()
    payload["current_stage"] = "planned"
    with pytest.raises(HarnessIOError, match="current_stage must equal replayed stage"):
        OuterLoopRunState.from_dict(payload)

    payload = dataset_ready.to_dict()
    payload["artifacts"]["policy_checkpoint"] = _artifact(
        tmp_path,
        "unearned_policy.ckpt",
        "policy_checkpoint",
    ).to_dict()
    with pytest.raises(HarnessIOError, match="must equal replayed transition artifacts"):
        OuterLoopRunState.from_dict(payload)


def test_outer_loop_state_rejects_bad_artifacts_and_timestamps(tmp_path: Path) -> None:
    registry_artifact, _, dataset_artifact = _candidate_registry_artifact(tmp_path)
    state = initialize_outer_loop_state(
        loop_id="outer-loop-a",
        candidate_registry=registry_artifact,
        created_at_utc="2026-04-25T00:00:00Z",
    )

    wrong_kind = _artifact(tmp_path, "wrong_kind.json", "policy_checkpoint")
    with pytest.raises(HarnessIOError, match="training_dataset_report'.*sha256|kind must be"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="2026-04-25T00:01:00Z",
            reason="wrong role artifact",
            artifacts={OuterLoopArtifactRole.training_dataset_report: wrong_kind},
        )

    with pytest.raises(HarnessIOError, match="sha256 must match"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="2026-04-25T00:01:00Z",
            reason="digest mismatch",
            artifacts={
                OuterLoopArtifactRole.training_dataset_report: ArtifactRef(
                    kind=dataset_artifact.kind,
                    path=dataset_artifact.path,
                    sha256="0" * 64,
                    provenance=dataset_artifact.provenance,
                )
            },
        )

    with pytest.raises(HarnessIOError, match="must be local byte-verifiable"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="2026-04-25T00:01:00Z",
            reason="remote-only required artifact",
            artifacts={
                OuterLoopArtifactRole.training_dataset_report: ArtifactRef(
                    kind="training_dataset_report",
                    uri="gs://aic-bucket/training_dataset_report.json",
                    sha256="a" * 64,
                    provenance={"producer": "pytest", "derivation": "fixture"},
                )
            },
        )

    with pytest.raises(HarnessIOError, match="UTC timestamp ending in Z"):
        advance_outer_loop_state(
            state,
            to_stage=OuterLoopStage.dataset_ready,
            recorded_at_utc="not-a-timestamp",
            reason="malformed time",
            artifacts={OuterLoopArtifactRole.training_dataset_report: dataset_artifact},
        )


def test_outer_loop_state_artifact_reader_fails_closed_on_provenance(tmp_path: Path) -> None:
    registry_artifact, _, _ = _candidate_registry_artifact(tmp_path)
    state = initialize_outer_loop_state(
        loop_id="outer-loop-a",
        candidate_registry=registry_artifact,
        created_at_utc="2026-04-25T00:00:00Z",
    )
    artifact = write_outer_loop_state(tmp_path / "outer_loop_state.json", state)

    with pytest.raises(HarnessIOError, match="provenance producer must be"):
        read_outer_loop_state_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "pytest",
                    "derivation": "write_outer_loop_state",
                    "loop_id": state.loop_id,
                    "current_stage": state.current_stage.value,
                    "selected_candidate_id": state.selected_candidate_id,
                },
            )
        )

    with pytest.raises(HarnessIOError, match="provenance derivation must be"):
        read_outer_loop_state_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "aic_signal_harness.outer_loop_state",
                    "derivation": "manual",
                    "loop_id": state.loop_id,
                    "current_stage": state.current_stage.value,
                    "selected_candidate_id": state.selected_candidate_id,
                },
            )
        )

    with pytest.raises(HarnessIOError, match="selected_candidate_id"):
        read_outer_loop_state_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "aic_signal_harness.outer_loop_state",
                    "derivation": "write_outer_loop_state",
                    "loop_id": state.loop_id,
                    "current_stage": state.current_stage.value,
                },
            )
        )
    with pytest.raises(HarnessIOError, match="current_stage must match"):
        read_outer_loop_state_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "aic_signal_harness.outer_loop_state",
                    "derivation": "write_outer_loop_state",
                    "loop_id": state.loop_id,
                    "current_stage": "promoted",
                    "selected_candidate_id": state.selected_candidate_id,
                },
            )
        )


def test_outer_loop_transition_id_covers_notes() -> None:
    first = OuterLoopTransition(
        recorded_at_utc="2026-04-25T00:01:00Z",
        from_stage=OuterLoopStage.planned,
        to_stage=OuterLoopStage.failed,
        reason="same reason",
        notes=("first note",),
    )
    second = OuterLoopTransition(
        recorded_at_utc="2026-04-25T00:01:00Z",
        from_stage=OuterLoopStage.planned,
        to_stage=OuterLoopStage.failed,
        reason="same reason",
        notes=("second note",),
    )

    assert first.transition_id != second.transition_id


def test_outer_loop_state_read_rejects_missing_persisted_transition_id(tmp_path: Path) -> None:
    registry_artifact, _, dataset_artifact = _candidate_registry_artifact(tmp_path)
    state = initialize_outer_loop_state(
        loop_id="outer-loop-a",
        candidate_registry=registry_artifact,
        created_at_utc="2026-04-25T00:00:00Z",
    )
    state = advance_outer_loop_state(
        state,
        to_stage=OuterLoopStage.dataset_ready,
        recorded_at_utc="2026-04-25T00:01:00Z",
        reason="dataset materialized",
        artifacts={OuterLoopArtifactRole.training_dataset_report: dataset_artifact},
    )
    payload = state.to_dict()
    del payload["transitions"][0]["transition_id"]

    with pytest.raises(HarnessIOError, match="transition_id"):
        OuterLoopRunState.from_dict(payload)


def _candidate_registry_artifact(tmp_path: Path) -> tuple[ArtifactRef, str, ArtifactRef]:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    candidate = CandidatePolicyRecord(
        generated_at_utc="2026-04-25T00:00:00Z",
        source_training_dataset_report=dataset_artifact,
        trainer=TrainerInvocation(
            trainer_id="lewm-trainer-v1",
            trainer_name="LEWM trainer",
            backend_kind=BackendKind.lewm_world_model,
            command=(sys.executable, "-c", "print('train')"),
            config={"learning_rate": 0.0001},
            seed=11,
            offline_only=True,
            runtime_allowed=False,
            uses_online_language_model_control=False,
        ),
        policy_template=CandidatePolicyTemplate(
            backend_kind=BackendKind.lewm_world_model,
            name="lewm-candidate",
            training_sources=(TrainingSourceKind.official_demo,),
            simulator_sources=(SimulatorKind.offline_replay,),
            legal_observation_contract="official aic_model observations only",
            description="Candidate produced by the outer-loop contract.",
            config={"planner_mode": "lewm_mpc"},
            deterministic=True,
        ),
        eval=CandidateEvalSpec(
            mode=CandidateEvalMode.local_live_eval,
            gate_id="trained_policy_live_eval",
            planner_mode="lewm_mpc",
            min_improvement=1.0,
        ),
    )
    registry = CandidatePolicyRegistry(
        registry_id="registry-gate2-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
    )
    artifact = write_candidate_policy_registry(tmp_path / "candidate_registry.json", registry)
    assert candidate.candidate_id is not None
    return artifact, candidate.candidate_id, dataset_artifact


def _training_dataset_artifact(tmp_path: Path) -> ArtifactRef:
    signal_report = TrainingSignalReport(
        run_id="source-run-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={"producer": "pytest", "derivation": "fixture", "run_id": "source-run-a"},
        ),
        signals=(
            TrainingSignal(
                signal_id="signal-action-1",
                kind=TrainingSignalKind.behavior_clone_action,
                target="policy.action",
                weight=1.0,
                source="episode_trace",
                source_event_indices=(1,),
                extraction_method="episode_trace.v1.behavior_clone_action",
                leakage_class=LeakageClass.legal_policy_action_output,
                evidence={"payload": {"linear": [0.1, 0.0, 0.0]}},
            ),
        ),
    )
    signal_path = tmp_path / "training_signal_report.json"
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
    dataset_path = tmp_path / "training_dataset_report.json"
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


def _artifact(
    tmp_path: Path,
    name: str,
    kind: str,
    *,
    payload: dict[str, Any] | None = None,
) -> ArtifactRef:
    path = tmp_path / name
    write_json(path, {"kind": kind, "payload": {} if payload is None else payload}, overwrite=True)
    return ArtifactRef(
        kind=kind,
        path=str(path),
        sha256=sha256_file(path),
        provenance={"producer": "pytest", "derivation": "fixture"},
    )
