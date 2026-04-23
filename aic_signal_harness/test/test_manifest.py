import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    LeakageClass,
    PolicyBackendSpec,
    RunManifest,
    RunStatus,
    RuntimeBoundaryProof,
    RuntimeRole,
    SchemaValidationError,
    ScoreReport,
    SimulatorKind,
    TrainingSourceKind,
    TrialScore,
)


def _live_backend(**overrides):
    values = {
        "backend_kind": BackendKind.replay_servo,
        "name": "replay-servo-baseline",
        "runtime_role": RuntimeRole.live_policy,
        "training_sources": (TrainingSourceKind.official_demo,),
        "simulator_sources": (SimulatorKind.offline_replay,),
        "runtime_allowed": True,
        "leakage_class": LeakageClass.legal_policy_input,
        "runtime_boundary": RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
        ),
        "description": "Replay a screened demonstration through the official runtime path.",
        "config": {"control_hz": 10},
        "provenance": {"producer": "pytest"},
    }
    values.update(overrides)
    return PolicyBackendSpec(**values)


def _score_report() -> ScoreReport:
    return ScoreReport(
        source="eval/scoring.yaml",
        parsed_at_utc="2026-04-23T00:00:00Z",
        total=7.5,
        trials={"trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)},
    )


def test_planned_manifest_round_trip() -> None:
    manifest = RunManifest(
        run_id="gate0-planned",
        status=RunStatus.planned,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        experiment_id="gate0",
        hypothesis="A neutral manifest can plan a screened run.",
        notes=("awaiting evaluator slot",),
    )

    assert RunManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.to_dict()["status"] == "planned"
    assert manifest.to_dict()["artifacts"] == []
    assert manifest.to_dict()["score"] is None


def test_completed_manifest_round_trip_with_lerobot_backend_policy_artifact() -> None:
    policy_artifact = ArtifactRef(
        kind="policy_checkpoint",
        uri="s3://bucket/act_policy.pt",
        sha256="a" * 64,
    )
    backend = _live_backend(
        backend_kind=BackendKind.lerobot_act,
        name="isaac-act-policy",
        training_sources=(TrainingSourceKind.official_demo, TrainingSourceKind.isaac_synthetic),
        simulator_sources=(SimulatorKind.isaac_lab,),
        runtime_boundary=RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
            policy_artifact=policy_artifact,
        ),
        config={"architecture": "act"},
    )
    manifest = RunManifest(
        run_id="gate1-act-complete",
        status=RunStatus.completed,
        backend=backend,
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:10:00Z",
        artifacts=(
            ArtifactRef(kind="scoring_yaml", path="eval/scoring.yaml", sha256="b" * 64),
        ),
        score=_score_report(),
        experiment_id="gate1",
        hypothesis="ACT can be evaluated through official scoring.",
    )

    decoded = RunManifest.from_dict(manifest.to_dict())

    assert decoded == manifest
    assert decoded.backend.backend_kind is BackendKind.lerobot_act
    assert decoded.backend.runtime_boundary.policy_artifact == policy_artifact
    assert decoded.score.total == 7.5


def test_completed_manifest_requires_score_and_artifacts() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        RunManifest(
            run_id="missing-evidence",
            status=RunStatus.completed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
        )

    message = str(exc_info.value)
    assert "completed manifest must include at least one artifact" in message
    assert "completed manifest must include a score report" in message


def test_completed_manifest_accepts_matching_scoring_artifact_path() -> None:
    manifest = RunManifest(
        run_id="matching-scoring-path",
        status=RunStatus.completed,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(ArtifactRef(kind="scoring_yaml", path="eval/scoring.yaml"),),
        score=_score_report(),
    )

    assert manifest.artifacts[0].path == manifest.score.source


def test_promoted_manifest_accepts_matching_scoring_artifact_uri() -> None:
    source = "s3://bucket/eval/scoring.yaml"
    manifest = RunManifest(
        run_id="matching-scoring-uri",
        status=RunStatus.promoted,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(ArtifactRef(kind="scoring_yaml", uri=source),),
        score=ScoreReport(
            source=source,
            parsed_at_utc="2026-04-23T00:00:00Z",
            total=7.5,
            trials={"trial_1": TrialScore(total=7.5, tier_1=7.5)},
        ),
    )

    assert manifest.artifacts[0].uri == manifest.score.source


def test_completed_manifest_rejects_unrelated_scoring_artifact() -> None:
    with pytest.raises(SchemaValidationError, match="scoring_yaml artifact"):
        RunManifest(
            run_id="unrelated-scoring-artifact",
            status=RunStatus.completed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(ArtifactRef(kind="scoring_yaml", path="other/scoring.yaml"),),
            score=_score_report(),
        )


def test_completed_manifest_rejects_matching_path_with_wrong_artifact_kind() -> None:
    with pytest.raises(SchemaValidationError, match="scoring_yaml artifact"):
        RunManifest(
            run_id="wrong-scoring-artifact-kind",
            status=RunStatus.completed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(ArtifactRef(kind="log", path="eval/scoring.yaml"),),
            score=_score_report(),
        )


def test_manifest_from_dict_rejects_unknown_fields() -> None:
    payload = RunManifest(
        run_id="unknown-field",
        status=RunStatus.planned,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
    ).to_dict()
    payload["policy_path"] = "s3://bucket/hidden-policy.pt"

    with pytest.raises(SchemaValidationError, match="run manifest has unknown fields"):
        RunManifest.from_dict(payload)
