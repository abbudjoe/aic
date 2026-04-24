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

_ABS_SCORE_SOURCE = "/tmp/eval/scoring.yaml"


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
        source=_ABS_SCORE_SOURCE,
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
            ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE, sha256="b" * 64),
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
        artifacts=(ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE, sha256="a" * 64),),
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
        artifacts=(ArtifactRef(kind="scoring_yaml", uri=source, sha256="a" * 64),),
        score=ScoreReport(
            source=source,
            parsed_at_utc="2026-04-23T00:00:00Z",
            total=7.5,
            trials={"trial_1": TrialScore(total=7.5, tier_1=7.5, tier_2=0.0, tier_3=0.0)},
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
            artifacts=(ArtifactRef(kind="scoring_yaml", path="/tmp/other/scoring.yaml"),),
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
            artifacts=(ArtifactRef(kind="log", path=_ABS_SCORE_SOURCE),),
            score=_score_report(),
        )


def test_completed_manifest_rejects_duplicate_scoring_artifacts() -> None:
    with pytest.raises(SchemaValidationError, match="duplicate scoring_yaml artifacts"):
        RunManifest(
            run_id="duplicate-scoring-artifacts",
            status=RunStatus.completed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(
                ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE),
                ArtifactRef(kind="scoring_yaml", uri="s3://bucket/eval/scoring.yaml"),
            ),
            score=_score_report(),
        )


def test_planned_manifest_rejects_duplicate_scoring_artifacts() -> None:
    with pytest.raises(SchemaValidationError, match="duplicate scoring_yaml artifacts"):
        RunManifest(
            run_id="planned-duplicate-scoring-artifacts",
            status=RunStatus.planned,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(
                ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE),
                ArtifactRef(kind="scoring_yaml", uri="s3://bucket/eval/scoring.yaml"),
            ),
        )


def test_completed_manifest_rejects_scoring_artifact_without_sha256() -> None:
    with pytest.raises(SchemaValidationError, match="manifests with score must bind"):
        RunManifest(
            run_id="undigested-scoring-artifact",
            status=RunStatus.completed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE),),
            score=_score_report(),
        )


def test_failed_manifest_with_score_requires_bound_scoring_artifact() -> None:
    with pytest.raises(SchemaValidationError, match="manifests with score must include"):
        RunManifest(
            run_id="failed-with-unbound-score",
            status=RunStatus.failed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
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


def test_manifest_rejects_malformed_artifact_entries_fail_closed() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "bad-artifact-entry",
        "status": "completed",
        "created_at_utc": "2026-04-23T00:00:00Z",
        "updated_at_utc": "2026-04-23T00:00:00Z",
        "backend": _live_backend().to_dict(),
        "artifacts": [{}],
        "score": _score_report().to_dict(),
        "experiment_id": None,
        "hypothesis": None,
        "notes": [],
    }

    with pytest.raises(SchemaValidationError, match="artifact must set path or uri"):
        RunManifest.from_dict(payload)


def test_manifest_rejects_relative_scoring_artifact_path() -> None:
    with pytest.raises(
        SchemaValidationError,
        match="scoring_yaml artifact.path must be an absolute local path when set",
    ):
        RunManifest(
            run_id="relative-scoring-artifact",
            status=RunStatus.planned,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(ArtifactRef(kind="scoring_yaml", path="eval/scoring.yaml"),),
        )


def test_manifest_rejects_colon_bearing_relative_scoring_artifact_path() -> None:
    with pytest.raises(
        SchemaValidationError,
        match="scoring_yaml artifact.path must be an absolute local path when set",
    ):
        RunManifest(
            run_id="colon-relative-scoring-artifact",
            status=RunStatus.planned,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(ArtifactRef(kind="scoring_yaml", path="logs:scoring.yaml"),),
        )


def test_manifest_rejects_relative_score_source() -> None:
    with pytest.raises(
        SchemaValidationError,
        match="score.source must be an absolute local path or uri",
    ):
        RunManifest(
            run_id="relative-score-source",
            status=RunStatus.completed,
            backend=_live_backend(),
            created_at_utc="2026-04-23T00:00:00Z",
            updated_at_utc="2026-04-23T00:00:00Z",
            artifacts=(ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE, sha256="a" * 64),),
            score=ScoreReport(
                source="eval/scoring.yaml",
                parsed_at_utc="2026-04-23T00:00:00Z",
                total=7.5,
                trials={
                    "trial_1": TrialScore(
                        total=7.5,
                        tier_1=1.0,
                        tier_2=2.5,
                        tier_3=4.0,
                    )
                },
            ),
        )


def test_manifest_from_dict_rejects_relative_scoring_artifact_path() -> None:
    payload = RunManifest(
        run_id="relative-scoring-artifact-payload",
        status=RunStatus.planned,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
    ).to_dict()
    payload["artifacts"] = [{"kind": "scoring_yaml", "path": "eval/scoring.yaml"}]

    with pytest.raises(
        SchemaValidationError,
        match="scoring_yaml artifact.path must be an absolute local path when set",
    ):
        RunManifest.from_dict(payload)


def test_manifest_from_dict_rejects_colon_bearing_relative_scoring_artifact_path() -> None:
    payload = RunManifest(
        run_id="colon-relative-scoring-artifact-payload",
        status=RunStatus.planned,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
    ).to_dict()
    payload["artifacts"] = [{"kind": "scoring_yaml", "path": "logs:scoring.yaml"}]

    with pytest.raises(
        SchemaValidationError,
        match="scoring_yaml artifact.path must be an absolute local path when set",
    ):
        RunManifest.from_dict(payload)


def test_manifest_from_dict_rejects_relative_score_source() -> None:
    payload = RunManifest(
        run_id="relative-score-source-payload",
        status=RunStatus.completed,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE, sha256="a" * 64),),
        score=_score_report(),
    ).to_dict()
    payload["score"]["source"] = "eval/scoring.yaml"

    with pytest.raises(
        SchemaValidationError,
        match="score.source must be an absolute local path or uri",
    ):
        RunManifest.from_dict(payload)


def test_manifest_from_dict_rejects_colon_bearing_relative_score_source() -> None:
    payload = RunManifest(
        run_id="colon-relative-score-source-payload",
        status=RunStatus.completed,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE, sha256="a" * 64),),
        score=_score_report(),
    ).to_dict()
    payload["score"]["source"] = "logs:scoring.yaml"

    with pytest.raises(
        SchemaValidationError,
        match="score.source must be an absolute local path or uri",
    ):
        RunManifest.from_dict(payload)


def test_manifest_from_dict_rejects_partial_legacy_score_payload_under_current_schema() -> None:
    payload = RunManifest(
        run_id="legacy-score-payload",
        status=RunStatus.completed,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(ArtifactRef(kind="scoring_yaml", path=_ABS_SCORE_SOURCE, sha256="a" * 64),),
        score=_score_report(),
    ).to_dict()
    payload["score"] = {
        "schema_version": 1,
        "source": _ABS_SCORE_SOURCE,
        "parsed_at_utc": None,
        "total": 1.0,
        "trial_count": 1,
        "trials": {"trial_1": {"total": 1.0, "tier_1": 1.0}},
    }

    with pytest.raises(SchemaValidationError, match="must include tier_1, tier_2, and tier_3"):
        RunManifest.from_dict(payload)
