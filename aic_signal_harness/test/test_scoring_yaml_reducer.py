from pathlib import Path

import pytest
import aic_signal_harness.reducers.scoring_yaml as scoring_yaml_reducer

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    HarnessIOError,
    LeakageClass,
    PolicyBackendSpec,
    RunManifest,
    RunStatus,
    RuntimeBoundaryProof,
    RuntimeRole,
    SchemaValidationError,
    ScoreReport,
    ScoringYamlReduction,
    SimulatorKind,
    TrainingSourceKind,
    TrialScore,
    attach_scoring_yaml_reduction,
    reduce_scoring_yaml,
    sha256_file,
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


def _manifest(**overrides) -> RunManifest:
    values = {
        "run_id": "gate2-backfill",
        "status": RunStatus.planned,
        "backend": _live_backend(),
        "created_at_utc": "2026-04-23T00:00:00Z",
        "updated_at_utc": "2026-04-23T00:00:00Z",
    }
    values.update(overrides)
    return RunManifest(**values)


def _score_report(
    source: str,
    *,
    total: float = 7.5,
    parsed_at_utc: str | None = "2026-04-23T00:10:00Z",
) -> ScoreReport:
    return ScoreReport(
        source=source,
        parsed_at_utc=parsed_at_utc,
        total=total,
        trials={
            "trial_1": TrialScore(
                total=total,
                tier_1=1.0,
                tier_2=2.5,
                tier_3=total - 3.5,
            )
        },
    )


def _write_scoring_yaml(path: Path, *, total: float = 7.5) -> None:
    path.write_text(
        f"""
total: {total}
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 2.5
  tier_3:
    score: {total - 3.5}
""",
        encoding="utf-8",
    )


def test_reduce_scoring_yaml_captures_digest_and_metadata(tmp_path: Path) -> None:
    scoring = tmp_path / "eval" / "scoring.yaml"
    scoring.parent.mkdir()
    _write_scoring_yaml(scoring)

    reduction = reduce_scoring_yaml(
        scoring,
        parsed_at_utc="2026-04-23T00:10:00Z",
        uri="s3://bucket/eval/scoring.yaml",
        provenance={"producer": "pytest", "run_id": "gate2"},
    )

    assert reduction == ScoringYamlReduction(
        artifact=ArtifactRef(
            kind="scoring_yaml",
            path=str(scoring),
            uri="s3://bucket/eval/scoring.yaml",
            sha256=sha256_file(scoring),
            provenance={"producer": "pytest", "run_id": "gate2"},
        ),
        score=_score_report(str(scoring)),
    )


def test_reduce_scoring_yaml_rejects_split_path_file_uri_identity(tmp_path: Path) -> None:
    scoring = tmp_path / "eval" / "scoring.yaml"
    other_scoring = tmp_path / "other" / "scoring.yaml"
    scoring.parent.mkdir()
    other_scoring.parent.mkdir()
    _write_scoring_yaml(scoring)
    _write_scoring_yaml(other_scoring)

    with pytest.raises(HarnessIOError, match="path and file URI"):
        reduce_scoring_yaml(scoring, uri=other_scoring.as_uri())


def test_reduce_scoring_yaml_rejects_malformed_uri_as_harness_error(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "eval" / "scoring.yaml"
    scoring.parent.mkdir()
    _write_scoring_yaml(scoring)

    for uri, message in (
        ("not-a-uri", "URI must include a scheme"),
        ("s3:path", "URI must include a network location"),
        ("x:", "URI must include a network location"),
        ("file:///tmp/%00x", "must not contain NUL"),
        ("file:///tmp/%ZZ", "invalid percent escape"),
    ):
        with pytest.raises(HarnessIOError, match=message):
            reduce_scoring_yaml(scoring, uri=uri)


def test_reduce_scoring_yaml_rejects_file_changed_after_snapshot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring, total=7.5)
    original_read_scoring_yaml_snapshot = scoring_yaml_reducer.read_scoring_yaml_snapshot

    def read_and_mutate(path: str | Path) -> tuple[bytes, str]:
        loaded_bytes, canonical_path = original_read_scoring_yaml_snapshot(path)
        _write_scoring_yaml(Path(path), total=1.0)
        return loaded_bytes, canonical_path

    monkeypatch.setattr(
        scoring_yaml_reducer,
        "read_scoring_yaml_snapshot",
        read_and_mutate,
    )

    with pytest.raises(HarnessIOError, match="sha256 must match source file"):
        reduce_scoring_yaml(scoring)


def test_scoring_yaml_reduction_rejects_uri_only_decoded_nul_as_harness_error() -> None:
    source = "file:///tmp/%00x"

    with pytest.raises(SchemaValidationError, match="must not contain NUL"):
        ArtifactRef(
            kind="scoring_yaml",
            uri=source,
            sha256="a" * 64,
        )


def test_scoring_yaml_reduction_rejects_uri_only_invalid_escape_as_harness_error() -> None:
    source = "file:///tmp/%ZZ"

    with pytest.raises(SchemaValidationError, match="invalid percent escape"):
        ArtifactRef(
            kind="scoring_yaml",
            uri=source,
            sha256="a" * 64,
        )


def test_scoring_yaml_reduction_rejects_forged_local_digest(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    forged_sha256 = "0" * 64
    if forged_sha256 == sha256_file(scoring):
        forged_sha256 = "1" * 64

    with pytest.raises(HarnessIOError, match="sha256 must match source file"):
        ScoringYamlReduction(
            artifact=ArtifactRef(
                kind="scoring_yaml",
                path=str(scoring),
                sha256=forged_sha256,
            ),
            score=_score_report(str(scoring)),
        )


def test_attach_scoring_yaml_reduction_adds_score_and_artifact(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring)
    manifest = _manifest()

    attached = attach_scoring_yaml_reduction(manifest, reduction)

    assert attached.score == reduction.score
    assert attached.artifacts == (reduction.artifact,)
    assert attached.updated_at_utc == manifest.updated_at_utc
    assert manifest.score is None
    assert manifest.artifacts == ()


def test_attach_scoring_yaml_reduction_is_idempotent(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring)

    attached_once = attach_scoring_yaml_reduction(_manifest(), reduction)
    attached_twice = attach_scoring_yaml_reduction(attached_once, reduction)

    assert attached_twice is attached_once


def test_attach_scoring_yaml_reduction_accepts_equivalent_existing_score_state(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring, uri="s3://bucket/eval/scoring.yaml")
    manifest = _manifest(
        artifacts=(reduction.artifact,),
        score=_score_report("s3://bucket/eval/scoring.yaml"),
    )

    attached = attach_scoring_yaml_reduction(manifest, reduction)

    assert attached is not manifest
    assert attached.score.source == reduction.score.source
    assert attached.score.total == reduction.score.total
    assert attached.score.trials == reduction.score.trials
    assert attached.score.parsed_at_utc == manifest.score.parsed_at_utc


def test_attach_scoring_yaml_reduction_rejects_conflicting_existing_score(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring)
    manifest = _manifest(
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                path=str(scoring),
                sha256=reduction.artifact.sha256,
            ),
        ),
        score=_score_report(str(scoring), total=1.0),
    )

    with pytest.raises(HarnessIOError, match="conflicting score report state"):
        attach_scoring_yaml_reduction(manifest, reduction)


def test_reduce_scoring_yaml_canonicalizes_local_path_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring = tmp_path / "eval" / "scoring.yaml"
    scoring.parent.mkdir()
    _write_scoring_yaml(scoring)
    monkeypatch.chdir(tmp_path)

    relative_reduction = reduce_scoring_yaml(Path("eval/scoring.yaml"))
    absolute_reduction = reduce_scoring_yaml(scoring)

    assert relative_reduction == absolute_reduction


def test_attach_scoring_yaml_reduction_rejects_relative_manifest_scoring_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring = tmp_path / "eval" / "scoring.yaml"
    scoring.parent.mkdir()
    _write_scoring_yaml(scoring)
    monkeypatch.chdir(tmp_path)
    reduction = reduce_scoring_yaml(Path("eval/scoring.yaml"))
    with pytest.raises(
        SchemaValidationError,
        match="scoring_yaml artifact.path must be an absolute local path when set",
    ):
        _manifest(
            artifacts=(
                ArtifactRef(
                    kind="scoring_yaml",
                    path="eval/scoring.yaml",
                    sha256=reduction.artifact.sha256,
                ),
            ),
            score=_score_report("eval/scoring.yaml"),
        )


def test_attach_scoring_yaml_reduction_merges_uri_only_existing_scoring_artifact(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    uri = "s3://bucket/eval/scoring.yaml"
    reduction = reduce_scoring_yaml(
        scoring,
        uri=uri,
        provenance={"producer": "pytest", "run_id": "gate2"},
    )
    manifest = _manifest(
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                uri=uri,
                provenance={"producer": "pytest"},
            ),
        ),
    )

    attached = attach_scoring_yaml_reduction(manifest, reduction)

    assert attached is not manifest
    assert attached.score == reduction.score
    assert attached.artifacts == (
        ArtifactRef(
            kind="scoring_yaml",
            path=str(scoring),
            uri=uri,
            sha256=sha256_file(scoring),
            provenance={"producer": "pytest", "run_id": "gate2"},
        ),
    )


def test_attach_scoring_yaml_reduction_upgrades_equivalent_partial_legacy_score(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 0
  tier_3:
    score: 0
""",
        encoding="utf-8",
    )
    reduction = reduce_scoring_yaml(scoring)
    legacy_score = ScoreReport.from_legacy_v1_dict(
        {
            "schema_version": 1,
            "source": reduction.score.source,
            "parsed_at_utc": None,
            "total": reduction.score.total,
            "trial_count": 1,
            "trials": {
                "trial_1": {"total": reduction.score.total, "tier_1": reduction.score.total},
            },
        }
    )
    manifest = _manifest(
        artifacts=(reduction.artifact,),
        score=legacy_score,
    )

    attached = attach_scoring_yaml_reduction(manifest, reduction)

    assert attached is manifest
    assert attached.score == reduction.score


def test_attach_scoring_yaml_reduction_merges_path_only_existing_scoring_artifact(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring)
    manifest = _manifest(
        artifacts=(ArtifactRef(kind="scoring_yaml", path=str(scoring)),),
    )

    attached = attach_scoring_yaml_reduction(manifest, reduction)

    assert attached is not manifest
    assert attached.score == reduction.score
    assert attached.artifacts == (reduction.artifact,)


def test_attach_scoring_yaml_reduction_rejects_conflicting_existing_scoring_artifact_uri(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring, uri="s3://bucket/eval/scoring.yaml")
    manifest = _manifest(
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                path=str(scoring),
                uri="s3://bucket/other/scoring.yaml",
            ),
        ),
    )

    with pytest.raises(
        HarnessIOError,
        match="conflicting scoring_yaml artifact state: uri",
    ):
        attach_scoring_yaml_reduction(manifest, reduction)


def test_attach_scoring_yaml_reduction_rejects_conflicting_existing_scoring_artifact_sha256(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring)
    wrong_sha256 = "0" * 64 if reduction.artifact.sha256 != "0" * 64 else "1" * 64
    manifest = _manifest(
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                path=str(scoring),
                sha256=wrong_sha256,
            ),
        ),
    )

    with pytest.raises(
        HarnessIOError,
        match="conflicting scoring_yaml artifact state: sha256",
    ):
        attach_scoring_yaml_reduction(manifest, reduction)


def test_attach_scoring_yaml_reduction_rejects_conflicting_existing_scoring_artifact_provenance(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(
        scoring,
        provenance={"producer": "pytest", "run_id": "gate2"},
    )
    manifest = _manifest(
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                path=str(scoring),
                provenance={"producer": "other"},
            ),
        ),
    )

    with pytest.raises(
        HarnessIOError,
        match="conflicting scoring_yaml artifact state: provenance.producer",
    ):
        attach_scoring_yaml_reduction(manifest, reduction)


def test_manifest_rejects_duplicate_existing_scoring_artifacts_before_attach(
    tmp_path: Path,
) -> None:
    scoring = tmp_path / "scoring.yaml"
    _write_scoring_yaml(scoring)
    reduction = reduce_scoring_yaml(scoring)

    with pytest.raises(SchemaValidationError, match="duplicate scoring_yaml artifacts"):
        _manifest(artifacts=(reduction.artifact, reduction.artifact))


def test_scoring_yaml_reduction_rejects_bad_contracts() -> None:
    score = _score_report("eval/scoring.yaml")
    uri_score = _score_report("s3://bucket/eval/scoring.yaml")

    reduction = ScoringYamlReduction(
        artifact=ArtifactRef(
            kind="scoring_yaml",
            uri="s3://bucket/eval/scoring.yaml",
            sha256="a" * 64,
        ),
        score=uri_score,
    )

    assert reduction.score is uri_score

    with pytest.raises(HarnessIOError, match="artifact.kind must be 'scoring_yaml'"):
        ScoringYamlReduction(
            artifact=ArtifactRef(kind="log", path="eval/scoring.yaml", sha256="a" * 64),
            score=score,
        )

    with pytest.raises(HarnessIOError, match="artifact.sha256 must be set"):
        ScoringYamlReduction(
            artifact=ArtifactRef(kind="scoring_yaml", path="eval/scoring.yaml"),
            score=score,
        )

    with pytest.raises(HarnessIOError, match="artifact path or uri must match score.source"):
        ScoringYamlReduction(
            artifact=ArtifactRef(
                kind="scoring_yaml",
                path="other/scoring.yaml",
                uri="s3://bucket/other/scoring.yaml",
                sha256="a" * 64,
            ),
            score=score,
        )
