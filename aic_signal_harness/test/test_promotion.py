from pathlib import Path
import math

import pytest

from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
    PromotionMetric,
    read_promotion_decision,
    write_baseline_decision,
    write_promotion_decision,
)


def _decision(
    *,
    decision: PromotionDecisionKind = PromotionDecisionKind.accepted,
    accepted: bool = True,
    eligible_for_submission: bool = True,
) -> PromotionDecision:
    return PromotionDecision(
        decided_at_utc="2026-04-23T12:00:00Z",
        decision=decision,
        accepted=accepted,
        eligible_for_submission=eligible_for_submission,
        reason="127.8 >= 122.3 + 1.0",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=127.8,
            baseline_value=122.3,
            min_improvement=1.0,
        ),
        run_id="gate2-run-a",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-a/manifest.json",
            sha256="a" * 64,
        ),
        baseline_run_id="gate2-run-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-baseline/manifest.json",
            sha256="b" * 64,
        ),
        notes=("kept as the current replay baseline",),
    )


def test_promotion_decision_round_trips_strictly() -> None:
    decision = _decision()

    assert PromotionDecision.from_dict(decision.to_dict()) == decision


def test_promotion_decision_rejects_incoherent_acceptance_state() -> None:
    with pytest.raises(HarnessIOError, match="accepted/bootstrap promotion decisions must set accepted=true"):
        PromotionDecision(
            decided_at_utc="2026-04-23T12:00:00Z",
            decision=PromotionDecisionKind.accepted,
            accepted=False,
            eligible_for_submission=False,
            reason="bad",
            metric=PromotionMetric(
                name="evaluation.score.total",
                goal=PromotionGoal.max,
                value=10.0,
                baseline_value=5.0,
                min_improvement=1.0,
            ),
            run_id="gate2-run-a",
            manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            baseline_run_id="baseline",
            baseline_manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/baseline/manifest.json",
                sha256="b" * 64,
            ),
        )


def test_promotion_decision_rejects_metric_snapshot_that_loses() -> None:
    with pytest.raises(
        HarnessIOError,
        match="promotion decision acceptance does not match metric comparison",
    ):
        PromotionDecision(
            decided_at_utc="2026-04-23T12:00:00Z",
            decision=PromotionDecisionKind.accepted,
            accepted=True,
            eligible_for_submission=True,
            reason="bad accepted snapshot",
            metric=PromotionMetric(
                name="evaluation.score.total",
                goal=PromotionGoal.max,
                value=10.0,
                baseline_value=12.0,
                min_improvement=1.0,
            ),
            run_id="gate2-run-a",
            manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            baseline_run_id="baseline",
            baseline_manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/baseline/manifest.json",
                sha256="b" * 64,
            ),
        )


def test_promotion_decision_rejects_goal_min_metric_snapshot_that_loses() -> None:
    with pytest.raises(
        HarnessIOError,
        match="promotion decision acceptance does not match metric comparison",
    ):
        PromotionDecision(
            decided_at_utc="2026-04-23T12:00:00Z",
            decision=PromotionDecisionKind.accepted,
            accepted=True,
            eligible_for_submission=True,
            reason="bad accepted min snapshot",
            metric=PromotionMetric(
                name="evaluation.force.max",
                goal=PromotionGoal.min,
                value=10.0,
                baseline_value=5.0,
                min_improvement=1.0,
            ),
            run_id="gate2-run-a",
            manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            baseline_run_id="baseline",
            baseline_manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/baseline/manifest.json",
                sha256="b" * 64,
            ),
        )


def test_promotion_decision_rejects_baseline_refs_set_one_sided() -> None:
    with pytest.raises(HarnessIOError, match="must be set together"):
        PromotionDecision(
            decided_at_utc="2026-04-23T12:00:00Z",
            decision=PromotionDecisionKind.rejected,
            accepted=False,
            eligible_for_submission=False,
            reason="bad",
            metric=PromotionMetric(
                name="evaluation.score.total",
                goal=PromotionGoal.max,
                value=10.0,
                baseline_value=12.0,
                min_improvement=1.0,
            ),
            run_id="gate2-run-a",
            manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            baseline_run_id="baseline",
        )


def test_promotion_decision_rejects_nonbootstrap_without_baseline_context() -> None:
    with pytest.raises(
        HarnessIOError,
        match="accepted/rejected promotion decisions must set metric.baseline_value",
    ):
        PromotionDecision(
            decided_at_utc="2026-04-23T12:00:00Z",
            decision=PromotionDecisionKind.accepted,
            accepted=True,
            eligible_for_submission=True,
            reason="missing baseline context",
            metric=PromotionMetric(
                name="evaluation.score.total",
                goal=PromotionGoal.max,
                value=10.0,
                baseline_value=None,
                min_improvement=1.0,
            ),
            run_id="gate2-run-a",
            manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
        )


def test_promotion_decision_from_dict_rejects_malformed_metric_with_harness_error() -> None:
    payload = _decision().to_dict()
    payload["metric"] = {}

    with pytest.raises(HarnessIOError, match="promotion metric"):
        PromotionDecision.from_dict(payload)


def test_promotion_decision_from_dict_rejects_malformed_manifest_with_harness_error() -> None:
    payload = _decision().to_dict()
    payload["manifest"] = {}

    with pytest.raises(HarnessIOError):
        PromotionDecision.from_dict(payload)


def test_promotion_decision_from_dict_rejects_malformed_baseline_manifest() -> None:
    payload = _decision().to_dict()
    payload["baseline_manifest"] = {}

    with pytest.raises(HarnessIOError):
        PromotionDecision.from_dict(payload)


def test_promotion_decision_rejects_self_baseline_identity() -> None:
    with pytest.raises(HarnessIOError, match="must not compare a run against itself"):
        PromotionDecision(
            decided_at_utc="2026-04-23T12:00:00Z",
            decision=PromotionDecisionKind.accepted,
            accepted=True,
            eligible_for_submission=True,
            reason="same run twice",
            metric=PromotionMetric(
                name="evaluation.score.total",
                goal=PromotionGoal.max,
                value=10.0,
                baseline_value=9.0,
                min_improvement=1.0,
            ),
            run_id="gate2-run-a",
            manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            baseline_run_id="gate2-run-a",
            baseline_manifest=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a-copy/manifest.json",
                sha256="a" * 64,
            ),
        )


def test_write_and_read_promotion_decision_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    decision = _decision()

    write_promotion_decision(path, decision)

    assert read_promotion_decision(path) == decision


def test_write_and_read_bootstrap_promotion_decision_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "bootstrap_promotion.json"
    decision = PromotionDecision(
        decided_at_utc="2026-04-23T12:00:00Z",
        decision=PromotionDecisionKind.bootstrap,
        accepted=True,
        eligible_for_submission=False,
        reason="bootstrap baseline",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=97.8,
            baseline_value=None,
            min_improvement=0.0,
        ),
        run_id="gate2-bootstrap",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-bootstrap/manifest.json",
            sha256="d" * 64,
        ),
    )

    write_promotion_decision(path, decision)

    assert read_promotion_decision(path) == decision


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, True])
def test_promotion_metric_rejects_non_finite_min_improvement(bad_value: float) -> None:
    with pytest.raises(HarnessIOError, match="promotion metric.min_improvement must be a finite number"):
        PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=10.0,
            baseline_value=9.0,
            min_improvement=bad_value,
        )


def test_write_baseline_decision_requires_accepted_decision(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    rejected = PromotionDecision(
        decided_at_utc="2026-04-23T12:00:00Z",
        decision=PromotionDecisionKind.rejected,
        accepted=False,
        eligible_for_submission=False,
        reason="loses the margin",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=122.9,
            baseline_value=122.3,
            min_improvement=1.0,
        ),
        run_id="gate2-run-a",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-a/manifest.json",
            sha256="a" * 64,
        ),
        baseline_run_id="gate2-run-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-baseline/manifest.json",
            sha256="b" * 64,
        ),
    )

    with pytest.raises(HarnessIOError, match="baseline decisions must be accepted"):
        write_baseline_decision(path, rejected)
