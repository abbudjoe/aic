import math

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    HarnessIOError,
    LedgerEntry,
    LedgerMetric,
    LedgerMetricGoal,
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
    PromotionMetric,
    RunStatus,
    promote_ledger_entry,
)


def _entry(
    *,
    value: float | None = 127.8,
    goal: LedgerMetricGoal = LedgerMetricGoal.max,
    status: RunStatus = RunStatus.completed,
) -> LedgerEntry:
    return LedgerEntry(
        recorded_at_utc="2026-04-23T12:00:00Z",
        run_id="gate2-run-a",
        status=status,
        backend_kind=BackendKind.replay_servo,
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-a/manifest.json",
            sha256="a" * 64,
        ),
        metric=LedgerMetric(
            name="evaluation.score.total",
            goal=goal,
            value=value,
        ),
        notes=("candidate ready for gate",),
    )


def _baseline(
    *,
    value: float = 122.3,
    goal: PromotionGoal = PromotionGoal.max,
    accepted: bool = True,
) -> PromotionDecision:
    return PromotionDecision(
        decided_at_utc="2026-04-23T11:00:00Z",
        decision=PromotionDecisionKind.accepted if accepted else PromotionDecisionKind.rejected,
        accepted=accepted,
        eligible_for_submission=accepted,
        reason="prior baseline",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=goal,
            value=value,
            baseline_value=100.0,
            min_improvement=1.0,
        ),
        run_id="gate2-run-baseline",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-baseline/manifest.json",
            sha256="b" * 64,
        ),
        baseline_run_id="gate1-run-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate1-run-baseline/manifest.json",
            sha256="c" * 64,
        ),
    )


def test_promote_ledger_entry_bootstraps_when_requested() -> None:
    decision = promote_ledger_entry(
        _entry(value=97.8),
        bootstrap=True,
        eligible_for_submission=False,
        decided_at_utc="2026-04-23T12:00:00Z",
    )

    assert decision.decision is PromotionDecisionKind.bootstrap
    assert decision.accepted is True
    assert decision.metric.baseline_value is None


def test_promote_ledger_entry_rejects_missing_baseline_without_bootstrap() -> None:
    with pytest.raises(HarnessIOError, match="bootstrap=True is required"):
        promote_ledger_entry(_entry())


def test_promote_ledger_entry_requires_completed_status() -> None:
    with pytest.raises(HarnessIOError, match="entry.status=completed"):
        promote_ledger_entry(_entry(status=RunStatus.failed), bootstrap=True)


def test_promote_ledger_entry_accepts_candidate_above_margin() -> None:
    decision = promote_ledger_entry(
        _entry(value=127.8),
        baseline=_baseline(value=122.3),
        min_improvement=1.0,
        eligible_for_submission=True,
        notes=("Accepted as the current replay baseline.",),
        decided_at_utc="2026-04-23T12:00:00Z",
    )

    assert decision.decision is PromotionDecisionKind.accepted
    assert decision.accepted is True
    assert decision.eligible_for_submission is True
    assert decision.reason == "127.8 >= 122.3 + 1.0"
    assert decision.baseline_run_id == "gate2-run-baseline"


def test_promote_ledger_entry_rejects_candidate_below_margin() -> None:
    decision = promote_ledger_entry(
        _entry(value=122.9),
        baseline=_baseline(value=122.3),
        min_improvement=1.0,
        eligible_for_submission=True,
        decided_at_utc="2026-04-23T12:00:00Z",
    )

    assert decision.decision is PromotionDecisionKind.rejected
    assert decision.accepted is False
    assert decision.eligible_for_submission is False
    assert decision.reason == "122.9 < 122.3 + 1.0"


def test_promote_ledger_entry_accepts_goal_min_candidates_below_margin() -> None:
    decision = promote_ledger_entry(
        _entry(value=3.0, goal=LedgerMetricGoal.min),
        baseline=_baseline(value=5.0, goal=PromotionGoal.min),
        min_improvement=1.0,
        decided_at_utc="2026-04-23T12:00:00Z",
    )

    assert decision.decision is PromotionDecisionKind.accepted
    assert decision.reason == "3.0 <= 5.0 - 1.0"


def test_promote_ledger_entry_rejects_stale_baseline_metric_name() -> None:
    stale = PromotionDecision.from_dict(
        {
            **_baseline().to_dict(),
            "metric": {
                **_baseline().metric.to_dict(),
                "name": "other.metric",
            },
        }
    )

    with pytest.raises(HarnessIOError, match="baseline metric.name does not match"):
        promote_ledger_entry(_entry(), baseline=stale)


def test_promote_ledger_entry_rejects_malformed_baseline_mapping() -> None:
    baseline = _baseline().to_dict()
    baseline["baseline_manifest"] = {}

    with pytest.raises(HarnessIOError):
        promote_ledger_entry(_entry(), baseline=baseline)


def test_promote_ledger_entry_rejects_unaccepted_baseline() -> None:
    rejected_baseline = PromotionDecision(
        decided_at_utc="2026-04-23T11:00:00Z",
        decision=PromotionDecisionKind.rejected,
        accepted=False,
        eligible_for_submission=False,
        reason="prior candidate lost",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=100.5,
            baseline_value=100.0,
            min_improvement=1.0,
        ),
        run_id="gate2-run-baseline",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-baseline/manifest.json",
            sha256="b" * 64,
        ),
        baseline_run_id="gate1-run-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate1-run-baseline/manifest.json",
            sha256="c" * 64,
        ),
    )

    with pytest.raises(HarnessIOError, match="baseline decision must be accepted"):
        promote_ledger_entry(_entry(), baseline=rejected_baseline)


def test_promote_ledger_entry_rejects_self_baseline_run() -> None:
    baseline = PromotionDecision.from_dict(
        {
            **_baseline().to_dict(),
            "run_id": "gate2-run-a",
            "manifest": ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="b" * 64,
            ).to_dict(),
        }
    )

    with pytest.raises(HarnessIOError, match="candidate run_id matches baseline run_id"):
        promote_ledger_entry(_entry(), baseline=baseline)


def test_promote_ledger_entry_requires_candidate_metric_value() -> None:
    with pytest.raises(HarnessIOError, match="requires entry.metric.value"):
        promote_ledger_entry(_entry(value=None), bootstrap=True)


def test_promote_ledger_entry_rejects_non_numeric_min_improvement() -> None:
    with pytest.raises(HarnessIOError, match="min_improvement must be a finite number"):
        promote_ledger_entry(
            _entry(),
            baseline=_baseline(),
            min_improvement="1.0",
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, True])
def test_promote_ledger_entry_rejects_non_finite_min_improvement(bad_value: float) -> None:
    with pytest.raises(HarnessIOError, match="min_improvement must be a finite number"):
        promote_ledger_entry(
            _entry(),
            baseline=_baseline(),
            min_improvement=bad_value,
        )
