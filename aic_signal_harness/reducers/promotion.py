"""Reducer slice for typed promotion and baseline gate decisions."""

from __future__ import annotations

import math
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.ledger import LedgerEntry
from aic_signal_harness.manifest import RunStatus
from aic_signal_harness.promotion import (
    PromotionDecision,
    PromotionDecisionKind,
    PromotionMetric,
)
from aic_signal_harness.schemas import utc_now_iso


def promote_ledger_entry(
    entry: LedgerEntry,
    *,
    baseline: PromotionDecision | Mapping[str, Any] | None = None,
    min_improvement: float = 0.0,
    bootstrap: bool = False,
    eligible_for_submission: bool = False,
    notes: tuple[str, ...] | list[str] = (),
    decided_at_utc: str | None = None,
) -> PromotionDecision:
    """Compare one ledger entry against the current baseline and return a typed decision."""

    if not isinstance(entry, LedgerEntry):
        raise HarnessIOError("promote_ledger_entry requires a LedgerEntry")
    if entry.status is not RunStatus.completed:
        raise HarnessIOError("promote_ledger_entry requires entry.status=completed")
    if entry.metric.value is None:
        raise HarnessIOError("promote_ledger_entry requires entry.metric.value")
    if (
        isinstance(min_improvement, bool)
        or not isinstance(min_improvement, (int, float))
        or not math.isfinite(min_improvement)
    ):
        raise HarnessIOError(
            "promote_ledger_entry min_improvement must be a finite number"
        )
    min_improvement = float(min_improvement)
    if min_improvement < 0.0:
        raise HarnessIOError("promote_ledger_entry min_improvement must be nonnegative")

    typed_baseline = (
        None
        if baseline is None
        else baseline
        if isinstance(baseline, PromotionDecision)
        else PromotionDecision.from_dict(baseline)
    )

    if typed_baseline is None:
        if not bootstrap:
            raise HarnessIOError("baseline does not exist; bootstrap=True is required for the first one")
        return PromotionDecision(
            decided_at_utc=utc_now_iso() if decided_at_utc is None else decided_at_utc,
            decision=PromotionDecisionKind.bootstrap,
            accepted=True,
            eligible_for_submission=eligible_for_submission,
            reason="bootstrap baseline",
            metric=PromotionMetric(
                name=entry.metric.name,
                goal=entry.metric.goal.value,
                value=entry.metric.value,
                baseline_value=None,
                min_improvement=min_improvement,
            ),
            run_id=entry.run_id,
            manifest=entry.manifest,
            notes=tuple(notes),
        )

    if not typed_baseline.accepted:
        raise HarnessIOError("baseline decision must be accepted")
    if typed_baseline.metric.value is None:
        raise HarnessIOError("baseline decision must set metric.value")
    if typed_baseline.metric.name != entry.metric.name:
        raise HarnessIOError("baseline metric.name does not match candidate metric.name")
    if typed_baseline.metric.goal.value != entry.metric.goal.value:
        raise HarnessIOError("baseline metric.goal does not match candidate metric.goal")
    if typed_baseline.run_id == entry.run_id:
        raise HarnessIOError("candidate run_id matches baseline run_id")
    if typed_baseline.manifest.sha256 == entry.manifest.sha256:
        raise HarnessIOError("candidate manifest matches baseline manifest")

    goal = entry.metric.goal.value
    candidate = entry.metric.value
    baseline_value = typed_baseline.metric.value
    if goal == "max":
        accepted = candidate >= baseline_value + min_improvement
        comparator = ">=" if accepted else "<"
        reason = f"{candidate} {comparator} {baseline_value} + {min_improvement}"
    else:
        accepted = candidate <= baseline_value - min_improvement
        comparator = "<=" if accepted else ">"
        reason = f"{candidate} {comparator} {baseline_value} - {min_improvement}"

    return PromotionDecision(
        decided_at_utc=utc_now_iso() if decided_at_utc is None else decided_at_utc,
        decision=PromotionDecisionKind.accepted if accepted else PromotionDecisionKind.rejected,
        accepted=accepted,
        eligible_for_submission=eligible_for_submission if accepted else False,
        reason=reason,
        metric=PromotionMetric(
            name=entry.metric.name,
            goal=goal,
            value=candidate,
            baseline_value=baseline_value,
            min_improvement=min_improvement,
        ),
        run_id=entry.run_id,
        manifest=entry.manifest,
        baseline_run_id=typed_baseline.run_id,
        baseline_manifest=typed_baseline.manifest,
        notes=tuple(notes),
    )
