"""Derive reward and failure reports from typed AIC evidence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.failure import (
    FailureKind,
    FailureLabel,
    FailureReport,
    FailureSeverity,
)
from aic_signal_harness.manifest import RunManifest
from aic_signal_harness.mcap_eval import McapEvalBundleReport
from aic_signal_harness.promotion import (
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
)
from aic_signal_harness.reward import RewardReport, RewardSignalKind, RewardTerm
from aic_signal_harness.schemas import LeakageClass, utc_now_iso
from aic_signal_harness.scoring import ScoreReport, TrialScore


_PARTIAL_INSERTION_TIER3_THRESHOLD = 38.0
_TRIAL_ID_RE = re.compile(r"^trial_([1-9][0-9]*)$")


@dataclass(frozen=True)
class RewardFailureReduction:
    """Pair of offline reward and failure reports for one run."""

    reward_report: RewardReport
    failure_report: FailureReport

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.reward_report, RewardReport):
            errors.append("reward_failure reduction reward_report must be a RewardReport")
        if not isinstance(self.failure_report, FailureReport):
            errors.append("reward_failure reduction failure_report must be a FailureReport")
        if (
            isinstance(self.reward_report, RewardReport)
            and isinstance(self.failure_report, FailureReport)
            and self.reward_report.run_id != self.failure_report.run_id
        ):
            errors.append("reward and failure reports must describe the same run_id")
        if errors:
            raise HarnessIOError("; ".join(errors))


def derive_reward_failure_reports(
    manifest: RunManifest | Mapping[str, Any],
    *,
    mcap_eval: McapEvalBundleReport | Mapping[str, Any] | None = None,
    promotion: PromotionDecision | Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> RewardFailureReduction:
    """Derive deterministic offline labels from already-typed harness evidence."""

    typed_manifest = manifest if isinstance(manifest, RunManifest) else RunManifest.from_dict(manifest)
    typed_mcap = (
        None
        if mcap_eval is None
        else mcap_eval
        if isinstance(mcap_eval, McapEvalBundleReport)
        else McapEvalBundleReport.from_dict(mcap_eval)
    )
    typed_promotion = (
        None
        if promotion is None
        else promotion
        if isinstance(promotion, PromotionDecision)
        else PromotionDecision.from_dict(promotion)
    )
    generated_at = utc_now_iso() if generated_at_utc is None else generated_at_utc

    reward_terms: list[RewardTerm] = []
    failure_labels: list[FailureLabel] = []

    if typed_manifest.score is None:
        failure_labels.append(
            FailureLabel(
                kind=FailureKind.inconclusive_missing_evidence,
                severity=FailureSeverity.blocker,
                summary="Run manifest has no official score report.",
                source="run_manifest",
                leakage_class=LeakageClass.post_hoc_label,
                evidence={"missing": "score"},
            )
        )
    else:
        reward_terms.extend(_official_score_terms(typed_manifest.score))
        failure_labels.extend(_score_failure_labels(typed_manifest.score))

    if typed_promotion is not None:
        _validate_promotion_matches_manifest(typed_promotion, typed_manifest)
        reward_terms.extend(_promotion_reward_terms(typed_promotion))
        failure_labels.extend(_promotion_failure_labels(typed_promotion))

    if typed_mcap is not None:
        reward_terms.extend(_mcap_reward_terms(typed_mcap))
        failure_labels.extend(_mcap_failure_labels(typed_mcap))

    return RewardFailureReduction(
        reward_report=RewardReport(
            run_id=typed_manifest.run_id,
            generated_at_utc=generated_at,
            terms=tuple(reward_terms),
            notes=(
                "Derived offline from typed harness evidence; not a live policy input.",
            ),
        ),
        failure_report=FailureReport(
            run_id=typed_manifest.run_id,
            generated_at_utc=generated_at,
            labels=tuple(failure_labels),
            notes=(
                "Derived offline from typed harness evidence; labels are post-hoc diagnostics.",
            ),
        ),
    )


def _official_score_terms(score: ScoreReport) -> tuple[RewardTerm, ...]:
    terms = [
        RewardTerm(
            name="official.score.total",
            value=score.total,
            signal_kind=RewardSignalKind.official_score,
            leakage_class=LeakageClass.privileged_eval_signal,
            source=score.source,
            provenance={"trial_count": score.trial_count},
        )
    ]
    for trial_id, trial_score in _sorted_trial_scores(score):
        terms.append(
            RewardTerm(
                name="official.trial.total",
                value=trial_score.total,
                signal_kind=RewardSignalKind.official_score,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=score.source,
                trial_id=trial_id,
            )
        )
        for tier_name in ("tier_1", "tier_2", "tier_3"):
            tier_value = getattr(trial_score, tier_name)
            if tier_value is None:
                continue
            terms.append(
                RewardTerm(
                    name=f"official.trial.{tier_name}",
                    value=tier_value,
                    signal_kind=RewardSignalKind.official_score,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    source=score.source,
                    trial_id=trial_id,
                )
            )
    return tuple(terms)


def _score_failure_labels(score: ScoreReport) -> tuple[FailureLabel, ...]:
    labels: list[FailureLabel] = []
    tier3_by_trial = {
        trial_id: trial_score.tier_3
        for trial_id, trial_score in _sorted_trial_scores(score)
        if trial_score.tier_3 is not None
    }
    if tier3_by_trial and max(tier3_by_trial.values()) < _PARTIAL_INSERTION_TIER3_THRESHOLD:
        labels.append(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary=(
                    "Tier 3 evidence is proximity-only; no partial or full insertion "
                    "credit reached the 38-point threshold."
                ),
                source=score.source,
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence={
                    "partial_insertion_tier3_threshold": _PARTIAL_INSERTION_TIER3_THRESHOLD,
                    "max_tier_3": max(tier3_by_trial.values()),
                    "tier_3_by_trial": tier3_by_trial,
                },
            )
        )
    return tuple(labels)


def _promotion_reward_terms(promotion: PromotionDecision) -> tuple[RewardTerm, ...]:
    if promotion.metric.baseline_value is None:
        return ()
    improvement = _promotion_improvement(promotion)
    margin_gap = improvement - promotion.metric.min_improvement
    return (
        RewardTerm(
            name="promotion.metric.improvement",
            value=improvement,
            signal_kind=RewardSignalKind.diagnostic,
            leakage_class=LeakageClass.privileged_eval_signal,
            source="promotion_decision",
            provenance={
                "run_id": promotion.run_id,
                "baseline_run_id": promotion.baseline_run_id,
                "metric_name": promotion.metric.name,
                "goal": promotion.metric.goal.value,
            },
        ),
        RewardTerm(
            name="promotion.metric.margin_gap",
            value=margin_gap,
            signal_kind=RewardSignalKind.diagnostic,
            leakage_class=LeakageClass.privileged_eval_signal,
            source="promotion_decision",
            provenance={
                "run_id": promotion.run_id,
                "baseline_run_id": promotion.baseline_run_id,
                "min_improvement": promotion.metric.min_improvement,
            },
        ),
    )


def _promotion_failure_labels(promotion: PromotionDecision) -> tuple[FailureLabel, ...]:
    if promotion.metric.baseline_value is None:
        return ()
    labels: list[FailureLabel] = []
    improvement = _promotion_improvement(promotion)
    margin_gap = improvement - promotion.metric.min_improvement
    evidence = {
        "decision": promotion.decision.value,
        "accepted": promotion.accepted,
        "metric_name": promotion.metric.name,
        "goal": promotion.metric.goal.value,
        "value": promotion.metric.value,
        "baseline_value": promotion.metric.baseline_value,
        "min_improvement": promotion.metric.min_improvement,
        "improvement": improvement,
        "margin_gap": margin_gap,
    }
    if improvement < 0.0:
        labels.append(
            FailureLabel(
                kind=FailureKind.score_regression,
                severity=FailureSeverity.blocker,
                summary="Candidate official score regressed versus the declared baseline.",
                source="promotion_decision",
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence=evidence,
            )
        )
    if (
        promotion.decision is PromotionDecisionKind.rejected
        and not promotion.accepted
        and margin_gap < 0.0
    ):
        labels.append(
            FailureLabel(
                kind=FailureKind.below_promotion_margin,
                severity=FailureSeverity.blocker,
                summary="Candidate did not clear the deterministic promotion margin.",
                source="promotion_decision",
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence=evidence,
            )
        )
    return tuple(labels)


def _mcap_reward_terms(bundle: McapEvalBundleReport) -> tuple[RewardTerm, ...]:
    terms: list[RewardTerm] = []
    total_contacts = sum(trial.off_limit_contact_count for trial in bundle.trials)
    terms.append(
        RewardTerm(
            name="diagnostic.off_limit_contact_count.total",
            value=float(total_contacts),
            signal_kind=RewardSignalKind.diagnostic,
            leakage_class=LeakageClass.privileged_eval_signal,
            source=bundle.source,
            provenance={"trial_count": len(bundle.trials)},
        )
    )
    for trial in bundle.trials:
        terms.append(
            RewardTerm(
                name="diagnostic.off_limit_contact_count",
                value=float(trial.off_limit_contact_count),
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=trial.source,
                trial_id=trial.trial_id,
            )
        )
        if trial.final_tcp_error is not None:
            terms.append(
                RewardTerm(
                    name="diagnostic.final_tcp_error_norm",
                    value=_vec_norm(trial.final_tcp_error),
                    signal_kind=RewardSignalKind.diagnostic,
                    leakage_class=LeakageClass.post_hoc_label,
                    source=trial.source,
                    trial_id=trial.trial_id,
                )
            )
    return tuple(terms)


def _mcap_failure_labels(bundle: McapEvalBundleReport) -> tuple[FailureLabel, ...]:
    labels: list[FailureLabel] = []
    for trial in bundle.trials:
        if trial.off_limit_contact_count <= 0:
            continue
        evidence: dict[str, Any] = {
            "off_limit_contact_count": trial.off_limit_contact_count,
        }
        if trial.first_off_limit_contact is not None:
            evidence["first_off_limit_contact"] = trial.first_off_limit_contact.to_dict()
        labels.append(
            FailureLabel(
                kind=FailureKind.off_limit_contact,
                severity=FailureSeverity.blocker,
                summary="Off-limit contact was observed in MCAP evaluation evidence.",
                source=trial.source,
                leakage_class=LeakageClass.privileged_eval_signal,
                trial_id=trial.trial_id,
                evidence=evidence,
            )
        )
        if (
            trial.first_off_limit_contact is not None
            and trial.first_off_limit_contact.recommended_stop_sec is not None
        ):
            labels.append(
                FailureLabel(
                    kind=FailureKind.contact_guard_stop_recommended,
                    severity=FailureSeverity.warning,
                    summary="MCAP contact evidence recommends an earlier safety stop.",
                    source=trial.source,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    trial_id=trial.trial_id,
                    evidence={
                        "recommended_stop_sec": trial.first_off_limit_contact.recommended_stop_sec,
                        "nearest_controller_elapsed_sec": (
                            trial.first_off_limit_contact.nearest_controller_elapsed_sec
                        ),
                    },
                )
            )
    return tuple(labels)


def _validate_promotion_matches_manifest(
    promotion: PromotionDecision,
    manifest: RunManifest,
) -> None:
    if promotion.run_id != manifest.run_id:
        raise HarnessIOError("promotion.run_id does not match manifest.run_id")
    if promotion.metric.name != "evaluation.score.total":
        return
    if manifest.score is None:
        raise HarnessIOError(
            "promotion metric evaluation.score.total requires manifest.score"
        )
    if not math.isclose(
        promotion.metric.value,
        manifest.score.total,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "promotion metric.value does not match manifest score.total"
        )


def _promotion_improvement(promotion: PromotionDecision) -> float:
    if promotion.metric.baseline_value is None:
        raise HarnessIOError("promotion improvement requires metric.baseline_value")
    if promotion.metric.goal is PromotionGoal.max:
        return promotion.metric.value - promotion.metric.baseline_value
    return promotion.metric.baseline_value - promotion.metric.value


def _sorted_trial_scores(score: ScoreReport) -> tuple[tuple[str, TrialScore], ...]:
    return tuple(
        sorted(
            score.trials.items(),
            key=lambda item: _trial_sort_key(item[0]),
        )
    )


def _trial_sort_key(trial_id: str) -> tuple[int, str]:
    match = _TRIAL_ID_RE.match(trial_id)
    if match:
        return (int(match.group(1)), trial_id)
    return (10**9, trial_id)


def _vec_norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))
