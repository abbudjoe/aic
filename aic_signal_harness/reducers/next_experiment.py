"""Generate deterministic next-experiment reports from typed harness evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.failure import FailureKind, FailureLabel, FailureReport
from aic_signal_harness.manifest import RunManifest
from aic_signal_harness.next_experiment import (
    NextExperimentAction,
    NextExperimentCandidate,
    NextExperimentCandidateKind,
    NextExperimentDecision,
    NextExperimentPriority,
    NextExperimentReport,
)
from aic_signal_harness.promotion import (
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
)
from aic_signal_harness.reward import RewardReport, RewardSignalKind
from aic_signal_harness.schemas import LeakageClass


_OFFICIAL_TOTAL_TERM = "official.score.total"
_PROMOTION_IMPROVEMENT_TERM = "promotion.metric.improvement"
_PROMOTION_MARGIN_TERM = "promotion.metric.margin_gap"
_NO_INSERTION = FailureKind.no_partial_or_full_insertion.value
_BELOW_MARGIN = FailureKind.below_promotion_margin.value
_OFF_LIMIT_CONTACT = FailureKind.off_limit_contact.value
_MISSING_EVIDENCE = FailureKind.inconclusive_missing_evidence.value
_SCORE_REGRESSION = FailureKind.score_regression.value
_PROMOTION_FAILURE_LABELS = frozenset({_BELOW_MARGIN, _SCORE_REGRESSION})
_RUNTIME_ENV_FIELD_PREFIX = "backend.provenance.legacy_runtime_env"


def derive_next_experiment_report(
    manifest: RunManifest | Mapping[str, Any],
    *,
    reward_report: RewardReport | Mapping[str, Any],
    failure_report: FailureReport | Mapping[str, Any],
    gate_id: str,
    promotion: PromotionDecision | Mapping[str, Any] | None = None,
    objective: str | None = None,
    generated_at_utc: str | None = None,
) -> NextExperimentReport:
    """Build ``next_experiment.json`` content from typed offline evidence."""

    typed_manifest = manifest if isinstance(manifest, RunManifest) else RunManifest.from_dict(manifest)
    typed_reward = (
        reward_report
        if isinstance(reward_report, RewardReport)
        else RewardReport.from_dict(reward_report)
    )
    typed_failure = (
        failure_report
        if isinstance(failure_report, FailureReport)
        else FailureReport.from_dict(failure_report)
    )
    typed_promotion = (
        None
        if promotion is None
        else promotion
        if isinstance(promotion, PromotionDecision)
        else PromotionDecision.from_dict(promotion)
    )
    _validate_inputs(
        typed_manifest,
        typed_reward,
        typed_failure,
        typed_promotion,
    )
    generated_at = typed_reward.generated_at_utc if generated_at_utc is None else generated_at_utc
    failure_kinds = {label.kind.value for label in typed_failure.labels}
    reward_terms = {term.name for term in typed_reward.terms}

    candidates = _candidate_plan(
        failure_kinds=failure_kinds,
        reward_terms=reward_terms,
        manifest=typed_manifest,
        promotion=typed_promotion,
    )
    decision = _report_decision(failure_kinds, typed_promotion)

    return NextExperimentReport(
        run_id=typed_manifest.run_id,
        gate_id=gate_id,
        generated_at_utc=generated_at,
        decision=decision,
        objective=objective,
        candidates=candidates,
        notes=(
            "Generated from typed reward/failure/promotion evidence; this report does not run experiments.",
        ),
    )


def _validate_inputs(
    manifest: RunManifest,
    reward_report: RewardReport,
    failure_report: FailureReport,
    promotion: PromotionDecision | None,
) -> None:
    if reward_report.run_id != manifest.run_id:
        raise HarnessIOError("reward_report.run_id does not match manifest.run_id")
    if failure_report.run_id != manifest.run_id:
        raise HarnessIOError("failure_report.run_id does not match manifest.run_id")
    failure_kinds = {label.kind.value for label in failure_report.labels}
    if not reward_report.terms and _MISSING_EVIDENCE not in failure_kinds:
        raise HarnessIOError("next experiment generation requires reward terms")
    _validate_official_score_evidence(manifest, reward_report, failure_kinds)
    if promotion is None:
        _reject_unbound_promotion_evidence(reward_report, failure_kinds)
        if not failure_report.labels:
            raise HarnessIOError("next experiment generation requires failure labels")
        return
    if promotion.run_id != manifest.run_id:
        raise HarnessIOError("promotion.run_id does not match manifest.run_id")
    if promotion.metric.name != "evaluation.score.total":
        raise HarnessIOError(
            "next experiment promotion decisions must use evaluation.score.total"
        )
    if promotion.metric.goal is not PromotionGoal.max:
        raise HarnessIOError(
            "next experiment promotion decisions for evaluation.score.total must use goal=max"
        )
    if manifest.score is None:
        raise HarnessIOError("promotion metric evaluation.score.total requires manifest.score")
    if not math.isclose(
        promotion.metric.value,
        manifest.score.total,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError("promotion metric.value does not match manifest score.total")
    _validate_promotion_reward_evidence(promotion, reward_report)
    _validate_promotion_failure_evidence(promotion, failure_report)
    if not failure_report.labels and promotion.decision is not PromotionDecisionKind.accepted:
        raise HarnessIOError("next experiment generation requires failure labels")


def _validate_official_score_evidence(
    manifest: RunManifest,
    reward_report: RewardReport,
    failure_kinds: set[str],
) -> None:
    official_total_terms = tuple(
        term for term in reward_report.terms if term.name == _OFFICIAL_TOTAL_TERM
    )
    if manifest.score is None:
        if official_total_terms:
            raise HarnessIOError("official.score.total reward term requires manifest.score")
        if _MISSING_EVIDENCE not in failure_kinds:
            raise HarnessIOError(
                "next experiment generation requires manifest.score or inconclusive_missing_evidence"
            )
        return
    if len(official_total_terms) != 1:
        raise HarnessIOError(
            "next experiment generation requires exactly one official.score.total reward term"
        )
    official_total = official_total_terms[0]
    if official_total.signal_kind is not RewardSignalKind.official_score:
        raise HarnessIOError(
            "official.score.total reward term must have signal_kind official_score"
        )
    if official_total.leakage_class is not LeakageClass.privileged_eval_signal:
        raise HarnessIOError(
            "official.score.total reward term must have leakage_class privileged_eval_signal"
        )
    if official_total.source != manifest.score.source:
        raise HarnessIOError(
            "official.score.total reward term source does not match manifest score.source"
        )
    if not math.isclose(
        official_total.value,
        manifest.score.total,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "official.score.total reward term value does not match manifest score.total"
        )


def _validate_promotion_reward_evidence(
    promotion: PromotionDecision,
    reward_report: RewardReport,
) -> None:
    if promotion.metric.baseline_value is None:
        _reject_present_reward_terms(
            reward_report,
            ("promotion.metric.improvement", "promotion.metric.margin_gap"),
        )
        return
    improvement = _promotion_improvement(promotion)
    expected_values = {
        "promotion.metric.improvement": improvement,
        "promotion.metric.margin_gap": improvement - promotion.metric.min_improvement,
    }
    for term_name, expected_value in expected_values.items():
        matches = tuple(term for term in reward_report.terms if term.name == term_name)
        if len(matches) != 1:
            raise HarnessIOError(
                f"next experiment generation requires exactly one {term_name} reward term"
            )
        term = matches[0]
        if term.signal_kind is not RewardSignalKind.diagnostic:
            raise HarnessIOError(f"{term_name} reward term must have signal_kind diagnostic")
        if term.leakage_class is not LeakageClass.privileged_eval_signal:
            raise HarnessIOError(
                f"{term_name} reward term must have leakage_class privileged_eval_signal"
            )
        if term.source != "promotion_decision":
            raise HarnessIOError(f"{term_name} reward term source must be promotion_decision")
        if not math.isclose(
            term.value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise HarnessIOError(f"{term_name} reward term value does not match promotion decision")


def _reject_unbound_promotion_evidence(
    reward_report: RewardReport,
    failure_kinds: set[str],
) -> None:
    promotion_terms = tuple(
        term.name for term in reward_report.terms if term.name.startswith("promotion.")
    )
    errors: list[str] = []
    if promotion_terms:
        errors.append(
            "promotion reward terms require a supplied PromotionDecision: "
            + ", ".join(sorted(set(promotion_terms)))
        )
    promotion_labels = tuple(sorted(failure_kinds & _PROMOTION_FAILURE_LABELS))
    if promotion_labels:
        errors.append(
            "promotion failure labels require a supplied PromotionDecision: "
            + ", ".join(promotion_labels)
        )
    if errors:
        raise HarnessIOError("; ".join(errors))


def _validate_promotion_failure_evidence(
    promotion: PromotionDecision,
    failure_report: FailureReport,
) -> None:
    present_labels = tuple(
        label for label in failure_report.labels if label.kind.value in _PROMOTION_FAILURE_LABELS
    )
    present_kind_values = tuple(label.kind.value for label in present_labels)
    present_kinds = {label.kind.value for label in present_labels}
    expected_kinds = _expected_promotion_failure_kinds(promotion)
    unexpected_kinds = sorted(present_kinds - expected_kinds)
    missing_kinds = sorted(expected_kinds - present_kinds)
    errors: list[str] = []
    duplicate_kinds = sorted(
        kind for kind in set(present_kind_values) if present_kind_values.count(kind) > 1
    )
    if duplicate_kinds:
        errors.append(
            "duplicate promotion-derived failure labels: "
            + ", ".join(duplicate_kinds)
        )
    if unexpected_kinds:
        errors.append(
            "promotion failure labels contradict promotion decision: "
            + ", ".join(unexpected_kinds)
        )
    if missing_kinds:
        errors.append(
            "failure report missing promotion-derived labels: "
            + ", ".join(missing_kinds)
        )
    for label in present_labels:
        errors.extend(_promotion_label_evidence_errors(label, promotion))
    if errors:
        raise HarnessIOError("; ".join(errors))


def _expected_promotion_failure_kinds(promotion: PromotionDecision) -> set[str]:
    if promotion.metric.baseline_value is None:
        return set()
    improvement = _promotion_improvement(promotion)
    margin_gap = improvement - promotion.metric.min_improvement
    expected: set[str] = set()
    if improvement < 0.0:
        expected.add(_SCORE_REGRESSION)
    if (
        promotion.decision is PromotionDecisionKind.rejected
        and not promotion.accepted
        and margin_gap < 0.0
    ):
        expected.add(_BELOW_MARGIN)
    return expected


def _promotion_label_evidence_errors(
    label: FailureLabel,
    promotion: PromotionDecision,
) -> list[str]:
    errors: list[str] = []
    label_name = label.kind.value
    if label.source != "promotion_decision":
        errors.append(f"{label_name} label source must be promotion_decision")
    if label.leakage_class is not LeakageClass.privileged_eval_signal:
        errors.append(
            f"{label_name} label leakage_class must be privileged_eval_signal"
        )
    baseline_value = promotion.metric.baseline_value
    if baseline_value is None:
        return errors
    improvement = _promotion_improvement(promotion)
    expected_evidence = {
        "decision": promotion.decision.value,
        "accepted": promotion.accepted,
        "metric_name": promotion.metric.name,
        "goal": promotion.metric.goal.value,
        "value": promotion.metric.value,
        "baseline_value": baseline_value,
        "min_improvement": promotion.metric.min_improvement,
        "improvement": improvement,
        "margin_gap": improvement - promotion.metric.min_improvement,
    }
    for key, expected_value in expected_evidence.items():
        if key not in label.evidence:
            errors.append(f"{label_name} label evidence missing {key}")
            continue
        actual_value = label.evidence[key]
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (int, float)) or not math.isclose(
                float(actual_value),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(f"{label_name} label evidence.{key} does not match promotion decision")
        elif actual_value != expected_value:
            errors.append(f"{label_name} label evidence.{key} does not match promotion decision")
    return errors


def _reject_present_reward_terms(
    reward_report: RewardReport,
    term_names: tuple[str, ...],
) -> None:
    present_names = sorted({term.name for term in reward_report.terms if term.name in term_names})
    if present_names:
        raise HarnessIOError(
            "promotion reward terms require promotion metric baseline_value: "
            + ", ".join(present_names)
        )


def _promotion_improvement(promotion: PromotionDecision) -> float:
    baseline_value = promotion.metric.baseline_value
    if baseline_value is None:
        raise HarnessIOError("promotion metric baseline_value is required")
    if promotion.metric.goal.value == "max":
        return promotion.metric.value - baseline_value
    return baseline_value - promotion.metric.value


def _candidate_plan(
    *,
    failure_kinds: set[str],
    reward_terms: set[str],
    manifest: RunManifest,
    promotion: PromotionDecision | None,
) -> tuple[NextExperimentCandidate, ...]:
    candidates: list[NextExperimentCandidate] = []
    fixed_servo_fields = _fixed_final_servo_manifest_fields(manifest)
    force_guard_fields = _baseline_force_guard_manifest_fields(manifest)

    if _MISSING_EVIDENCE in failure_kinds:
        candidates.append(
            NextExperimentCandidate(
                candidate_id="repair_missing_evidence",
                kind=NextExperimentCandidateKind.evidence_repair,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.block,
                title="Repair missing required evidence",
                rationale="The harness cannot choose a scientifically grounded next experiment until required score evidence is present.",
                evidence_labels=(_MISSING_EVIDENCE,),
                blocked_by_labels=(_MISSING_EVIDENCE,),
                acceptance_checks=(
                    "Produce a valid manifest, reward report, failure report, and official score evidence before launching another candidate.",
                ),
                runtime_notes="No policy runtime change is recommended while evidence is incomplete.",
                leakage_notes="Evidence repair is an offline harness action.",
            )
        )
        return tuple(candidates)

    if _NO_INSERTION in failure_kinds and _BELOW_MARGIN in failure_kinds and fixed_servo_fields:
        candidates.append(
            NextExperimentCandidate(
                candidate_id="reject_fixed_lateral_bias_push",
                kind=NextExperimentCandidateKind.reject_control_primitive,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.reject,
                title="Reject fixed lateral-bias final-servo push",
                rationale=(
                    "The candidate did not clear the promotion margin and did not earn "
                    "partial or full insertion credit, so another hand-chosen constant "
                    "final-servo vector is not a justified Gate 2 solution."
                ),
                evidence_labels=_present((_NO_INSERTION, _BELOW_MARGIN), failure_kinds),
                evidence_terms=_present((_OFFICIAL_TOTAL_TERM, _PROMOTION_MARGIN_TERM), reward_terms),
                evidence_manifest_fields=fixed_servo_fields,
                acceptance_checks=(
                    "Next candidate must replace fixed final-servo vectors with evidence-driven correction.",
                    "Do not launch another official run whose only change is a constant lateral-bias push.",
                ),
                runtime_notes="This rejects a policy primitive; it does not change the official AIC lifecycle.",
                leakage_notes="Decision uses privileged official score and post-hoc labels only for offline experiment selection.",
            )
        )

    if (_NO_INSERTION in failure_kinds or _BELOW_MARGIN in failure_kinds) and force_guard_fields:
        candidates.append(
            NextExperimentCandidate(
                candidate_id="keep_baseline_relative_force_guard",
                kind=NextExperimentCandidateKind.safety_contract,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.keep,
                title="Keep baseline-relative force guard",
                rationale=(
                    "Gate 2 should preserve the safety contract while changing the "
                    "final-centimeter behavior, because the rejected candidate remains "
                    "useful negative evidence rather than a reason to remove guarding."
                ),
                evidence_labels=_present((_NO_INSERTION, _BELOW_MARGIN, _OFF_LIMIT_CONTACT), failure_kinds),
                evidence_terms=_present((_PROMOTION_MARGIN_TERM,), reward_terms),
                evidence_manifest_fields=force_guard_fields,
                acceptance_checks=(
                    "Future final-centimeter primitive keeps a baseline-relative force/contact guard.",
                    "Offline gate checks that the candidate does not increase contact risk before official evaluation.",
                ),
                runtime_notes="Safety guard must remain inside the policy backend and emit legal commands through aic_model/aic_controller.",
                leakage_notes="Guard selection is decided offline from privileged eval evidence; live guard may only consume legal runtime observations.",
            )
        )
    if _NO_INSERTION in failure_kinds or _BELOW_MARGIN in failure_kinds:
        candidates.append(
            NextExperimentCandidate(
                candidate_id="build_adaptive_final_centimeter_primitive",
                kind=NextExperimentCandidateKind.controller_development,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.build,
                title="Build adaptive final-centimeter insertion primitive",
                rationale=(
                    "The failure state points at final alignment/insertion rather than "
                    "lifecycle or router discovery, so the next behavior should estimate "
                    "lateral error and insertion axis online from legal observations."
                ),
                evidence_labels=_present((_NO_INSERTION, _BELOW_MARGIN), failure_kinds),
                evidence_terms=_present((_OFFICIAL_TOTAL_TERM, _PROMOTION_MARGIN_TERM), reward_terms),
                acceptance_checks=(
                    "Primitive estimates lateral error and insertion axis from legal observations.",
                    "Primitive remains deterministic enough for the official runtime deadline.",
                    "Primitive is wrapped by baseline-relative force/contact guarding.",
                ),
                runtime_notes="Compiled policy behavior may run live only through the official aic_model policy boundary.",
                leakage_notes="Training or diagnosis may use privileged evidence, but live action cannot consume post-hoc labels or evaluator-only state.",
            )
        )
        candidates.append(
            NextExperimentCandidate(
                candidate_id="add_offline_final_centimeter_acceptance_gate",
                kind=NextExperimentCandidateKind.offline_acceptance_gate,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.require,
                title="Add offline acceptance gate before another official run",
                rationale=(
                    "The previous official run produced negative evidence, so the next "
                    "candidate should pass an offline check before spending another "
                    "evaluation attempt."
                ),
                evidence_labels=_present((_NO_INSERTION, _BELOW_MARGIN), failure_kinds),
                evidence_terms=_present((_PROMOTION_MARGIN_TERM,), reward_terms),
                acceptance_checks=(
                    "Offline gate predicts improved final distance on the hard SFP trial.",
                    "Offline gate rejects candidates that trip the force/contact guard window.",
                    "Gate output is recorded as a typed artifact before the next official run.",
                ),
                runtime_notes="The acceptance gate is offline analysis, not an inserted real-time controller.",
                leakage_notes="Offline gate may inspect privileged labels; live policy must not.",
            )
        )

    if _OFF_LIMIT_CONTACT in failure_kinds:
        candidates.append(
            NextExperimentCandidate(
                candidate_id="mine_contact_failure_window",
                kind=NextExperimentCandidateKind.failure_analysis,
                priority=NextExperimentPriority.medium,
                action=NextExperimentAction.analyze,
                title="Mine the contact failure window",
                rationale="Off-limit contact evidence should be reduced into a safer stop or correction window before another run.",
                evidence_labels=(_OFF_LIMIT_CONTACT,),
                acceptance_checks=(
                    "Identify first contact timing and nearest command context.",
                    "Add or tune an offline gate that rejects candidates with matching contact risk.",
                ),
                runtime_notes="Any resulting live stop rule must use legal controller or force observations only.",
                leakage_notes="MCAP contact evidence is privileged evaluation signal and stays offline.",
            )
        )

    if (
        not candidates
        and not failure_kinds
        and promotion is not None
        and promotion.decision is PromotionDecisionKind.accepted
    ):
        candidates.append(
            NextExperimentCandidate(
                candidate_id="prepare_promoted_candidate_review",
                kind=NextExperimentCandidateKind.offline_acceptance_gate,
                priority=NextExperimentPriority.medium,
                action=NextExperimentAction.require,
                title="Prepare promoted candidate review",
                rationale="Promotion evidence passed; prepare submission-readiness checks before changing behavior again.",
                evidence_terms=_present(
                    (
                        _OFFICIAL_TOTAL_TERM,
                        _PROMOTION_IMPROVEMENT_TERM,
                        _PROMOTION_MARGIN_TERM,
                    ),
                    reward_terms,
                ),
                acceptance_checks=(
                    "Run lifecycle, no-leakage, and artifact completeness checks.",
                    "Record whether this candidate should close the current gate.",
                ),
                runtime_notes="No runtime change is proposed until submission-readiness checks pass.",
                leakage_notes="Promotion review uses official score evidence offline.",
            )
        )

    if not candidates:
        candidates.append(
            NextExperimentCandidate(
                candidate_id="inspect_unclassified_failure_state",
                kind=NextExperimentCandidateKind.failure_analysis,
                priority=NextExperimentPriority.medium,
                action=NextExperimentAction.analyze,
                title="Inspect unclassified failure state",
                rationale="Reward and failure evidence exists, but no specialized Gate 2 recommendation matched it.",
                evidence_labels=tuple(sorted(failure_kinds)),
                evidence_terms=tuple(sorted(reward_terms)),
                acceptance_checks=(
                    "Classify the failure mode before launching another official run.",
                ),
                runtime_notes="No live policy change is recommended before the failure is classified.",
                leakage_notes="This is offline analysis over recorded evidence.",
            )
        )
    return tuple(candidates)


def _fixed_final_servo_manifest_fields(manifest: RunManifest) -> tuple[str, ...]:
    runtime_env = manifest.backend.provenance.get("legacy_runtime_env")
    if not isinstance(runtime_env, Mapping):
        return ()
    if not _truthy_env(runtime_env.get("AIC_LEWM_FINAL_SERVO_ENABLED")):
        return ()
    vector_fields = tuple(
        f"{_RUNTIME_ENV_FIELD_PREFIX}.{key}"
        for key in (
            "AIC_LEWM_SFP_FINAL_SERVO_LINEAR",
            "AIC_LEWM_SC_FINAL_SERVO_LINEAR",
        )
        if _looks_like_nonzero_vector(runtime_env.get(key))
    )
    if not vector_fields:
        return ()
    return (
        f"{_RUNTIME_ENV_FIELD_PREFIX}.AIC_LEWM_FINAL_SERVO_ENABLED",
        *vector_fields,
    )


def _baseline_force_guard_manifest_fields(manifest: RunManifest) -> tuple[str, ...]:
    runtime_env = manifest.backend.provenance.get("legacy_runtime_env")
    if not isinstance(runtime_env, Mapping):
        return ()
    if not _truthy_env(runtime_env.get("AIC_LEWM_FINAL_SERVO_ENABLED")):
        return ()
    if runtime_env.get("AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE") != "delta":
        return ()
    if not _looks_like_positive_number(runtime_env.get("AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N")):
        return ()
    return (
        f"{_RUNTIME_ENV_FIELD_PREFIX}.AIC_LEWM_FINAL_SERVO_ENABLED",
        f"{_RUNTIME_ENV_FIELD_PREFIX}.AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
        f"{_RUNTIME_ENV_FIELD_PREFIX}.AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N",
    )


def _truthy_env(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_positive_number(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return float(value) > 0.0
    except ValueError:
        return False


def _looks_like_nonzero_vector(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 3:
        return False
    try:
        return any(abs(float(part)) > 1e-12 for part in parts)
    except ValueError:
        return False


def _report_decision(
    failure_kinds: set[str],
    promotion: PromotionDecision | None,
) -> NextExperimentDecision:
    if _MISSING_EVIDENCE in failure_kinds:
        return NextExperimentDecision.block
    if (
        promotion is not None
        and promotion.decision is PromotionDecisionKind.accepted
        and not failure_kinds
    ):
        return NextExperimentDecision.promote
    return NextExperimentDecision.iterate


def _present(values: tuple[str, ...], available: set[str]) -> tuple[str, ...]:
    return tuple(value for value in values if value in available)
