"""Post-eval finalization for live AIC runs.

This module runs after the official evaluator exits. It observes official
artifacts, reduces them into neutral harness contracts, and writes decision
evidence. It does not participate in the live policy control path.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError, sha256_file, write_json
from aic_signal_harness.ledger import append_ledger_entry, read_ledger_entries
from aic_signal_harness.manifest import RunManifest, RunStatus
from aic_signal_harness.promotion import (
    PromotionDecision,
    read_promotion_decision,
    write_baseline_decision,
    write_promotion_decision,
)
from aic_signal_harness.reducers import (
    PolicyTraceReduction,
    attach_scoring_yaml_reduction,
    build_ledger_entry,
    derive_next_experiment_report,
    derive_reward_failure_reports,
    promote_ledger_entry,
    reduce_policy_trace_jsonl,
    reduce_scoring_yaml,
)
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    LeakageClass,
    PolicyBackendSpec,
    RuntimeBoundaryProof,
    RuntimeRole,
    SimulatorKind,
    TrainingSourceKind,
    utc_now_iso,
)


_DEFAULT_LEGAL_OBSERVATION_CONTRACT = (
    "official aic_model task, observation, and action callbacks only"
)
_RUNTIME_ENV_KEYS = (
    "AIC_DOCKER_GPUS",
    "AIC_EVAL_IMAGE",
    "AIC_EVAL_TIMEOUT_SEC",
    "AIC_EVAL_USE_LOCAL_LAUNCH",
    "AIC_GZ_VERBOSITY_LEVEL",
    "AIC_LEWM_ANGULAR_VEL_LIMIT",
    "AIC_LEWM_COMMAND_FRAME",
    "AIC_LEWM_CONTROL_HZ",
    "AIC_LEWM_CHECKPOINT",
    "AIC_LEWM_DEVICE",
    "AIC_LEWM_FINAL_SERVO_DURATION_SEC",
    "AIC_LEWM_FINAL_SERVO_ENABLED",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N",
    "AIC_LEWM_LINEAR_VEL_LIMIT",
    "AIC_LEWM_MAX_RUNTIME_SEC",
    "AIC_LEWM_NUM_ACTION_CANDIDATES",
    "AIC_LEWM_PLANNER_MODE",
    "AIC_LEWM_PLANNING_HORIZON",
    "AIC_LEWM_GOAL_DATASET",
    "AIC_LEWM_REPLAY_ACTION_GAIN",
    "AIC_LEWM_REPLAY_DATASET",
    "AIC_LEWM_REPLAY_HZ",
    "AIC_LEWM_REPLAY_TIME_SCALE",
    "AIC_LEWM_REQUIRE_CHECKPOINT",
    "AIC_LEWM_SC_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SC_FINAL_SERVO_LINEAR",
    "AIC_LEWM_SC_REPLAY_STOP_SEC",
    "AIC_LEWM_SFP_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SFP_FINAL_SERVO_LINEAR",
)
_CONFIG_ENV_KEYS = (
    "AIC_LEWM_ANGULAR_VEL_LIMIT",
    "AIC_LEWM_COMMAND_FRAME",
    "AIC_LEWM_CONTROL_HZ",
    "AIC_LEWM_DEVICE",
    "AIC_LEWM_FINAL_SERVO_DURATION_SEC",
    "AIC_LEWM_FINAL_SERVO_ENABLED",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N",
    "AIC_LEWM_LINEAR_VEL_LIMIT",
    "AIC_LEWM_MAX_RUNTIME_SEC",
    "AIC_LEWM_NUM_ACTION_CANDIDATES",
    "AIC_LEWM_PLANNER_MODE",
    "AIC_LEWM_PLANNING_HORIZON",
    "AIC_LEWM_REPLAY_ACTION_GAIN",
    "AIC_LEWM_REPLAY_HZ",
    "AIC_LEWM_REPLAY_TIME_SCALE",
    "AIC_LEWM_SC_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SC_FINAL_SERVO_LINEAR",
    "AIC_LEWM_SC_REPLAY_STOP_SEC",
    "AIC_LEWM_SFP_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SFP_FINAL_SERVO_LINEAR",
)


@dataclass(frozen=True)
class LiveEvalFinalization:
    """Paths and typed outputs produced by one live eval finalization."""

    manifest: RunManifest
    manifest_artifact: ArtifactRef
    ledger_entry_path: Path
    manifest_path: Path
    score_report_path: Path
    scoring_artifact_path: Path
    reward_report_path: Path
    failure_report_path: Path
    summary_path: Path
    policy_trace_report_path: Path | None = None
    policy_trace_artifact_path: Path | None = None
    promotion_path: Path | None = None
    next_experiment_path: Path | None = None
    ledger_path: Path | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.manifest.run_id,
            "score_total": None if self.manifest.score is None else self.manifest.score.total,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_artifact.sha256,
            "score_report_path": str(self.score_report_path),
            "scoring_artifact_path": str(self.scoring_artifact_path),
            "policy_trace_report_path": _optional_path(self.policy_trace_report_path),
            "policy_trace_artifact_path": _optional_path(self.policy_trace_artifact_path),
            "ledger_entry_path": str(self.ledger_entry_path),
            "ledger_path": _optional_path(self.ledger_path),
            "promotion_path": _optional_path(self.promotion_path),
            "reward_report_path": str(self.reward_report_path),
            "failure_report_path": str(self.failure_report_path),
            "next_experiment_path": _optional_path(self.next_experiment_path),
        }


def finalize_live_eval_run(
    *,
    run_id: str,
    result_root: str | Path,
    harness_root: str | Path,
    scoring_yaml: str | Path,
    policy_trace: str | Path | None = None,
    ledger_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    update_baseline_path: str | Path | None = None,
    bootstrap_promotion: bool = False,
    min_improvement: float = 0.0,
    eligible_for_submission: bool = False,
    gate_id: str = "live_eval",
    experiment_id: str | None = None,
    hypothesis: str | None = None,
    model_image: str | None = None,
    model_image_id: str | None = None,
    backend_kind: str | BackendKind | None = None,
    planner_mode: str | None = None,
    runtime_env: Mapping[str, str] | None = None,
    overwrite: bool = False,
    append_ledger: bool = True,
    write_next_experiment: bool = True,
    generated_at_utc: str | None = None,
) -> LiveEvalFinalization:
    """Reduce official live eval artifacts into the neutral harness record."""

    run_id = _require_text(run_id, "run_id")
    result_root = _resolve_existing_dir(result_root, "result_root")
    harness_root = Path(harness_root).expanduser().resolve(strict=False)
    generated_at = utc_now_iso() if generated_at_utc is None else generated_at_utc
    env = _runtime_env_from_mapping(runtime_env)
    baseline = _load_baseline_decision(baseline_path=baseline_path)
    _validate_declared_policy_trace(policy_trace)

    scoring_reduction = reduce_scoring_yaml(
        scoring_yaml,
        parsed_at_utc=generated_at,
        provenance={"producer": "aic_signal_harness.live_eval", "run_id": run_id},
    )
    policy_trace_reduction = _reduce_optional_policy_trace(
        policy_trace=policy_trace,
        run_id=run_id,
    )
    backend = _backend_spec(
        run_id=run_id,
        model_image=model_image,
        model_image_id=model_image_id,
        backend_kind=backend_kind,
        planner_mode=planner_mode,
        runtime_env=env,
    )
    score_report_path = harness_root / "score_report.json"
    scoring_artifact_path = harness_root / "scoring_yaml_artifact.json"
    policy_trace_report_path = (
        None if policy_trace_reduction is None else harness_root / "policy_trace_report.json"
    )
    policy_trace_artifact_path = (
        None if policy_trace_reduction is None else harness_root / "policy_trace_artifact.json"
    )
    manifest_path = harness_root / "run_manifest.json"
    promotion_path = (
        harness_root / "promotion_report.json"
        if baseline is not None or bootstrap_promotion
        else None
    )
    reward_report_path = harness_root / "reward_report.json"
    failure_report_path = harness_root / "failure_report.json"
    next_experiment_path = (
        harness_root / "next_experiment.json" if write_next_experiment else None
    )
    ledger_entry_path = harness_root / "ledger_entry.json"
    summary_path = harness_root / "live_eval_summary.json"
    typed_ledger_path = None if ledger_path is None else Path(ledger_path).expanduser()
    _preflight_live_eval_writes(
        run_id=run_id,
        output_paths=(
            score_report_path,
            scoring_artifact_path,
            policy_trace_report_path,
            policy_trace_artifact_path,
            manifest_path,
            promotion_path,
            reward_report_path,
            failure_report_path,
            next_experiment_path,
            ledger_entry_path,
            summary_path,
        ),
        ledger_path=typed_ledger_path,
        append_ledger=append_ledger,
        overwrite=overwrite,
    )

    harness_root.mkdir(parents=True, exist_ok=True)
    write_json(score_report_path, scoring_reduction.score.to_dict(), overwrite=overwrite)
    write_json(scoring_artifact_path, scoring_reduction.artifact.to_dict(), overwrite=overwrite)
    if policy_trace_reduction is not None:
        assert policy_trace_report_path is not None
        assert policy_trace_artifact_path is not None
        write_json(
            policy_trace_report_path,
            policy_trace_reduction.report.to_dict(),
            overwrite=overwrite,
        )
        write_json(
            policy_trace_artifact_path,
            policy_trace_reduction.artifact.to_dict(),
            overwrite=overwrite,
        )

    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.completed,
        backend=backend,
        created_at_utc=generated_at,
        updated_at_utc=generated_at,
        artifacts=(
            (scoring_reduction.artifact,)
            + (
                ()
                if policy_trace_reduction is None
                else (policy_trace_reduction.artifact,)
            )
        ),
        score=scoring_reduction.score,
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        notes=(
            "Generated after official AIC evaluator completion; harness outputs are post-hoc evidence.",
            f"Official eval result root: {result_root}",
        ),
    )
    manifest = attach_scoring_yaml_reduction(
        manifest,
        scoring_reduction,
        updated_at_utc=generated_at,
    )
    write_json(manifest_path, manifest.to_dict(), overwrite=overwrite)
    manifest_artifact = ArtifactRef(
        kind="run_manifest",
        path=str(manifest_path),
        sha256=sha256_file(manifest_path),
        provenance={"producer": "aic_signal_harness.live_eval", "run_id": run_id},
    )

    pending_ledger_entry = build_ledger_entry(
        manifest,
        manifest_artifact=manifest_artifact,
        recorded_at_utc=generated_at,
    )
    promotion = _promotion_decision(
        pending_ledger_entry,
        baseline=baseline,
        bootstrap_promotion=bootstrap_promotion,
        min_improvement=min_improvement,
        eligible_for_submission=eligible_for_submission,
        generated_at_utc=generated_at,
    )
    accepted_baseline_update = None
    if promotion is not None:
        if promotion.accepted and update_baseline_path is not None:
            accepted_baseline_update = Path(update_baseline_path).expanduser()

    reward_failure = derive_reward_failure_reports(
        manifest,
        promotion=promotion,
        generated_at_utc=generated_at,
    )
    write_json(
        reward_report_path,
        reward_failure.reward_report.to_dict(),
        overwrite=overwrite,
    )
    write_json(
        failure_report_path,
        reward_failure.failure_report.to_dict(),
        overwrite=overwrite,
    )

    if write_next_experiment:
        assert next_experiment_path is not None
        next_experiment = derive_next_experiment_report(
            manifest,
            reward_report=reward_failure.reward_report,
            failure_report=reward_failure.failure_report,
            gate_id=gate_id,
            promotion=promotion,
            objective="Improve official AIC qualification score without violating runtime boundaries.",
            generated_at_utc=generated_at,
        )
        write_json(next_experiment_path, next_experiment.to_dict(), overwrite=overwrite)

    ledger_entry = build_ledger_entry(
        manifest,
        manifest_artifact=manifest_artifact,
        promotion={} if promotion is None else promotion.to_dict(),
        recorded_at_utc=generated_at,
    )
    write_json(ledger_entry_path, ledger_entry.to_dict(), overwrite=overwrite)

    if append_ledger and typed_ledger_path is not None:
        append_ledger_entry(typed_ledger_path, ledger_entry, allow_duplicate=False)

    finalization = LiveEvalFinalization(
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        manifest_path=manifest_path,
        score_report_path=score_report_path,
        scoring_artifact_path=scoring_artifact_path,
        policy_trace_report_path=policy_trace_report_path,
        policy_trace_artifact_path=policy_trace_artifact_path,
        ledger_entry_path=ledger_entry_path,
        ledger_path=typed_ledger_path,
        promotion_path=promotion_path,
        reward_report_path=reward_report_path,
        failure_report_path=failure_report_path,
        next_experiment_path=next_experiment_path,
        summary_path=summary_path,
    )
    write_json(finalization.summary_path, finalization.to_summary(), overwrite=overwrite)
    if promotion is not None and promotion_path is not None:
        write_promotion_decision(str(promotion_path), promotion, overwrite=overwrite)
    if accepted_baseline_update is not None and promotion is not None:
        write_baseline_decision(
            str(accepted_baseline_update),
            promotion,
            overwrite=True,
        )
    return finalization


def _reduce_optional_policy_trace(
    *,
    policy_trace: str | Path | None,
    run_id: str,
) -> PolicyTraceReduction | None:
    if policy_trace is None:
        return None
    trace_path = Path(policy_trace).expanduser()
    reduction = reduce_policy_trace_jsonl(
        trace_path,
        provenance={"producer": "aic_signal_harness.live_eval", "run_id": run_id},
    )
    if reduction.report.run_id != run_id:
        raise HarnessIOError(
            f"policy trace run_id {reduction.report.run_id!r} "
            f"does not match finalizer run_id {run_id!r}"
        )
    return reduction


def _preflight_live_eval_writes(
    *,
    run_id: str,
    output_paths: tuple[Path | None, ...],
    ledger_path: Path | None,
    append_ledger: bool,
    overwrite: bool,
) -> None:
    if not overwrite:
        existing_paths = tuple(path for path in output_paths if path is not None and path.exists())
        if existing_paths:
            raise HarnessIOError(
                "live eval output already exists; pass overwrite=True to replace it: "
                + ", ".join(str(path) for path in existing_paths)
            )
    if append_ledger and ledger_path is not None:
        if any(entry.run_id == run_id for entry in read_ledger_entries(ledger_path)):
            raise HarnessIOError(f"ledger already contains run_id={run_id}")


def _load_baseline_decision(
    *,
    baseline_path: str | Path | None,
) -> PromotionDecision | None:
    if baseline_path is None:
        return None
    path = Path(baseline_path).expanduser()
    if path.exists():
        return read_promotion_decision(str(path))
    raise HarnessIOError(f"baseline does not exist: {path}")


def _validate_declared_policy_trace(policy_trace: str | Path | None) -> None:
    if policy_trace is None:
        return
    trace_path = Path(policy_trace).expanduser()
    if not trace_path.is_file():
        raise HarnessIOError(f"policy trace does not exist: {trace_path}")
    if trace_path.stat().st_size <= 0:
        raise HarnessIOError(f"policy trace is empty: {trace_path}")


def _promotion_decision(
    entry,
    *,
    baseline: PromotionDecision | None,
    bootstrap_promotion: bool,
    min_improvement: float,
    eligible_for_submission: bool,
    generated_at_utc: str,
) -> PromotionDecision | None:
    if baseline is None and not bootstrap_promotion:
        return None
    return promote_ledger_entry(
        entry,
        baseline=baseline,
        bootstrap=bootstrap_promotion and baseline is None,
        min_improvement=min_improvement,
        eligible_for_submission=eligible_for_submission,
        notes=("Generated by live eval harness finalizer.",),
        decided_at_utc=generated_at_utc,
    )


def _backend_spec(
    *,
    run_id: str,
    model_image: str | None,
    model_image_id: str | None,
    backend_kind: str | BackendKind | None,
    planner_mode: str | None,
    runtime_env: Mapping[str, str],
) -> PolicyBackendSpec:
    mode = (planner_mode or runtime_env.get("AIC_LEWM_PLANNER_MODE") or "lewm_mpc").strip()
    kind = (
        BackendKind.parse(backend_kind)
        if backend_kind is not None
        else BackendKind.replay_servo
        if mode == "replay"
        else BackendKind.lewm_world_model
    )
    image = _optional_text(model_image, "model_image") or "aic-lewm-learned:latest"
    image_sha256 = _docker_image_sha256(model_image_id) if kind is BackendKind.lewm_world_model else None
    policy_artifact = ArtifactRef(
        kind="docker_image",
        uri=_docker_uri(image),
        sha256=image_sha256,
        provenance={"run_id": run_id},
    )
    runtime_boundary = RuntimeBoundaryProof(
        deterministic=True,
        uses_online_language_model_control=False,
        legal_observation_contract=_DEFAULT_LEGAL_OBSERVATION_CONTRACT,
        policy_artifact=policy_artifact if kind is BackendKind.lewm_world_model else None,
        notes="Harness finalizer records this after the official live eval; no online LLM control.",
    )
    return PolicyBackendSpec(
        backend_kind=kind,
        name=f"aic-lewm-{mode}",
        runtime_role=RuntimeRole.live_policy,
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.gazebo,),
        runtime_allowed=True,
        leakage_class=LeakageClass.legal_policy_input,
        runtime_boundary=runtime_boundary,
        description="AIC LEWM policy evaluated through the official AIC runtime.",
        config={
            "planner_mode": mode,
            "runtime_env": {
                key: runtime_env[key]
                for key in _CONFIG_ENV_KEYS
                if key in runtime_env
            },
        },
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "model_image": image,
            "model_image_id": _optional_text(model_image_id, "model_image_id"),
            "runtime_env": dict(runtime_env),
        },
    )


def _docker_uri(image: str) -> str:
    return "docker://" + image.lstrip("/")


def _docker_image_sha256(image_id: str | None) -> str:
    value = _require_text(image_id, "model_image_id")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise HarnessIOError("model_image_id must be a sha256 Docker image id")
    return digest.lower()


def _runtime_env_from_mapping(runtime_env: Mapping[str, str] | None) -> dict[str, str]:
    source = os.environ if runtime_env is None else runtime_env
    return {
        key: str(source[key])
        for key in _RUNTIME_ENV_KEYS
        if key in source and str(source[key]).strip()
    }


def _resolve_existing_dir(path: str | Path, field_name: str) -> Path:
    directory = Path(path).expanduser().resolve(strict=False)
    if not directory.is_dir():
        raise HarnessIOError(f"{field_name} must be an existing directory: {directory}")
    return directory


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _optional_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _cmd_finalize(args: argparse.Namespace) -> int:
    finalization = finalize_live_eval_run(
        run_id=args.run_id,
        result_root=args.result_root,
        harness_root=args.harness_root,
        scoring_yaml=args.scoring_yaml,
        policy_trace=args.policy_trace,
        ledger_path=args.ledger,
        baseline_path=args.baseline,
        update_baseline_path=args.update_baseline,
        bootstrap_promotion=args.bootstrap_promotion,
        min_improvement=args.min_improvement,
        eligible_for_submission=args.eligible_for_submission,
        gate_id=args.gate_id,
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        model_image=args.model_image,
        model_image_id=args.model_image_id,
        backend_kind=args.backend_kind,
        planner_mode=args.planner_mode,
        overwrite=args.overwrite,
        append_ledger=not args.no_append_ledger,
        write_next_experiment=not args.no_next_experiment,
    )
    print(f"AIC_HARNESS_MANIFEST_PATH={finalization.manifest_path}")
    print(f"AIC_HARNESS_LEDGER_ENTRY_PATH={finalization.ledger_entry_path}")
    if finalization.ledger_path is not None:
        print(f"AIC_HARNESS_LEDGER_PATH={finalization.ledger_path}")
    if finalization.promotion_path is not None:
        print(f"AIC_HARNESS_PROMOTION_PATH={finalization.promotion_path}")
    if finalization.next_experiment_path is not None:
        print(f"AIC_HARNESS_NEXT_EXPERIMENT_PATH={finalization.next_experiment_path}")
    print(f"AIC_HARNESS_SUMMARY_PATH={finalization.summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize a live AIC eval into harness evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize", help="Reduce one completed live eval")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--result-root", required=True)
    finalize.add_argument("--harness-root", required=True)
    finalize.add_argument("--scoring-yaml", required=True)
    finalize.add_argument("--policy-trace")
    finalize.add_argument("--ledger")
    finalize.add_argument("--baseline")
    finalize.add_argument("--update-baseline")
    finalize.add_argument("--bootstrap-promotion", action="store_true")
    finalize.add_argument("--min-improvement", type=float, default=0.0)
    finalize.add_argument("--eligible-for-submission", action="store_true")
    finalize.add_argument("--gate-id", default="live_eval")
    finalize.add_argument("--experiment-id")
    finalize.add_argument("--hypothesis")
    finalize.add_argument("--model-image")
    finalize.add_argument("--model-image-id")
    finalize.add_argument("--backend-kind")
    finalize.add_argument("--planner-mode")
    finalize.add_argument("--overwrite", action="store_true", default=_bool_env("AIC_HARNESS_OVERWRITE"))
    finalize.add_argument("--no-append-ledger", action="store_true")
    finalize.add_argument("--no-next-experiment", action="store_true")
    finalize.set_defaults(func=_cmd_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HarnessIOError as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
