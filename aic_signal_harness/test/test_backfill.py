from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import pytest

from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
    PromotionMetric,
    RunStatus,
    backfill_legacy_replay_policy_eval,
    read_json,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "aic_lewm_policy" / "experiments" / "runs"


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _promotion_report(run_id: str) -> dict:
    return read_json(_run_dir(run_id) / "promotion_report.json")


def _ledger_entry(run_id: str) -> dict:
    return read_json(_run_dir(run_id) / "ledger_entry.json")


def _append_only_ledger_entry(run_id: str) -> dict:
    ledger_path = REPO_ROOT / "aic_lewm_policy" / "experiments" / "ledger.jsonl"
    matches = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["run_id"] == run_id
    ]
    assert len(matches) == 1
    return matches[0]


def _repo_with_mutated_ledger(
    tmp_path: Path,
    mutate_entry: Callable[[dict], None],
) -> Path:
    repo_root = tmp_path / "repo"
    experiments_root = repo_root / "aic_lewm_policy" / "experiments"
    experiments_root.mkdir(parents=True)
    (experiments_root / "runs").symlink_to(RUNS_ROOT, target_is_directory=True)
    for name in ("baselines", "gate_reviews"):
        source = REPO_ROOT / "aic_lewm_policy" / "experiments" / name
        if source.exists():
            (experiments_root / name).symlink_to(source, target_is_directory=True)
    for name in ("roadmap.json",):
        source = REPO_ROOT / "aic_lewm_policy" / "experiments" / name
        if source.exists():
            (experiments_root / name).symlink_to(source)
    (repo_root / "aic_lewm_policy" / "runtime_artifacts").symlink_to(
        REPO_ROOT / "aic_lewm_policy" / "runtime_artifacts",
        target_is_directory=True,
    )
    (repo_root / "artifacts").symlink_to(
        REPO_ROOT / "artifacts",
        target_is_directory=True,
    )
    ledger_path = REPO_ROOT / "aic_lewm_policy" / "experiments" / "ledger.jsonl"
    mutated_entries = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        mutate_entry(entry)
        mutated_entries.append(entry)
    (experiments_root / "ledger.jsonl").write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in mutated_entries),
        encoding="utf-8",
    )
    return repo_root


def _repo_with_isolated_run(
    tmp_path: Path,
    run_id: str,
) -> tuple[Path, Path]:
    repo_root = tmp_path / f"repo-{run_id}"
    experiments_root = repo_root / "aic_lewm_policy" / "experiments"
    runs_root = experiments_root / "runs"
    runs_root.mkdir(parents=True)
    for source in RUNS_ROOT.iterdir():
        if source.is_dir():
            (runs_root / source.name).symlink_to(source, target_is_directory=True)
    isolated_run_dir = runs_root / run_id
    isolated_run_dir.unlink()
    shutil.copytree(RUNS_ROOT / run_id, isolated_run_dir)
    for name in ("baselines", "gate_reviews"):
        source = REPO_ROOT / "aic_lewm_policy" / "experiments" / name
        if source.exists():
            (experiments_root / name).symlink_to(source, target_is_directory=True)
    for name in ("roadmap.json", "ledger.jsonl"):
        source = REPO_ROOT / "aic_lewm_policy" / "experiments" / name
        if source.exists():
            (experiments_root / name).symlink_to(source)
    (repo_root / "aic_lewm_policy" / "runtime_artifacts").symlink_to(
        REPO_ROOT / "aic_lewm_policy" / "runtime_artifacts",
        target_is_directory=True,
    )
    (repo_root / "artifacts").symlink_to(
        REPO_ROOT / "artifacts",
        target_is_directory=True,
    )
    return repo_root, isolated_run_dir


def _historical_baseline_seed(
    *,
    run_id: str,
    value: float,
    baseline_value: float,
    manifest_path: Path | None = None,
) -> PromotionDecision:
    baseline_manifest_path = _run_dir(run_id) / "manifest.json" if manifest_path is None else manifest_path
    if manifest_path is not None and not manifest_path.exists():
        manifest_path.write_text("{}\n", encoding="utf-8")
    return PromotionDecision(
        decided_at_utc="2026-04-22T07:23:04Z",
        decision=PromotionDecisionKind.accepted,
        accepted=True,
        eligible_for_submission=True,
        reason="seeded historical baseline before the backfilled chain",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=value,
            baseline_value=baseline_value,
            min_improvement=1.0,
        ),
        run_id=run_id,
        manifest=ArtifactRef(
            kind="run_manifest",
            path=str(baseline_manifest_path),
            sha256=sha256_file(baseline_manifest_path),
        ),
        baseline_run_id="gate0-prehistory-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path=str(baseline_manifest_path.with_name("gate0-prehistory-baseline.run_manifest.json")),
            sha256="e" * 64,
        ),
        notes=(
            "Seeded from the recorded Gate 1 baseline score because prehistory is not backfilled yet.",
        ),
    )


def _copy_run_evidence(source_run_dir: Path, isolated_run_dir: Path) -> None:
    isolated_run_dir.mkdir()
    for filename in [
        "manifest.json",
        "dataset_report.json",
        "score_report.json",
        "replay_check.json",
        "bag_analysis.json",
        "lateral_bias_analysis.json",
        "promotion_report.json",
    ]:
        source = source_run_dir / filename
        if source.exists():
            (isolated_run_dir / filename).write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )


def test_backfill_gate2_chain_matches_recorded_ledgers_and_promotion_reports(
    tmp_path: Path,
) -> None:
    gate1_seed = _historical_baseline_seed(
        run_id="replay-20260422T063538Z-gate1-sim2p5-full",
        value=106.5764522524607,
        baseline_value=97.819241,
    )
    replay6p5 = backfill_legacy_replay_policy_eval(
        _run_dir("gate2-20260422T135752Z-replay6p5"),
        manifest_output_path=tmp_path / "gate2-20260422T135752Z-replay6p5.run_manifest.json",
        baseline=gate1_seed,
        repo_root=REPO_ROOT,
    )
    scstop = backfill_legacy_replay_policy_eval(
        _run_dir("gate2-20260422T172605Z-scstop2p75"),
        manifest_output_path=tmp_path / "gate2-20260422T172605Z-scstop2p75.run_manifest.json",
        baseline=replay6p5.promotion_decision,
        repo_root=REPO_ROOT,
    )
    latbias = backfill_legacy_replay_policy_eval(
        _run_dir("gate2-20260422T230909Z-latbias"),
        manifest_output_path=tmp_path / "gate2-20260422T230909Z-latbias.run_manifest.json",
        baseline=scstop.promotion_decision,
        repo_root=REPO_ROOT,
    )

    assert replay6p5.promotion_decision is not None
    replay6p5_report = _promotion_report("gate2-20260422T135752Z-replay6p5")
    replay6p5_ledger = _ledger_entry("gate2-20260422T135752Z-replay6p5")
    assert replay6p5.promotion_decision.decision is PromotionDecisionKind.accepted
    assert replay6p5.promotion_decision.accepted is True
    assert replay6p5.promotion_decision.reason == replay6p5_report["reason"]
    assert replay6p5.promotion_decision.metric.baseline_value == pytest.approx(
        replay6p5_report["metric"]["baseline_value"]
    )
    assert replay6p5.ledger_entry.recorded_at_utc == replay6p5_ledger["recorded_at_utc"]
    assert Path(replay6p5.manifest_artifact.path).exists()
    assert sha256_file(replay6p5.manifest_artifact.path) == replay6p5.manifest_artifact.sha256
    assert replay6p5.ledger_entry.manifest == replay6p5.manifest_artifact
    assert replay6p5.promotion_decision.manifest == replay6p5.manifest_artifact

    scstop_report = _promotion_report("gate2-20260422T172605Z-scstop2p75")
    scstop_ledger = _ledger_entry("gate2-20260422T172605Z-scstop2p75")
    assert scstop.promotion_decision is not None
    assert scstop.promotion_decision.decision is PromotionDecisionKind.accepted
    assert scstop.promotion_decision.accepted is scstop_report["accepted"]
    assert scstop.promotion_decision.eligible_for_submission is scstop_report[
        "eligible_for_submission"
    ]
    assert scstop.promotion_decision.reason == scstop_report["reason"]
    assert scstop.promotion_decision.metric.value == pytest.approx(
        scstop_report["metric"]["value"]
    )
    assert scstop.promotion_decision.metric.baseline_value == pytest.approx(
        scstop_report["metric"]["baseline_value"]
    )
    assert scstop.promotion_decision.metric.min_improvement == pytest.approx(
        scstop_report["metric"]["min_improvement"]
    )
    assert tuple(scstop_report["notes"]) == scstop.promotion_decision.notes
    assert scstop.ledger_entry.recorded_at_utc == scstop_ledger["recorded_at_utc"]
    assert scstop.ledger_entry.metric.value == pytest.approx(
        scstop_ledger["metric"]["value"]
    )
    assert scstop.ledger_entry.dataset is not None
    assert scstop.ledger_entry.dataset.episode_count == scstop_ledger["dataset"][
        "episode_count"
    ]
    assert scstop.ledger_entry.dataset.step_count == scstop_ledger["dataset"]["step_count"]

    latbias_report = _promotion_report("gate2-20260422T230909Z-latbias")
    latbias_ledger = _ledger_entry("gate2-20260422T230909Z-latbias")
    assert latbias.promotion_decision is not None
    assert latbias.promotion_decision.decision is PromotionDecisionKind.rejected
    assert latbias.promotion_decision.accepted is latbias_report["accepted"]
    assert latbias.promotion_decision.eligible_for_submission is latbias_report[
        "eligible_for_submission"
    ]
    assert latbias.promotion_decision.reason == latbias_report["reason"]
    assert latbias.promotion_decision.metric.value == pytest.approx(
        latbias_report["metric"]["value"]
    )
    assert latbias.promotion_decision.metric.baseline_value == pytest.approx(
        latbias_report["metric"]["baseline_value"]
    )
    assert latbias.promotion_decision.metric.min_improvement == pytest.approx(
        latbias_report["metric"]["min_improvement"]
    )
    assert tuple(latbias_report["notes"]) == latbias.promotion_decision.notes
    assert latbias.ledger_entry.recorded_at_utc == latbias_ledger["recorded_at_utc"]
    assert latbias.ledger_entry.metric.value == pytest.approx(
        latbias_ledger["metric"]["value"]
    )
    assert latbias.ledger_entry.dataset is not None
    assert latbias.ledger_entry.dataset.episode_count == latbias_ledger["dataset"][
        "episode_count"
    ]
    assert latbias.ledger_entry.dataset.step_count == latbias_ledger["dataset"]["step_count"]


def test_backfill_gate2_latbias_manifest_preserves_canonical_and_legacy_artifacts(
    tmp_path: Path,
) -> None:
    baseline = _historical_baseline_seed(
        run_id="gate2-20260422T172605Z-scstop2p75",
        value=127.78018667287431,
        baseline_value=122.39001511101944,
    )
    backfill = backfill_legacy_replay_policy_eval(
        _run_dir("gate2-20260422T230909Z-latbias"),
        manifest_output_path=tmp_path / "gate2-20260422T230909Z-latbias.run_manifest.json",
        baseline=baseline,
        repo_root=REPO_ROOT,
    )

    assert backfill.manifest.status is RunStatus.completed
    assert backfill.manifest.score is not None
    assert backfill.manifest.score.total == pytest.approx(128.00872420948235)
    assert backfill.manifest.experiment_id == "gate2"

    artifacts_by_kind = {}
    for artifact in backfill.manifest.artifacts:
        artifacts_by_kind.setdefault(artifact.kind, []).append(artifact)

    assert len(artifacts_by_kind["scoring_yaml"]) == 1
    assert artifacts_by_kind["scoring_yaml"][0].path == str(
        REPO_ROOT
        / "artifacts"
        / "cloud-logs"
        / "gate2-20260422T230909Z-latbias"
        / "eval"
        / "scoring.yaml"
    )
    assert len(artifacts_by_kind["hdf5_dataset"]) == 1
    assert artifacts_by_kind["hdf5_dataset"][0].path == str(
        REPO_ROOT / "aic_lewm_policy" / "runtime_artifacts" / "aic_qualification_train.h5"
    )
    assert len(artifacts_by_kind["promotion_report"]) == 1
    assert artifacts_by_kind["promotion_report"][0].path == str(
        _run_dir("gate2-20260422T230909Z-latbias") / "promotion_report.json"
    )
    assert len(artifacts_by_kind["eval_bundle"]) == 1
    assert artifacts_by_kind["eval_bundle"][0].path == str(
        REPO_ROOT
        / "artifacts"
        / "cloud-logs"
        / "gate2-20260422T230909Z-latbias"
    )


def test_backfill_rejects_bootstrap_when_legacy_promotion_is_not_bootstrap(
    tmp_path: Path,
) -> None:
    with pytest.raises(HarnessIOError, match="bootstrap=True conflicts"):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T135752Z-replay6p5"),
            manifest_output_path=tmp_path / "gate2-20260422T135752Z-replay6p5.run_manifest.json",
            bootstrap=True,
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_missing_sidecar_artifacts(tmp_path: Path) -> None:
    source = _run_dir("gate2-20260422T230909Z-latbias") / "manifest.json"
    isolated_run_dir = tmp_path / "isolated_run"
    isolated_run_dir.mkdir()
    (isolated_run_dir / "manifest.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(HarnessIOError, match="legacy artifact path does not exist"):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "isolated.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_score_matching_but_wrong_baseline_identity(
    tmp_path: Path,
) -> None:
    wrong_baseline = _historical_baseline_seed(
        run_id="totally-wrong-baseline",
        value=122.39001511101944,
        baseline_value=97.819241,
        manifest_path=tmp_path / "wrong-baseline.run_manifest.json",
    )

    with pytest.raises(
        HarnessIOError,
        match="baseline run_id does not match recorded historical ledger",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "gate2-20260422T172605Z-scstop2p75.run_manifest.json",
            baseline=wrong_baseline,
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_bogus_baseline_manifest_provenance(tmp_path: Path) -> None:
    bogus_manifest = tmp_path / "bogus-replay6p5-baseline-manifest.json"
    bogus_manifest.write_text(
        json.dumps(
            {
                "evaluation": {"score": {"total": 122.39001511101944}},
                "run_id": "gate2-20260422T135752Z-replay6p5",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = _historical_baseline_seed(
        run_id="gate2-20260422T135752Z-replay6p5",
        value=122.39001511101944,
        baseline_value=106.5764522524607,
        manifest_path=bogus_manifest,
    )

    with pytest.raises(
        HarnessIOError,
        match="baseline manifest provenance does not match recorded historical ledger",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "gate2-20260422T172605Z-scstop2p75.run_manifest.json",
            baseline=baseline,
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_fake_neutral_baseline_manifest(tmp_path: Path) -> None:
    fake_neutral_manifest = tmp_path / "fake-neutral-replay6p5-baseline-manifest.json"
    fake_neutral_manifest.write_text(
        json.dumps(
            {
                "artifacts": [],
                "backend": {
                    "provenance": {
                        "legacy_manifest_path": str(
                            _run_dir("gate2-20260422T135752Z-replay6p5") / "manifest.json"
                        )
                    }
                },
                "created_at_utc": "2026-04-22T15:23:44Z",
                "run_id": "gate2-20260422T135752Z-replay6p5",
                "schema_version": 1,
                "score": {"total": 122.39001511101944},
                "status": "completed",
                "updated_at_utc": "2026-04-22T15:24:20Z",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = _historical_baseline_seed(
        run_id="gate2-20260422T135752Z-replay6p5",
        value=122.39001511101944,
        baseline_value=106.5764522524607,
        manifest_path=fake_neutral_manifest,
    )

    with pytest.raises(
        HarnessIOError,
        match="neutral manifest must be a valid RunManifest",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "gate2-20260422T172605Z-scstop2p75.run_manifest.json",
            baseline=baseline,
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_contradictory_historical_baseline_ledger_row(
    tmp_path: Path,
) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T135752Z-replay6p5":
            entry["promotion"]["decision"] = "rejected"
            entry["promotion"]["metric"]["value"] = 999.0

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(
        HarnessIOError,
        match="legacy baseline ledger does not match canonical ledger evidence",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "gate2-20260422T172605Z-scstop2p75.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T135752Z-replay6p5",
                value=122.39001511101944,
                baseline_value=106.5764522524607,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_undeclared_promotion_sidecar(tmp_path: Path) -> None:
    repo_root, isolated_run_dir = _repo_with_isolated_run(
        tmp_path,
        "gate2-20260422T230909Z-latbias",
    )
    manifest_payload = json.loads((isolated_run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["artifacts"] = [
        artifact
        for artifact in manifest_payload["artifacts"]
        if artifact["kind"] != "promotion_report"
    ]
    (isolated_run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy manifest does not match append-only ledger manifest evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "undeclared_promotion.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_legacy_promotion_schema_version_mismatch(
    tmp_path: Path,
) -> None:
    repo_root, isolated_run_dir = _repo_with_isolated_run(
        tmp_path,
        "gate2-20260422T230909Z-latbias",
    )
    manifest_payload = json.loads((isolated_run_dir / "manifest.json").read_text(encoding="utf-8"))
    promotion_payload = json.loads((isolated_run_dir / "promotion_report.json").read_text(encoding="utf-8"))
    promotion_payload["schema_version"] = 999
    (isolated_run_dir / "promotion_report.json").write_text(
        json.dumps(promotion_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    del manifest_payload
    with pytest.raises(
        HarnessIOError,
        match="legacy manifest.artifacts\\[6\\].path does not match canonical legacy artifact evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_promotion_schema.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_mismatched_legacy_promotion_binding(tmp_path: Path) -> None:
    repo_root, isolated_run_dir = _repo_with_isolated_run(
        tmp_path,
        "gate2-20260422T230909Z-latbias",
    )
    manifest_payload = json.loads((isolated_run_dir / "manifest.json").read_text(encoding="utf-8"))
    promotion_payload = json.loads((isolated_run_dir / "promotion_report.json").read_text(encoding="utf-8"))
    promotion_payload["run_id"] = "other-run"
    (isolated_run_dir / "promotion_report.json").write_text(
        json.dumps(promotion_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    del manifest_payload
    with pytest.raises(
        HarnessIOError,
        match="legacy manifest.artifacts\\[6\\].path does not match canonical legacy artifact evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_promotion.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_copied_manifest_metadata_rewrite(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_manifest_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    manifest_payload = json.loads((isolated_run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["description"] = "invented description"
    manifest_payload["evaluation"]["planner_mode"] = "invented-planner"
    (isolated_run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy manifest does not match append-only ledger manifest evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_manifest.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_rebased_repo_root_copied_manifest_rewrite(
    tmp_path: Path,
) -> None:
    repo_root, isolated_run_dir = _repo_with_isolated_run(
        tmp_path,
        "gate2-20260422T230909Z-latbias",
    )
    manifest_payload = json.loads((isolated_run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["description"] = "invented description"
    manifest_payload["evaluation"]["planner_mode"] = "invented-planner"
    (isolated_run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy manifest does not match append-only ledger manifest evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "rebased_mutated_manifest.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_append_only_ledger_manifest_path_contradiction(
    tmp_path: Path,
) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T230909Z-latbias":
            entry["manifest_path"] = "runs/gate2-20260422T172605Z-scstop2p75/manifest.json"

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(
        HarnessIOError,
        match="legacy append-only ledger does not match canonical ledger evidence",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T230909Z-latbias"),
            manifest_output_path=tmp_path / "mutated_append_manifest_path.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_append_only_ledger_promotion_manifest_path_contradiction(
    tmp_path: Path,
) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T230909Z-latbias":
            entry["promotion"]["manifest_path"] = (
                "aic_lewm_policy/experiments/runs/"
                "gate2-20260422T172605Z-scstop2p75/manifest.json"
            )

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(
        HarnessIOError,
        match="legacy append-only ledger does not match canonical ledger evidence",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T230909Z-latbias"),
            manifest_output_path=tmp_path / "mutated_append_promotion_path.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_baseline_ledger_promotion_manifest_path_contradiction(
    tmp_path: Path,
) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T135752Z-replay6p5":
            entry["promotion"]["manifest_path"] = (
                "aic_lewm_policy/experiments/runs/"
                "gate2-20260422T172605Z-scstop2p75/manifest.json"
            )

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(
        HarnessIOError,
        match="legacy baseline ledger does not match canonical ledger evidence",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "mutated_baseline_promotion_path.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T135752Z-replay6p5",
                value=122.39001511101944,
                baseline_value=106.5764522524607,
            ),
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    ("mutate_promotion", "match"),
    [
        (
            lambda promotion: promotion.__setitem__("report_path", "wrong_report.json"),
            "legacy baseline ledger does not match canonical ledger evidence",
        ),
        (
            lambda promotion: promotion.__setitem__("checkpoint_uri", "gs://wrong"),
            "legacy baseline ledger does not match canonical ledger evidence",
        ),
    ],
)
def test_backfill_rejects_baseline_ledger_promotion_provenance_contradictions(
    tmp_path: Path,
    mutate_promotion: Callable[[dict], None],
    match: str,
) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T135752Z-replay6p5":
            mutate_promotion(entry["promotion"])

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(HarnessIOError, match=match):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "mutated_baseline_promotion_provenance.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T135752Z-replay6p5",
                value=122.39001511101944,
                baseline_value=106.5764522524607,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_mutated_score_report_sidecar(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_score_report_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    (isolated_run_dir / "score_report.json").write_text(
        json.dumps({"parsed_at_utc": "2026-04-23T00:49:23Z"}, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy score_report does not match canonical score_report evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_score_report.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_rebound_official_scoring_artifact(tmp_path: Path) -> None:
    repo_root, isolated_run_dir = _repo_with_isolated_run(
        tmp_path,
        "gate2-20260422T230909Z-latbias",
    )
    artifacts_root = repo_root / "artifacts"
    artifacts_root.unlink()
    source_scoring = (
        REPO_ROOT
        / "artifacts"
        / "cloud-logs"
        / "gate2-20260422T230909Z-latbias"
        / "eval"
        / "scoring.yaml"
    )
    copied_scoring = (
        artifacts_root
        / "cloud-logs"
        / "gate2-20260422T230909Z-latbias"
        / "eval"
        / "scoring.yaml"
    )
    copied_scoring.parent.mkdir(parents=True)
    copied_scoring.write_text(
        source_scoring.read_text(encoding="utf-8") + "\n# semantic no-op mutation\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy score_report.path does not match scoring artifact",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "rebound_scoring.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_mutated_generic_sidecar_artifact(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_generic_sidecar_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    bag_payload = json.loads((isolated_run_dir / "bag_analysis.json").read_text(encoding="utf-8"))
    bag_payload["finding"] = "invented finding"
    (isolated_run_dir / "bag_analysis.json").write_text(
        json.dumps(bag_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="does not match canonical legacy artifact evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_generic_sidecar.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_synthetic_neutral_baseline_manifest(tmp_path: Path) -> None:
    gate1_seed = _historical_baseline_seed(
        run_id="replay-20260422T063538Z-gate1-sim2p5-full",
        value=106.5764522524607,
        baseline_value=97.819241,
    )
    replay6p5 = backfill_legacy_replay_policy_eval(
        _run_dir("gate2-20260422T135752Z-replay6p5"),
        manifest_output_path=tmp_path / "gate2-20260422T135752Z-replay6p5.run_manifest.json",
        baseline=gate1_seed,
        repo_root=REPO_ROOT,
    )
    assert replay6p5.promotion_decision is not None
    assert replay6p5.promotion_decision.manifest.path is not None
    synthetic_manifest_path = tmp_path / "synthetic_replay6p5.run_manifest.json"
    neutral_payload = json.loads(
        Path(replay6p5.promotion_decision.manifest.path).read_text(encoding="utf-8")
    )
    neutral_payload["notes"] = ["invented neutral manifest note"]
    neutral_payload["backend"]["description"] = "invented backend description"
    synthetic_manifest_path.write_text(
        json.dumps(neutral_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_payload = replay6p5.promotion_decision.to_dict()
    baseline_payload["manifest"]["path"] = str(synthetic_manifest_path)
    baseline_payload["manifest"]["sha256"] = sha256_file(synthetic_manifest_path)

    with pytest.raises(
        HarnessIOError,
        match="legacy promotion baseline neutral manifest does not match reconstructed historical ledger",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T172605Z-scstop2p75"),
            manifest_output_path=tmp_path / "scstop_with_synthetic_baseline.run_manifest.json",
            baseline=PromotionDecision.from_dict(baseline_payload),
            repo_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    ("mutate_promotion", "match"),
    [
        (
            lambda promotion: promotion.__setitem__("report_path", "wrong_report.json"),
            "legacy append-only ledger does not match canonical ledger evidence",
        ),
        (
            lambda promotion: promotion.__setitem__("checkpoint_uri", "gs://wrong"),
            "legacy append-only ledger does not match canonical ledger evidence",
        ),
    ],
)
def test_backfill_rejects_append_only_ledger_promotion_provenance_contradictions(
    tmp_path: Path,
    mutate_promotion: Callable[[dict], None],
    match: str,
) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T230909Z-latbias":
            mutate_promotion(entry["promotion"])

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(HarnessIOError, match=match):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T230909Z-latbias"),
            manifest_output_path=tmp_path / "mutated_append_promotion_provenance.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_malformed_append_only_ledger_timestamp(tmp_path: Path) -> None:
    def mutate(entry: dict) -> None:
        if entry["run_id"] == "gate2-20260422T230909Z-latbias":
            entry["recorded_at_utc"] = "not-a-timestamp"

    repo_root = _repo_with_mutated_ledger(tmp_path, mutate)

    with pytest.raises(
        HarnessIOError,
        match="legacy append-only ledger does not match canonical ledger evidence",
    ):
        backfill_legacy_replay_policy_eval(
            _run_dir("gate2-20260422T230909Z-latbias"),
            manifest_output_path=tmp_path / "malformed_append_timestamp.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_malformed_legacy_ledger_sidecar_timestamp(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "malformed_ledger_sidecar_timestamp_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["recorded_at_utc"] = "not-a-timestamp"
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy ledger.recorded_at_utc must be a UTC timestamp ending in Z",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "malformed_sidecar_timestamp.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_noncanonical_legacy_ledger_timestamp(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "noncanonical_ledger_timestamp_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["recorded_at_utc"] = "2026-4-23T00:53:28Z"
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy ledger.recorded_at_utc must be a canonical UTC timestamp",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "noncanonical_sidecar_timestamp.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_contradictory_legacy_ledger_sidecar(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_ledger_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["metric"]["value"] = 999.0
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="legacy ledger.metric.value does not match"):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_ledger.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_mutated_legacy_ledger_timestamp(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_ledger_timestamp_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["recorded_at_utc"] = "2099-01-01T00:00:00Z"
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="recorded_at_utc does not match append-only ledger"):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_ledger_timestamp.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_caller_supplied_ledger_chronology_rewrite(
    tmp_path: Path,
) -> None:
    run_id = "gate2-20260422T230909Z-latbias"
    repo_root, isolated_run_dir = _repo_with_isolated_run(tmp_path, run_id)

    ledger_path = repo_root / "aic_lewm_policy" / "experiments" / "ledger.jsonl"
    ledger_path.unlink()
    canonical_ledger_path = REPO_ROOT / "aic_lewm_policy" / "experiments" / "ledger.jsonl"
    mutated_entries = []
    for line in canonical_ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["run_id"] == run_id:
            entry["recorded_at_utc"] = "2099-01-01T00:00:00Z"
        mutated_entries.append(entry)
    ledger_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in mutated_entries),
        encoding="utf-8",
    )

    ledger_payload = json.loads((isolated_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["recorded_at_utc"] = "2099-01-01T00:00:00Z"
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HarnessIOError,
        match="legacy append-only ledger does not match canonical ledger evidence",
    ):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "caller_ledger_chronology.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=repo_root,
        )


def test_backfill_rejects_legacy_ledger_schema_version_mismatch(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_ledger_schema_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["schema_version"] = 999
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="legacy ledger.schema_version must be 1"):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_ledger_schema.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    ("field_name", "mutate_ledger", "match"),
    [
        (
            "kind",
            lambda payload: payload.__setitem__("kind", "not_replay_policy_eval"),
            "legacy ledger.kind does not match legacy manifest",
        ),
        (
            "git_commit",
            lambda payload: payload.__setitem__("git_commit", "bad-commit"),
            "legacy ledger.git_commit does not match legacy manifest",
        ),
        (
            "git_dirty",
            lambda payload: payload.__setitem__("git_dirty", False),
            "legacy ledger.git_dirty does not match legacy manifest",
        ),
        (
            "training",
            lambda payload: payload.__setitem__(
                "training",
                {"checkpoint_uri": "gs://wrong", "max_epochs": 1, "run_name": "wrong"},
            ),
            "legacy ledger.training does not match legacy manifest",
        ),
    ],
)
def test_backfill_rejects_legacy_ledger_metadata_contradictions(
    tmp_path: Path,
    field_name: str,
    mutate_ledger: Callable[[dict], None],
    match: str,
) -> None:
    del field_name
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_ledger_metadata_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    mutate_ledger(ledger_payload)
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match=match):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_ledger_metadata.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_uses_append_only_ledger_timestamp_when_sidecar_missing(
    tmp_path: Path,
) -> None:
    run_id = "gate2-20260422T230909Z-latbias"
    source_run_dir = _run_dir(run_id)
    isolated_run_dir = tmp_path / "missing_ledger_sidecar_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)

    backfill = backfill_legacy_replay_policy_eval(
        isolated_run_dir,
        manifest_output_path=tmp_path / "missing_ledger_sidecar.run_manifest.json",
        baseline=_historical_baseline_seed(
            run_id="gate2-20260422T172605Z-scstop2p75",
            value=127.78018667287431,
            baseline_value=122.39001511101944,
        ),
        repo_root=REPO_ROOT,
    )

    assert backfill.ledger_entry.recorded_at_utc == _append_only_ledger_entry(run_id)[
        "recorded_at_utc"
    ]


def test_backfill_rejects_contradictory_legacy_ledger_promotion(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_ledger_promotion_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["promotion"]["reason"] = "mutated historical ledger reason"
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="legacy ledger.promotion.reason does not match"):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_ledger_promotion.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )


def test_backfill_rejects_contradictory_legacy_ledger_dataset_identity(tmp_path: Path) -> None:
    source_run_dir = _run_dir("gate2-20260422T230909Z-latbias")
    isolated_run_dir = tmp_path / "mutated_ledger_dataset_run"
    _copy_run_evidence(source_run_dir, isolated_run_dir)
    ledger_payload = json.loads((source_run_dir / "ledger_entry.json").read_text(encoding="utf-8"))
    ledger_payload["dataset"]["uri"] = "aic_lewm_policy/experiments/ledger.jsonl"
    (isolated_run_dir / "ledger_entry.json").write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="legacy ledger.dataset.uri does not match"):
        backfill_legacy_replay_policy_eval(
            isolated_run_dir,
            manifest_output_path=tmp_path / "mutated_ledger_dataset.run_manifest.json",
            baseline=_historical_baseline_seed(
                run_id="gate2-20260422T172605Z-scstop2p75",
                value=127.78018667287431,
                baseline_value=122.39001511101944,
            ),
            repo_root=REPO_ROOT,
        )
