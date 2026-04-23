from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_lewm_policy.experiment_harness import (  # noqa: E402
    HarnessError,
    append_ledger,
    parse_scoring_yaml,
    promote_candidate,
    validate_dataset,
    validate_manifest,
    write_json,
)


def _write_minimal_dataset(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([2], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
        handle.create_dataset("ep_idx", data=np.asarray([0, 0], dtype=np.int32))
        handle.create_dataset("episode_idx", data=np.asarray([0, 0], dtype=np.int32))
        handle.create_dataset("step_idx", data=np.asarray([0, 1], dtype=np.int32))
        handle.create_dataset("pixels", data=np.zeros((2, 4, 5, 3), dtype=np.uint8))
        handle.create_dataset("left_pixels", data=np.zeros((2, 4, 5, 3), dtype=np.uint8))
        handle.create_dataset("right_pixels", data=np.zeros((2, 4, 5, 3), dtype=np.uint8))
        handle.create_dataset("proprio", data=np.zeros((2, 32), dtype=np.float32))
        handle.create_dataset("state", data=np.zeros((2, 32), dtype=np.float32))
        handle.create_dataset("action", data=np.zeros((2, 6), dtype=np.float32))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key in ("task_id", "plug_type", "port_type", "target_module_name"):
            handle.create_dataset(key, data=np.asarray(["a", "a"], dtype=object), dtype=string_dtype)


def _manifest(run_id: str, score: float) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "smoke_train",
        "status": "completed",
        "created_at_utc": "2026-04-21T20:00:00Z",
        "code": {"commit": "abc", "dirty": True},
        "dataset": {
            "uri": "gs://bucket/datasets/data.h5",
            "sha256": "0" * 64,
            "episode_count": 1,
            "step_count": 2,
            "validation": {"ok": True},
        },
        "training": {
            "run_name": run_id,
            "max_epochs": 1,
            "output": {"checkpoint_uri": "gs://bucket/runs/checkpoint.ckpt"},
        },
        "evaluation": {"score": {"total": score}},
        "artifacts": [{"kind": "checkpoint", "uri": "gs://bucket/runs/checkpoint.ckpt"}],
        "promotion": {"decision": "not_evaluated"},
    }


def test_validate_dataset_accepts_expected_aic_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "rollout.h5"
    _write_minimal_dataset(dataset)

    report = validate_dataset(dataset)

    assert report["ok"] is True
    assert report["episode_count"] == 1
    assert report["step_count"] == 2
    assert report["datasets"]["action"]["shape"] == [2, 6]


def test_validate_dataset_reports_missing_required_key(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "broken.h5"
    with h5py.File(dataset, "w") as handle:
        handle.create_dataset("ep_len", data=[1])

    report = validate_dataset(dataset)

    assert report["ok"] is False
    assert any("missing required datasets" in error for error in report["errors"])


def test_parse_scoring_yaml_normalizes_total_and_trials(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 12.5
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 2.5
  tier_3:
    score: 9
""",
        encoding="utf-8",
    )

    report = parse_scoring_yaml(scoring)

    assert report["score"]["total"] == 12.5
    assert report["score"]["trial_count"] == 1
    assert report["score"]["trials"]["trial_1"]["total"] == 12.5


def test_completed_manifest_and_ledger_duplicate_gate(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"
    manifest_path = experiments_dir / "runs" / "run-a" / "manifest.json"
    ledger_path = experiments_dir / "ledger.jsonl"
    write_json(manifest_path, _manifest("run-a", 10.0))

    assert validate_manifest(_manifest("run-a", 10.0)) == []
    entry = append_ledger(
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        experiments_dir=experiments_dir,
    )

    assert entry["run_id"] == "run-a"
    with pytest.raises(HarnessError):
        append_ledger(
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            experiments_dir=experiments_dir,
        )


def test_promotion_gate_bootstraps_then_requires_improvement(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    first_path = tmp_path / "first.json"
    lower_path = tmp_path / "lower.json"
    better_path = tmp_path / "better.json"
    write_json(first_path, _manifest("first", 10.0))
    write_json(lower_path, _manifest("lower", 9.0))
    write_json(better_path, _manifest("better", 11.5))

    first = promote_candidate(
        manifest_path=first_path,
        baseline_path=baseline_path,
        bootstrap=True,
    )
    lower = promote_candidate(
        manifest_path=lower_path,
        baseline_path=baseline_path,
        min_improvement=1.0,
    )
    better = promote_candidate(
        manifest_path=better_path,
        baseline_path=baseline_path,
        min_improvement=1.0,
    )

    assert first["accepted"] is True
    assert lower["accepted"] is False
    assert lower["reason"] == "9.0 < 10.0 + 1.0"
    assert better["accepted"] is True
    assert better["reason"] == "11.5 >= 10.0 + 1.0"
