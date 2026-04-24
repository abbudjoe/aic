import json
from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    HarnessIOError,
    LedgerDatasetSummary,
    LedgerEntry,
    LedgerMetric,
    LedgerMetricGoal,
    RunStatus,
    append_ledger_entry,
    read_ledger_entries,
)


def _entry(run_id: str = "gate2-run-a") -> LedgerEntry:
    return LedgerEntry(
        recorded_at_utc="2026-04-23T12:00:00Z",
        run_id=run_id,
        status=RunStatus.completed,
        backend_kind=BackendKind.replay_servo,
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-a/manifest.json",
            sha256="a" * 64,
        ),
        metric=LedgerMetric(
            name="evaluation.score.total",
            goal=LedgerMetricGoal.max,
            value=127.78,
        ),
        dataset=LedgerDatasetSummary(
            artifact=ArtifactRef(
                kind="hdf5_dataset",
                path="/tmp/data/demo.h5",
                sha256="b" * 64,
            ),
            episode_count=3,
            step_count=207,
        ),
        experiment_id="gate2",
        notes=("kept as current replay baseline",),
        promotion={"decision": "accepted", "eligible_for_submission": True},
    )


def test_ledger_entry_round_trips_strictly() -> None:
    entry = _entry()

    assert LedgerEntry.from_dict(entry.to_dict()) == entry


def test_read_ledger_entries_rejects_malformed_jsonl(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(_entry().to_dict(), sort_keys=True) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(HarnessIOError, match="invalid JSONL"):
        read_ledger_entries(ledger_path)


def test_read_ledger_entries_rejects_invalid_entry_payload(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text('{"ok": true}\n', encoding="utf-8")

    with pytest.raises(HarnessIOError, match="invalid ledger entry"):
        read_ledger_entries(ledger_path)


def test_append_and_read_ledger_entries_round_trip(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    entry = _entry()

    append_ledger_entry(ledger_path, entry)

    assert read_ledger_entries(ledger_path) == (entry,)


def test_append_ledger_entry_rejects_duplicate_run_id(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    entry = _entry()
    append_ledger_entry(ledger_path, entry)

    with pytest.raises(HarnessIOError, match="ledger already contains run_id=gate2-run-a"):
        append_ledger_entry(ledger_path, entry)


def test_append_ledger_entry_allows_duplicate_when_requested(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    entry = _entry()
    append_ledger_entry(ledger_path, entry)
    append_ledger_entry(ledger_path, entry, allow_duplicate=True)

    assert len(read_ledger_entries(ledger_path)) == 2
