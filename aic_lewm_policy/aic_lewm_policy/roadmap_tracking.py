"""Validation helpers for the AIC LEWM roadmap and gate reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_GATE_FIELDS = (
    "id",
    "name",
    "status",
    "objective",
    "entry_criteria",
    "exit_criteria",
    "metric",
    "next_gate",
)
VALID_GATE_STATUS = {"planned", "in_progress", "completed", "blocked", "skipped"}
VALID_DECISIONS = {"proceed", "iterate", "block", "skip"}


class RoadmapError(RuntimeError):
    """Raised when roadmap or gate-review data breaks the tracking contract."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_roadmap(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    roadmap = load_json(path)
    errors: list[str] = []
    gates = roadmap.get("gates")

    if roadmap.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(roadmap.get("objective"), str) or not roadmap["objective"].strip():
        errors.append("objective is required")
    if not isinstance(roadmap.get("decision_policy"), dict):
        errors.append("decision_policy object is required")
    if not isinstance(gates, list) or not gates:
        errors.append("gates must be a non-empty list")
        gates = []

    gate_ids: set[str] = set()
    completed_missing_reviews: list[str] = []
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(f"gate[{index}] must be an object")
            continue
        for field in REQUIRED_GATE_FIELDS:
            if field not in gate:
                errors.append(f"gate[{index}] missing {field}")
        gate_id = str(gate.get("id", ""))
        if not gate_id:
            errors.append(f"gate[{index}] id is required")
        elif gate_id in gate_ids:
            errors.append(f"duplicate gate id {gate_id}")
        gate_ids.add(gate_id)

        status = gate.get("status")
        if status not in VALID_GATE_STATUS:
            errors.append(f"{gate_id or f'gate[{index}]'} has invalid status {status!r}")
        for list_field in ("entry_criteria", "exit_criteria"):
            if not _non_empty_string_list(gate.get(list_field)):
                errors.append(f"{gate_id or f'gate[{index}]'} {list_field} must be non-empty strings")
        metric = gate.get("metric")
        if not isinstance(metric, dict) or not metric.get("name") or metric.get("goal") not in {"max", "min"}:
            errors.append(f"{gate_id or f'gate[{index}]'} metric requires name and goal=max|min")
        if status == "completed" and not gate.get("review_path"):
            completed_missing_reviews.append(gate_id)

    for gate_id in completed_missing_reviews:
        errors.append(f"{gate_id} is completed but has no review_path")

    return {
        "ok": not errors,
        "path": str(path),
        "gate_count": len(gates),
        "errors": errors,
    }


def validate_gate_review(path: str | Path, *, roadmap_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    review = load_json(path)
    errors: list[str] = []
    if review.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(review.get("gate_id"), str) or not review["gate_id"].strip():
        errors.append("gate_id is required")
    if review.get("decision") not in VALID_DECISIONS:
        errors.append(f"decision must be one of {sorted(VALID_DECISIONS)}")
    if not _non_empty_string_list(review.get("evidence")):
        errors.append("evidence must be a non-empty string list")
    if not _non_empty_string_list(review.get("next_requirements")):
        errors.append("next_requirements must be a non-empty string list")

    if roadmap_path is not None:
        roadmap = load_json(roadmap_path)
        gate_ids = {gate.get("id") for gate in roadmap.get("gates", []) if isinstance(gate, dict)}
        if review.get("gate_id") not in gate_ids:
            errors.append(f"gate_id {review.get('gate_id')!r} is not in roadmap")

    return {
        "ok": not errors,
        "path": str(path),
        "gate_id": review.get("gate_id"),
        "errors": errors,
    }


def _non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_validate_roadmap(args: argparse.Namespace) -> int:
    report = validate_roadmap(args.path)
    _print_json(report)
    return 0 if report["ok"] else 1


def _cmd_validate_gate_review(args: argparse.Namespace) -> int:
    report = validate_gate_review(args.path, roadmap_path=args.roadmap)
    _print_json(report)
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AIC LEWM roadmap tracking files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    roadmap_parser = subparsers.add_parser("validate-roadmap")
    roadmap_parser.add_argument("path")
    roadmap_parser.set_defaults(func=_cmd_validate_roadmap)

    review_parser = subparsers.add_parser("validate-gate-review")
    review_parser.add_argument("path")
    review_parser.add_argument("--roadmap")
    review_parser.set_defaults(func=_cmd_validate_gate_review)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RoadmapError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
