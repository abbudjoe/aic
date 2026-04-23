"""Experiment record keeping for AIC LEWM runs.

This module intentionally avoids ROS imports so it can run in the local Pixi
environment, a cloud VM bootstrap shell, or a small CI check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_METRIC_PATH = "evaluation.score.total"

REQUIRED_DATASET_KEYS = (
    "ep_len",
    "ep_offset",
    "ep_idx",
    "episode_idx",
    "step_idx",
    "pixels",
    "left_pixels",
    "right_pixels",
    "proprio",
    "state",
    "action",
    "task_id",
    "plug_type",
    "port_type",
    "target_module_name",
)

PER_STEP_DATASET_KEYS = (
    "ep_idx",
    "episode_idx",
    "step_idx",
    "pixels",
    "left_pixels",
    "right_pixels",
    "proprio",
    "state",
    "action",
    "task_id",
    "plug_type",
    "port_type",
    "target_module_name",
)


class HarnessError(RuntimeError):
    """Raised for expected command failures."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: dict[str, Any], *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise HarnessError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    os.replace(tmp_path, path)


def resolve_experiments_dir(path: str | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    if os.getenv("AIC_LEWM_EXPERIMENTS_DIR"):
        return Path(os.environ["AIC_LEWM_EXPERIMENTS_DIR"]).expanduser()
    cwd = Path.cwd()
    if (cwd / "aic_lewm_policy" / "experiments").exists() or (cwd / "aic_lewm_policy").exists():
        return cwd / "aic_lewm_policy" / "experiments"
    return cwd / "experiments"


def relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def git_context(repo_root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(repo_root)
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _run_git(["status", "--short"], repo_root) or ""
    diff_stat = _run_git(["diff", "--stat", "--", "."], repo_root) or ""
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_short": [line for line in status.splitlines() if line],
        "diff_stat": diff_stat,
    }


def validate_dataset(
    dataset_path: str | Path,
    *,
    min_episodes: int = 1,
    min_steps: int = 1,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise HarnessError("validate-dataset requires h5py") from exc

    dataset_path = Path(dataset_path).expanduser()
    if not dataset_path.exists():
        raise HarnessError(f"Dataset not found: {dataset_path}")

    errors: list[str] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "path": str(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "sha256": sha256_file(dataset_path),
        "validated_at_utc": utc_now_iso(),
        "required_keys": list(REQUIRED_DATASET_KEYS),
    }

    with h5py.File(dataset_path, "r") as handle:
        present_keys = tuple(sorted(handle.keys()))
        missing = [key for key in REQUIRED_DATASET_KEYS if key not in handle]
        if missing:
            errors.append(f"missing required datasets: {', '.join(missing)}")

        report["datasets"] = {
            key: {
                "shape": list(handle[key].shape),
                "dtype": str(handle[key].dtype),
            }
            for key in present_keys
        }
        report["attrs"] = {
            key: to_jsonable(value)
            for key, value in handle.attrs.items()
        }

        if not missing:
            ep_len = handle["ep_len"][()]
            ep_offset = handle["ep_offset"][()]
            episode_count = int(ep_len.shape[0])
            step_count = int(ep_len.sum())

            report["episode_count"] = episode_count
            report["step_count"] = step_count
            report["episode_lengths"] = [int(value) for value in ep_len.tolist()]
            report["episode_offsets"] = [int(value) for value in ep_offset.tolist()]

            if episode_count < min_episodes:
                errors.append(f"episode_count {episode_count} < required {min_episodes}")
            if step_count < min_steps:
                errors.append(f"step_count {step_count} < required {min_steps}")
            if ep_offset.shape[0] != episode_count:
                errors.append(
                    f"ep_offset length {ep_offset.shape[0]} does not match ep_len length {episode_count}"
                )

            for key in PER_STEP_DATASET_KEYS:
                shape = handle[key].shape
                if len(shape) < 1 or int(shape[0]) != step_count:
                    errors.append(f"{key} first dimension {shape[:1]} does not match steps {step_count}")

            for key in ("pixels", "left_pixels", "right_pixels"):
                shape = handle[key].shape
                dtype = str(handle[key].dtype)
                if len(shape) != 4 or int(shape[-1]) != 3:
                    errors.append(f"{key} must be THWC images with 3 channels, got {shape}")
                if dtype != "uint8":
                    errors.append(f"{key} must be uint8, got {dtype}")

            action_shape = handle["action"].shape
            if len(action_shape) != 2 or int(action_shape[1]) != 6:
                errors.append(f"action must have shape (T, 6), got {action_shape}")

            for key in ("state", "proprio"):
                shape = handle[key].shape
                if len(shape) != 2 or int(shape[1]) != 32:
                    errors.append(f"{key} must have shape (T, 32), got {shape}")

    report["ok"] = not errors
    report["errors"] = errors
    return report


def parse_scoring_yaml(scoring_path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise HarnessError("parse-scoring requires PyYAML") from exc

    scoring_path = Path(scoring_path).expanduser()
    if not scoring_path.exists():
        raise HarnessError(f"Scoring file not found: {scoring_path}")

    with scoring_path.open("r", encoding="utf-8") as handle:
        scoring = yaml.safe_load(handle) or {}
    if not isinstance(scoring, dict):
        raise HarnessError(f"Scoring file must contain a YAML mapping: {scoring_path}")
    if "total" not in scoring:
        raise HarnessError("Scoring file is missing required key: total")

    trials: dict[str, Any] = {}
    for key, value in scoring.items():
        if key == "total":
            continue
        if not isinstance(value, dict):
            continue
        trials[key] = {
            "total": _trial_total(value),
            "tier_1": _tier_score(value.get("tier_1")),
            "tier_2": _tier_score(value.get("tier_2")),
            "tier_3": _tier_score(value.get("tier_3")),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(scoring_path),
        "parsed_at_utc": utc_now_iso(),
        "score": {
            "source": str(scoring_path),
            "total": float(scoring["total"]),
            "trial_count": len(trials),
            "trials": trials,
        },
    }


def _tier_score(value: Any) -> float | None:
    if not isinstance(value, dict) or "score" not in value:
        return None
    try:
        return float(value["score"])
    except (TypeError, ValueError):
        return None


def _trial_total(value: dict[str, Any]) -> float:
    return float(
        sum(
            score
            for score in (
                _tier_score(value.get("tier_1")),
                _tier_score(value.get("tier_2")),
                _tier_score(value.get("tier_3")),
            )
            if score is not None
        )
    )


def create_manifest(
    *,
    run_id: str,
    kind: str,
    description: str,
    hypothesis: str,
    status: str = "planned",
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": kind,
        "status": status,
        "created_at_utc": now,
        "updated_at_utc": now,
        "description": description,
        "hypothesis": hypothesis,
        "code": git_context(repo_root),
        "dataset": {},
        "training": {},
        "evaluation": {},
        "artifacts": [],
        "promotion": {"decision": "not_evaluated"},
        "notes": [],
    }


def get_metric(manifest: dict[str, Any], metric_path: str = DEFAULT_METRIC_PATH) -> float:
    value: Any = manifest
    for part in metric_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise HarnessError(f"Metric path {metric_path!r} is missing at {part!r}")
        value = value[part]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"Metric path {metric_path!r} is not numeric: {value!r}") from exc


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "schema_version",
        "run_id",
        "kind",
        "status",
        "created_at_utc",
        "code",
        "dataset",
        "training",
        "evaluation",
        "artifacts",
        "promotion",
    ):
        if key not in manifest:
            errors.append(f"missing manifest key: {key}")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    status = manifest.get("status")
    if status not in {"planned", "running", "completed", "failed", "rejected", "promoted"}:
        errors.append(f"invalid status: {status!r}")

    if status in {"completed", "promoted"}:
        dataset = manifest.get("dataset", {})
        validation = dataset.get("validation", {})
        if validation.get("ok") is not True:
            errors.append("completed manifests require dataset.validation.ok=true")
        if int(dataset.get("episode_count", 0) or 0) < 1:
            errors.append("completed manifests require dataset.episode_count >= 1")
        if int(dataset.get("step_count", 0) or 0) < 1:
            errors.append("completed manifests require dataset.step_count >= 1")

        artifacts = manifest.get("artifacts", [])
        if not isinstance(artifacts, list) or not artifacts:
            errors.append("completed manifests require at least one artifact")

        training = manifest.get("training", {})
        output = training.get("output", {})
        if manifest.get("kind", "").find("train") >= 0 and not output.get("checkpoint_uri"):
            errors.append("training manifests require training.output.checkpoint_uri")

        try:
            get_metric(manifest)
        except HarnessError as exc:
            errors.append(str(exc))

    return errors


def ledger_entry(manifest: dict[str, Any], manifest_path: Path, experiments_dir: Path) -> dict[str, Any]:
    dataset = manifest.get("dataset", {})
    training = manifest.get("training", {})
    output = training.get("output", {})
    metric_value: float | None
    try:
        metric_value = get_metric(manifest)
    except HarnessError:
        metric_value = None

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": utc_now_iso(),
        "run_id": manifest["run_id"],
        "kind": manifest["kind"],
        "status": manifest["status"],
        "manifest_path": relative_to(manifest_path, experiments_dir),
        "git_commit": manifest.get("code", {}).get("commit"),
        "git_dirty": manifest.get("code", {}).get("dirty"),
        "dataset": {
            "uri": dataset.get("uri"),
            "sha256": dataset.get("sha256"),
            "episode_count": dataset.get("episode_count"),
            "step_count": dataset.get("step_count"),
        },
        "training": {
            "run_name": training.get("run_name"),
            "max_epochs": training.get("max_epochs"),
            "checkpoint_uri": output.get("checkpoint_uri"),
        },
        "metric": {
            "name": DEFAULT_METRIC_PATH,
            "goal": "max",
            "value": metric_value,
        },
        "promotion": manifest.get("promotion", {}),
    }


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HarnessError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return entries


def append_ledger(
    *,
    manifest_path: str | Path,
    ledger_path: str | Path,
    experiments_dir: str | Path,
    allow_duplicate: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    ledger_path = Path(ledger_path)
    experiments_dir = Path(experiments_dir)
    manifest = read_json(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise HarnessError("Manifest validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    entries = read_ledger(ledger_path)
    if not allow_duplicate and any(entry.get("run_id") == manifest["run_id"] for entry in entries):
        raise HarnessError(f"Ledger already contains run_id={manifest['run_id']}")

    entry = ledger_entry(manifest, manifest_path, experiments_dir)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=json_default))
        handle.write("\n")
    return entry


def promote_candidate(
    *,
    manifest_path: str | Path,
    baseline_path: str | Path,
    metric_path: str = DEFAULT_METRIC_PATH,
    goal: str = "max",
    min_improvement: float = 0.0,
    bootstrap: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    baseline_path = Path(baseline_path)
    manifest = read_json(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise HarnessError("Manifest validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    score = get_metric(manifest, metric_path)
    baseline: dict[str, Any] | None = None
    if baseline_path.exists():
        baseline = read_json(baseline_path)
    elif not bootstrap:
        raise HarnessError(f"Baseline does not exist: {baseline_path}. Use --bootstrap for the first one.")

    accepted = False
    reason = "bootstrap baseline"
    baseline_score = None
    if baseline is not None:
        baseline_score = float(baseline["metric"]["value"])
        if goal == "max":
            accepted = score >= baseline_score + min_improvement
            comparator = ">=" if accepted else "<"
            reason = f"{score} {comparator} {baseline_score} + {min_improvement}"
        elif goal == "min":
            accepted = score <= baseline_score - min_improvement
            comparator = "<=" if accepted else ">"
            reason = f"{score} {comparator} {baseline_score} - {min_improvement}"
        else:
            raise HarnessError("--goal must be max or min")
    else:
        accepted = True

    decision = {
        "schema_version": SCHEMA_VERSION,
        "decided_at_utc": utc_now_iso(),
        "accepted": accepted,
        "reason": reason,
        "metric": {
            "name": metric_path,
            "goal": goal,
            "value": score,
            "baseline_value": baseline_score,
            "min_improvement": min_improvement,
        },
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path),
        "checkpoint_uri": manifest.get("training", {}).get("output", {}).get("checkpoint_uri"),
        "notes": manifest.get("promotion", {}).get("notes", []),
    }
    if accepted:
        write_json(baseline_path, decision, overwrite=overwrite or baseline_path.exists())
    return decision


def summarize_ledger(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    entries = list(entries)
    scored = [
        entry
        for entry in entries
        if entry.get("metric", {}).get("value") is not None
    ]
    best = max(scored, key=lambda entry: float(entry["metric"]["value"]), default=None)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_count": len(entries),
        "scored_run_count": len(scored),
        "best": best,
    }


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=json_default))


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    report = validate_dataset(
        args.dataset,
        min_episodes=args.min_episodes,
        min_steps=args.min_steps,
    )
    if args.output:
        write_json(args.output, report, overwrite=args.overwrite)
    _print_json(report)
    return 0 if report["ok"] else 1


def _cmd_parse_scoring(args: argparse.Namespace) -> int:
    report = parse_scoring_yaml(args.scoring_yaml)
    if args.output:
        write_json(args.output, report, overwrite=args.overwrite)
    _print_json(report)
    return 0


def _cmd_create_manifest(args: argparse.Namespace) -> int:
    experiments_dir = resolve_experiments_dir(args.experiments_dir)
    output = Path(args.output) if args.output else experiments_dir / "runs" / args.run_id / "manifest.json"
    manifest = create_manifest(
        run_id=args.run_id,
        kind=args.kind,
        description=args.description,
        hypothesis=args.hypothesis,
        status=args.status,
        repo_root=args.repo_root,
    )
    write_json(output, manifest, overwrite=args.overwrite)
    print(output)
    return 0


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    manifest = read_json(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"manifest ok: {manifest['run_id']}")
    return 0


def _cmd_append_ledger(args: argparse.Namespace) -> int:
    experiments_dir = resolve_experiments_dir(args.experiments_dir)
    ledger_path = Path(args.ledger) if args.ledger else experiments_dir / "ledger.jsonl"
    entry = append_ledger(
        manifest_path=args.manifest,
        ledger_path=ledger_path,
        experiments_dir=experiments_dir,
        allow_duplicate=args.allow_duplicate,
    )
    _print_json(entry)
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    experiments_dir = resolve_experiments_dir(args.experiments_dir)
    baseline_path = Path(args.baseline) if args.baseline else experiments_dir / "baselines" / "current.json"
    decision = promote_candidate(
        manifest_path=args.manifest,
        baseline_path=baseline_path,
        metric_path=args.metric_path,
        goal=args.goal,
        min_improvement=args.min_improvement,
        bootstrap=args.bootstrap,
        overwrite=args.overwrite,
    )
    _print_json(decision)
    return 0 if decision["accepted"] else 1


def _cmd_summarize_ledger(args: argparse.Namespace) -> int:
    experiments_dir = resolve_experiments_dir(args.experiments_dir)
    ledger_path = Path(args.ledger) if args.ledger else experiments_dir / "ledger.jsonl"
    summary = summarize_ledger(read_ledger(ledger_path))
    if args.json:
        _print_json(summary)
    else:
        print(f"runs: {summary['run_count']}")
        print(f"scored runs: {summary['scored_run_count']}")
        best = summary.get("best")
        if best:
            print(
                "best: "
                f"{best['run_id']} {best['metric']['name']}={best['metric']['value']}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIC LEWM experiment harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate an AIC LEWM HDF5 dataset and optionally write a JSON report.",
    )
    validate_dataset_parser.add_argument("dataset")
    validate_dataset_parser.add_argument("--output")
    validate_dataset_parser.add_argument("--min-episodes", type=int, default=1)
    validate_dataset_parser.add_argument("--min-steps", type=int, default=1)
    validate_dataset_parser.add_argument("--overwrite", action="store_true")
    validate_dataset_parser.set_defaults(func=_cmd_validate_dataset)

    parse_scoring_parser = subparsers.add_parser(
        "parse-scoring",
        help="Parse AIC scoring.yaml into normalized JSON for a manifest.",
    )
    parse_scoring_parser.add_argument("scoring_yaml")
    parse_scoring_parser.add_argument("--output")
    parse_scoring_parser.add_argument("--overwrite", action="store_true")
    parse_scoring_parser.set_defaults(func=_cmd_parse_scoring)

    create_manifest_parser = subparsers.add_parser(
        "create-manifest",
        help="Create a skeleton run manifest with git context.",
    )
    create_manifest_parser.add_argument("--run-id", required=True)
    create_manifest_parser.add_argument("--kind", required=True)
    create_manifest_parser.add_argument("--description", default="")
    create_manifest_parser.add_argument("--hypothesis", default="")
    create_manifest_parser.add_argument("--status", default="planned")
    create_manifest_parser.add_argument("--repo-root", default=".")
    create_manifest_parser.add_argument("--experiments-dir")
    create_manifest_parser.add_argument("--output")
    create_manifest_parser.add_argument("--overwrite", action="store_true")
    create_manifest_parser.set_defaults(func=_cmd_create_manifest)

    validate_manifest_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate a completed run manifest.",
    )
    validate_manifest_parser.add_argument("manifest")
    validate_manifest_parser.set_defaults(func=_cmd_validate_manifest)

    append_ledger_parser = subparsers.add_parser(
        "append-ledger",
        help="Append a validated manifest summary to the JSONL ledger.",
    )
    append_ledger_parser.add_argument("manifest")
    append_ledger_parser.add_argument("--experiments-dir")
    append_ledger_parser.add_argument("--ledger")
    append_ledger_parser.add_argument("--allow-duplicate", action="store_true")
    append_ledger_parser.set_defaults(func=_cmd_append_ledger)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Gate a candidate against the current baseline.",
    )
    promote_parser.add_argument("manifest")
    promote_parser.add_argument("--experiments-dir")
    promote_parser.add_argument("--baseline")
    promote_parser.add_argument("--metric-path", default=DEFAULT_METRIC_PATH)
    promote_parser.add_argument("--goal", choices=("max", "min"), default="max")
    promote_parser.add_argument("--min-improvement", type=float, default=0.0)
    promote_parser.add_argument("--bootstrap", action="store_true")
    promote_parser.add_argument("--overwrite", action="store_true")
    promote_parser.set_defaults(func=_cmd_promote)

    summarize_parser = subparsers.add_parser(
        "summarize-ledger",
        help="Summarize the experiment ledger.",
    )
    summarize_parser.add_argument("--experiments-dir")
    summarize_parser.add_argument("--ledger")
    summarize_parser.add_argument("--json", action="store_true")
    summarize_parser.set_defaults(func=_cmd_summarize_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
