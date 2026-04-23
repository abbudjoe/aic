# AIC Signal Harness Tombstone

Status: tombstone reference
Date: 2026-04-23

This document records the current architecture decision for building an AIC
experiment and learning harness from the inside out. It is intentionally a
tombstone: future work should point here when explaining why the harness is
centered on AIC-native multimodal episodes rather than directly adopting Fawx
or Codex internals.

## Decision

Build a new AIC signal harness on top of the official AIC runtime. Use Fawx as
architectural inspiration for the signal flywheel, but do not make Fawx's
current coding-agent signal schema the core AIC substrate. Use Codex as the
current outer-loop operator and future-compatible orchestration shell, but do
not put Codex in the real-time robot control loop.

The stable center of the system should be typed AIC episodes:

```text
Run
  Trial / Episode
    TaskSpec
    Observation stream
    Action stream
    Controller state
    Planner/model decisions
    Contact and force events
    Score terms
    Artifacts
    Provenance
    Reward labels
    Failure analysis
```

Raw evidence remains in native formats. Offline reducers convert that evidence
into typed traces and reports.

## Bootstrap Strategy

Build from the inside out.

The current Codex session acts as a faithful proxy for a future outer Codex
shell because it already operates through the same basic harness capabilities:
inspect repository state, edit files, launch runs, read artifacts, summarize
failures, update ledgers, and choose the next experiment.

The important constraint is that Codex's judgment must be forced through
explicit contracts. Each run should produce durable artifacts that a future
standalone harness can consume without hidden conversational context:

```text
experiment_spec.json
run_manifest.json
raw artifacts
episode_trace.jsonl
reward_report.json
failure_report.json
promotion_report.json
next_experiment.json
```

Today, Codex plus the human operates the outer loop. Later, a standalone Codex
harness, OpenAI Agents SDK harness, DeepAgents harness, Fawx-derived loop, or
custom Rust orchestrator can operate the same contracts.

## Runtime Boundary

The real-time control path must stay inside the official AIC lifecycle:

```text
AIC evaluator
  -> aic_model
    -> policy backend
      -> aic_controller
        -> official scoring.yaml
```

The offline empirical path is separate:

```text
MCAP / HDF5 / logs / scores
  -> canonical episode trace
    -> reward and failure labels
      -> promotion gate
        -> next experiment
```

Codex controls experiments. The AIC policy controls the robot.

## What Has Been Inspected

This is not a claim that every line of AIC, Fawx, and Codex has been audited.
The inspected surfaces are the parts needed to define the architecture boundary.

AIC surfaces:

- `aic_model/aic_model/aic_model.py`: official policy lifecycle, observations,
  action callbacks, and dynamic policy loading.
- `aic_lewm_policy/aic_lewm_policy/schemas.py`: typed task, observation, and
  runtime policy contracts.
- `aic_lewm_policy/aic_lewm_policy/experiment_harness.py`: dataset validation,
  scoring parsing, manifests, ledger entries, and promotion gates.
- `aic_lewm_policy/aic_lewm_policy/eval_bag_analysis.py`: offline MCAP
  analysis for controller state, commands, and off-limit contacts.
- `aic_lewm_policy/aic_lewm_policy/replay_policy_check.py`: replay planner
  validation over HDF5 datasets.
- `aic_lewm_policy/aic_lewm_policy/roadmap_tracking.py`: gate and review
  validation.
- `aic_lewm_policy/experiments/roadmap.json`: current empirical gate structure.
- `docs/submission.md`, `docs/build_eval.md`, and `docs/policy.md`: official
  submission, local evaluation, and policy integration contracts.

Fawx surfaces:

- `fx-core::signals`: loop step, signal kind, severity, metadata, causal ids,
  durations, and typed control-plane markers.
- `fx-memory::signal_store`: append-only JSONL signal persistence by session.
- `fx-analysis`: LLM-powered recurring-pattern analysis over stored signals.
- `fx-improve`: signal -> finding -> candidate -> plan -> proposal loop with
  thresholds, fingerprints, deduplication, and execution modes.
- `fx-training`: dataset curation and training-example extraction.
- `fx-journal`: append-only reflective memory and searchable lessons.

Codex surfaces:

- `codex-rs/rollout-trace`: opt-in local raw event capture, payload store, and
  offline reduction into semantic rollout graphs.

## Keep

Keep the official AIC runtime as the source of truth:

- ROS 2 and Gazebo evaluation loop.
- `aic_engine`.
- `aic_model` policy lifecycle.
- `aic_controller`.
- official `scoring.yaml`.
- official container/submission boundary.

Keep the existing experiment discipline from `aic_lewm_policy`:

- dataset validation;
- scoring parse;
- run manifests;
- append-only ledgers;
- promotion gates;
- roadmap and gate reviews;
- GCP run scripts and artifact upload patterns.

Keep native artifact formats:

- MCAP for ROS bags;
- HDF5 for datasets;
- YAML for official scoring;
- JSON/JSONL for manifests, traces, labels, and policy/model events.

## Modify

The current harness is too LEWM-specific. Extract the general pieces into a
neutral AIC harness package and leave `aic_lewm_policy` as one policy backend.

The roadmap should eventually be broadened from "LEWM policy work" into
"submission candidate work" with LEWM represented as one gate or backend rather
than the central assumption.

Policies should emit explicit policy traces instead of relying on prose logs.
The first version can be JSONL with stable event types:

```text
policy.started
task.received
observation.sampled
planner.selected_action
action.published
guard.triggered
policy.completed
policy.failed
```

## Borrow From Fawx

Borrow the flywheel, not the exact schema:

```text
signals
  -> analysis findings
    -> improvement candidates
      -> plans or next experiments
        -> evaluated runs
          -> training/eval data
```

Specific concepts worth borrowing:

- append-only signal/event logs;
- typed signal categories and severity;
- causal links and spans;
- evidence-backed analysis findings;
- confidence thresholds;
- fingerprint deduplication;
- bounded improvement candidates;
- explicit human/agent review gates;
- training data extraction from accepted and rejected attempts;
- durable lessons/journal entries.

Fawx's current signals are coding-agent signals. AIC needs robotics and
multimodal signals, so the canonical schema must be AIC-native.

## Borrow From Codex

Codex contributes the first working outer loop:

- repository inspection;
- code edits;
- command execution;
- artifact reading;
- cloud orchestration;
- failure summarization;
- next-experiment planning.

Codex's `rollout-trace` is useful as an adapter for tracing outer-loop agent
behavior. It should not become the canonical AIC episode format.

## Build

Create a neutral package:

```text
aic_signal_harness/
  __init__.py
  schemas.py
  artifacts.py
  manifest.py
  scoring.py
  signals.py
  reducers/
    hdf5.py
    mcap.py
    scoring_yaml.py
    policy_trace.py
  analysis/
    reward_labeler.py
    failure_labeler.py
    next_experiment.py
  gates.py
  cli.py
```

The package should avoid ROS imports unless a reducer requires them. This keeps
manifest validation, scoring parsing, reports, and gate checks runnable in local
Pixi, cloud bootstrap shells, or CI.

## Canonical Objects

The first schema pass should define these objects:

- `ExperimentSpec`: hypothesis, policy backend, config, expected artifacts,
  run command, acceptance criteria.
- `RunManifest`: run identity, status, code provenance, environment, artifacts,
  dataset refs, score refs, reports, promotion decision.
- `ArtifactRef`: kind, path, URI, checksum, size, source, optional media type.
- `EpisodeTrace`: one run's typed trial collection.
- `TrialTrace`: task, time bounds, observations, actions, contacts, scores,
  failure labels.
- `TimelineEvent`: timestamped normalized event with source and payload ref.
- `ObservationSummary`: camera/state/wrench/controller summary without forcing
  all raw tensors into JSON.
- `ActionEvent`: high-level primitive and low-level command.
- `ContactEvent`: legal or off-limit contact evidence.
- `RewardReport`: official and shaped reward terms.
- `FailureReport`: evidence-backed failure classes and suggested next action.
- `GateDecision`: proceed, iterate, block, skip, promote, or reject.

## Execution Plan

1. Create `aic_signal_harness` with schemas and CLI skeleton.

2. Move or wrap reusable logic from `aic_lewm_policy.experiment_harness`:
   dataset validation, scoring parsing, manifests, ledgers, and promotion gates.

3. Add reducers for existing artifact types:
   HDF5, official `scoring.yaml`, MCAP eval bundles, and policy trace JSONL.

4. Backfill existing runs into the new format:
   Gate 0 learned baseline, Gate 1 replay baseline, and the latest Gate 2
   lat-bias run.

5. Add reward labels:
   official total, per-trial score, tier 1/2/3 terms, score delta, insertion
   evidence, contact penalties, runtime/lifecycle pass/fail.

6. Add failure labels:
   alignment drift, insertion miss, force guard trip, off-limit contact,
   lifecycle failure, dataset/schema failure, score regression, unsupported
   evidence, and inconclusive.

7. Add `next_experiment.json` generation as a report, not an autonomous action.
   The current Codex session can review and execute it manually.

8. Wire live GCP/local eval scripts to call the new CLI after each run.

9. Add tests before relying on the harness:
   schema round trips, reducer fixtures, scoring parse, manifest validation,
   ledger duplicate protection, promotion gates, and backfilled run acceptance.

10. After the contracts are stable, add adapters:
    Codex rollout trace adapter, Fawx signal import/export adapter, training
    dataset exporter, and optional standalone outer-loop executor.

## First Definition Of Done

The first implementation pass is done when one existing Gate 2 run can be
processed end to end by the new harness:

```text
existing raw artifacts
  -> canonical episode_trace.jsonl
  -> reward_report.json
  -> failure_report.json
  -> validated run_manifest.json
  -> ledger entry
  -> promotion_report.json
```

The output must preserve all evidence currently available in the old harness and
make the failure state more explicit than the current manifest alone.

## Initial Backfill Target

Use the latest Gate 2 run as the first backfill target:

```text
gate2-20260422T230909Z-latbias
```

Known evidence from that run:

- official total score: `128.00872420948235`;
- no insertion/contact credit;
- final distances remained nonzero;
- trial 1 tripped the delta force guard;
- the run was rejected by promotion gate because it did not beat the baseline by
  the required margin.

This is useful because it contains both positive evidence and negative evidence:
the lifecycle works, scoring works, replay works, and the candidate did not solve
final insertion.

## Non-Goals

Do not put an LLM directly in the 20 Hz real-time control loop.

Do not depend on Fawx's current coding-agent signal schema for robotics
episodes.

Do not make LEWM the mandatory core planner before it earns that role through
the gate process.

Do not let prose replace contracts. If a rule matters, it needs to appear in a
schema, reducer, test, gate, or executable report.

## Open Questions

- Should the neutral harness live as a top-level package or under
  `aic_lewm_policy` during the first extraction?
- Should policy trace JSONL be written from every backend immediately, or only
  from the active experimental backend first?
- Which shaped reward terms should be allowed to influence planning before they
  have been validated against official score movement?
- How much of Codex `rollout-trace` should be imported into AIC reports versus
  stored as a separate outer-loop trace artifact?

## Guiding Principle

The AIC signal harness should make every experiment replayable as evidence. If
Codex can make a decision today by reading a run, a future harness should be
able to make the same decision from typed artifacts without needing the original
conversation.
