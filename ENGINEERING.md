# ENGINEERING.md - AIC Signal Harness Engineering Doctrine

Effective 2026-04-22. These standards apply to all work in this AIC repository.

This file defines durable engineering rules for the AIC signal harness effort.
For immutable runtime and empirical-loop invariants, see `DOCTRINE.md`. For
evolving judgment and style preferences, see `TASTE.md`.

## 0. Core Doctrine

### Fix Root Causes

Fix the primitive that failed, not the symptom that surfaced first. If a live
failure reveals a missing schema, hidden contract, broken reducer, or ambiguous
artifact boundary, repair that boundary and add a focused regression.

### Explicit Contracts Over Inference

Control-plane contracts must be visible to code. Use typed schemas, enums,
versioned manifests, and structured validation instead of prose, file-name
heuristics, or loosely parsed logs. If the harness cannot inspect the real
contract, the contract surface is incomplete.

### Official Runtime Is The Source Of Truth

The official AIC execution path remains authoritative:

```text
AIC evaluator
  -> aic_model
    -> policy backend
      -> aic_controller
        -> official scoring.yaml
```

Harness code may observe, reduce, replay, label, and gate experiments. It must
not redefine official challenge scoring, bypass lifecycle behavior, or insert an
unofficial controller between the policy and `aic_controller`.

### Empirical Discipline

Every experiment should leave enough evidence for a fresh worker to understand
what ran, why it ran, what artifacts were produced, what score was official, what
labels were derived, and why the next gate passed or failed. If the next
experiment cannot be justified from recorded artifacts, the loop is not
scientific enough.

### No Slop

Slop means ambiguous ownership, duplicated parsing, silent fallback, untyped
state, flaky gates, hand-waved provenance, or reports that require oral context
to interpret. Slop in an experiment harness compounds into false confidence, so
remove it early.

## 1. AIC Signal Harness Architecture

The harness is an offline empirical flywheel around typed multimodal episodes:

```text
MCAP / HDF5 / logs / scoring.yaml / policy traces
  -> canonical episode trace
    -> reward and failure labels
      -> promotion gate
        -> next experiment
```

The stable center is the canonical episode, not any single policy backend,
runner script, notebook, or model checkpoint.

Core artifact families:

- `RunManifest`: hypothesis, environment, policy backend, configuration,
  expected artifacts, artifact digests, command provenance, and run status.
- `EpisodeTrace`: time-aligned multimodal observations, actions, controller
  state, policy events, scoring events, and derived episode boundaries.
- `RewardReport`: official score terms plus shaped reward terms with explicit
  provenance and leakage classification.
- `FailureLabel`: typed failure modes tied to evidence windows, not free-form
  impressions.
- `LedgerEntry`: append-only experiment accounting, promotion result, and link
  to the next decision.

Reducers translate source artifacts into canonical traces:

- MCAP reducers for ROS bags and evaluation recordings.
- HDF5 reducers for datasets and replayable demonstrations.
- Scoring reducers for official `scoring.yaml`.
- Policy trace reducers for backend-emitted JSONL events.
- Log reducers only when structured sources are unavailable; logs are evidence,
  not the canonical schema.

## 2. Runtime Boundaries

Codex and other workers are outer-loop experiment operators. They may edit code,
run offline analysis, launch evaluations, summarize traces, and propose the next
experiment. They are not the real-time policy controller.

Policy backends may consume legal observations and emit legal robot commands
through the `aic_model` contract. The official runtime owns lifecycle,
observation delivery, action invocation, controller integration, and scoring.

Do not introduce a real-time language-model control loop for robot actions.
Language models can help design experiments, analyze traces, generate reports,
or edit policy code offline. They must not sit in the live control path.

## 3. Code Quality

- Every module has a clear owner and purpose.
- No catch-all `utils` modules for new harness primitives; name modules after
  the contract they own.
- Prefer typed dataclasses, pydantic models, message schemas, or equivalent
  structured contracts for artifacts and state.
- Use explicit error types or structured error payloads where practical.
- Fail fast on missing required artifacts, malformed scores, schema version
  mismatches, duplicate ledger entries, and illegal provenance.
- No silent fallbacks from official scores to shaped rewards.
- No stringly typed policy modes, failure labels, gate states, or artifact kinds
  when an enum or schema is possible.
- No TODO, FIXME, or HACK without a linked task or an explicit expiration plan.
- Remove dead adapters and obsolete gates in the same change that makes them
  unused.

## 4. Testing And Verification

Tests scale with risk and contract surface.

Required regression patterns:

- Schema round trips for run manifests, episode traces, reward reports, failure
  labels, and ledger entries.
- Reducer fixtures for MCAP, HDF5, official scoring YAML, policy traces, and
  representative logs.
- Gate tests for pass, fail, missing-artifact, duplicate-run, and stale-baseline
  cases.
- Leakage tests proving that ground truth, evaluator-only state, and shaped
  labels cannot enter legal policy observations or action decisions.
- Boundary tests around `aic_model`, `aic_controller`, and official scoring
  assumptions whenever those surfaces are touched.

Bug fixes require a focused regression that would have caught the bug. If a bug
exists because the contract was implicit, add the missing contract and test that
contract directly.

For docs-only changes, use narrow verification such as file listing, grep, link
checks, or spelling checks as appropriate. Do not run broad test suites for
governance edits unless the user asks.

## 5. Experiment And Promotion Discipline

Every promoted candidate must be backed by recorded evidence:

- The run manifest validates.
- Required artifacts exist and are digest-addressed or otherwise uniquely
  identified.
- Official scoring was parsed from official output.
- Reward and failure labels name their source artifacts and derivation method.
- The promotion gate compares against the declared baseline.
- The ledger records pass/fail, reason, and next action.

Promotion gates must be deterministic. If a gate depends on stochastic sampling,
the random seed, sample set, and confidence rule must be part of the manifest.

Never let shaped reward terms silently override official scoring. Shaped rewards
are useful for learning and diagnosis; official scoring remains the competition
truth.

## 6. Assembly-Line Agent Execution Model

The main Codex session owns orchestration: it keeps state, assigns stages, reads
review results, decides when the loop is complete, and reports the outcome. For
massive harness work, implementation, review, and fix stages should be delegated
to single-purpose workers.

Default loop:

```text
main Codex session
  -> implementation worker
  -> independent review worker
  -> fix worker
  -> fresh review worker
  -> repeat until no findings remain, including nits
```

Execution rules:

- Workers receive explicit scope, file boundaries, acceptance criteria, and
  relevant governance files.
- Implementers and fixers may write code within their assigned scope.
- Reviewers should be fresh subagents that did not author the change.
- Reviewers are generic fresh review workers; no provider or model family is
  doctrine.
- Fresh implementation/review workers or currently available subagents may be
  used when appropriate, but worker choice is operational taste, not
  repository law.
- Review findings are not optional backlog unless the orchestrator explicitly
  reclassifies them with a reason.
- Nits are fixed before the loop closes.
- The main session may make small direct edits when that is the cleanest path,
  but for large harness changes it should preserve the assembly-line separation
  of implementation, review, and fix stages.

## 7. Review Standards

Fresh reviewers check:

- Correctness against the official AIC runtime boundary.
- Artifact provenance and schema completeness.
- Leakage risks from ground truth, evaluator-only state, or labels into legal
  policy execution.
- Reducer determinism and reproducibility.
- Gate logic, baseline comparison, and ledger updates.
- Test coverage for touched contracts and failure paths.
- Simplicity, readability, and removal of obsolete paths.

Findings should be ordered by severity and grounded in concrete files and lines.
When no issues remain, reviewers should say so clearly and name any residual
risk.
