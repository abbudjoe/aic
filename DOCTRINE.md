# DOCTRINE.md - AIC Runtime And Empirical Invariants

Effective 2026-04-22. This file defines immutable invariants for the AIC signal
harness. `ENGINEERING.md` governs how changes are built. `DOCTRINE.md` governs
what must remain true.

Violation of any invariant is a critical bug.

## 0. Anchor

AIC policy development is an empirical control problem around a fixed official
runtime. The harness improves the outer experiment loop by turning multimodal
execution evidence into typed traces, labels, gates, and ledgers. It does not
become the runtime authority.

## 1. Official Evaluator Boundary

- The official AIC evaluator is the authority for trial lifecycle and official
  score production.
- Official score truth comes from official `scoring.yaml` output.
- The legal runtime path is `AIC evaluator -> aic_model -> policy backend ->
  aic_controller -> official scoring.yaml`.
- Harness code must not replace, fork, reinterpret, or short-circuit official
  scoring.
- Local scoring helpers may parse, validate, summarize, and compare official
  scores. They may not claim authority over challenge outcome.

## 2. No Real-Time LLM Control Loop

- No language model may sit in the live robot action loop.
- Codex and other agents are outer-loop operators: they edit code, run
  experiments, reduce artifacts, review results, and propose next steps.
- A policy backend may use learned models, planners, controllers, or scripted
  logic that comply with AIC rules. It must emit legal commands through
  `aic_model` and `aic_controller`.
- Natural-language reasoning belongs outside real-time control unless it is
  compiled into a deterministic, tested policy artifact before the run.

## 3. Legal Observation Boundary

- Ground truth, evaluator-only state, scoring internals, and post-hoc labels
  must not leak into legal policy observations or online action decisions.
- Ground-truth data may be used for training, debugging, annotation, or offline
  diagnosis only when its provenance is explicit and the artifact is marked as
  non-policy-runtime input.
- Reward shaping must carry a leakage classification. Terms derived from
  privileged signals cannot be used as if they were legal online observations.
- Any code path that can influence live actions must be auditable back to legal
  inputs.

## 4. Typed Artifact Provenance

- Every canonical artifact has a type, schema version, source path or URI,
  creation command, timestamp, producer, and validation status.
- Derived artifacts must name their source artifacts and derivation method.
- Episode traces are appendable evidence, not editable narratives.
- Ledgers are append-only. Corrections are new entries that supersede prior
  entries with an explicit reason.
- Missing provenance invalidates promotion.

## 5. Canonical Episode Invariant

The canonical episode trace is the stable interchange format for empirical
signal. It must be able to represent:

- Multimodal observations delivered to the policy.
- Policy decisions and backend trace events.
- Commands emitted toward `aic_controller`.
- Controller state and safety-relevant feedback.
- Official scoring events and final score terms.
- Episode boundaries, timing, failure windows, and artifact links.

MCAP, HDF5, logs, scores, and backend traces are source formats. They are not
the canonical contract until reduced into typed episode traces.

## 6. Promotion Gate Invariant

No candidate is promoted without a deterministic gate result.

A valid promotion decision requires:

- A valid run manifest.
- Available required artifacts.
- Parsed official scoring.
- Reward and failure labels with provenance.
- A declared baseline or acceptance threshold.
- A ledger entry recording the result and reason.

If any required input is missing, malformed, stale, or ambiguous, the gate fails
closed.

## 7. Reducer Invariant

Reducers are part of the control plane and must be deterministic.

- The same source artifact and reducer version must produce the same canonical
  trace.
- Reducers must preserve enough timing and provenance to audit labels and gates.
- Reducers must reject incompatible schema versions instead of guessing.
- Log parsing is allowed only as a compatibility adapter and must not outrank
  structured MCAP, HDF5, scoring, or policy trace inputs.

## 8. Runtime Immutability

- The official AIC runtime contracts are not rewritten by harness convenience.
- `aic_model` lifecycle and action semantics remain the policy integration
  boundary.
- `aic_controller` remains the robot command execution boundary.
- Official `scoring.yaml` remains the scoring output boundary.
- Harness adapters may wrap or observe these surfaces; they may not silently
  redefine them.

## 9. Subagent Process Invariants

The assembly-line process is part of governance for substantial harness work.

- The main Codex session orchestrates implementation, independent review, fix,
  and fresh review stages.
- Implementers and fixers may write within assigned scope.
- Reviewers do not author the diff they review.
- Fresh review means a new review pass that is not merely the fixer declaring
  success.
- The loop continues until no findings remain, including nits, or until the
  user explicitly accepts a known residual risk.
- Subagent roles and permissions are process invariants. A worker cannot grant
  itself broader scope than the orchestrator assigned.

## 10. Invariant Summary

1. Official evaluator and official `scoring.yaml` remain authoritative.
2. No real-time language-model control loop drives robot actions.
3. Legal policy execution cannot consume ground truth, evaluator-only state, or
   post-hoc labels.
4. Every canonical artifact carries typed provenance.
5. Canonical episode traces are the stable empirical signal contract.
6. Promotion gates fail closed on missing or ambiguous evidence.
7. Reducers are deterministic and versioned.
8. Harness code observes and improves the loop; it does not redefine the AIC
   runtime.
9. Ledgers are append-only.
10. Substantial harness work uses implementation, review, fix, and fresh-review
    stages until clean.
