# TASTE.md - AIC Signal Harness Taste

Effective 2026-04-22. This file captures evolving preferences for AIC docs,
code, reports, gates, and process. It guides judgment when `ENGINEERING.md` and
`DOCTRINE.md` leave room for choice.

## Philosophy

### Signal First

The point of the harness is not to produce more files. It is to make experiment
signal sharper: what happened, why it happened, whether it improved, and what to
try next.

### Small Durable Steps

Prefer a small reducer, schema, gate, or report that becomes trustworthy over a
large speculative harness that nobody can validate. The flywheel earns speed by
keeping each turn legible.

### Make The Boundary Obvious

Readers should be able to tell at a glance whether code belongs to official AIC
runtime integration, policy backend execution, offline reduction, reward
labeling, gate evaluation, or reporting. When those boundaries blur, rename,
split, or document the owner.

### Reports Are For Decisions

A report should help decide the next experiment. It should not bury the lead in
raw logs or ornate prose. State the score, the comparison, the failures, the
evidence, and the recommended next move.

## Documentation Style

- Use concrete AIC artifact names: MCAP, HDF5, `scoring.yaml`, policy trace
  JSONL, `run_manifest.json`, `episode_trace.jsonl`, reward reports, failure
  labels, and ledgers.
- Prefer diagrams of data flow over paragraphs describing invisible contracts.
- Keep docs close to the artifact or gate they describe.
- Record why a gate or reward term exists, what evidence it consumes, and what
  failure it is meant to prevent.
- Do not write provider-specific process doctrine into repo governance. Worker
  availability changes; runtime invariants should not.

## Code Style Preferences

- Put schemas near the reducer or gate that owns them unless a shared package has
  already been established.
- Keep schema migrations explicit and boring.
- Prefer typed parse functions that return structured results over functions
  that print warnings and continue.
- Name gates after decisions, not implementation details:
  `passes_contact_safety_gate` is better than `check_yaml_fields`.
- Keep command-line tools narrow: one command should validate, reduce, label, or
  gate, not secretly do all four.
- Reports should have stable machine-readable output first and human-readable
  summaries second when both are needed.

## Gate Iteration

- Start with gates that catch known failures from recorded runs.
- Add one gate per clear failure mode when possible.
- Store the baseline, threshold, artifact set, and reducer versions in the
  manifest or adjacent gate config.
- A gate that nobody can explain should be simplified or removed.
- Shaped reward terms should graduate only after they predict useful next-step
  decisions without contradicting official scores.

## Reward And Failure Design

- Separate official score terms from shaped reward terms.
- Label failures with typed categories tied to time windows and evidence links.
- Prefer labels that imply an action: `missed_port_alignment`,
  `excess_contact_force`, `controller_tracking_reset`, `policy_timeout`.
- Keep privileged-label provenance visible. A label derived from ground truth is
  useful, but it is not a legal online observation.
- When labels disagree with official scores, preserve both and investigate. Do
  not average away the contradiction.

## PR Shape

- Small PRs are the default. A good harness PR usually changes one contract and
  its tests: schema plus fixtures, reducer plus round trip, gate plus report, or
  policy trace event plus parser.
- Architectural PRs should include a short written map of the new boundary and
  the migration path from existing code.
- Avoid mixing policy-performance experimentation with harness-control-plane
  changes unless the PR explicitly exists to connect the two.
- Delete obsolete paths as soon as their replacement is proven.

## Review And Subagent Cadence

For substantial harness work, use the assembly line:

```text
orchestrate -> implement -> review -> fix -> fresh review -> repeat until clean
```

Preferences:

- Give workers exact file boundaries and the artifact contract they are changing.
- Give reviewers the diff, the relevant invariant list, and any known concerns.
- Fix all findings, including nits, before asking for a clean re-review.
- Use a fresh reviewer for the clean pass.
- Status updates should name the current stage and the next gate, not provide a
  transcript of every command.
- Do not wait for user acknowledgment between stages unless a real decision,
  risk acceptance, or scope change is needed.

## Experiment Reports

A readable run report usually contains:

- Run id, manifest path, policy backend, and environment.
- Official score and important tier breakdowns.
- Gate verdict and baseline comparison.
- Top failure labels with evidence windows.
- Reward terms that changed materially.
- Artifact links or paths.
- Recommended next experiment.

Keep the prose plain. A fresh worker should be able to pick up the report and
continue the loop without asking what the numbers meant.
