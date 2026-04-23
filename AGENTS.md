# AGENTS.md

These root governance files are authoritative within this repository:

- `ENGINEERING.md`: engineering standards, architecture boundaries, testing expectations, and agent execution model.
- `DOCTRINE.md`: immutable AIC runtime and empirical-loop invariants.
- `TASTE.md`: evolving preferences for documentation, code style, reports, gates, and subagent cadence.

Agents working in this repo must read and obey all three before making changes.
Repository-local guidance in deeper `AGENTS.md`, `ENGINEERING.md`, `DOCTRINE.md`,
`TASTE.md`, or similar files applies within that subtree when it does not weaken
the root doctrine.

The AIC signal harness exists to make real experiment signal legible and
actionable. It does not replace the official AIC runtime, does not control the
robot in real time, and does not bypass the official evaluator or scoring path.
