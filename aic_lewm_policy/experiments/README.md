# AIC LEWM Experiment Ledger

This directory is the empirical record for LEWM-based AIC qualification work.
It is intentionally small and append-only:

- `ledger.jsonl` contains one summary row per completed run.
- `runs/<run_id>/manifest.json` contains the full evidence record for a run.
- `runs/<run_id>/dataset_report.json` contains HDF5 schema and checksum validation.
- `baselines/current.json` is the current promotion target.

Use the harness CLI from the repo root:

```bash
HARNESS_PY=${AIC_LEWM_HARNESS_PYTHON:-.pixi/envs/default/bin/python}

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness validate-dataset /path/to/data.h5 \
  --output aic_lewm_policy/experiments/runs/<run_id>/dataset_report.json

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness parse-scoring /path/to/scoring.yaml \
  --output aic_lewm_policy/experiments/runs/<run_id>/score_report.json

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness validate-manifest \
  aic_lewm_policy/experiments/runs/<run_id>/manifest.json

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness append-ledger \
  aic_lewm_policy/experiments/runs/<run_id>/manifest.json

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness promote \
  aic_lewm_policy/experiments/runs/<run_id>/manifest.json \
  --min-improvement 1.0
```

Promotion means only that the run beat the current tracked baseline on the
declared metric. It does not make a smoke run submission-ready.

For new live evals, prefer the backend-neutral finalizer:

```bash
PYTHONPATH=. "$HARNESS_PY" -m aic_signal_harness.live_eval finalize \
  --run-id <run_id> \
  --result-root /path/to/aic_results/<run_id> \
  --harness-root /path/to/aic_results/<run_id>/harness \
  --scoring-yaml /path/to/aic_results/<run_id>/eval/scoring.yaml \
  --policy-trace /path/to/aic_results/<run_id>/harness/policy_trace.jsonl \
  --ledger /path/to/aic_results/<run_id>/harness/ledger.jsonl \
  --model-image aic-lewm-learned:<run_id> \
  --model-image-id sha256:<docker-image-id>
```

This writes the neutral `run_manifest.json`, reward/failure reports,
`next_experiment.json`, and ledger entry from official eval evidence.
