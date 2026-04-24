# GCP L4 Training Lane

This directory contains the repeatable GCP path for training LEWM on AIC
rollouts. It targets one NVIDIA L4 GPU on a `g2-standard-16` VM in `us-west4-a`.

The default region here is `us-west4` because an existing bucket in this
project is also in `US-WEST4`.

## Setup

```bash
cd /Users/joseph/.codex/worktrees/0587/aic/aic_lewm_policy/cloud/gcp
cp env.example .env
```

Edit `.env` if you want a different bucket, zone, machine size, run name, or
local dataset path.

## Flow

Create and prepare the VM:

```bash
./check_quota.sh
./launch_l4_vm.sh
./bootstrap_vm.sh
./sync_sources.sh
```

Collect an official-style three-trial ground-truth rollout dataset:

```bash
# In .env or the shell:
# AIC_EXPERIMENT_RUN_ID=<run-id>
./run_recording.sh
```

Upload an AIC rollout dataset after it exists:

```bash
# In .env:
# AIC_DATASET_LOCAL=/absolute/path/to/aic_qualification_train.h5
# AIC_EXPERIMENT_RUN_ID=<run-id>
./upload_dataset.sh
```

`upload_dataset.sh` validates the HDF5 with the experiment harness before it
uploads anything. The validation report is written to
`aic_lewm_policy/experiments/runs/<run-id>/dataset_report.json`.

Run training:

```bash
./run_training.sh
```

Fetch checkpoints:

```bash
./fetch_outputs.sh
```

Stop the VM when idle:

```bash
./stop_vm.sh
```

## Notes

- `run_training.sh` refuses to start unless the AIC HDF5 exists in GCS.
- `upload_dataset.sh` refuses to upload an invalid HDF5 dataset.
- `launch_l4_vm.sh` refuses to start unless both global GPU quota and regional
  L4 quota are available.
- The dataset must be named `aic_qualification_train.h5` unless
  `AIC_DATASET_NAME` is changed.
- On the VM, LEWM reads datasets from `$STABLEWM_HOME/datasets/`.
- Training outputs are copied to
  `$GCP_BUCKET/$GCP_PREFIX/runs/checkpoints/$LEWM_RUN_NAME/`.
- Keep the VM stopped when not actively training; the boot disk remains until
  the instance is deleted.

## Experiment Records

Every data collection, training, and evaluation attempt should get a run id and
manifest under `aic_lewm_policy/experiments/runs/<run-id>/`. A completed run is
only part of the empirical record after:

```bash
HARNESS_PY=${AIC_LEWM_HARNESS_PYTHON:-.pixi/envs/default/bin/python}

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness \
  validate-manifest aic_lewm_policy/experiments/runs/<run-id>/manifest.json

PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness \
  append-ledger aic_lewm_policy/experiments/runs/<run-id>/manifest.json
```

Candidate checkpoints must pass the promotion gate before replacing the tracked
baseline:

```bash
PYTHONPATH=aic_lewm_policy "$HARNESS_PY" -m aic_lewm_policy.experiment_harness \
  promote aic_lewm_policy/experiments/runs/<run-id>/manifest.json \
  --min-improvement 1.0
```

Learned-policy evals now call the backend-neutral live eval finalizer after the
official evaluator exits. The finalizer writes typed harness outputs under
`$AIC_EVAL_RESULT_ROOT/harness/`, including:

- `run_manifest.json`
- `score_report.json`
- `policy_trace_report.json` and `policy_trace_artifact.json` when tracing is enabled
- `episode_trace.json` and `training_signal_report.json` when tracing is enabled
- `reward_report.json` and `failure_report.json`
- `promotion_report.json` when `AIC_HARNESS_BASELINE_PATH` or
  `AIC_HARNESS_BOOTSTRAP_PROMOTION=1` is set
- `next_experiment.json`
- `ledger_entry.json` and the run-local `ledger.jsonl`

Useful knobs:

```bash
export AIC_HARNESS_BASELINE_PATH=/absolute/path/to/neutral_baseline.json
export AIC_HARNESS_UPDATE_BASELINE_PATH=/absolute/path/to/neutral_baseline.json
export AIC_HARNESS_BOOTSTRAP_PROMOTION=0
export AIC_HARNESS_MIN_IMPROVEMENT=1.0
export AIC_HARNESS_NO_NEXT_EXPERIMENT=0
```

For GCP runs, harness paths are interpreted on the remote VM. The launcher
records the immutable Docker image id for the learned-policy image and passes it
to the finalizer as policy artifact provenance.
