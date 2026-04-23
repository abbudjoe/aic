# AIC LEWM Policy

This package is the qualification-round integration surface for adapting
`/Users/joseph/le-wm` to the AIC policy API.

Current state:

- Defines an explicit AIC observation schema: three wrist images plus a fixed
  controller/joint/wrench state vector.
- Defines the action schema as Cartesian TCP velocity commands.
- Provides `dataset_schema.write_lewm_hdf5()` for writing flat HDF5 files that
  `stable_worldmodel.data.HDF5Dataset` can read.
- Provides `RecordingCheatCode`, a simulation-only ground-truth recorder that
  runs the reference insertion behavior and writes LEWM HDF5 episodes.
- Vendors the small LEWM model-definition files needed by object checkpoints.
- Loads a LEWM object checkpoint from `AIC_LEWM_CHECKPOINT` and runs a
  goal-image MPC planner when `AIC_LEWM_GOAL_DATASET` is set.
- Can run a dataset-backed `replay` planner for Gate 1 via
  `AIC_LEWM_PLANNER_MODE=replay`; replay uses public task metadata and recorded
  actions, not eval-time ground-truth TF.
- Defaults to a lifecycle-safe hold-still planner unless
  `AIC_LEWM_REQUIRE_CHECKPOINT=1` is set, which makes submission images fail
  fast if the learned runtime is missing.
- Provides `aic_lewm_policy.experiment_harness`, a CLI for validating HDF5
  datasets, recording run manifests, appending the experiment ledger, and
  gating checkpoint promotion.

Run locally with:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_lewm_policy.LewmMpcPolicy
```

Useful runtime variables:

- `AIC_LEWM_CHECKPOINT`: LEWM checkpoint name/path to load during configure.
- `AIC_LEWM_PLANNER_MODE`: `lewm_mpc`, `replay`, or `hold`; default
  `lewm_mpc`.
- `AIC_LEWM_GOAL_DATASET`: AIC LEWM HDF5 file used for terminal goal images
  and action normalization.
- `AIC_LEWM_REPLAY_DATASET`: optional HDF5 path for `replay`; defaults to
  `AIC_LEWM_GOAL_DATASET`.
- `AIC_LEWM_REPLAY_HZ`, `AIC_LEWM_REPLAY_TIME_SCALE`,
  `AIC_LEWM_REPLAY_ACTION_GAIN`: replay timing and action-scale knobs.
- `AIC_LEWM_REQUIRE_CHECKPOINT`: set to `1` for submission or learned-policy
  evaluation so missing checkpoint/runtime state is a hard failure.
- `AIC_LEWM_CACHE_DIR`: checkpoint cache root when using stable-worldmodel names.
- `AIC_LEWM_MAX_RUNTIME_SEC`: task loop duration for local runs, default `8`.
- `AIC_LEWM_CONTROL_HZ`: command rate, default `10`.
- `AIC_LEWM_COMMAND_FRAME`: command frame, default `gripper/tcp`.
- `AIC_LEWM_NUM_ACTION_CANDIDATES`: random-shooting MPC candidate count,
  default `32` in local config and `4` in the submission/eval Docker path.
- `AIC_LEWM_PLANNING_HORIZON`: future action horizon, default `5` in local
  config and `1` in the submission/eval Docker path.

The submission/eval Docker path uses a monotonic wall-clock action deadline,
publishes a final zero-velocity command on exit, and reinstalls the policy
package from copied source after `pixi install` so image builds cannot reuse a
stale local conda package.

Offline learned-policy smoke check:

```bash
PYTHONPATH=aic_lewm_policy python -m aic_lewm_policy.offline_policy_check \
  --checkpoint aic_lewm_policy/runtime_artifacts/aic_lewm_epoch_100_object.ckpt \
  --dataset aic_lewm_policy/runtime_artifacts/aic_qualification_train.h5
```

Offline replay gate check:

```bash
PYTHONPATH=aic_lewm_policy python -m aic_lewm_policy.replay_policy_check \
  --dataset aic_lewm_policy/runtime_artifacts/aic_qualification_train.h5
```

For local submission-image builds, stage the trained object checkpoint and
goal dataset under `aic_lewm_policy/runtime_artifacts/`. The directory is
intentionally ignored except for `.gitkeep` so large binary run artifacts stay
out of Git.

Collect a first ground-truth rollout dataset with:

```bash
export AIC_LEWM_DATASET_PATH=~/aic_results/aic_qualification_train.h5
export AIC_LEWM_RECORD_HZ=10
export AIC_LEWM_RECORD_IMAGE_SCALE=0.25

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_lewm_policy.RecordingCheatCode
```

Run the simulator/engine separately with `ground_truth:=true`; the recorder
uses ground-truth TF through the reference policy and is only for training data.

Training data written by this package should be placed under
`$STABLEWM_HOME/datasets/`, for example:

```text
$STABLEWM_HOME/datasets/aic_qualification_train.h5
```

Then copy or reference `lewm_configs/aic_data.yaml` from the LEWM training config
tree and launch LEWM with `data=aic_data`.

Every dataset/train/eval iteration should be recorded under
`aic_lewm_policy/experiments/`. See
`aic_lewm_policy/experiments/roadmap.json` for the gate plan and
`aic_lewm_policy/experiments/README.md` for the ledger and promotion commands.
