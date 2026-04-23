#!/usr/bin/env bash
set -euo pipefail

LEWM_REMOTE_ROOT="${LEWM_REMOTE_ROOT:-$HOME/src/le-wm}"
STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable-wm}"
AIC_DATASET_NAME="${AIC_DATASET_NAME:-aic_qualification_train}"
AIC_DATASET_GCS_URI="${AIC_DATASET_GCS_URI:-}"
GCS_OUTPUT_PREFIX="${GCS_OUTPUT_PREFIX:-}"
LEWM_RUN_NAME="${LEWM_RUN_NAME:-aic-lewm-baseline}"
LEWM_MAX_EPOCHS="${LEWM_MAX_EPOCHS:-100}"
LEWM_BATCH_SIZE="${LEWM_BATCH_SIZE:-64}"

if [[ ! -d "$LEWM_REMOTE_ROOT" ]]; then
  echo "Missing LEWM repo at $LEWM_REMOTE_ROOT. Run sync_sources.sh first." >&2
  exit 2
fi

if [[ ! -f "$LEWM_REMOTE_ROOT/config/train/data/aic.yaml" ]]; then
  echo "Missing $LEWM_REMOTE_ROOT/config/train/data/aic.yaml. Run sync_sources.sh first." >&2
  exit 2
fi

if [[ -z "$AIC_DATASET_GCS_URI" ]]; then
  echo "Set AIC_DATASET_GCS_URI before training." >&2
  exit 2
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to start a GPU training run." >&2
  exit 2
fi

nvidia-smi

mkdir -p "$STABLEWM_HOME/datasets"
gcloud storage cp "$AIC_DATASET_GCS_URI" "$STABLEWM_HOME/datasets/$AIC_DATASET_NAME.h5"
cp "$STABLEWM_HOME/datasets/$AIC_DATASET_NAME.h5" \
  "$STABLEWM_HOME/$AIC_DATASET_NAME.h5"

cd "$LEWM_REMOTE_ROOT"
uv venv --python=3.10 --clear .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install "stable-worldmodel[train,env]"
# stable-pretraining imports datasets.config, which is present in the
# Hugging Face datasets 2.x API but absent from the older transitive pin.
uv pip install "datasets==2.21.0"

export STABLEWM_HOME
python train.py \
  data=aic \
  output_model_name=aic_lewm \
  subdir="$LEWM_RUN_NAME" \
  wandb.enabled=false \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs="$LEWM_MAX_EPOCHS" \
  loader.batch_size="$LEWM_BATCH_SIZE"

if [[ -n "$GCS_OUTPUT_PREFIX" ]]; then
  gcloud storage cp --recursive \
    "$STABLEWM_HOME/$LEWM_RUN_NAME" \
    "${GCS_OUTPUT_PREFIX%/}/checkpoints/"
  echo "Synced run output to ${GCS_OUTPUT_PREFIX%/}/checkpoints/$LEWM_RUN_NAME."
fi
