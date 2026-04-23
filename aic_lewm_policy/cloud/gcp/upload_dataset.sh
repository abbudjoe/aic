#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_gcp_project
require_bucket

if [[ -z "$AIC_DATASET_LOCAL" ]]; then
  echo "Set AIC_DATASET_LOCAL in $ENV_FILE before uploading." >&2
  exit 2
fi

if [[ ! -f "$AIC_DATASET_LOCAL" ]]; then
  echo "AIC_DATASET_LOCAL is not a file: $AIC_DATASET_LOCAL" >&2
  exit 2
fi

EXPERIMENT_RUN_ID="${AIC_EXPERIMENT_RUN_ID:-$LEWM_RUN_NAME}"
EXPERIMENTS_DIR="${AIC_LEWM_EXPERIMENTS_DIR:-$AIC_ROOT/aic_lewm_policy/experiments}"
DATASET_REPORT_PATH="$EXPERIMENTS_DIR/runs/$EXPERIMENT_RUN_ID/dataset_report.json"

HARNESS_PYTHON="${AIC_LEWM_HARNESS_PYTHON:-}"
if [[ -z "$HARNESS_PYTHON" ]]; then
  if [[ -x "$AIC_ROOT/.pixi/envs/default/bin/python" ]]; then
    HARNESS_PYTHON="$AIC_ROOT/.pixi/envs/default/bin/python"
  else
    HARNESS_PYTHON="python3"
  fi
fi

PYTHONPATH="$AIC_ROOT/aic_lewm_policy" "$HARNESS_PYTHON" \
  -m aic_lewm_policy.experiment_harness validate-dataset \
  "$AIC_DATASET_LOCAL" \
  --output "$DATASET_REPORT_PATH" \
  --overwrite >/dev/null

gcloud storage buckets describe "$GCP_BUCKET" --project "$GCP_PROJECT" >/dev/null
gcloud storage cp "$AIC_DATASET_LOCAL" "$AIC_DATASET_GCS_URI" --project "$GCP_PROJECT"

echo "Uploaded dataset to $AIC_DATASET_GCS_URI."
echo "Validated dataset report at $DATASET_REPORT_PATH."
