#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_instance
require_dataset_uri

if ! gcloud storage ls "$AIC_DATASET_GCS_URI" --project "$GCP_PROJECT" >/dev/null 2>&1; then
  echo "Dataset not found at $AIC_DATASET_GCS_URI. Run upload_dataset.sh first." >&2
  exit 2
fi

gcloud compute scp "$SCRIPT_DIR/vm_train_aic_lewm.sh" "$GCP_INSTANCE:~/vm_train_aic_lewm.sh" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

remote_stablewm_export=""
if [[ -n "$REMOTE_STABLEWM_HOME" ]]; then
  remote_stablewm_export="export STABLEWM_HOME='$(quote_for_remote "$REMOTE_STABLEWM_HOME")'"
fi

remote_command="
set -euo pipefail
chmod +x ~/vm_train_aic_lewm.sh
export AIC_DATASET_GCS_URI='$(quote_for_remote "$AIC_DATASET_GCS_URI")'
export AIC_DATASET_NAME='$(quote_for_remote "$AIC_DATASET_NAME")'
export GCS_OUTPUT_PREFIX='$(quote_for_remote "$GCS_OUTPUT_PREFIX")'
export LEWM_RUN_NAME='$(quote_for_remote "$LEWM_RUN_NAME")'
export LEWM_MAX_EPOCHS='$(quote_for_remote "$LEWM_MAX_EPOCHS")'
export LEWM_BATCH_SIZE='$(quote_for_remote "$LEWM_BATCH_SIZE")'
$remote_stablewm_export
~/vm_train_aic_lewm.sh
"

gcloud compute ssh "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command "$remote_command"
