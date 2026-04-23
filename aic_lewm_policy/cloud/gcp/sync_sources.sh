#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_instance

if [[ ! -d "$LEWM_LOCAL_ROOT" ]]; then
  echo "LEWM_LOCAL_ROOT does not exist: $LEWM_LOCAL_ROOT" >&2
  exit 2
fi

if [[ ! -f "$AIC_ROOT/aic_lewm_policy/lewm_configs/aic_data.yaml" ]]; then
  echo "Missing AIC LEWM data config under $AIC_ROOT." >&2
  exit 2
fi

gcloud compute ssh "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command "rm -rf ~/src/le-wm && mkdir -p ~/src"

gcloud compute scp --recurse "$LEWM_LOCAL_ROOT" "$GCP_INSTANCE:~/src" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

gcloud compute ssh "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command "mkdir -p ~/src/le-wm/config/train/data"

gcloud compute scp "$AIC_ROOT/aic_lewm_policy/lewm_configs/aic_data.yaml" \
  "$GCP_INSTANCE:~/src/le-wm/config/train/data/aic.yaml" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

echo "Synced $LEWM_LOCAL_ROOT and installed config/train/data/aic.yaml."
