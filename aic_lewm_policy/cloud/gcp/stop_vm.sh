#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_instance

gcloud compute instances stop "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

echo "Stopped $GCP_INSTANCE. Delete it with gcloud compute instances delete if you no longer need the disk."
