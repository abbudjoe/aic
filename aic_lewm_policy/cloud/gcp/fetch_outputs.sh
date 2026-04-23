#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_gcp_project
require_bucket

OUTPUT_DIR="${OUTPUT_DIR:-$AIC_ROOT/artifacts/lewm-runs}"
mkdir -p "$OUTPUT_DIR"

gcloud storage cp --recursive \
  "${GCS_OUTPUT_PREFIX%/}/checkpoints/$LEWM_RUN_NAME" \
  "$OUTPUT_DIR/" \
  --project "$GCP_PROJECT"

echo "Fetched outputs into $OUTPUT_DIR/$LEWM_RUN_NAME."
