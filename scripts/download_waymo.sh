#!/usr/bin/env bash
# Download a few Waymo Open Dataset v2 segments (Perception) for one split.
# Needs: Waymo Open Dataset terms accepted with your Google account, `gcloud auth login`, and gsutil.
#
#   ./scripts/download_waymo.sh list                 # show first 20 segment ids
#   ./scripts/download_waymo.sh get <segment> [...]  # download components for those segments
set -euo pipefail
BUCKET="gs://waymo_open_dataset_v_2_0_0"
SPLIT="${SPLIT:-training}"
DEST="${DEST:-data/waymo_v2/${SPLIT}}"

case "${1:-}" in
  list)
    gsutil ls "${BUCKET}/${SPLIT}/stats/" | head -20 | sed -E 's#.*/([^/]+)\.parquet#\1#'
    ;;
  get)
    shift
    for seg in "$@"; do
      for comp in camera_image stats camera_box; do
        mkdir -p "${DEST}/${comp}"
        gsutil cp "${BUCKET}/${SPLIT}/${comp}/${seg}.parquet" "${DEST}/${comp}/" || echo "WARN: ${comp}/${seg} not available"
      done
    done
    echo "next: python -m waymo_enrich extract --root ${DEST} --segment <id> --out data/clips"
    ;;
  *)
    echo "usage: $0 list | get <segment> [<segment> ...]"; exit 1 ;;
esac
