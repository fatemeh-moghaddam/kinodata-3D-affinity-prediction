#!/usr/bin/env bash
# Move every probe-result directory, data/probing/<gnn>/rmsd_cutoff_<x>/<split>/<target>/,
# into data/probing_archive/<timestamp>/ with the same layout. That takes the tuned
# best params (reports/*_best_params.json, shared_best_params/), predictions, summaries
# and experiments/summary_runs.csv with it, so the next probe run starts clean.
#
# Kept in place: the extraction outputs (layer_*.pt, ids.pt, fold dirs 0-4,
# manifest.json, predictions.csv) and data/probing/probe_split.csv.
#
# Dry run by default (only lists what would move). To actually move:
#   DRY_RUN=0 bash prob/cluster/archive_probe_results.sh
set -euo pipefail

# PROJ="${HOME_PROJ_DIR:-$HOME/kinodata-3D-affinity-prediction}"
# for running on local
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$PROJ/data/probing"
DEST="$PROJ/data/probing_archive/$(date +%Y%m%d_%H%M%S)"
DRY_RUN="${DRY_RUN:-1}"

n=0
while IFS= read -r -d '' dir; do
  rel="${dir#"$SRC"/}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "would move: $rel"
  else
    mkdir -p "$DEST/$(dirname "$rel")"
    mv "$dir" "$DEST/$rel"
    echo "moved: $rel"
  fi
  n=$((n + 1))
done < <(find "$SRC" -mindepth 4 -maxdepth 4 -type d \
           -path "$SRC/*/rmsd_cutoff_*/*-k-fold/*" ! -name '[0-9]' -print0)

if [ "$DRY_RUN" = 1 ]; then
  echo "$n directories would move to $DEST (dry run; rerun with DRY_RUN=0)"
else
  echo "$n directories moved to $DEST"
  echo "best_params files left under data/probing: $(find "$SRC" -name '*_best_params.json' | wc -l)"
fi
