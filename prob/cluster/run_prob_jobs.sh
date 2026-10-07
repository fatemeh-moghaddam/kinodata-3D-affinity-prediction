#!/usr/bin/env bash
# Prints one queue line per job for run_prob.sub (read via `queue ... from bash run_prob_jobs.sh |`).
# Preview the job list with: bash run_prob_jobs.sh
# Columns: script, split, rmsd, gnn, target, run_linear, run_nonlinear, run_baseline, nonlinear_models
set -euo pipefail

SCRIPT=prob_orchestrate
SPLIT_TYPES=(scaffold-k-fold pocket-k-fold random-k-fold)
RMSD_CUTOFFS=(2 4 6)
GNN_MODELS=(CGNN-3D CGNN DTI)
TARGETS=(hb_score.pt hb_score_mw_weighted.pt hb_score_mw_subtracted.pt)
RUN_LINEAR=1
RUN_NONLINEAR=1
RUN_BASELINE=1
NONLINEAR_MODELS=mlp

for split_type in "${SPLIT_TYPES[@]}"; do
  for rmsd_cutoff in "${RMSD_CUTOFFS[@]}"; do
    for gnn_model in "${GNN_MODELS[@]}"; do
      for target in "${TARGETS[@]}"; do
        echo "${SCRIPT}, ${split_type}, ${rmsd_cutoff}, ${gnn_model}, ${target}, ${RUN_LINEAR}, ${RUN_NONLINEAR}, ${RUN_BASELINE}, ${NONLINEAR_MODELS}"
      done
    done
  done
done
