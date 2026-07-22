#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Phase 1: Per-fold computation"
echo "  Split: $MASTER_SPLIT"
echo "=============================================="

if [ -z "$MASTER_SPLIT" ]; then
    echo "ERROR: MASTER_SPLIT must be set."
    exit 1
fi

# Pretrained method: aggregate runs and summarise
declare -ax MASTER_APPS=("nnintv1")
declare -ax MASTER_RUN_NUMS=("1" "2" "3")
source ./result_summarisation/results_aggregation_run.sh

declare -ax MASTER_RUN_NUMS=("1" "2" "3" "-aggregated")
source ./result_summarisation/standard_metric_summary_run.sh
source ./oracle_metrics/num_of_interactions_run.sh

# Adaptive apps: per-run episodic metrics
declare -ax MASTER_APPS=("adatest2")
declare -ax MASTER_RUN_NUMS=("1" "2" "3")
source ./result_summarisation/standard_metric_summary_run.sh
source ./oracle_metrics/num_of_interactions_run.sh

# Episode-level aggregation
source ./result_summarisation/results_aggregation_run.sh
source ./result_summarisation/results_aggregation_run_final_episode.sh

declare -ax MASTER_RUN_NUMS=("-aggregated")
source ./result_summarisation/standard_metric_summary_run.sh
source ./oracle_metrics/num_of_interactions_run.sh

declare -ax MASTER_FINAL_EPISODE_ONLY_FLAG=true
source ./result_summarisation/standard_metric_summary_run.sh
source ./oracle_metrics/num_of_interactions_run.sh
unset MASTER_FINAL_EPISODE_ONLY_FLAG

# Pseudotime: merge episodes, aggregate, compute AUC
declare -ax MASTER_APPS=("nnintv1" "adatest2")
declare -ax MASTER_RUN_NUMS=("1" "2" "3")
source ./trajectory_metrics/merge_episodes_run.sh

declare -ax MASTER_RUN_NUMS=("1" "2" "3")
source ./trajectory_metrics/pseudotime_results_aggregation_run.sh

declare -ax MASTER_RUN_NUMS=("-aggregated")
source ./trajectory_metrics/pseudotime_auc_run.sh

# Per-fold trajectory plots (for reference)
declare -ax MASTER_RUN_NUMS=("-aggregated")
declare -ax MASTER_APPS=("nnintv1" "adatest2")
# This script is `source`d once per fold from call_summarisation_stack_crossfold.sh, so
# any change to MASTER_CONFIG_NAME here would otherwise leak back into the caller and
# corrupt the CONFIG_NAME that cross_fold_run.sh (Phase 2) reads afterwards. Save/restore
# around the plot-only override.
_PHASE1_SAVED_CONFIG_NAME="$MASTER_CONFIG_NAME"
declare -ax MASTER_CONFIG_NAME=$CONF_NAME"_traj"
source ./trajectory_metrics/plot_pseudotime_auc_run.sh
source ./trajectory_metrics/plot_pseudotime_auc_selection_run.sh
declare -ax MASTER_CONFIG_NAME="$_PHASE1_SAVED_CONFIG_NAME"
unset _PHASE1_SAVED_CONFIG_NAME

echo "=============================================="
echo "Phase 1 complete for split: $MASTER_SPLIT"
echo "=============================================="
