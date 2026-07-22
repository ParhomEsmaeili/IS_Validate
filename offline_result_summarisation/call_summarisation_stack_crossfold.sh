#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Cross-Fold Summarisation Stack"
echo "=============================================="

#=========================================
# Common configuration
#=========================================
export MASTER_RESULTS_ROOT="/home/parhomesmaeili/IS-Validation-Framework/MICCAI_2026_Storage"
export MASTER_NNUNET_METRICS_ROOT="/home/parhomesmaeili/Helmholtz Group/nnUNet_miccai_main_2026"
export MASTER_NNUNET_METRICS_SUBFOLDER="nnUNet_Metrics/holdoutset"
export MASTER_PROJECT_ROOT="/home/parhomesmaeili/IS-Validation-Framework"
export MASTER_PRETRAINED="nnintv1"

declare -ax MASTER_DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
declare -ax MASTER_DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010")

declare -ax MASTER_FOLDS=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")

CONF_NAME="adatest2_nnint_comparison_means"

declare -ax MASTER_NNUNET_STATISTIC=("gaussian")
declare -ax MASTER_NNUNET_BOUND=("0")
declare -ax MASTER_DICE_AUC_STATISTIC=("mean")
declare -ax MASTER_NSD_AUC_STATISTIC=("mean")
declare -ax MASTER_DICE_ITERATION_STATISTIC=("mean")
declare -ax MASTER_NSD_ITERATION_STATISTIC=("mean")
declare -ax MASTER_NOI_STATISTIC=("mean")
declare -ax MASTER_PSEUDOTIME_NOS_EPOCH=("Interactive Edit Iter 100")

export MASTER_METRIC_STATISTIC_CHOSEN="mean"
export MASTER_CONFIG_NAME=$CONF_NAME

declare -ax MASTER_RUN_NUMS=("1" "2" "3")

echo "Configuration:"
echo "  Folds: ${MASTER_FOLDS[@]}"
echo "  Datasets: ${MASTER_DATASET_NAMES[@]}"
echo "  Config Name: $CONF_NAME"
echo "=============================================="

#=========================================
# Phase 1: Per-fold computation
#=========================================
for FOLD in "${MASTER_FOLDS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Phase 1 for fold: $FOLD"
    echo "=============================================="
    export MASTER_SPLIT="$FOLD"
    source ./call_summarisation_stack_phase1.sh
    echo "✓ Phase 1 complete for fold: $FOLD"
done

#=========================================
# Phase 2: Cross-fold summarisation
#=========================================
echo ""
echo "=============================================="
echo "Phase 2: Cross-fold Summarisation"
echo "=============================================="

source ./cross_fold_summarisation/cross_fold_run.sh

echo ""
echo "=============================================="
echo "Cross-fold pipeline complete!"
echo "Results in: $MASTER_RESULTS_ROOT/Results_CrossSplit/$CONF_NAME"
echo "=============================================="
