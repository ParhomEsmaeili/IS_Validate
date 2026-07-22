#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

echo "=============================================="
echo "Phase 2: Cross-fold Summarisation"
echo "=============================================="

FOLDS=(${MASTER_FOLDS[@]})
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
APPS=("${MASTER_APPS[@]}")
PROMPTER="pointsonly"
CONFIG_NAME="$MASTER_CONFIG_NAME"
METRIC_STATISTICS_CHOSEN=$MASTER_METRIC_STATISTIC_CHOSEN

NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}")
DICE_AUC_STATISTIC=("${MASTER_DICE_AUC_STATISTIC[@]}")
NSD_AUC_STATISTIC=("${MASTER_NSD_AUC_STATISTIC[@]}")
DICE_ITERATION_STATISTIC=("${MASTER_DICE_ITERATION_STATISTIC[@]}")
NSD_ITERATION_STATISTIC=("${MASTER_NSD_ITERATION_STATISTIC[@]}")
NOI_STATISTIC=("${MASTER_NOI_STATISTIC[@]}")

export EXP_DICE_AUC_STATISTIC="${DICE_AUC_STATISTIC[0]}"
export EXP_NSD_AUC_STATISTIC="${NSD_AUC_STATISTIC[0]}"
export EXP_DICE_ITERATION_STATISTIC="${DICE_ITERATION_STATISTIC[0]}"
export EXP_NSD_ITERATION_STATISTIC="${NSD_ITERATION_STATISTIC[0]}"
export EXP_NOI_STATISTIC="${NOI_STATISTIC[0]}"

declare -A TRANSLATE_NNUNET_STATISTIC_TO_NAME
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.25"]="LQ"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.5"]="Median"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.75"]="UQ"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["gaussian_0"]="Mean"
NNUNET_STATISTIC_NAME_CALL="${TRANSLATE_NNUNET_STATISTIC_TO_NAME[${NNUNET_STATISTIC}_"${NNUNET_BOUND}"]}"

METRICS_ROOT="$MASTER_RESULTS_ROOT/Results_Summary"
PSEUDOTIME_ROOT="$MASTER_RESULTS_ROOT/Results_Pseudotime"
OUTPUT_ROOT="$MASTER_RESULTS_ROOT/Results_CrossSplit/$CONFIG_NAME"

NNUNET_ROOT_PATH="$MASTER_NNUNET_METRICS_ROOT/$MASTER_NNUNET_METRICS_SUBFOLDER"

SNAPSHOT_SIG_METRICS_CONFIG="{\"NOI\": {\"subpath\": \"metrics/NOI/casewise_noi.csv\", \"confs\": {\"NOI\": {\"cols\": [\"NOI_Dice_thr_${NNUNET_STATISTIC_NAME_CALL}\"]}, \"NoF\": {\"cols\": [\"Fail_Dice_thr_${NNUNET_STATISTIC_NAME_CALL}\"]}}}, \"AUC\": {\"subpath\": \"metrics/AUC/casewise_aucs.csv\", \"confs\": {\"Dice_AUC\": {\"cols\": [\"Dice_auc_scores\"]}, \"NSD_AUC\": {\"cols\": [\"NSD_auc_scores\"]}}}, \"Dice\": {\"subpath\": \"metrics/Dice/cross_class_scores.csv\", \"confs\": {\"Dice Init.\": {\"cols\": [\"Interactive Init\"]}, \"Dice Interactive Edit Iter 100\": {\"cols\": [\"Interactive Edit Iter 100\"]}}}, \"NSD\": {\"subpath\": \"metrics/NSD/cross_class_scores.csv\", \"confs\": {\"NSD Init.\": {\"cols\": [\"Interactive Init\"]}, \"NSD Interactive Edit Iter 100\": {\"cols\": [\"Interactive Edit Iter 100\"]}}}}"

TRAJ_SIG_METRICS_CONFIG="{\"NOI\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"NOI\": {\"cols\": [\"Normalised_${NOI_STATISTIC[0]^}_NOI\"]}, \"NoF\": {\"cols\": [\"Failure_Cases_Fraction\"]}}}, \"AUC\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"Dice_AUC\": {\"cols\": [\"Dice_auc_${DICE_AUC_STATISTIC[0]}\"]}, \"NSD_AUC\": {\"cols\": [\"NSD_auc_${NSD_AUC_STATISTIC[0]}\"]}}}, \"Dice\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"Dice Init.\": {\"cols\": [\"Dice_${DICE_ITERATION_STATISTIC[0]} Init.\"]}, \"Dice Interactive Edit Iter 100\": {\"cols\": [\"Dice_${DICE_ITERATION_STATISTIC[0]} Interactive Edit Iter 100\"]}}}, \"NSD\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"NSD Init.\": {\"cols\": [\"NSD_${NSD_ITERATION_STATISTIC[0]} Init.\"]}, \"NSD Interactive Edit Iter 100\": {\"cols\": [\"NSD_${NSD_ITERATION_STATISTIC[0]} Interactive Edit Iter 100\"]}}}}"

RANKING_METRICS=(
    'NOI'
    'NoF'
    'Dice_AUC'
    'NSD_AUC'
    'Dice Init.'
    'Dice Interactive Edit Iter 100'
    'NSD Init.'
    'NSD Interactive Edit Iter 100'
)

echo "Configuration:"
echo "  Folds: ${FOLDS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "  Algorithms: ${APPS[@]}"
echo "  Metrics Root: $METRICS_ROOT"
echo "  Pseudotime Root: $PSEUDOTIME_ROOT"
echo "  Output Root: $OUTPUT_ROOT"
echo "=============================================="

# Step 1: Cross-fold significance and betterness
echo ""
echo "--- Step 1: Cross-fold Significance and Betterness ---"
python3 $SCRIPT_DIR/cross_fold_sig_and_ranking.py \
    --folds "${FOLDS[@]}" \
    --dataset_names "${DATASET_NAMES[@]}" \
    --algorithm_names "${APPS[@]}" \
    --metrics_root "$METRICS_ROOT" \
    --pseudotime_root "$PSEUDOTIME_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --prompter "$PROMPTER" \
    --snapshot_metrics_config "$SNAPSHOT_SIG_METRICS_CONFIG" \
    --trajectory_metrics_config "$TRAJ_SIG_METRICS_CONFIG"

if [ $? -ne 0 ]; then
    echo "ERROR: Cross-fold significance scoring failed."
    exit 1
fi
echo "✓ Cross-fold significance and betterness complete."

# Step 2: Cross-fold ranking
echo ""
echo "--- Step 2: Cross-fold Rankings ---"
python3 $SCRIPT_DIR/../ranking_and_stat_tests/generate_algo_rankings.py \
    --ref_root "$OUTPUT_ROOT" \
    --metrics "${RANKING_METRICS[@]}" \
    --task_names "${DATASET_NAMES[@]}" \
    --algorithm_names "${APPS[@]}" \
    --output_root "$OUTPUT_ROOT"

if [ $? -ne 0 ]; then
    echo "ERROR: Cross-fold ranking generation failed."
    exit 1
fi
echo "✓ Cross-fold rankings complete."

# Step 3: Cross-fold trajectory plots and NoS
echo ""
echo "--- Step 3: Cross-fold Trajectory Plots and NoS ---"
EPOCH=("${MASTER_PSEUDOTIME_NOS_EPOCH[@]}")
TRAJ_PLOT_METRICS_CONFIG="{\"Dice_auc_${DICE_AUC_STATISTIC[0]}\": null, \"NSD_auc_${NSD_AUC_STATISTIC[0]}\": null, \"Dice_${DICE_ITERATION_STATISTIC[0]}\": [\"Interactive Init\", \"Interactive Edit Iter 100\"], \"NSD_${NSD_ITERATION_STATISTIC[0]}\": [\"Interactive Init\", \"Interactive Edit Iter 100\"], \"Normalised_${NOI_STATISTIC[0]^}_NOI\": null, \"Failure_Cases_Fraction\": null}"

python3 $SCRIPT_DIR/cross_fold_trajectory_plots.py \
    --folds "${FOLDS[@]}" \
    --dataset_names "${DATASET_NAMES[@]}" \
    --algorithm_names "${APPS[@]}" \
    --pseudotime_root "$PSEUDOTIME_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --prompter "$PROMPTER" \
    --nnunet_root "$NNUNET_ROOT_PATH" \
    --nnunet_reference_filename "cross_class_scores.csv" \
    --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
    --nnunet_bound "${NNUNET_BOUND[@]}" \
    --metrics_config "$TRAJ_PLOT_METRICS_CONFIG" \
    --nos_epoch "${EPOCH[0]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Cross-fold trajectory plotting failed."
    exit 1
fi
echo "✓ Cross-fold trajectory plots complete."

# Step 4: Cross-fold tables
echo ""
echo "--- Step 4: Cross-fold Tables ---"
# Table generator's cross-fold mode reads directly from cross_fold_summary_*.csv, whose
# 'metric' values are the exact RANKING_METRICS names (as written by
# cross_fold_sig_and_ranking.py) -- so the table config must be that same flat name set,
# not the all_pseudotime_metrics.csv-style column names used by the trajectory plot config.
TABLE_METRICS_CONFIG=$(python3 -c "import json,sys; print(json.dumps({m: None for m in sys.argv[1:]}))" "${RANKING_METRICS[@]}")

python3 $SCRIPT_DIR/../table_generation/table_generator.py \
    --cross_fold_mode \
    --cross_fold_root "$OUTPUT_ROOT" \
    --ranking_root "$OUTPUT_ROOT" \
    --algorithm_names "${APPS[@]}" \
    --metrics_config "$TABLE_METRICS_CONFIG" \
    --output_root "$OUTPUT_ROOT/tables"

if [ $? -ne 0 ]; then
    echo "ERROR: Cross-fold table generation failed."
    exit 1
fi
echo "✓ Cross-fold tables complete."

echo ""
echo "=============================================="
echo "Phase 2 pipeline complete."
echo "Outputs in: $OUTPUT_ROOT"
echo "=============================================="
