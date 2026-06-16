#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
# Set your arguments here
# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
APPS=("${MASTER_APPS[@]}")
METRIC_STATISTICS_CHOSEN=$MASTER_METRIC_STATISTIC_CHOSEN
# APPS=("nnintv1" "adadesign1")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1")
# SPLIT_NAME="designset"
# SPLIT_NAME="holdoutset"
SPLIT_NAME=$MASTER_SPLIT
PROMPTER="pointsonly"
RUN_NAME="-aggregated" 


CONFIG_NAME="$MASTER_CONFIG_NAME"
# Base paths
BASIC_RESULTS_SUMMARY_ROOT="$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME"
# TRAJ_RESULTS_SUMMARY_ROOT="$MASTER_RESULTS_ROOT/Results_Pseudotime/$SPLIT_NAME"
BASIC_STATS_SIG_OUTPUT_ROOT="$MASTER_RESULTS_ROOT/Results_StatSig/$SPLIT_NAME/$CONFIG_NAME"
# TRAJ_STATS_SIG_OUTPUT_ROOT="$MASTER_RESULTS_ROOT/Results_StatSig/$SPLIT_NAME/$CONFIG_NAME"
BASIC_RANKING_OUTPUT_ROOT="$MASTER_RESULTS_ROOT/Results_Ranking/$SPLIT_NAME/$CONFIG_NAME"
# TRAJ_RANKING_OUTPUT_ROOT="$MASTER_RESULTS_ROOT/Results_Ranking/$SPLIT_NAME/$CONFIG_NAME"
NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}") #("quantile")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}") #("0.5")
declare -A TRANSLATE_NNUNET_STATISTIC_TO_NAME
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.25"]="LQ"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.5"]="Median"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.75"]="UQ"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["gaussian_0"]="Mean"

NNUNET_STATISTIC_NAME_CALL="${TRANSLATE_NNUNET_STATISTIC_TO_NAME[${NNUNET_STATISTIC}_"${NNUNET_BOUND}"]}"

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

#SIG SUBPATHS
BASIC_SIG_METRICS_CONFIG="{\"NOI\": {\"subpath\": \"metrics/NOI/casewise_noi.csv\", \"confs\": {\"NOI\": {\"cols\": [\"NOI_Dice_thr_${NNUNET_STATISTIC_NAME_CALL}\"]}, \"NoF\": {\"cols\": [\"Fail_Dice_thr_${NNUNET_STATISTIC_NAME_CALL}\"]}}}, \"AUC\": {\"subpath\": \"metrics/AUC/casewise_aucs.csv\", \"confs\": {\"Dice_AUC\": {\"cols\": [\"Dice_auc_scores\"]}, \"NSD_AUC\": {\"cols\": [\"NSD_auc_scores\"]}}}, \"Dice\": {\"subpath\": \"metrics/Dice/cross_class_scores.csv\", \"confs\": {\"Dice Init.\": {\"cols\": [\"Interactive Init\"]}, \"Dice Interactive Edit Iter 100\": {\"cols\": [\"Interactive Edit Iter 100\"]}}}, \"NSD\": {\"subpath\": \"metrics/NSD/cross_class_scores.csv\", \"confs\": {\"NSD Init.\": {\"cols\": [\"Interactive Init\"]}, \"NSD Interactive Edit Iter 100\": {\"cols\": [\"Interactive Edit Iter 100\"]}}}}"
# TRAJ_SIG_METRICS_CONFIG="{\"NOI\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"NOI\": {\"cols\": [\"Normalised_${NOI_STATISTIC[0]^}_NOI\"]}, \"NoF\": {\"cols\": [\"Failure_Cases_Fraction\"]}}}, \"AUC\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"Dice_AUC\": {\"cols\": [\"Dice_auc_${DICE_AUC_STATISTIC[0]}\"]}, \"NSD_AUC\": {\"cols\": [\"NSD_auc_${NSD_AUC_STATISTIC[0]}\"]}}}, \"Dice\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"Dice Init.\": {\"cols\": [\"Dice_${DICE_ITERATION_STATISTIC[0]} Init.\"]}, \"Dice Interactive Edit Iter 100\": {\"cols\": [\"Dice_${DICE_ITERATION_STATISTIC[0]} Interactive Edit Iter 100\"]}}}, \"NSD\": {\"subpath\": \"all_pseudotime_metrics.csv\", \"confs\": {\"NSD Init.\": {\"cols\": [\"NSD_${NSD_ITERATION_STATISTIC[0]} Init.\"]}, \"NSD Interactive Edit Iter 100\": {\"cols\": [\"NSD_${NSD_ITERATION_STATISTIC[0]} Interactive Edit Iter 100\"]}}}}"

# TRAJ_SIG_METRICS_CONFIG="{\"all_auc_summaries.csv\": {\"Dice_auc_${DICE_AUC_STATISTIC[0]}\": null, \"NSD_auc_${NSD_AUC_STATISTIC[0]}\": null}, \"all_iteration_summaries.csv\": {\"Dice_${DICE_ITERATION_STATISTIC[0]}\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}, \"NSD_${NSD_ITERATION_STATISTIC[0]}\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}}, \"summarised_num_interactions_fitting.csv\": {\"Normalised_${NOI_STATISTIC[0]^}_NOI\": {\"rows\": [\"${NNUNET_STATISTIC_NAME_CALL}\"]}, \"Failure_Cases_Fraction\": {\"rows\": [\"${NNUNET_STATISTIC_NAME_CALL}\"]}}}"


SNAPSHOT_SIG_SUBPATHS=()
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    SNAPSHOT_SIG_SUBPATHS+=("$DATASET_NAME/$PROMPTER/run$RUN_NAME")
done    

#RANKING METRICS
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
echo "=============================================="
echo "Running Snapshot Algorithm-wise Significance Scoring"
echo "=============================================="
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "  Input Root: $BASIC_RESULTS_SUMMARY_ROOT"
echo "  Output Root: $BASIC_STATS_SIG_OUTPUT_ROOT"
echo "  Config Name: $CONFIG_NAME"
echo "=============================================="

# Run significance scoring for the snapshot.
python3 $SCRIPT_DIR/algo_wise_sig_score.py \
    --dataset_names "${DATASET_NAMES[@]}" \
    --metrics_root "$BASIC_RESULTS_SUMMARY_ROOT" \
    --metrics_config "${BASIC_SIG_METRICS_CONFIG}" \
    --algorithm_names "${APPS[@]}" \
    --experiment_subpath "${SNAPSHOT_SIG_SUBPATHS[@]}" \
    --output_root "$BASIC_STATS_SIG_OUTPUT_ROOT"
    
if [ $? -eq 0 ]; then
    echo "✓ Significance scoring completed successfully"
else
    echo "✗ Significance scoring failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Running Snapshot Algorithm Ranking Generation"
echo "=============================================="
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "  Input Root: $BASIC_STATS_SIG_OUTPUT_ROOT"
echo "  Output Root: $BASIC_RANKING_OUTPUT_ROOT"
echo "=============================================="



# Run ranking generation for snapshot
python3 $SCRIPT_DIR/generate_algo_rankings.py \
    --ref_root "$BASIC_STATS_SIG_OUTPUT_ROOT" \
    --metrics "${RANKING_METRICS[@]}" \
    --task_names "${DATASET_NAMES[@]}" \
    --algorithm_names "${APPS[@]}" \
    --output_root "$BASIC_RANKING_OUTPUT_ROOT"

if [ $? -eq 0 ]; then
    echo "✓ Ranking generation completed successfully"
else
    echo "✗ Ranking generation failed"
    exit 1
fi


echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Output locations for snapshot:"
echo "  Significance Scores: $BASIC_STATS_SIG_OUTPUT_ROOT"
echo "  Rankings: $BASIC_RANKING_OUTPUT_ROOT"
echo "=============================================="