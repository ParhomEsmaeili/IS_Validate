#!/bin/bash
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
# Set your arguments here
# APPS=("sammed2dv1" "sam2v1" "sammed3dv1" "segvolv1"  "nnintv1")
APPS=("${MASTER_APPS[@]}")
# APPS=("nnintv1" "adadesign1")
CONFIG_NAME="${MASTER_CONFIG_NAME[@]}"

SPLIT_NAME=$MASTER_SPLIT

echo "=============================================="
echo "Running Result Table Generation"
echo "=============================================="
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}") #("quantile")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}") #("0.5")
#assert length of both are 1. 
if [ ${#NNUNET_STATISTIC[@]} -ne 1 ] || [ ${#NNUNET_BOUND[@]} -ne 1 ]; then
    echo "Error: NNUNET_STATISTIC and NNUNET_BOUND arrays must both have exactly one element. Current values: NNUNET_STATISTIC=${NNUNET_STATISTIC[@]}, NNUNET_BOUND=${NNUNET_BOUND[@]}"
    exit 1
fi
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

RANKING_ROOT="$MASTER_RESULTS_ROOT/Results_Ranking/$SPLIT_NAME/$CONFIG_NAME"
REFERENCE_METRICS_ROOT="$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME"
# QUANTILE="0.5"

AXIS_OF_COMPLEXITY_NAME="All Tasks Visualised, nnU-Net Threshold: $NNUNET_STATISTIC_NAME_CALL" #Quantile $QUANTILE"
# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")


OUTPUT_ROOT="$MASTER_RESULTS_ROOT/PrintedResults/$SPLIT_NAME/$CONFIG_NAME/$AXIS_OF_COMPLEXITY_NAME"
CAPTION="${AXIS_OF_COMPLEXITY_NAME} on $SPLIT_NAME Comparison Table"
LABEL="tab:${AXIS_OF_COMPLEXITY_NAME}_${SPLIT_NAME}_comparison_${CONFIG_NAME[@]}"


PROMPTER="pointsonly"
# RUN_NUMS=("1" "2" "3")
RUN_NAME="-aggregated"
EXPERIMENT_SUBPATHS=()
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    EXPERIMENT_SUBPATHS+=("$DATASET_NAME/$PROMPTER/run$RUN_NAME")
done
# METRICS_CONFIG='{"all_auc_summaries.csv": {"Dice_auc_median": null, "NSD_auc_median": null}, "all_iteration_summaries.csv": {"Dice_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}, "NSD_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}}, "summarised_num_interactions_fitting.csv": {"Normalised_Median_NOI": null, "Failure_Cases_Fraction": null}}'
# METRICS_CONFIG='{"all_auc_summaries.csv": {"Dice_auc_median": null, "NSD_auc_median": null}, "all_iteration_summaries.csv": {"Dice_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}, "NSD_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}}, "summarised_num_interactions_fitting.csv": {"Normalised_Median_NOI": {"rows": ["quantile_Q = '${QUANTILE}'"]}, "Failure_Cases_Fraction": {"rows": ["quantile_Q = '${QUANTILE}'"]}}}'

# METRICS_CONFIG="{\"all_auc_summaries.csv\": {\"Dice_auc_mean\": null, \"NSD_auc_mean\": null}, \"all_iteration_summaries.csv\": {\"Dice_mean\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}, \"NSD_mean\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}}, \"summarised_num_interactions_fitting.csv\": {\"Normalised_Mean_NOI\": {\"rows\": [\"quantile_Q = '${QUANTILE}'\"]}, \"Failure_Cases_Fraction\": {\"rows\": [\"quantile_Q = '${QUANTILE}'\"]}}}"
METRICS_CONFIG="{\"all_auc_summaries.csv\": {\"Dice_auc_${DICE_AUC_STATISTIC[0]}\": null, \"NSD_auc_${NSD_AUC_STATISTIC[0]}\": null}, \"all_iteration_summaries.csv\": {\"Dice_${DICE_ITERATION_STATISTIC[0]}\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}, \"NSD_${NSD_ITERATION_STATISTIC[0]}\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}}, \"summarised_num_interactions_fitting.csv\": {\"Normalised_${NOI_STATISTIC[0]^}_NOI\": {\"rows\": [\"${NNUNET_STATISTIC_NAME_CALL}\"]}, \"Failure_Cases_Fraction\": {\"rows\": [\"${NNUNET_STATISTIC_NAME_CALL}\"]}}}"


# echo "${EXPERIMENT_SUBPATHS[@]}"
python3 "$SCRIPT_DIR/table_generator.py" \
    --ranking_root="$RANKING_ROOT" \
    --metrics_root="$REFERENCE_METRICS_ROOT" \
    --algorithm_names "${APPS[@]}" \
    --experiment_subpath "${EXPERIMENT_SUBPATHS[@]}" \
    --metrics_config="$METRICS_CONFIG" \
    --output_root="$OUTPUT_ROOT" \
    --caption="$CAPTION" \
    --label="$LABEL"
  