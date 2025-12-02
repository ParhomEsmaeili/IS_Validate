#!/bin/bash

# Set your arguments here
SPLIT_NAME="designset"
REFERENCE_METRICS_ROOT="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results_Summary/$SPLIT_NAME"
QUANTILE="0.5"
# AXIS_OF_COMPLEXITY_NAME="Algorithm Complexity Variation in Image Voxel Count - Quantile ${QUANTILE}"
# DATASET_NAMES=("Dataset004_Hippocampus" "Dataset001_BrainTumour" "Dataset007_Pancreas"  "Dataset003_Liver" )

# AXIS_OF_COMPLEXITY_NAME="Algorithm Complexity Variation in Image Anisotropy - Quantile ${QUANTILE}"
# DATASET_NAMES=("Dataset005_Prostate" "Dataset001_BrainTumour") 

# AXIS_OF_COMPLEXITY_NAME="Algorithm Complexity Variation in target jaggedness or non-convexity - Quantile ${QUANTILE}"
# DATASET_NAMES=("Dataset005_Prostate" "Dataset001_BrainTumour" "Dataset008_HepaticVessel")

# AXIS_OF_COMPLEXITY_NAME="Algorithm Complexity Variation in target size - Quantile ${QUANTILE}"
# DATASET_NAMES=("Dataset006_Lung" "Dataset007_Pancreas" "Dataset003_Liver")

# AXIS_OF_COMPLEXITY_NAME="Algorithm Complexity Variation in target heterogeneity or heterogeneity in pathological presentation - Quantile ${QUANTILE}"
# DATASET_NAMES=("Dataset003_Liver" "Dataset010_Colon")

AXIS_OF_COMPLEXITY_NAME="All Tasks Visualised - Quantile $QUANTILE"
DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")



OUTPUT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_PrintedResults/$SPLIT_NAME/$AXIS_OF_COMPLEXITY_NAME"
CAPTION="${AXIS_OF_COMPLEXITY_NAME} on $SPLIT_NAME Comparison Table"
LABEL="tab:${AXIS_OF_COMPLEXITY_NAME}_${SPLIT_NAME}_comparison"

# APPS=("sammed2dv1" "sam2v1" "sammed3dv1" "segvolv1"  "nnintv1")
APPS=("nnintv1")
PROMPTER="pointsonly"
# RUN_NUMS=("1" "2" "3")
RUN_NAME="-aggregated"
EXPERIMENT_SUBPATHS=()
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    EXPERIMENT_SUBPATHS+=("$DATASET_NAME/$PROMPTER/run$RUN_NAME")
done
# METRICS_CONFIG='{"all_auc_summaries.csv": {"Dice_auc_median": null, "NSD_auc_median": null}, "all_iteration_summaries.csv": {"Dice_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}, "NSD_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}}, "summarised_num_interactions_fitting.csv": {"Normalised_Median_NOI": null, "Failure_Cases_Fraction": null}}'
METRICS_CONFIG='{"all_auc_summaries.csv": {"Dice_auc_median": null, "NSD_auc_median": null}, "all_iteration_summaries.csv": {"Dice_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}, "NSD_median": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}}, "summarised_num_interactions_fitting.csv": {"Normalised_Median_NOI": {"rows": ["quantile_Q = '${QUANTILE}'"]}, "Failure_Cases_Fraction": {"rows": ["quantile_Q = '${QUANTILE}'"]}}}'

# echo "${EXPERIMENT_SUBPATHS[@]}"
python3 "table_generator.py" \
    --metrics_root="$REFERENCE_METRICS_ROOT" \
    --algorithm_names "${APPS[@]}" \
    --experiment_subpath "${EXPERIMENT_SUBPATHS[@]}" \
    --metrics_config="$METRICS_CONFIG" \
    --output_root="$OUTPUT_ROOT" \
    --caption="$CAPTION" \
    --label="$LABEL"
  