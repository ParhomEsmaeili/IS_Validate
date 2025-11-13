#!/bin/bash

# Set your arguments here

NNUNET_ROOT_PATH="/home/parhomesmaeili/Helmholtz Group/MIDL2025_nnunet/nnUNet_Metrics"
#Experiment configuration variables.
DATASET_NAME="Dataset010_Colon"
DATASET_ID="010" 
APP=("sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
PROMPTER="pointsonly"
RUN_NUMS=("1") #"2" "3")
ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results/$DATASET_NAME"
INFER_INFO='{"init": "Interactive Init", "edit": 100}'

NNUNET_STATISTIC="quantile"
NNUNET_BOUND="0.25"
REFERENCE_METRIC="Dice"
REFERENCE_FILE="cross_class_scores.csv"
REFERENCE_COLUMN="Automatic Init"

for APP in "${APP[@]}"; do
  echo "Processing application: $APP"
  for RUN_NUM in "${RUN_NUMS[@]}"; do
    EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
    OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results_Summary/$EXPERIMENT_NAME";
    ALGORITHM_RESULTS_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
    REFERENCE_RESULT_PATH="$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
    python3 "num_of_interaction.py" \
        --algo_results_path="$ALGORITHM_RESULTS_PATH" \
        --output_result_root="$OUTPUT_RESULT_ROOT" \
        --metric="$REFERENCE_METRIC" \
        --reference_path="$REFERENCE_RESULT_PATH" \
        --nnunet_statistic="$NNUNET_STATISTIC" \
        --nnunet_bound="$NNUNET_BOUND" \
        --infer_info="$INFER_INFO"; 
    done;
done;
