#!/bin/bash

# Set your arguments here

#Experiment configuration variables.
DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010") 
# APPS=("sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
APPS=("nnintv1")
PROMPTER="pointsonly"
SPLIT_NAME="designset"
NNUNET_ROOT_PATH="/home/parhomesmaeili/Helmholtz Group/MICCAI2026_nnunet/nnUNet_Metrics_$SPLIT_NAME"
# RUN_NUMS=("1" "2" "3")
RUN_NUMS=("-aggregated")
INFER_INFO='{"init": "Interactive Init", "edit": 100}'

NNUNET_STATISTIC=("quantile" "quantile")
NNUNET_BOUND=("0.25" "0.5")
REFERENCE_METRIC="Dice"
REFERENCE_FILE="cross_class_scores.csv"
REFERENCE_COLUMN="Automatic Init"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results/$SPLIT_NAME/$DATASET_NAME"
  for APP in "${APPS[@]}"; do
    echo "Processing application: $APP"
    for RUN_NUM in "${RUN_NUMS[@]}"; do
      if [ "$RUN_NUM" == "-aggregated" ]; then
        ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results_Summary/$SPLIT_NAME";
        EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
      else
        ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results/$SPLIT_NAME/$DATASET_NAME";
        EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
      fi
      # EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
      OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
      OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
      ALGORITHM_RESULTS_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
      REFERENCE_RESULT_PATH="$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
      python3 "num_of_interaction.py" \
          --algo_results_path="$ALGORITHM_RESULTS_PATH" \
          --output_result_root="$OUTPUT_RESULT_ROOT" \
          --metric="$REFERENCE_METRIC" \
          --reference_path="$REFERENCE_RESULT_PATH" \
          --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
          --nnunet_bound "${NNUNET_BOUND[@]}" \
          --infer_info="$INFER_INFO"; 
      done;
  done;
done;