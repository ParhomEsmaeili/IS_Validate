#!/bin/bash

#Doubles as both a per-run summarisation script, and as a aggregated results summarisation script, depending on the "run_nums" arg.
# Set your arguments here
DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010") 
# APP=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
APP=("nnintv1")
PROMPTER="pointsonly"
RUN_NUMS=("-aggregated") #("1") #"2" "3")
INFER_INFO='{"init": "Interactive Init", "edit": 100}'
# SPLIT_NAME="holdout_set"
SPLIT_NAME="designset"
REFERENCE_FILE="cross_class_scores.csv"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  for APP in "${APP[@]}"; do
    echo "Processing application: $APP"
    for RUN_NUM in "${RUN_NUMS[@]}"; do
      if [ "$RUN_NUM" == "-aggregated" ]; then
        ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results_Summary/$SPLIT_NAME";
        EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
      else
        ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results/$SPLIT_NAME/$DATASET_NAME";
        EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
      fi
      OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
      OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/MIDL_DGX_Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
      ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics";
      echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT";
      echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
      echo filename: "$REFERENCE_FILE";
      python3 "standard_metric_summarisation.py" \
          --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
          --output_folder_root="$OUTPUT_RESULT_ROOT" \
          --filename="$REFERENCE_FILE" \
          --infer_info="$INFER_INFO"; 
      done;
  done;
done;