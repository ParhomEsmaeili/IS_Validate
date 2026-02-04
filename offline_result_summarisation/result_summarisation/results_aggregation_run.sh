#!/usr/bin/env bash

# Set your arguments here
DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010") 
APPS=("nnintv1" "adadesign1")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1")
SPLIT_NAME="designset"
# SPLIT_NAME="holdoutset"
PROMPTER="pointsonly"
RUN_NUMS=("1" "2" "3")
INFER_INFO='{"init": "Interactive Init", "edit": 100}'
METRICS=("Dice" "NSD")
REFERENCE_FILE="cross_class_scores.csv"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME"
  for METRIC in "${METRICS[@]}"; do
    for APP in "${APPS[@]}"; do
      echo "Processing application: $APP"
      EXPERIMENT_BASENAME="$APP-dataset${DATASET_ID}-$PROMPTER-run";
      ALGORITHM_RESULTS_ROOTS=();
      for RUN_NUM in "${RUN_NUMS[@]}"; do
        EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
        CANDIDATE_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics"
        
        # Check if path exists before adding
        if [ -d "$CANDIDATE_PATH" ]; then
          echo "Found run: $RUN_NUM at $CANDIDATE_PATH"
          ALGORITHM_RESULTS_ROOTS+=("$CANDIDATE_PATH");
        else
          echo "Skipping run $RUN_NUM (not found)"
        fi
      done;
      
      # Only process if we found at least one valid run
      if [ ${#ALGORITHM_RESULTS_ROOTS[@]} -gt 0 ]; then
        echo ALGORITHM_RESULTS_ROOTS: "${ALGORITHM_RESULTS_ROOTS[@]}"
        OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run-aggregated"
        OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";

        python3 "result_aggregation.py" \
            --algorithm_result_roots "${ALGORITHM_RESULTS_ROOTS[@]}" \
            --output_result_root="$OUTPUT_RESULT_ROOT" \
            --metric="$METRIC" \
            --filename="$REFERENCE_FILE" \
            --infer_info="$INFER_INFO"; 
      else
        echo "No valid runs found for $DATASET_NAME"
      fi
    done;
  done;
done;