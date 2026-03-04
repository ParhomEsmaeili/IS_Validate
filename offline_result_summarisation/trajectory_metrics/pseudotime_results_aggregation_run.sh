#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
# Set your arguments here
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
DATASET_IDS=("${MASTER_DATASET_IDS[@]}")
# APPS=("nnintv1" "adadesign1" "adadesign2")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1")

echo "=============================================="
echo "Running Algorithm-wise Result Aggregation"
APPS=("${MASTER_APPS[@]}")
SPLIT_NAME=$MASTER_SPLIT
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

# SPLIT_NAME="designset"
# SPLIT_NAME="holdoutset"
PROMPTER="pointsonly"
RUN_NUMS=("${MASTER_RUN_NUMS[@]}") #("1" "2" "3")
REFERENCE_FILE="all_pseudotime_metrics.csv"
#
for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME"

  for APP in "${APPS[@]}"; do
    echo "=============================================="
    echo "Processing application: $APP"
    echo "=============================================="
    EXPERIMENT_BASENAME="$APP/$DATASET_NAME/$PROMPTER";
    ALGORITHM_RESULTS_ROOTS=();
    for RUN_NUM in "${RUN_NUMS[@]}"; do
      EXPERIMENT_NAME="$EXPERIMENT_BASENAME/run$RUN_NUM";
      CANDIDATE_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/pseudotime_metrics/$REFERENCE_FILE";
      
      # Check if path exists before adding
      if [ -f "$CANDIDATE_PATH" ]; then
        echo "Found run: $RUN_NUM at $CANDIDATE_PATH"
        ALGORITHM_RESULTS_ROOTS+=("$CANDIDATE_PATH");
      else
        echo "Skipping run $RUN_NUM (not found)"
      fi
    done;
    
    # Only process if we found at least one valid run
    if [ ${#ALGORITHM_RESULTS_ROOTS[@]} -gt 0 ]; then
      echo ALGORITHM_RESULTS_ROOTS: "${ALGORITHM_RESULTS_ROOTS[@]}"
      OUTPUT_SUBPATH="$EXPERIMENT_BASENAME/run-aggregated"
      OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME/$OUTPUT_SUBPATH";

      python3 $SCRIPT_DIR/"pseudotime_result_aggregation.py" \
          --algorithm_result_roots "${ALGORITHM_RESULTS_ROOTS[@]}" \
          --output_result_root="$OUTPUT_RESULT_ROOT";
    else
      echo "No valid runs found for $DATASET_NAME"
    fi
  done;
done;