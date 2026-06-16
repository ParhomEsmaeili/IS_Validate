#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

# Set your arguments here

#Experiment configuration variables.
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
DATASET_IDS=("${MASTER_DATASET_IDS[@]}")
# APPS=("sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
# APPS=("nnintv1" "adadesign1")
PROMPTER="pointsonly"
echo "=============================================="
echo "Running Algorithm-wise Number of Samples Metrics Extraction"
APPS=("${MASTER_APPS[@]}")
SPLIT_NAME=$MASTER_SPLIT
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

NNUNET_ROOT_PATH="$MASTER_NNUNET_METRICS_ROOT/$MASTER_NNUNET_METRICS_SUBFOLDER"
# RUN_NUMS=("1" "2" "3")
RUN_NUMS=("${MASTER_RUN_NUMS[@]}") #("1" "2" "3" "-aggregated")

NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}") #("quantile")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}") #("0.5")
EPOCH=("${MASTER_PSEUDOTIME_NOS_EPOCH[@]}") #("Interactive Edit Iter 100")
PSEUDOTIME_METRICS_CONFIG="{\"Dice\": [\"${EPOCH[0]}\"], \"NSD\": [\"${EPOCH[0]}\"]}"
# PSEUDOTIME_REFERENCE_METRIC="all_pseudotime_metrics.csv"
NNUNET_REFERENCE_FILE="cross_class_scores.csv"
REFERENCE_COLUMN="Automatic Init"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  for APP in "${APPS[@]}"; do
    for RUN_NUM in "${RUN_NUMS[@]}"; do
      if [ "$RUN_NUM" == "-aggregated" ]; then
        ALGORITHM_RESULTS_ROOT_PATH="$MASTER_RESULTS_ROOT/Results_Pseudotime/$SPLIT_NAME";
        EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM";  
        OUTPUT_RESULT_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/pseudotime_metrics";
        ALGORITHM_RESULTS_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/pseudotime_metrics"
        REFERENCE_RESULT_ROOT="$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics"
          python3 "$SCRIPT_DIR/num_of_samples.py" \
              --algo_results_path="$ALGORITHM_RESULTS_PATH" \
              --output_result_root="$OUTPUT_RESULT_ROOT" \
              --metrics_config="$PSEUDOTIME_METRICS_CONFIG" \
              --reference_root="$REFERENCE_RESULT_ROOT" \
              --reference_filename="$NNUNET_REFERENCE_FILE" \
              --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
              --nnunet_bound "${NNUNET_BOUND[@]}";
      else
        #Raise exception, single runs can be too volatile to obtain a meaningful metric here.
        echo "Passed in a non-aggregated run num for num_of_samples extraction, this is not intended as single runs can be too volatile to obtain a meaningful metric here. Exiting." 
        exit 1
      fi
      
    done;
  done;
done;