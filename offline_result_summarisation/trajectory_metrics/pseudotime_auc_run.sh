#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

#Doubles as both a per-run summarisation script, and as a aggregated results summarisation script, depending on the "run_nums" arg.
# Set your arguments here
# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
# DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010") 
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
DATASET_IDS=("${MASTER_DATASET_IDS[@]}")


# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
# APPS=("nnintv1" "adadesign1")

echo "=============================================="
echo "Running Algorithm-wise Pseudo-time AUC Calculations for aggregated runs"
APPS=("${MASTER_APPS[@]}")
SPLIT_NAME=$MASTER_SPLIT
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

PROMPTER="pointsonly"
RUN_NUMS=("${MASTER_RUN_NUMS[@]}") #("-aggregated") 

SPLIT_NAME=$MASTER_SPLIT
REFERENCE_FILE="all_pseudotime_metrics.csv"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  for APP in "${APPS[@]}"; do
    echo "=============================================="
    echo "Processing trajectory aucs for dataset $DATASET_NAME application: $APP"
    echo "=============================================="
    for RUN_NUM in "${RUN_NUMS[@]}"; do
      ALGORITHM_RESULTS_ROOT_PATH="$MASTER_RESULTS_ROOT/Results_Pseudotime/$SPLIT_NAME";
      EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM";
      OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
      OUTPUT_RESULT_ROOT="$MASTER_RESULTS_ROOT/Results_Pseudotime/$SPLIT_NAME/$OUTPUT_SUBPATH/pseudotime_metrics";
      ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/pseudotime_metrics";
      echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT"
      echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT"
      echo filename: "$REFERENCE_FILE"
      if [ ! -d "$ALGORITHM_RESULTS_ROOT" ]; then
        echo "Skipping: $ALGORITHM_RESULTS_ROOT does not exist."
        continue
      fi
      python3 "$SCRIPT_DIR/pseudotime_auc_calc.py" \
        --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
        --output_folder_root="$OUTPUT_RESULT_ROOT" \
        --filename="$REFERENCE_FILE"
    done;
  done;
done;

