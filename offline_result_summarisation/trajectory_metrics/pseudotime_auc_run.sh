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
ORIGINAL_RUN_NUMS=("${MASTER_ORIGINAL_RUN_NUMS[@]}") #("1" "2" "3") #Specify the original run numbers that were used to generate the aggregated results
#This is intended so that we can extract some variability bounds on pseudo-time AUC.
RUN_NUMS=("${MASTER_RUN_NUMS[@]}") #("-aggregated") 

SPLIT_NAME=$MASTER_SPLIT
REFERENCE_FILE="all_pseudotime_metrics.csv"

QUANTILE="0.5"
METRICS_CONFIG='{"all_auc_summaries.csv": {"Dice_auc_mean": null, "NSD_auc_mean": null}, "all_iteration_summaries.csv": {"Dice_mean": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}, "NSD_mean": {"rows": ["Interactive Init", "Interactive Edit Iter 100"]}}, "summarised_num_interactions_fitting.csv": {"Normalised_Mean_NOI": {"rows": ["quantile_Q = '${QUANTILE}'"]}, "Failure_Cases_Fraction": {"rows": ["quantile_Q = '${QUANTILE}'"]}}}'


#Now we do per-run. 
for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  # DATASET_ID=${DATASET_IDS[$index]};
  for APP in "${APPS[@]}"; do
    echo "=============================================="
    echo "Processing trajectory aucs for dataset $DATASET_NAME application: $APP"
    echo "=============================================="

    for RUN_NUM in "${ORIGINAL_RUN_NUMS[@]}"; do
      ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME";
      EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM";
      OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
      OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME/$OUTPUT_SUBPATH/pseudotime_metrics";
      ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/pseudotime_metrics";
      echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT";
      echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
      echo filename: "$REFERENCE_FILE";
      if [ ! -d "$ALGORITHM_RESULTS_ROOT" ]; then
        echo "Skipping: $ALGORITHM_RESULTS_ROOT does not exist."
        continue
      fi
      python3 $SCRIPT_DIR/"pseudotime_auc_calc.py" \
        --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
        --output_folder_root="$OUTPUT_RESULT_ROOT" \
        --filename="$REFERENCE_FILE"
      done;
    
    for RUN_NUM in "${RUN_NUMS[@]}"; do
      if [ "$RUN_NUM" == "-aggregated" ]; then
        ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME";
        EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
      else
        #Raise Exception here because we should have already calculated the trajectory AUCs for the individual runs in the previous loop, and we don't want to accidentally recalculate them here.
        echo "Error: For non-aggregated runs, please run the script with the original run numbers specified in the previous loop. This is to ensure that we have the individual run trajectory AUCs calculated for sanity checking and variability estimation purposes. Exiting."
        exit 1
      fi

      AGGREGATE_EXPERIMENT_NAMES=()
      for original_run_num in "${ORIGINAL_RUN_NUMS[@]}"; do
          AGGREGATE_EXPERIMENT_NAMES+=("$APP/$DATASET_NAME/$PROMPTER/run$original_run_num")
      done
      
      OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
      OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME/$OUTPUT_SUBPATH/pseudotime_metrics";
      ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/pseudotime_metrics";
      REFERENCE_RESULTS_ROOTS=()
      for EXPERIMENT in "${AGGREGATE_EXPERIMENT_NAMES[@]}"; do
          REFERENCE_RESULTS_ROOTS+=("$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT/pseudotime_metrics")
      done

        echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT";
        echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
        echo filename: "$REFERENCE_FILE";
        echo REFERENCE EXPERIMENT ROOTS FOR AGGREGATED RUN: "${REFERENCE_RESULTS_ROOTS[@]}";
        if [ ! -d "$ALGORITHM_RESULTS_ROOT" ]; then
          if [ "$RUN_NUM" == "-aggregated" ]; then
            echo "Error: Aggregated results root $ALGORITHM_RESULTS_ROOT does not exist. Please run the pseudotime results aggregation script before running this script with the aggregated run number. Exiting."
            exit 1
          else
            echo "Error: Non-aggregated results root $ALGORITHM_RESULTS_ROOT does not exist"
            echo "Skipping: $ALGORITHM_RESULTS_ROOT does not exist." #Note that this only triggers when 
            #we are lookjing at a non-run aggregated run.
            continue
          fi
        fi
        python3 $SCRIPT_DIR/"pseudotime_auc_calc.py" \
          --aggregated_experiment_roots ${REFERENCE_RESULTS_ROOTS[@]} \
          --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
          --output_folder_root="$OUTPUT_RESULT_ROOT" \
          --filename="$REFERENCE_FILE";
    done;
  done;
done;

