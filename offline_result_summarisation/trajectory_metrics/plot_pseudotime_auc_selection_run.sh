#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

#Doubles as both a per-run summarisation script, and as a aggregated results summarisation script, depending on the "run_nums" arg.
# Set your arguments here
DATASET_NAMES=("Dataset008_HepaticVessel" "Dataset001_BrainTumour" "Dataset004_Hippocampus") #("${MASTER_DATASET_NAMES[@]}")
# DATASET_IDS=("${MASTER_DATASET_IDS[@]}")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
# APPS=("nnintv1" "adadesign1")

echo "=============================================="
echo "Running Algorithm-wise Pseudo-time AUC Plotter for aggregated runs for a selection of tasks to join together."
APPS=("${MASTER_APPS[@]}")
SPLIT_NAME=$MASTER_SPLIT

NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}")
DICE_AUC_STATISTIC=("${MASTER_DICE_AUC_STATISTIC[@]}")
NSD_AUC_STATISTIC=("${MASTER_NSD_AUC_STATISTIC[@]}")
DICE_ITERATION_STATISTIC=("${MASTER_DICE_ITERATION_STATISTIC[@]}")
NSD_ITERATION_STATISTIC=("${MASTER_NSD_ITERATION_STATISTIC[@]}")
NOI_STATISTIC=("${MASTER_NOI_STATISTIC[@]}")
METRICS_CONFIG="{\"Dice_auc_${DICE_AUC_STATISTIC[0]}\": null, \"NSD_auc_${NSD_AUC_STATISTIC[0]}\": null, \"Dice_${DICE_ITERATION_STATISTIC[0]}\": [\"Interactive Init\", \"Interactive Edit Iter 100\"], \"NSD_${NSD_ITERATION_STATISTIC[0]}\": [\"Interactive Init\", \"Interactive Edit Iter 100\"], \"Normalised_${NOI_STATISTIC[0]^}_NOI\": null, \"Failure_Cases_Fraction\": null}"
EPOCH=("${MASTER_PSEUDOTIME_NOS_EPOCH[@]}") 



# APPS=("nnintv1" "adadesign1" "adadesign2") #("${MASTER_APPS[@]}")
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "  Metric statistics chosen: $METRICS_CONFIG"
echo "=============================================="

PROMPTER="pointsonly"

#We actually keep these separate here because we need have a sanity check where we enforce that
#the trajectory AUC on aggregated run matches the mean of the trajectory AUCs for all of the original runs.
RUN_NUMS=$MASTER_RUN_NUMS #("-aggregated") #("1") #"2" "3")
REFERENCE_FILE="all_pseudotime_metrics.csv"
TRAJECTORY_REFERENCE_FILE="all_trajectory_aucs.csv"
# VARIABILITY_SUBPATH="pseudotime/variability/" #We can introduce this later. not relevant for NOW.


NNUNET_ROOT_PATH="$MASTER_NNUNET_METRICS_ROOT/$MASTER_NNUNET_METRICS_SUBFOLDER"
NNUNET_REFERENCE_FILE="cross_class_scores.csv"
REFERENCE_COLUMN="Automatic Init"

CONFIG_NAME="$MASTER_CONFIG_NAME"

  echo "=============================================="
  echo "Processing applications: ${APPS[@]}" for runs on dataset $DATASET_NAME for pseudotime plotting
  echo "=============================================="
  for RUN_NUM in "${RUN_NUMS[@]}"; do
      ALGORITHM_RESULTS_ROOT_PATH="$MASTER_RESULTS_ROOT/Results_Pseudotime/$SPLIT_NAME";
      if [ "$RUN_NUM" == "-aggregated" ]; then
        EXPERIMENT_NAMES=()
        for DATASET_NAME in "${DATASET_NAMES[@]}"; do
          EXPERIMENT_NAMES+=("$DATASET_NAME/$PROMPTER/run$RUN_NUM")
        done;
        # EXPERIMENT_NAME="$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
      else
        exit 1 #We should only be plotting the aggregated runs, as the single runs can be too volatile to obtain a meaningful plot here.
      fi
      OUTPUT_SUBPATH="$SPLIT_NAME/$PROMPTER/run$RUN_NUM/$CONFIG_NAME/selection_plot" #We can specify the config name here to avoid overwriting when we run different configurations of the plot.
      OUTPUT_RESULT_ROOT="$MASTER_RESULTS_ROOT/Results_Plotting/$OUTPUT_SUBPATH" #$OUTPUT_SUBPATH"
      REFERENCE_RESULT_ROOTS=()
      for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        REFERENCE_RESULT_ROOTS+=("$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics")
      done
      echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT_PATH"; #Route to the base folder containing results
      #across all algorithms/
      echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
      echo linegraph_filename: "$REFERENCE_FILE";
      echo DATASET_NAMES: "${DATASET_NAMES[@]}";
      echo APPS: "${APPS[@]}";
      python3 $SCRIPT_DIR/"plot_pseudotime_auc_selection.py" \
          --algorithm_results_root="$ALGORITHM_RESULTS_ROOT_PATH" \
          --experiment_names "${EXPERIMENT_NAMES[@]}" \
          --apps ${APPS[@]} \
          --output_folder_root="$OUTPUT_RESULT_ROOT" \
          --linegraph_filename="$REFERENCE_FILE" \
          --trajectory_filename="$TRAJECTORY_REFERENCE_FILE" \
          --metrics_config="$METRICS_CONFIG" \
          --nnunet_result_root "${REFERENCE_RESULT_ROOTS[@]}" \
          --nnunet_reference_filename="$NNUNET_REFERENCE_FILE" \
          --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
          --nnunet_bound "${NNUNET_BOUND[@]}" \
          --nos_epoch "${EPOCH[0]}";
  done;