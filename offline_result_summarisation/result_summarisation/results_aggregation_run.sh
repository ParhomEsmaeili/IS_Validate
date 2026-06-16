#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
# Set your arguments here
# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
# DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010") 
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
INFER_INFO='{"init": "Interactive Init", "edit": 100}'
METRICS=("Dice" "NSD")
REFERENCE_FILE="cross_class_scores.csv"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  EPISODES_ROOT_PATH="$MASTER_RESULTS_ROOT/Results/$SPLIT_NAME/$DATASET_NAME"
  for METRIC in "${METRICS[@]}"; do
    
    for APP in "${APPS[@]}"; do
      
      ALGORITHM_RESULTS_ROOT_PATH="$MASTER_RESULTS_ROOT/Results/$SPLIT_NAME/$DATASET_NAME"
  
      if [[ "$APP" == *"adatest"* ]]; then
        echo "========================================="
        echo "Checking variable number of episodes for app $APP under dataset $DATASET_NAME across runs"
        EPISODE_NUMS=();
        for RUN_NUM in "${RUN_NUMS[@]}"; do
          echo "Checking run: $RUN_NUM for app $APP under dataset $DATASET_NAME"
          # Extract episode numbers for each run using grep and sed
          EPISODE_STRINGS=$(ls -1 "$EPISODES_ROOT_PATH" 2>/dev/null | grep "$APP" | grep -oP "${APP}-episode[0-9]+.*-run${RUN_NUM}")
          EPISODE_NUM_LIST=$(echo "$EPISODE_STRINGS" | sed -n 's/.*-episode\([0-9]\+\).*\-run.*/\1/p' | sort -u | tr '\n' ' ')
          echo "Found run: $RUN_NUM with episode numbers: $EPISODE_NUM_LIST"
          EPISODE_NUMS+=("$EPISODE_NUM_LIST");
        done;
        # Check if all runs have the same set of episode numbers
        NORMALIZED_EPISODE_NUMS=()
        for ep_list in "${EPISODE_NUMS[@]}"; do
          norm=$(echo "$ep_list" | tr ' ' '\n' | sort -n | uniq | tr '\n' ' ' | sed 's/^ *//;s/ *$//')
          NORMALIZED_EPISODE_NUMS+=("$norm")
        done
        ref_list="${NORMALIZED_EPISODE_NUMS[0]}"
        for norm_list in "${NORMALIZED_EPISODE_NUMS[@]}"; do
          if [ "$norm_list" != "$ref_list" ]; then
            echo "Error: Not all runs have the same set of episode numbers for app $APP under dataset $DATASET_NAME. Found episode sets: ${NORMALIZED_EPISODE_NUMS[*]}. Please check the results folder and ensure consistency across runs. Exiting."
            exit 1
          fi
        done
        
        echo "=============================================="
        #Extract the matching episode app results from the app substring. 
        EPISODE_NUM=$(ls -1 "$EPISODES_ROOT_PATH" 2>/dev/null | grep "$APP" | grep -oP "${APP}-episode\d+" | sort -u | wc -l)
        echo "Processing application: $APP for dataset $DATASET_NAME for episode-wise aggregation with episodes:"
        echo "$EPISODE_NUM"
        if [ "$EPISODE_NUM" == "0" ]; then
          echo "Skipping because no episodes found for app $APP under dataset $DATASET_NAME"
          continue 1
        fi
        echo "=============================================="
      else
        echo "=============================================="
        echo "Processing application: $APP for dataset $DATASET_NAME for aggregation"
        EPISODE_NUM="0"
        echo "=============================================="
      
      fi
      
      if [ "$EPISODE_NUM" == "0" ]; then
        echo "==============================="
        echo "Running aggregation on non-episodic app: $APP"
        echo "==============================="
        EXPERIMENT_BASENAME="$APP-dataset${DATASET_ID}-$PROMPTER-run";
        ALGORITHM_RESULTS_ROOTS=();
        for RUN_NUM in "${RUN_NUMS[@]}"; do
          EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
          CANDIDATE_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics"
          echo "Checking for run: $RUN_NUM at $CANDIDATE_PATH"
          # Check if path exists before adding
          if [ -d "$CANDIDATE_PATH" ]; then
            echo "Found run: $RUN_NUM at $CANDIDATE_PATH"
            ALGORITHM_RESULTS_ROOTS+=("$CANDIDATE_PATH");
          else
            echo "Skipping run $RUN_NUM (not found)"
          fi
        done;
        
        # Only process if we found all valid run
        if [ ${#ALGORITHM_RESULTS_ROOTS[@]} -gt 0 ] && [ ${#ALGORITHM_RESULTS_ROOTS[@]} -eq ${#RUN_NUMS[@]} ]; then
          # #First lets move over the per-run results to results summary for ease of downstream use......
          # cp -r "${ALGORITHM_RESULTS_ROOTS[@]}" "$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME/$APP/$DATASET_NAME/$PROMPTER/"
          echo ALGORITHM_RESULTS_ROOTS: "${ALGORITHM_RESULTS_ROOTS[@]}"
          OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run-aggregated"
          OUTPUT_RESULT_ROOT="$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";

          python3 $SCRIPT_DIR/"result_aggregation.py" \
              --algorithm_result_roots "${ALGORITHM_RESULTS_ROOTS[@]}" \
              --output_result_root="$OUTPUT_RESULT_ROOT" \
              --metric="$METRIC" \
              --filename="$REFERENCE_FILE" \
              --infer_info="$INFER_INFO"; 
        else
          echo "No valid runs found for $DATASET_NAME"
        fi
      else
        echo "Running aggregation on episodic app: $APP for episode num: $EPISODE_NUM"
        for ((ADAPTATION_EPISODE=0; ADAPTATION_EPISODE<$EPISODE_NUM; ADAPTATION_EPISODE++)); do
          echo "Running aggregation on episodic app: $APP for episode $ADAPTATION_EPISODE"
          # EXPERIMENT_BASENAME="$APP-episode$ADAPTATION_EPISODE-dataset${DATASET_ID}-$PROMPTER-run";
          ALGORITHM_RESULTS_ROOTS=();
          for RUN_NUM in "${RUN_NUMS[@]}"; do
            EXPERIMENT_NAME="$APP-episode$ADAPTATION_EPISODE-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
            CANDIDATE_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics"
            
            # Check if path exists before adding
            if [ -d "$CANDIDATE_PATH" ]; then
              echo "Found run: $RUN_NUM at $CANDIDATE_PATH"
              ALGORITHM_RESULTS_ROOTS+=("$CANDIDATE_PATH");
            else
              echo "Skipping run $RUN_NUM (not found)"
            fi
          done;
          
          # Only process if we found all valid run
          if [ ${#ALGORITHM_RESULTS_ROOTS[@]} -gt 0 ] && [ ${#ALGORITHM_RESULTS_ROOTS[@]} -eq ${#RUN_NUMS[@]} ]; then
            # #First lets move over the per-run results to results summary for ease of downstream use......
            # cp -r "${ALGORITHM_RESULTS_ROOTS[@]}" "$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME/$APP/$DATASET_NAME/$PROMPTER/"
            echo ALGORITHM_RESULTS_ROOTS: "${ALGORITHM_RESULTS_ROOTS[@]}"
            OUTPUT_SUBPATH="$APP-episode$ADAPTATION_EPISODE/$DATASET_NAME/$PROMPTER/run-aggregated"
            OUTPUT_RESULT_ROOT="$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";

            python3 $SCRIPT_DIR/"result_aggregation.py" \
                --algorithm_result_roots "${ALGORITHM_RESULTS_ROOTS[@]}" \
                --output_result_root="$OUTPUT_RESULT_ROOT" \
                --metric="$METRIC" \
                --filename="$REFERENCE_FILE" \
                --infer_info="$INFER_INFO"; 
          else
            echo "No valid runs found for $DATASET_NAME"
          fi
        done;
      fi
    done;
  done;
done;