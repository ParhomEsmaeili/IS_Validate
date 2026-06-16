#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
# Set your arguments here
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
DATASET_IDS=("${MASTER_DATASET_IDS[@]}")
# APPS=("nnintv1" "adadesign1" "adadesign2")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1")

echo "=============================================="
echo "Running Algorithm-wise Result Aggregation on FINAL episode across runs"
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
        #First lets assert that the episode strings cannot be empty.
        if [ ${#EPISODE_NUMS[@]} -eq 0 ]; then
          echo "Error: No episode numbers found for app $APP under dataset $DATASET_NAME across any runs. Please check the results folder and ensure that episodes are named in the format ${APP}-episode{{EPISODE_NUM}}-run{{RUN_NUM}}. Exiting."
          exit 1
        fi
        #Lets extract the final episode number for each run
        FINAL_EPISODE_NUMS=()
        for ep_list in "${EPISODE_NUMS[@]}"; do
          norm=$(echo "$ep_list" | tr ' ' '\n' | sort -n | uniq | tail -n 1)
          FINAL_EPISODE_NUMS+=("$norm")
        done
        echo "Final episode numbers for each run: ${FINAL_EPISODE_NUMS[*]}"

        #Lets assert that the length of the final episode numbers matches the number of runs we expect to aggregate over.
        if [ ${#FINAL_EPISODE_NUMS[@]} -ne ${#RUN_NUMS[@]} ]; then
          echo "Error: Number of final episode numbers (${#FINAL_EPISODE_NUMS[@]}) does not match number of runs (${#RUN_NUMS[@]}). Please check the results folder and ensure consistency across runs. Exiting."
          exit 1
        fi
        
        # Check if all runs have the same set of episode numbers
        NORMALIZED_EPISODE_NUMS=()
        for ep_list in "${EPISODE_NUMS[@]}"; do
          norm=$(echo "$ep_list" | tr ' ' '\n' | sort -n | uniq | tr '\n' ' ' | sed 's/^ *//;s/ *$//')
          NORMALIZED_EPISODE_NUMS+=("$norm")
        done
        ref_list="${NORMALIZED_EPISODE_NUMS[0]}"
        for norm_list in "${NORMALIZED_EPISODE_NUMS[@]}"; do
          if [ "$norm_list" != "$ref_list" ]; then
            echo "==============================="
            echo "Error: Not all runs have the same set of episode numbers for app $APP under dataset $DATASET_NAME. Found episode sets: ${NORMALIZED_EPISODE_NUMS[*]}. Please check the results folder and ensure consistency across runs. Exiting."
            echo "Please check that our script will still work for variable episode numbers, this code SHOULD WORK without a hard exit but we put it here for security for now"
            exit 1
            echo "==============================="
          fi
        done
       echo "=============================================="

      else
        echo "=============================================="
        echo "Called final-episode aggregation on non-episodic app: $APP for dataset $DATASET_NAME."
        echo "exiting..."
        exit 1
      fi
      
      if [ "$EPISODE_NUM" == "0" ]; then
        echo "Somehow ended up in a non-episodic app, exiting..."
        exit 1
      
      else
        echo "Running aggregation on episodic app: $APP for episode nums: ${FINAL_EPISODE_NUMS[*]} across runs"
        ALGORITHM_RESULTS_ROOTS=();
        for index in "${!FINAL_EPISODE_NUMS[@]}"; do
          ADAPTATION_EPISODE=${FINAL_EPISODE_NUMS[$index]}
          RUN_NUM=${RUN_NUMS[$index]}
          echo "Running aggregation on episodic app: $APP, pulling final episode: $ADAPTATION_EPISODE on run: $RUN_NUM"
          
          EXPERIMENT_NAME="$APP-episode$ADAPTATION_EPISODE-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
          CANDIDATE_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics"
            
          # Check if path exists before adding
          if [ -d "$CANDIDATE_PATH" ]; then
            echo "Found run: $RUN_NUM at $CANDIDATE_PATH"
            ALGORITHM_RESULTS_ROOTS+=("$CANDIDATE_PATH");
          else
            # echo "Skipping run $RUN_NUM (not found)"
            echo "Error: Expected to find results for run $RUN_NUM at $CANDIDATE_PATH but it does not exist. Please check the results folder and ensure that the expected path exists. Exiting."
            exit 1
          fi
        done;
          
        # Only process if we found all valid run
        if [ ${#ALGORITHM_RESULTS_ROOTS[@]} -gt 0 ] && [ ${#ALGORITHM_RESULTS_ROOTS[@]} -eq ${#RUN_NUMS[@]} ]; then
          # #First lets move over the per-run results to results summary for ease of downstream use......
          # cp -r "${ALGORITHM_RESULTS_ROOTS[@]}" "$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME/$APP/$DATASET_NAME/$PROMPTER/"
          echo ALGORITHM_RESULTS_ROOTS: "${ALGORITHM_RESULTS_ROOTS[@]}"
          OUTPUT_SUBPATH="$APP-episodefinal/$DATASET_NAME/$PROMPTER/run-aggregated"
          OUTPUT_RESULT_ROOT="$MASTER_RESULTS_ROOT/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";

          # Write reference paths JSON before aggregation (no trailing comma)
          REFERENCE_JSON_PATH="$OUTPUT_RESULT_ROOT/reference_paths.json"
          mkdir -p "$OUTPUT_RESULT_ROOT"
          json_entries=()
          for i in "${!ALGORITHM_RESULTS_ROOTS[@]}"; do
            path="${ALGORITHM_RESULTS_ROOTS[$i]}"
            run="${RUN_NUMS[$i]}"
            json_entries+=("\"run_$run\": \"$path\"")
          done
          {
            echo '{'
            printf '  %s\n' "$(IFS=,; echo "${json_entries[*]}")"
            echo '}'
          } > "$REFERENCE_JSON_PATH"

          python3 $SCRIPT_DIR/"result_aggregation.py" \
              --algorithm_result_roots "${ALGORITHM_RESULTS_ROOTS[@]}" \
              --output_result_root="$OUTPUT_RESULT_ROOT" \
              --metric="$METRIC" \
              --filename="$REFERENCE_FILE" \
              --infer_info="$INFER_INFO"; 
        else
          echo "No valid runs found for $DATASET_NAME"
        fi
      fi
    done;
  done;
done;