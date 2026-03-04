#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

#Doubles as both a per-run summarisation script, and as a aggregated results summarisation script, depending on the "run_nums" arg.
# Set your arguments here
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
DATASET_IDS=("${MASTER_DATASET_IDS[@]}")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
# APPS=("nnintv1" "adadesign1")

echo "=============================================="
echo "Running Algorithm-wise Result Summarisation"
APPS=("${MASTER_APPS[@]}")
SPLIT_NAME=$MASTER_SPLIT
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

PROMPTER="pointsonly"
RUN_NUMS=("${MASTER_RUN_NUMS[@]}") #("1" "2" "3" "-aggregated") #We are going to run summaries on ALL #("1") #"2" "3")
INFER_INFO='{"init": "Interactive Init", "edit": 100}'
# SPLIT_NAME="holdoutset"
# SPLIT_NAME="designset"
SPLIT_NAME=$MASTER_SPLIT
REFERENCE_FILE="cross_class_scores.csv"
FINAL_EPISODE_ONLY_FLAG=$MASTER_FINAL_EPISODE_ONLY_FLAG
echo "FINAL_EPISODE_ONLY_FLAG: $FINAL_EPISODE_ONLY_FLAG"

for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  if [ "$FINAL_EPISODE_ONLY_FLAG" = true ]; then
    echo "Running summarisation only on final episode for dataset: $DATASET_NAME"
    for APP in "${APPS[@]}"; do
      for RUN_NUM in "${RUN_NUMS[@]}"; do
        if [ "$RUN_NUM" != "-aggregated" ]; then
          exit 1 #We should only be running final episode summarisation on the aggregated runs.
        else
          echo "Running summarisation on final episode for app $APP under dataset $DATASET_NAME for run $RUN_NUM"
        fi

        if [[ "$APP" == *"adatest"* ]]; then
          echo "=============================================="
          #Checking that the final episode app
          echo "=============================================="
        else
          echo "=============================================="
          echo "Ended up in a final episode flag scenario with a non-periodic app, exiting"
          exit 1
        fi      
        echo "Running summarisation on episodic app: $APP for FINAL episode across runs"
        ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
        EXPERIMENT_NAME="$APP-episodefinal/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
          
        OUTPUT_SUBPATH="$APP-episodefinal/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
        OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
        ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics";
        echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT";
        echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
        echo filename: "$REFERENCE_FILE";
        python3 $SCRIPT_DIR/"standard_metric_summarisation.py" \
            --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
            --output_folder_root="$OUTPUT_RESULT_ROOT" \
            --filename="$REFERENCE_FILE" \
            --infer_info="$INFER_INFO"; 
      done;
    done;
  else
    echo "Running summarisation on all episodes for dataset: $DATASET_NAME"
  
    for APP in "${APPS[@]}"; do
      for RUN_NUM in "${RUN_NUMS[@]}"; do
        
        if [ "$RUN_NUM" == "-aggregated" ]; then
          EPISODE_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
        else
          EPISODE_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
          # EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
        fi
        
        if [[ "$APP" == *"adatest"* ]]; then
          echo "=============================================="
          #Extract the matching episode app results from the app substring. 
          EPISODE_NUM=$(ls -1 "$EPISODE_ROOT_PATH" 2>/dev/null | grep "$APP" | grep -oP "${APP}-episode\d+" | sort -u | wc -l)
          echo "Processing application: $APP for dataset $DATASET_NAME for episode-wise aggregation with episodes:"
          echo "$EPISODE_NUM"
          if [ "$EPISODE_NUM" == "0" ]; then
            echo "Skipping because no episodes found for app $APP under dataset $DATASET_NAME for run $RUN_NUM"
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
          echo "Running summarisation on non-episodic app: $APP"
          OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
          OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
                  
          if [ "$RUN_NUM" == "-aggregated" ]; then
            ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
            EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM";
          else
            ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
            EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
          fi
          ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics";
    
          echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT";
          echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
          echo filename: "$REFERENCE_FILE";
          python3 $SCRIPT_DIR/"standard_metric_summarisation.py" \
              --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
              --output_folder_root="$OUTPUT_RESULT_ROOT" \
              --filename="$REFERENCE_FILE" \
              --infer_info="$INFER_INFO"; 
        else
          echo "Running summarisation on episodic app: $APP for episode num: $EPISODE_NUM"
          for ((ADAPTATION_EPISODE=0; ADAPTATION_EPISODE<$EPISODE_NUM; ADAPTATION_EPISODE++)); do
            if [ "$RUN_NUM" == "-aggregated" ]; then
              ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
              EXPERIMENT_NAME="$APP-episode$ADAPTATION_EPISODE/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
            else
              ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
              EXPERIMENT_NAME="$APP-episode$ADAPTATION_EPISODE-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
            fi
            OUTPUT_SUBPATH="$APP-episode$ADAPTATION_EPISODE/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
            OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
            ALGORITHM_RESULTS_ROOT="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics";
            echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT";
            echo OUTPUT_RESULT_ROOT: "$OUTPUT_RESULT_ROOT";
            echo filename: "$REFERENCE_FILE";
            python3 $SCRIPT_DIR/"standard_metric_summarisation.py" \
                --algorithm_results_root="$ALGORITHM_RESULTS_ROOT" \
                --output_folder_root="$OUTPUT_RESULT_ROOT" \
                --filename="$REFERENCE_FILE" \
                --infer_info="$INFER_INFO"; 
          done;
        fi
      done;
    done;
  fi
done;