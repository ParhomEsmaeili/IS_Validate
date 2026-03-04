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
echo "Running Algorithm-wise Number of Interactions Metrics Extraction"
APPS=("${MASTER_APPS[@]}")
SPLIT_NAME=$MASTER_SPLIT
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

NNUNET_ROOT_PATH="/home/parhomesmaeili/Helmholtz Group/MICCAI2026_nnunet/nnUNet_Metrics_$SPLIT_NAME"
# RUN_NUMS=("1" "2" "3")
RUN_NUMS=("${MASTER_RUN_NUMS[@]}") #("1" "2" "3" "-aggregated")
INFER_INFO='{"init": "Interactive Init", "edit": 100}'

NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}") #("quantile")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}") #("0.5")
REFERENCE_METRIC="Dice"
REFERENCE_FILE="cross_class_scores.csv"
REFERENCE_COLUMN="Automatic Init"
FINAL_EPISODE_ONLY_FLAG=$MASTER_FINAL_EPISODE_ONLY_FLAG


for index in ${!DATASET_NAMES[@]}; do
  DATASET_NAME=${DATASET_NAMES[$index]};
  DATASET_ID=${DATASET_IDS[$index]};
  if [ "$FINAL_EPISODE_ONLY_FLAG" = true ]; then
    echo "Running NoI extraction only on aggregated final episodes for dataset: $DATASET_NAME"
    for APP in "${APPS[@]}"; do
      for RUN_NUM in "${RUN_NUMS[@]}"; do
        if [ "$RUN_NUM" == "-aggregated" ]; then
          ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
          EXPERIMENT_NAME="$APP-episodefinal/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
          # EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
        else
          echo "Ended up in a final episode flag scenario with a non-aggregated run, exiting"
          exit 1 #We should only be running final episode summarisation on the aggregated runs.
        fi

        OUTPUT_SUBPATH=$EXPERIMENT_NAME #"$APP-episodefinal/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
        OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
        ALGORITHM_RESULTS_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
        REFERENCE_RESULT_PATH="$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
        python3 "$SCRIPT_DIR/num_of_interaction.py" \
            --algo_results_path="$ALGORITHM_RESULTS_PATH" \
            --output_result_root="$OUTPUT_RESULT_ROOT" \
            --metric="$REFERENCE_METRIC" \
            --reference_path="$REFERENCE_RESULT_PATH" \
            --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
            --nnunet_bound "${NNUNET_BOUND[@]}" \
            --infer_info="$INFER_INFO";  
      done;
    done;
  else

    EPISODES_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME"
    for APP in "${APPS[@]}"; do
      for RUN_NUM in "${RUN_NUMS[@]}"; do
        if [ "$RUN_NUM" == "-aggregated" ]; then
          ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
          # EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
        else
          ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
          # EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
        fi
        
        if [[ "$APP" == *"adatest"* ]]; then
          echo "=============================================="
          #Extract the matching episode app results from the app substring. 
          EPISODE_NUM=$(ls -1 "$EPISODES_ROOT_PATH" 2>/dev/null | grep "$APP" | grep -oP "${APP}-episode\d+" | sort -u | wc -l)
          echo "Processing application: $APP for dataset $DATASET_NAME for episode-wise NoI with episodes:"
          echo "$EPISODE_NUM"
          if [ "$EPISODE_NUM" == "0" ]; then
            echo "Skipping because no episodes found for app $APP under dataset $DATASET_NAME for run $RUN_NUM"
            continue 1
          fi
          echo "=============================================="
        else
          echo "=============================================="
          echo "Processing application: $APP for dataset $DATASET_NAME for NoI extraction"
          EPISODE_NUM="0"
          echo "=============================================="
        
        fi
        
        if [ "$EPISODE_NUM" == "0" ]; then
          echo "Running NoI extraction on non-episodic app: $APP"
          
          if [ "$RUN_NUM" == "-aggregated" ]; then
            # ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
            EXPERIMENT_NAME="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"; #Aggregated runs are stored in summary folder.
          else
            # ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
            EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
          fi
          # EXPERIMENT_NAME="$APP-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM";
          OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
          OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$OUTPUT_SUBPATH";
          ALGORITHM_RESULTS_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
          REFERENCE_RESULT_PATH="$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
          python3 "$SCRIPT_DIR/num_of_interaction.py" \
              --algo_results_path="$ALGORITHM_RESULTS_PATH" \
              --output_result_root="$OUTPUT_RESULT_ROOT" \
              --metric="$REFERENCE_METRIC" \
              --reference_path="$REFERENCE_RESULT_PATH" \
              --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
              --nnunet_bound "${NNUNET_BOUND[@]}" \
              --infer_info="$INFER_INFO"; 
        else
          echo "Running NoI extraction on episodic app: $APP for episode num: $EPISODE_NUM"
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
            ALGORITHM_RESULTS_PATH="$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME/metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
            REFERENCE_RESULT_PATH="$NNUNET_ROOT_PATH/$DATASET_NAME/nnUNet_metrics/$REFERENCE_METRIC/$REFERENCE_FILE";
            python3 "$SCRIPT_DIR/num_of_interaction.py" \
                --algo_results_path="$ALGORITHM_RESULTS_PATH" \
                --output_result_root="$OUTPUT_RESULT_ROOT" \
                --metric="$REFERENCE_METRIC" \
                --reference_path="$REFERENCE_RESULT_PATH" \
                --nnunet_statistic "${NNUNET_STATISTIC[@]}" \
                --nnunet_bound "${NNUNET_BOUND[@]}" \
                --infer_info="$INFER_INFO"; 
          done;
        fi
      done;
    done;
  fi
done;