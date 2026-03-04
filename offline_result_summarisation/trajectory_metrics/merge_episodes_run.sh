#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"


MASTER_PRETRAINED=$MASTER_PRETRAINED

# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" 
# DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010")
DATASET_NAMES=("${MASTER_DATASET_NAMES[@]}")
DATASET_IDS=("${MASTER_DATASET_IDS[@]}")

# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1" "nnintv1")
# APPS=("nnintv1" "adadesign1")

echo "=============================================="
echo "Running merging into pseudotime across episodes. Using $MASTER_PRETRAINED as the pretrained base for padding the pre-adaptation phase."
APPS=("${MASTER_APPS[@]}")
TRAIN_SPLIT_NAME="designset"
SPLIT_NAME=$MASTER_SPLIT
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "=============================================="

PROMPTER="pointsonly"
RUN_NUMS=("${MASTER_RUN_NUMS[@]}")
echo RUN_NUMS: "${RUN_NUMS[@]}"
if [[ "${RUN_NUMS[@]}" =~ "-aggregated" ]]; then
    echo "Error: Please run the script with ONLY the original run numbers specified. We cannot reliably perform episode merging on aggregated results. Exiting."
    exit 1
fi
# SPLIT_NAME="holdoutset"
# SPLIT_NAME="designset"
SPLIT_NAME=$MASTER_SPLIT


REFERENCE_METRICS_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME"

NNUNET_STATISTIC=("${MASTER_NNUNET_STATISTIC[@]}")
NNUNET_BOUND=("${MASTER_NNUNET_BOUND[@]}")
DICE_AUC_STATISTIC=("${MASTER_DICE_AUC_STATISTIC[@]}")
NSD_AUC_STATISTIC=("${MASTER_NSD_AUC_STATISTIC[@]}")
DICE_ITERATION_STATISTIC=("${MASTER_DICE_ITERATION_STATISTIC[@]}")
NSD_ITERATION_STATISTIC=("${MASTER_NSD_ITERATION_STATISTIC[@]}")
NOI_STATISTIC=("${MASTER_NOI_STATISTIC[@]}")

declare -A TRANSLATE_NNUNET_STATISTIC_TO_NAME
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.25"]="LQ"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.5"]="Median"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["quantile_0.75"]="UQ"
TRANSLATE_NNUNET_STATISTIC_TO_NAME["gaussian_0"]="Mean"

NNUNET_STATISTIC_NAME_CALL="${TRANSLATE_NNUNET_STATISTIC_TO_NAME[${NNUNET_STATISTIC}_"${NNUNET_BOUND}"]}"

METRICS_CONFIG="{\"all_auc_summaries.csv\": {\"Dice_auc_${DICE_AUC_STATISTIC[0]}\": null, \"NSD_auc_${NSD_AUC_STATISTIC[0]}\": null}, \"all_iteration_summaries.csv\": {\"Dice_${DICE_ITERATION_STATISTIC[0]}\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}, \"NSD_${NSD_ITERATION_STATISTIC[0]}\": {\"rows\": [\"Interactive Init\", \"Interactive Edit Iter 100\"]}}, \"summarised_num_interactions_fitting.csv\": {\"Normalised_${NOI_STATISTIC[0]^}_NOI\": {\"rows\": [\"${NNUNET_STATISTIC_NAME_CALL}\"]}, \"Failure_Cases_Fraction\": {\"rows\": [\"${NNUNET_STATISTIC_NAME_CALL}\"]}}}"


### setting the namne of the datastream to plot as pseudo time
CONFIG="all_train"

for RUN_NUM in "${RUN_NUMS[@]}"; do
    echo "Processing run number: $RUN_NUM"
    #Now we do per-run. 
    for index in ${!DATASET_NAMES[@]}; do
        DATASET_NAME=${DATASET_NAMES[$index]};
        DATASET_ID=${DATASET_IDS[$index]};
        for APP in "${APPS[@]}"; do
            echo "=============================================="
            echo "Merging episode wise metrics for dataset $DATASET_NAME application: $APP"
            GREPPING_EPISODES_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME";
            # echo $GREPPING_EPISODES_PATH;
            echo "=============================================="

            if [[ "$APP" == *"adatest"* ]]; then
                echo "=============================================="
                #Extract the matching episode app results from the app substring. 
                EPISODE_NUM=$(ls -1 "$GREPPING_EPISODES_PATH" 2>/dev/null | grep "$APP" | grep -oP "${APP}-episode\d+" | sort -u | wc -l)
                echo "Processing application: $APP for dataset $DATASET_NAME for episode-wise merge with episodes:"
                echo "$EPISODE_NUM"
                if [ "$EPISODE_NUM" == "0" ]; then
                    echo "Skipping because no episodes found for app $APP under dataset $DATASET_NAME for run $RUN_NUM"
                    continue 1
                fi
                #Now lets construct the paths to each episode's metrics, which we will pass to the merging script.
                # EPISODE_METRIC_ROOTS=()
                #we will need to start off with the pre-adaptation phase.
                EPISODE_METRIC_ROOTS=("/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$MASTER_PRETRAINED/$DATASET_NAME/$PROMPTER/run$RUN_NUM")
                
                # LOGS_DIR="/home/parhomesmaeili/IS-Validation-Framework/Results/$TRAIN_SPLIT_NAME/Results/$DATASET_NAME/logs"
                TRAIN_LOGS_SUBPATH=()
                for ((EPISODE_INDEX=0; EPISODE_INDEX<$EPISODE_NUM; EPISODE_INDEX++)); do
                    EXP_NAME=$APP-episode$EPISODE_INDEX-dataset${DATASET_ID}-$PROMPTER-run$RUN_NUM
                    EPISODE_METRIC_ROOTS+=("/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$APP-episode$EPISODE_INDEX/$DATASET_NAME/$PROMPTER/run$RUN_NUM")
                    TEST_LOG_DIR="/home/parhomesmaeili/IS-Validation-Framework/Results/$SPLIT_NAME/$DATASET_NAME/$EXP_NAME"
                    TEST_LOG_PATH="$TEST_LOG_DIR/experiment_${EXP_NAME}_logs.log"
                    #Now we read from the log file. We grep the name of the design run.
                    REFERENCE_CHECKPOINT=$(grep "reference_experiment_checkpoint" "$TEST_LOG_PATH" | sed 's/.*\/\([^/]*\)\.pkl.*/\1/')    
                    TRAIN_LOGS_SUBPATH+=($REFERENCE_CHECKPOINT)
                done;
                #We assert that the is the same across episodes, as this is where we will be extracting the interaction timestamps from for the pseudo-time fitting in the downstream script. This is because the interaction timestamps should be the same across episodes, as they are generated by the meta-algorithm and not by the individual runs.
                if [ $(echo "${TRAIN_LOGS_SUBPATH[@]}" | tr ' ' '\n' | sort -u | wc -l) -ne 1 ]; then
                    echo "Error: Found multiple different reference checkpoints across episodes for app $APP under dataset $DATASET_NAME for run $RUN_NUM. This should not happen, as the reference checkpoint should be determined by the meta-algorithm and should be the same across episodes. Exiting."
                    exit 1
                fi
                echo "Reference checkpoint for all episodes: ${TRAIN_LOGS_SUBPATH[0]}"
                REFERENCE_LOGS_SUBPATH="${TRAIN_LOGS_SUBPATH[0]}"
                REFERENCE_LOG_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results/$TRAIN_SPLIT_NAME/$DATASET_NAME/$REFERENCE_LOGS_SUBPATH/experiment_${REFERENCE_LOGS_SUBPATH}_logs.log"
                
                echo "==============================================="
                echo "Reference log path for extracting adaptation timestamps for pseudotime fitting: $REFERENCE_LOG_PATH"
                echo "==============================================="
                
                # Extract sample counts from data_split blocks
                SAMPLE_COUNTS=$(python3 -c "
import json

with open('$REFERENCE_LOG_PATH') as f:
    content = f.read()

parts = content.split('data_split:')[1:]
counts = []
for part in parts:
    start = part.find('{')
    if start == -1:
        continue
    brace_count = 0
    end = start
    for i, c in enumerate(part[start:], start):
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    try:
        data = json.loads(part[start:end])
        total = 0
        for fold in data.values():
            total += len(fold.get('train', []))
            total += len(fold.get('val', []))
        counts.append(total)
    except:
        pass
print(' '.join(map(str, counts)))
")

                # Extract memory buffer num_samples for cross-referencing
                BUFFER_SAMPLE_COUNTS=$(python3 -c "
import re

with open('$REFERENCE_LOG_PATH') as f:
    content = f.read()

matches = re.findall(r'memory buffer num_samples[:\s]+(\d+)', content, re.IGNORECASE)
print(' '.join(matches))
")

                # Extract total samples from dataset_split.json using CONFIG
                DATASET_SPLIT_JSON="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/${DATASET_NAME}/dataset_split.json"
                TOTAL_SAMPLES=$(python3 -c "
import json
import sys
try:
    with open('$DATASET_SPLIT_JSON') as f:
        d = json.load(f)
    cases = d['sampling']['$CONFIG']['all_cases']
    print(len(cases))
except Exception as e:
    print(f'Error extracting total samples from {DATASET_SPLIT_JSON}: {e}', file=sys.stderr)
    print('', end='')
")
                echo "Sample counts across episodes from data_split: $SAMPLE_COUNTS"
                echo "Memory buffer sample counts: $BUFFER_SAMPLE_COUNTS"
                echo "Total samples from dataset_split.json [$CONFIG]: $TOTAL_SAMPLES"

                # Cross-check: SAMPLE_COUNTS and BUFFER_SAMPLE_COUNTS must be identical
                if [ "$SAMPLE_COUNTS" != "$BUFFER_SAMPLE_COUNTS" ]; then
                    echo "Error: Sample counts from data_split ($SAMPLE_COUNTS) do not match memory buffer sample counts ($BUFFER_SAMPLE_COUNTS) for app $APP under dataset $DATASET_NAME for run $RUN_NUM. Exiting."
                    exit 1
                fi
                echo "Cross-check passed: data_split and memory buffer sample counts match."

                # Cross-check: final sample count must be less than or equal to the total samples
                FINAL_SAMPLE_COUNT=$(echo "$SAMPLE_COUNTS" | awk '{print $NF}')
                if [ "$FINAL_SAMPLE_COUNT" -gt "$TOTAL_SAMPLES" ]; then
                    echo "Error: Final sample count ($FINAL_SAMPLE_COUNT) is not less than total samples ($TOTAL_SAMPLES) for app $APP under dataset $DATASET_NAME for run $RUN_NUM. Exiting."
                    exit 1
                fi
                echo "Cross-check passed: final sample count ($FINAL_SAMPLE_COUNT) < total samples ($TOTAL_SAMPLES)."

                echo "=============================================="
                echo "Merging episodes for app $APP with episode metric roots:"
                echo "${EPISODE_METRIC_ROOTS[@]}"
                echo "=============================================="
                ADAPTED_FLAG="true"
                
            else
                echo "=============================================="
                echo "Processing application: $APP for dataset $DATASET_NAME, pseudo merge required."
                if [[ "$APP" != "$MASTER_PRETRAINED" ]]; then
                    echo "Error: For non-episodic apps, we require that the app name: $APP matches the $MASTER_PRETRAINED app name for padding purposes. Exiting."
                    exit 1
                fi
                EPISODE_NUM="0"
                EPISODE_METRIC_ROOTS=("/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME/$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM")
                
                #NOTE: This will break if we have not passed 
                REFERENCE_LOGS_SUBPATH="${DUMMY_LOGS_SUBPATH[0]}"
                
                # Extract total samples for non-episodic case from dataset_split.json using CONFIG
                DATASET_SPLIT_JSON="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/${DATASET_NAME}/dataset_split.json"
                TOTAL_SAMPLES=$(python3 -c "
import json
import sys
try:
    with open('$DATASET_SPLIT_JSON') as f:
        d = json.load(f)
    cases = d['sampling']['$CONFIG']['all_cases']
    print(len(cases))
except Exception as e:
    print(f'Error extracting total samples from {DATASET_SPLIT_JSON}: {e}', file=sys.stderr)
    print('', end='')
")
                echo "Total samples from dataset_split.json [$CONFIG]: $TOTAL_SAMPLES"
                if [[ -z "$TOTAL_SAMPLES" ]]; then
                    echo "Error: TOTAL_SAMPLES is empty for $DATASET_SPLIT_JSON [$CONFIG]. The file may be missing, malformed, or the config key is wrong. Skipping this run."
                    continue 1
                fi
                
                echo "=============================================="
                echo "Merging pseudo-episodes for app $APP with metric roots:"
                echo "${EPISODE_METRIC_ROOTS[@]}"
                echo "=============================================="
                ADAPTED_FLAG="false"
            fi  
    
            OUTPUT_SUBPATH="$APP/$DATASET_NAME/$PROMPTER/run$RUN_NUM"
            OUTPUT_RESULT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Pseudotime/$SPLIT_NAME/$OUTPUT_SUBPATH";
            if [[ "$ADAPTED_FLAG" == "true" ]]; then
                echo "Running episode merging for episodic app: $APP with roots ${EPISODE_METRIC_ROOTS[@]}, which will involve merging across episodes with padding using the pretrained model metrics for the pre-adaptation phase."
                python3 $SCRIPT_DIR/"merge_episodes.py" \
                    --adaptive \
                    --pretrained_name="$MASTER_PRETRAINED" \
                    --metric_roots ${EPISODE_METRIC_ROOTS[@]} \
                    --samples_counts $SAMPLE_COUNTS \
                    --total_samples $TOTAL_SAMPLES \
                    --output_result_root="$OUTPUT_RESULT_ROOT" \
                    --metrics_config="$METRICS_CONFIG" \
                    --nnunet_statistic="${NNUNET_STATISTIC_NAME_CALL}";
            else
                echo "Running episode merging for non-episodic app: $APP, with roots ${EPISODE_METRIC_ROOTS[@]}, which is effectively a pseudo-episode merging with padding using the pretrained model metrics."
                python3 $SCRIPT_DIR/"merge_episodes.py" \
                    --pretrained_name="$MASTER_PRETRAINED" \
                    --metric_roots ${EPISODE_METRIC_ROOTS[@]} \
                    --total_samples $TOTAL_SAMPLES \
                    --output_result_root="$OUTPUT_RESULT_ROOT" \
                    --metrics_config="$METRICS_CONFIG" \
                    --nnunet_statistic="${NNUNET_STATISTIC_NAME_CALL}";
            fi
        done;
    done;
done;