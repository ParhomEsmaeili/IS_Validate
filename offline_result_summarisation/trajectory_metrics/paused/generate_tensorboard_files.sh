# #!/bin/bash

#TODO: Amend this after we have regenerated pseudo-time AUC results.


# SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

# # Generate TensorBoard files for trajectory metrics

# OUTPUT_BASE_ROOT="/home/parhomesmaeili/IS-Validation-Framework/inspect_training/tensorboard_files"

# DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
# DATASET_IDS=("001" "003" "004" "005" "006" "007" "008" "010") 



# echo "=============================================="
# echo "Running TensorBoard file generation for aggregated runs."
# APPS=("${MASTER_APPS[@]}")
# SPLIT_NAME=$MASTER_SPLIT
# echo "Configuration:"
# echo "  Split: $SPLIT_NAME"
# echo "  Apps: ${APPS[@]}"
# echo "  Datasets: ${DATASET_NAMES[@]}"
# echo "=============================================="

# PROMPTER="pointsonly"
# # ORIGINAL_RUN_NUMS=("1" "2" "3") #Specify the original run numbers that were used to generate the aggregated results
# #This gives us the tensorboard files for each of the runs too if we want to inspect.

# RUN_NUMS=("1" "2" "3" "-aggregated")

# OUTPUT_BASE_ROOT="/home/parhomesmaeili/IS-Validation-Framework/inspect_training/tensorboard_files/$SPLIT_NAME" 
# #We will put the tensorboard files in a subfolder for each split. 

# for index in ${!DATASET_NAMES[@]}; do
#     DATASET_NAME=${DATASET_NAMES[$index]};
#     DATASET_ID=${DATASET_IDS[$index]};

#     for APP in "${APPS[@]}"; do
#         echo "=============================================="
#         echo "Generating TensorBoard files for dataset $DATASET_NAME for application: $APP"
#         echo "=============================================="
#         for RUN_NUM in "${RUN_NUMS[@]}"; do
#                 #Pull the values.
#                 ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
#                 EXPERIMENT_NAME="$DATASET_NAME/$PROMPTER/run$RUN_NUM";
                
#                 echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT_PATH";
#                 echo OUTPUT_ROOT: "$OUTPUT_BASE_ROOT";
#                 echo EXPERIMENT_NAME: "$EXPERIMENT_NAME";
#                 echo APP: "$APP";
                
#                 python3 $SCRIPT_DIR/"generate_tensorboard_files.py" \
#                     --algorithm_results_root "$ALGORITHM_RESULTS_ROOT_PATH" \
#                     --experiment_names "$EXPERIMENT_NAME" \
#                     --apps "$APP" \
#                     --output_root="$OUTPUT_BASE_ROOT";
#         done;
    
#     echo "=============================================="
#     echo "Generating merged TensorBoard files for dataset $DATASET_NAME for all applications: ${APPS[@]}"
#     echo "=============================================="
#     for RUN_NUM in "${RUN_NUMS[@]}"; do
#         #Pull the values.
#         ALGORITHM_RESULTS_ROOT_PATH="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME";
#         # OUTPUT_FOLDER="$OUTPUT_BASE_ROOT/$EXPERIMENT_NAME" 
#         EXPERIMENT_NAMES=()
#         for APP in "${APPS[@]}"; do
#             EXPERIMENT_NAMES+=("$DATASET_NAME/$PROMPTER/run$RUN_NUM")
#         done
#         echo ALGORITHM_RESULTS_ROOT: "$ALGORITHM_RESULTS_ROOT_PATH/$EXPERIMENT_NAME";
#         echo OUTPUT_ROOT: "$OUTPUT_BASE_ROOT";
#         echo EXPERIMENT_NAMES: "$EXPERIMENT_NAMES";
#         echo APPS: "$APPS";
        
#         python3 $SCRIPT_DIR/"generate_tensorboard_files.py" \
#             --algorithm_results_root "$ALGORITHM_RESULTS_ROOT_PATH" \
#             --experiment_names ${EXPERIMENT_NAMES[@]} \
#             --apps "${APPS[@]}" \
#             --output_root="$OUTPUT_BASE_ROOT";
#     done;
    
#     #Now lets generate a tensorboard file with all of the apps. This will be useful for comparing the trajectories.

# done;