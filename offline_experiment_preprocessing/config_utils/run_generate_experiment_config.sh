#!/bin/bash 

# ============================================================================
# Configuration Section - Edit these variables to customise the script
# ============================================================================

# Dataset name (used to construct config paths)
# DATASETS=(
#  "Dataset001_BrainTumour"
#  "Dataset003_Liver"
#  "Dataset004_Hippocampus"
#  "Dataset005_Prostate"
#  "Dataset006_Lung"
#  "Dataset007_Pancreas"
#  "Dataset008_HepaticVessel"
#  "Dataset010_Colon"
#  "Dataset041_Parse"
#  "Dataset042_TopCowCT"
#  "Dataset043_TopCowMR"
#  "Dataset044_TopBrainCT"
#  "Dataset045_TopBrainMR"
# )

DATASETS=(
  "Dataset011_Kits23"
  "Dataset015_Kits23"
  "Dataset019_Kits23"
  "Dataset023_Kits23"
  "Dataset027_Kits23"
  "Dataset031_Kits23"
)

# Task config IDs (comma-separated or space-separated)
TASK_CONFIG_IDS=(
  "task_id_15"
  "task_id_16"
  "task_id_17"
  "task_id_18"
  "task_id_19"
  "task_id_20"
  "task_id_21"
  "task_id_22"
  "task_id_23"
  "task_id_24"
)

# Metrics config IDs (comma-separated or space-separated)
METRICS_CONFIG_IDS=("prototype_annotator_4")

# Prompter IDs (comma-separated or space-separated)
PROMPTER_IDS=("prompter_4" "prompter_5" "prompter_6" "prompter_7" "prompter_8" "prompter_9" "prompter_10" "prompter_11" "prompter_12" "prompter_13" "prompter_14" "prompter_15") 
# "prompter_3")

# ============================================================================
# Script Execution - Do not edit below this line unless needed
# ============================================================================

for DATASET in "${DATASETS[@]}"; do
  echo "Processing dataset: ${DATASET}"
  
  # Config file paths (relative to project root)
  TASK_CONFIGS_FILE="exp_configs/${DATASET}/task_configs.txt"
  PROMPTER_CONFIGS_FILE="exp_configs/${DATASET}/prompter_manifest.json"
  METRICS_CONFIGS_FILE="exp_configs/${DATASET}/metrics_configs.txt"

  # Experiment manifest path (will be created if missing)
  EXPERIMENTS_JSON="exp_configs/${DATASET}/experiment_manifest.json"

  # Output JSON path (optional - defaults to EXPERIMENTS_JSON if not specified)
  OUTPUT_JSON="exp_configs/${DATASET}/experiment_manifest.json"
  # Run the Python script with configured arguments
  python3 generate_experiment_config_json.py \
    --task-config-id ${TASK_CONFIG_IDS[@]} \
    --prompter-id ${PROMPTER_IDS[@]} \
    --metrics-config-id ${METRICS_CONFIG_IDS[@]} \
    --task-configs-file "${TASK_CONFIGS_FILE}" \
    --prompter-configs-file "${PROMPTER_CONFIGS_FILE}" \
    --metrics-configs-file "${METRICS_CONFIGS_FILE}" \
    --experiments-json "${EXPERIMENTS_JSON}" \
    --output-json "${OUTPUT_JSON}"
done

