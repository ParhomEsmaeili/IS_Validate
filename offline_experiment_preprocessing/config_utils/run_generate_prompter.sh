#!/bin/bash 

# ============================================================================
# Configuration Section - Edit these variables to customise the script
# ============================================================================

# Dataset name (used to construct config paths)
# DATASETS=(
#   "Dataset001_BrainTumour"
#   "Dataset003_Liver"
#   "Dataset004_Hippocampus"
#   "Dataset005_Prostate"
#   "Dataset006_Lung"
#   "Dataset007_Pancreas"
#   "Dataset008_HepaticVessel"
#   "Dataset010_Colon"
# )

DATASETS=(
  "Dataset011_Kits23"
  "Dataset015_Kits23"
  "Dataset019_Kits23"
  "Dataset023_Kits23"
  "Dataset027_Kits23"
  "Dataset031_Kits23"
)

# DATASETS=(
#   "Dataset040_MSMultispine"
#   "Dataset041_Parse"
#   "Dataset042_TopCowCT"
#   "Dataset043_TopCowMR"
#   "Dataset044_TopBrainCT"
#   "Dataset045_TopBrainMR"
# )

INIT_PROMPT_CONF_NAME="points_prototype_simplified"
EDIT_PROMPT_CONF_NAME="points_prototype_simplified"
INFER_EDIT_NUMS=100
USE_MEM_INF_EDIT=False
IM_CONF_REMOVE_INIT=True
IM_CONF_MEM_LEN=1
ANNOTATION_CONF='{"annotator": ["annotator_4"], "instance_id": ["instance_1"]}'

# ============================================================================
# Script Execution - Do not edit below this line unless needed
# ============================================================================

for DATASET in "${DATASETS[@]}"; do
  echo "Processing dataset: ${DATASET}"
  
  # Config file paths (relative to project root)
  PROMPT_CONFIGS_FILE="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs/${DATASET}/prompts_configs.txt"
  # Output JSON path
  OUTPUT_JSON="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs/${DATASET}/prompter_manifest.json"
  # Run the Python script with configured arguments
  python3 generate_prompter_manifest.py \
    --prompts-configs "${PROMPT_CONFIGS_FILE}" \
    --init-prompt-conf-name "${INIT_PROMPT_CONF_NAME}" \
    --edit-prompt-conf-name "${EDIT_PROMPT_CONF_NAME}" \
    --infer-edit-nums "${INFER_EDIT_NUMS}" \
    --use-mem-inf-edit "${USE_MEM_INF_EDIT}" \
    --im-conf-remove-init "${IM_CONF_REMOVE_INIT}" \
    --im-conf-mem-len "${IM_CONF_MEM_LEN}" \
    --annotation-conf "${ANNOTATION_CONF}" \
    --output "${OUTPUT_JSON}"
done

