#!/bin/bash

# Usage: ./temp_json_rename.sh folder1 folder2 ...

# Key to rename and its new name
OLD_KEY="numTraining"
NEW_KEY="numTrain"
# Example: List of folders to process
# Uncomment and edit the following line to specify folders manually
set -- Dataset001_BrainTumour Dataset002_Heart Dataset003_Liver Dataset004_Hippocampus Dataset005_Prostate Dataset006_Lung Dataset007_Pancreas Dataset008_HepaticVessel Dataset009_Spleen Dataset010_Colon Dataset019CorruptionMethodadd_gaussCorruptionParam200_Liver Dataset020CorruptionMethodadd_gaussCorruptionParam400_Liver Dataset021CorruptionMethodadd_gaussCorruptionParam600_Liver Dataset022CorruptionMethodadd_gaussCorruptionParam800_Liver Dataset023CorruptionMethodadd_gaussCorruptionParam200_Pancreas Dataset024CorruptionMethodadd_gaussCorruptionParam400_Pancreas Dataset025CorruptionMethodadd_gaussCorruptionParam600_Pancreas Dataset026CorruptionMethodadd_gaussCorruptionParam800_Pancreas

# Adjust JSON_FILE path to point to datasets folder
for folder in "$@"; do
    JSON_FILE="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/$folder/dataset.json"
    if [[ -f "$JSON_FILE" ]]; then
        jq --indent 2 "with_entries(if .key == \"$OLD_KEY\" then .key = \"$NEW_KEY\" else . end)" "$JSON_FILE" > "$JSON_FILE.tmp" && mv "$JSON_FILE.tmp" "$JSON_FILE"
        echo "Replaced $JSON_FILE with renamed key"
    else
        echo "No dataset.json found in $folder"
    fi
done
