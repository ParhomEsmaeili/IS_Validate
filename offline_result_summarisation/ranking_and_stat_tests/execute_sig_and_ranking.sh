#!/bin/bash

# Set your arguments here
DATASET_NAMES=("Dataset001_BrainTumour" "Dataset003_Liver" "Dataset004_Hippocampus" "Dataset005_Prostate" "Dataset006_Lung" "Dataset007_Pancreas" "Dataset008_HepaticVessel" "Dataset010_Colon")
APPS=("nnintv1" "adadesign1")
# APPS=("nnintv1" "sammed3dv1" "segvolv1" "sammed2dv1" "sam2v1")
SPLIT_NAME="designset"
# SPLIT_NAME="holdoutset"
PROMPTER="pointsonly"
RUN_NAME="-aggregated" 

# Base paths
RESULTS_SUMMARY_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/$SPLIT_NAME"
STATS_SIG_OUTPUT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_StatSig/$SPLIT_NAME"
RANKING_OUTPUT_ROOT="/home/parhomesmaeili/IS-Validation-Framework/Results_Ranking/$SPLIT_NAME"

#SIG SUBPATHS
SIG_METRICS_CONFIG='{"NOI": {"subpath": "metrics/NOI/casewise_noi.csv", "confs": {"NOI": {"cols": ["NOI_Dice_thr_quantile_Q = 0.5"]}, "NoF": {"cols": ["Fail_Dice_thr_quantile_Q = 0.5"]}}}, "AUC": {"subpath": "metrics/AUC/casewise_aucs.csv", "confs": {"Dice_AUC": {"cols": ["Dice_auc_scores"]}, "NSD_AUC": {"cols": ["NSD_auc_scores"]}}}, "Dice": {"subpath": "metrics/Dice/cross_class_scores.csv", "confs": {"Dice Interactive Init": {"cols": ["Interactive Init"]}, "Dice Interactive Edit Iter 100": {"cols": ["Interactive Edit Iter 100"]}}}, "NSD": {"subpath": "metrics/NSD/cross_class_scores.csv", "confs": {"NSD Interactive Init": {"cols": ["Interactive Init"]}, "NSD Interactive Edit Iter 100": {"cols": ["Interactive Edit Iter 100"]}}}}'
SIG_SUBPATHS=()
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    SIG_SUBPATHS+=("$DATASET_NAME/$PROMPTER/run$RUN_NAME")
done    

#RANKING METRICS
RANKING_METRICS=(
    'NOI'
    'NoF'
    'Dice_AUC'
    'NSD_AUC'
    'Dice Interactive Init'
    'Dice Interactive Edit Iter 100'
    'NSD Interactive Init'
    'NSD Interactive Edit Iter 100'
    )
echo "=============================================="
echo "Running Algorithm-wise Significance Scoring"
echo "=============================================="
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "  Input Root: $RESULTS_SUMMARY_ROOT"
echo "  Output Root: $STATS_SIG_OUTPUT_ROOT"
echo "=============================================="

# Run significance scoring
python3 algo_wise_sig_score.py \
    --metrics_root "$RESULTS_SUMMARY_ROOT" \
    --metrics_config "${SIG_METRICS_CONFIG}" \
    --algorithm_names "${APPS[@]}" \
    --experiment_subpath "${SIG_SUBPATHS[@]}" \
    --output_root "$STATS_SIG_OUTPUT_ROOT"
    
if [ $? -eq 0 ]; then
    echo "✓ Significance scoring completed successfully"
else
    echo "✗ Significance scoring failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Running Algorithm Ranking Generation"
echo "=============================================="
echo "Configuration:"
echo "  Split: $SPLIT_NAME"
echo "  Apps: ${APPS[@]}"
echo "  Datasets: ${DATASET_NAMES[@]}"
echo "  Input Root: $STATS_SIG_OUTPUT_ROOT"
echo "  Output Root: $RANKING_OUTPUT_ROOT"
echo "=============================================="

# Run ranking generation
python generate_algo_rankings.py \
    --ref_root "$STATS_SIG_OUTPUT_ROOT" \
    --metrics "${RANKING_METRICS[@]}" \
    --task_names "${DATASET_NAMES[@]}" \
    --algorithm_names "${APPS[@]}" \
    --output_root "$RANKING_OUTPUT_ROOT"

if [ $? -eq 0 ]; then
    echo "✓ Ranking generation completed successfully"
else
    echo "✗ Ranking generation failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Output locations:"
echo "  Significance Scores: $STATS_SIG_OUTPUT_ROOT"
echo "  Rankings: $RANKING_OUTPUT_ROOT"
echo "=============================================="