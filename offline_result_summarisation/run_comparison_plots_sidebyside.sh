# DATASET_NAMES="Dataset005_Prostate Dataset001_BrainTumour" 

# DATASET_NAMES="Dataset004_Hippocampus Dataset001_BrainTumour Dataset007_Pancreas"
DATASET_NAMES="Dataset006_Lung Dataset007_Pancreas" 

DATETIMES_prostate=(
    "20250620_091628 20250622_221819 20250620_010537 20250621_024859"
) #this is prostate.

DATETIMES_hippocampus=(
    "20250622_221704 20250622_170810 20250618_232125 20250621_025056"
) #hippocampus
 
DATETIMES_braintumour=(
    "20250622_221544 20250620_155654 20250619_000505 20250621_025202"
    
)

DATETIMES_lung=(
    "20250620_034742 20250625_162905 20250619_150953 20250621_040419"
)

DATETIMES_pancreas=(
    "20250623_012721 20250623_191026 20250619_000045 20250621_205840"
)

APP_NAMES="SAMMed2D SAM2 SAMMed3D SegVol"

# Run the Python script
python comparison_plots_sidebyside_format.py \
    --dataset_names $DATASET_NAMES \
    --datetimes_list $DATETIMES_lung \
    --datetimes_list $DATETIMES_pancreas \
    --app_names $APP_NAMES