upstream_dataset_name=Dataset007_Pancreas; 
downstream_dataset_names=(
#   Dataset019CorruptionMethodadd_gaussCorruptionParam200_Liver
#   Dataset020CorruptionMethodadd_gaussCorruptionParam400_Liver
#   Dataset021CorruptionMethodadd_gaussCorruptionParam600_Liver
#   Dataset022CorruptionMethodadd_gaussCorruptionParam800_Liver
  Dataset023CorruptionMethodadd_gaussCorruptionParam200_Pancreas
  Dataset024CorruptionMethodadd_gaussCorruptionParam400_Pancreas
  Dataset025CorruptionMethodadd_gaussCorruptionParam600_Pancreas
  Dataset026CorruptionMethodadd_gaussCorruptionParam800_Pancreas
)

for dataset_name in "${downstream_dataset_names[@]}"; do
    echo "Copying directory for $dataset_name"
  cp -rp "/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs/$upstream_dataset_name/"* "/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs/$dataset_name/"
done