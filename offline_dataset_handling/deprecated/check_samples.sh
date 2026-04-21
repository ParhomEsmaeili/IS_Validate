#!/bin/bash

largest=0
largest_file=""
dataset=Dataset005_Prostate
while IFS= read -r f; do
    dims=($(fslhd "$f" | grep '^dim[1-3]' | awk '{print $2}'))
    voxels=$(( ${dims[0]:-1} * ${dims[1]:-1} * ${dims[2]:-1} ))
    if [[ $voxels -gt $largest_voxels ]]; then
        largest_voxels=$voxels
        largest_shape="${dims[0]}x${dims[1]}x${dims[2]}"
        largest_file=$f
    fi
done < <(find /home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/$dataset/imagesTr/ -type f \( -name "*.nii" -o -name "*.nii.gz" \))

echo "Largest image: $largest_file with shape $largest_shape"