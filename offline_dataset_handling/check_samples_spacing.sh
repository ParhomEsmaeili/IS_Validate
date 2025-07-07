#!/bin/bash

dataset=Dataset004_Hippocampus
ref_spacing=(1.5 1.5 1.5)

echo "Case,OriginalShape,ResampledShape,ResampledVoxelCount"

max_voxels=-1
max_case=""
max_filename=""
max_shape=""
max_orig_shape=""

while read -r f; do
    dims=($(fslhd "$f" | grep '^dim[1-3]' | awk '{print $2}'))
    spc=($(fslhd "$f" | grep '^pixdim[1-3]' | awk '{print $2}'))
    if [[ ${#dims[@]} -ne 3 || ${#spc[@]} -ne 3 ]]; then
        echo "Skipping $f (could not read dims or spacing)" >&2
        continue
    fi

    read n0 n1 n2 vox <<< $(awk -v d0="${dims[0]}" -v d1="${dims[1]}" -v d2="${dims[2]}" \
        -v s0="${spc[0]}" -v s1="${spc[1]}" -v s2="${spc[2]}" \
        -v r0="${ref_spacing[0]}" -v r1="${ref_spacing[1]}" -v r2="${ref_spacing[2]}" \
        'BEGIN {
            n0 = int(d0 * s0 / r0 + 0.5)
            n1 = int(d1 * s1 / r1 + 0.5)
            n2 = int(d2 * s2 / r2 + 0.5)
            vox = n0 * n1 * n2
            printf "%d %d %d %d", n0, n1, n2, vox
        }')

    case_name=$(basename "$(dirname "$f")")
    orig_shape="${dims[0]}x${dims[1]}x${dims[2]}"
    resampled_shape="${n0}x${n1}x${n2}"
    echo "$case_name,$orig_shape,$resampled_shape,$vox"

    if (( vox > max_voxels )); then
        max_voxels=$vox
        max_case=$case_name
        max_shape=$resampled_shape
        max_orig_shape=$orig_shape
    fi
done < <(find /home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/$dataset/imagesTr/ -type f \( -name "*.nii" -o -name "*.nii.gz" \))

echo ""
echo "Largest resampled voxel count:"
echo "Case: $max_case"
# echo "Filename: $max_filename"
echo "OriginalShape: $max_orig_shape"
echo "ResampledShape: $max_shape"
echo "ResampledVoxelCount: $max_voxels"