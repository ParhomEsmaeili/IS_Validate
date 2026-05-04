#!/bin/bash

# Usage: ./check_cc_cases.sh <dataset_dir> <label_value> [class_subdir]
# Example: ./check_cc_cases.sh ./dataset 1 label

DATASET_DIR="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset006_Lung"
LABEL_VALUE="1"
CLASS_SUBDIR="annotator_1/semantic_class_cancer"
DATA_SPLIT="all_train"

if [[ -z "$DATASET_DIR" || -z "$LABEL_VALUE" ]]; then
    echo "Usage: $0 <dataset_dir> <label_value> <class_subdir> [data_split]"
    exit 1
fi

PYTHON_SCRIPT=$(cat <<'END'
import sys
import os
import numpy as np
from skimage import measure
import SimpleITK as sitk
import json 

dataset_dir = sys.argv[1]
label_value = int(sys.argv[2])
class_subdir = sys.argv[3] if len(sys.argv) > 3 else None
split_name = sys.argv[4] if len(sys.argv) > 4 else None 

datasplit_json = os.path.join(dataset_dir, 'dataset_split.json')
with open(datasplit_json, 'r') as f:
    datasplit = json.load(f)


dataset_dir = os.path.join(dataset_dir, 'labelsTr') 

if split_name == 'all_train':
    cases = datasplit['sampling']['all_train']['all_cases']
else:
    cases = sorted(os.listdir(dataset_dir))


# print(datasplit_json)
# print(cases)
print('Only printing for cases with more than one connected component')
for case in cases:
    case_path = os.path.join(dataset_dir, case)
    if not os.path.isdir(case_path):
        continue
    if class_subdir:
        case_path = os.path.join(case_path, class_subdir)
        if not os.path.isdir(case_path):
            continue
    for fname in sorted(os.listdir(case_path)):
        if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
            continue
        fpath = os.path.join(case_path, fname)
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(fpath))
            img = np.asarray(img, dtype=np.int32)
        except Exception as e:
            print(f"Could not read {fpath}: {e}", file=sys.stderr)
            continue
        mask = (img == label_value)
        cc = measure.label(mask, connectivity=3)
        n_cc = np.max(cc)
        if n_cc  > 1:
            print(f"{case}/{fname}: {n_cc} connected components for label {label_value}")
            print(f"Biggest component had voxel count: {np.max(np.bincount(cc.flat)[1:])}")
            # print(f"Total voxel count: {np.sum(mask)}")
            # print(f"diff between total and biggest component: {np.sum(mask) - np.max(np.bincount(cc.flat)[1:])}")
END
)

if [[ -z "$DATA_SPLIT" ]]; then
    python3 -c "$PYTHON_SCRIPT" "$DATASET_DIR" "$LABEL_VALUE" "$CLASS_SUBDIR"
else
    python3 -c "$PYTHON_SCRIPT" "$DATASET_DIR" "$LABEL_VALUE" "$CLASS_SUBDIR" "$DATA_SPLIT"
fi