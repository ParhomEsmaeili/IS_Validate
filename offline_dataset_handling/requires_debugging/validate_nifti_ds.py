#Script for checking that the converted dataset is of a valid structure for the validation framework to use, assuming the nifti folder-structure.

import os
import argparse
import json
import SimpleITK as sitk
import numpy as np
from collections import defaultdict
from IS_Validate.dataset_conversion.utils import is_supported_filetype

def load_image(path):
    try:
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        return data, img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None


def load_dataset_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "labels" not in data or not isinstance(data["labels"], dict):
        raise ValueError("Could not parse canonical class names from dataset.json")
    if "file_ext" not in data or not isinstance(data["file_ext"], str):
        raise ValueError("Missing or invalid 'file_ext' in dataset.json")
    return set(data["labels"].values()), data["file_ext"]


def validate_dataset_structure(root_dir, file_ext, strict=False, check_overlap=False, expected_classes=None):
    issues = defaultdict(list)
    all_class_names = set()

    for case_id in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case_id)
        if not os.path.isdir(case_path):
            issues[case_id].append("Not a directory.")
            continue

        # Check for presence of input channels
        input_path = os.path.join(case_path, "input")
        if not os.path.isdir(input_path):
            issues[case_id].append("Missing 'input' directory.")
        else:
            input_files = os.listdir(input_path)
            if not input_files:
                issues[case_id].append("No files in 'input' directory.")
            else:
                for f in input_files:
                    if not is_supported_filetype(f, file_ext):
                        issues[case_id].append(f"Inconsistent file extension in input: {f}")

        annotators = sorted([d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d)) and d != "input"])
        if not annotators:
            issues[case_id].append("No annotators found.")
            continue

        per_annotator_class_names = set()

        for annotator in annotators:
            annotator_path = os.path.join(case_path, annotator)
            if not os.path.isdir(annotator_path):
                issues[case_id].append(f"{annotator} is not a directory.")
                continue

            classes = sorted(os.listdir(annotator_path))
            if not classes:
                issues[case_id].append(f"No classes under {annotator}.")
                continue

            masks = []
            for class_name in classes:
                per_annotator_class_names.add(class_name)
                all_class_names.add(class_name)

                class_path = os.path.join(annotator_path, class_name)
                if not os.path.isdir(class_path):
                    issues[case_id].append(f"{class_name} under {annotator} is not a directory.")
                    continue

                files = sorted(os.listdir(class_path))
                if not files:
                    issues[case_id].append(f"Missing {file_ext} file for class {class_name} under {annotator}.")
                    continue

                for f in files:
                    if not is_supported_filetype(f, file_ext):
                        issues[case_id].append(f"Inconsistent file extension in {class_name}/{annotator}: {f}")
                        continue
                    data, img = load_image(os.path.join(class_path, f))
                    if data is None:
                        issues[case_id].append(f"Unreadable {file_ext} file: {f} in {class_name}/{annotator}.")
                        continue
                    if strict and np.all(data == 0):
                        issues[case_id].append(f"Empty mask: {f} in {class_name}/{annotator}.")
                    if check_overlap:
                        masks.append((class_name, data))

            if check_overlap and masks:
                combined = np.zeros_like(masks[0][1], dtype=np.uint8)
                for class_name, mask in masks:
                    overlap = np.logical_and(combined, mask)
                    if np.any(overlap):
                        issues[case_id].append(f"Overlap detected for annotator {annotator} in class {class_name}.")
                    combined = np.logical_or(combined, mask)

        if expected_classes is not None:
            missing_expected = expected_classes - per_annotator_class_names
            if missing_expected:
                issues[case_id].append(f"Missing expected classes: {sorted(missing_expected)}")
        elif len(per_annotator_class_names) != len(all_class_names):
            missing_classes = all_class_names - per_annotator_class_names
            if missing_classes:
                issues[case_id].append(f"Missing classes for this case: {sorted(missing_classes)}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate canonical dataset structure.")
    parser.add_argument("--dataset", required=True, help="Path to canonical dataset root.")
    parser.add_argument("--dataset-json", required=True, help="Path to dataset.json file.")
    parser.add_argument("--strict", action="store_true", help="Check for empty masks and completeness.")
    parser.add_argument("--check-overlap", action="store_true", help="Check for overlapping class masks.")
    args = parser.parse_args()

    try:
        expected_classes, file_ext = load_dataset_config(args.dataset_json)
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return

    issues = validate_dataset_structure(
        args.dataset,
        file_ext=file_ext,
        strict=args.strict,
        check_overlap=args.check_overlap,
        expected_classes=expected_classes
    )

    if not issues:
        print("Validation passed: no issues found.")
    else:
        print("Validation issues:")
        for case_id, problems in issues.items():
            print(f"- {case_id}:")
            for p in problems:
                print(f"  - {p}")


if __name__ == "__main__":
    main()
