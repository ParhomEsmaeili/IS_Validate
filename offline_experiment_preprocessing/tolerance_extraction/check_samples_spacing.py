#!/usr/bin/env python
import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

def calculate_median_spacing(dataset_name):
    """
    Calculate median voxel spacing for each dimension in a dataset.
    Determine if dataset is isotropic or anisotropic and save results to JSON.
    
    Args:
        dataset_name: Name of the dataset (e.g., "Dataset004_Hippocampus")
    """
    dataset_dir = f"/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/{dataset_name}"
    images_dir = os.path.join(dataset_dir, "imagesTr")
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory not found: {images_dir}")
        return
    
    print(f"Calculating median voxel spacing for {dataset_name}...")
    print("")
    
    # Lists to store spacing values for each dimension
    spacing_dim0 = []
    spacing_dim1 = []
    spacing_dim2 = []
    
    count = 0
    
    # Find all cases
    cases = os.listdir(images_dir) #cases are the directories which contain the nifti files.
    casewise_nifti_files = [[Path(os.path.join(images_dir, case, file)) for file in os.listdir(os.path.join(images_dir, case)) if file.endswith(("nii", "nii.gz"))] for case in cases]

    for casewise_file_list in sorted(casewise_nifti_files):
        current_im_spacing = None
        for nifti_file in sorted(casewise_file_list):
            img = sitk.ReadImage(str(nifti_file))
            spacing = img.GetSpacing()
            
            if len(spacing) != 3:
                raise Exception(f"Warning: {nifti_file.name} has fewer than 3 dimensions")
    
            if current_im_spacing is None:
                current_im_spacing = spacing
            else:
                if not np.isclose(current_im_spacing, spacing).all():
                    print(f"Warning: Spacing mismatch in {nifti_file.name}. Expected {current_im_spacing}, but got {spacing}")          
        
        spacing_dim0.append(spacing[0])
        spacing_dim1.append(spacing[1])
        spacing_dim2.append(spacing[2])

        count += 1

    if count == 0:
        print("No valid NIfTI files found!")
        return
    
    print(f"Processed {count} images")
    print("")
    
    # Calculate medians
    median_dim0 = float(np.median(spacing_dim0))
    median_dim1 = float(np.median(spacing_dim1))
    median_dim2 = float(np.median(spacing_dim2))
    
    median_spacings = [median_dim0, median_dim1, median_dim2]
    
    print("Median voxel spacing:")
    print(f"Dimension 0 (X): {median_dim0:.4f}")
    print(f"Dimension 1 (Y): {median_dim1:.4f}")
    print(f"Dimension 2 (Z): {median_dim2:.4f}")
    print("")
    
    # Determine isotropicity (anisotropic if max/min >= 3x)
    max_spacing = max(median_spacings)
    min_spacing = min(median_spacings)
    anisotropy_ratio = max_spacing / min_spacing
    is_isotropic = anisotropy_ratio < 3.0
    
    if is_isotropic:
        # Use median of all three dimensions
        reference_spacing = float(np.median(median_spacings))
        spacing_type = "isotropic"
        print(f"Dataset is ISOTROPIC (ratio: {anisotropy_ratio:.2f})")
        print(f"Reference spacing: {reference_spacing:.4f}")
    else:
        # Use median of in-plane dimensions (the two smaller ones)
        sorted_spacings = sorted(median_spacings)
        in_plane_spacings = sorted_spacings[:2]  # Two smallest dimensions
        reference_spacing = float(np.median(in_plane_spacings))
        spacing_type = "anisotropic"
        print(f"Dataset is ANISOTROPIC (ratio: {anisotropy_ratio:.2f})")
        print(f"Reference spacing (in-plane): {reference_spacing:.4f}")
        print(f"Out-of-plane spacing: {max_spacing:.4f}")
    print("")
    
    # Create output JSON
    config_data = {
        "dataset_name": dataset_name,
        "median_spacing": {
            "dim0": median_dim0,
            "dim1": median_dim1,
            "dim2": median_dim2
        },
        "spacing_type": spacing_type,
        "anisotropy_ratio": float(anisotropy_ratio),
        "reference_spacing": reference_spacing,
        "num_cases": count,
        "note": "Reference spacing: median of all dims (isotropic) or median of in-plane dims (anisotropic). SimpleITK GetSpacing() accounts for image orientation from NIfTI headers."
    }
    
    # Write to exp_configs folder
    exp_configs_dir = f"/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs/{dataset_name}"
    os.makedirs(exp_configs_dir, exist_ok=True)
    
    config_file = os.path.join(exp_configs_dir, "spacing_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")

if __name__ == "__main__":
    import sys
    import argparse

    argument_parser = argparse.ArgumentParser(description="Calculate median voxel spacing for a medical imaging dataset")
    argument_parser.add_argument("--dataset", type=str, default="Dataset040_MSMultispine", help="Name of the dataset (e.g., Dataset004_Hippocampus)")
    args = argument_parser.parse_args()
    # You can change this to any dataset name
    dataset = args.dataset
    
    calculate_median_spacing(dataset)
