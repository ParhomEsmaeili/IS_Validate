import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import shutil
import SimpleITK as sitk
import multiprocessing
import random
import sys 
import warnings 

#NOTE: We have a-priori enforced a mapping into binary sem seg formulation, rather than do this in the evaluation
#task config. This is because for the MSD datasets we could always guarantee the existence of the target, whereas 
#we may not necessarily for all subproblems that we might want to explore in kits. 

#We therefore will split a-priori, and filter out cases where the dataset fusion strategy results in empty foreground,
#as these cases would be skipped over in eval and would result in inconsistencies across splits & for use in training
#nnu-net models. 


# Get paths: current file is 3 levels deep in offline_dataset_handling
# offline_dataset_handling/dataset_specific_processing/kits23/convert_to_framework.py
offline_dhandling_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
datasets_path = os.path.join(os.path.dirname(offline_dhandling_dir), 'datasets')
sys.path.insert(0, offline_dhandling_dir)

# Now import from utils in the offline_dataset_handling directory
from utils import check_dataset_existence

def validate_case(case_dir):
    """
    Validate if a case has the required files (T2, LESIONMASK, and one complementary modality).
    
    Args:
        case_dir: Path to case directory (named sub-{case_name})
    
    Returns:
        Tuple of (is_valid, t2_file, lesion_mask_file, complementary_file) or (False, None, None, None)
    """
    case_dir = Path(case_dir)
    dir_name = case_dir.name
    
    # Extract case name from directory (sub-{case_name} -> {case_name})
    if dir_name.startswith("sub-"):
        case_name = dir_name[4:]  # Remove "sub-" prefix
    else:
        case_name = dir_name
    
    # Find T2 file
    t2_files = list(case_dir.glob(f"*{case_name}_T2.nii.gz"))
    if len(t2_files) != 1:
        return False, None, None, None
    
    # Find LESIONMASK file
    lesion_files = list(case_dir.glob(f"*{case_name}_LESIONMASK.nii.gz"))
    if len(lesion_files) != 1:
        return False, None, None, None
    
    # Find complementary modality (any .nii.gz file that's not T2 or LESIONMASK)
    all_nifti_files = list(case_dir.glob("*.nii.gz"))
    complementary_files = [f for f in all_nifti_files if f not in t2_files and f not in lesion_files]
    if len(complementary_files) != 1:
        return False, None, None, None
    
    return True, t2_files[0], lesion_files[0], complementary_files[0]


def process_single_case(case_dir, t2_file, lesion_mask_file, complementary_file, output_images_path, output_labels_path, semantic_class_mapping, input_to_output_class_mapping, final_semantic_class_mapping, modality_index_mapping):
    """
    Process a single case directory.
    
    Args:
        case_dir: Path to case directory
        t2_file: Path to T2 image file
        lesion_mask_file: Path to lesion mask file
        complementary_file: Path to complementary modality file
        output_images_path: Output directory for images
        output_labels_path: Output directory for labels
        semantic_class_mapping: Dict mapping INPUT semantic class names to INPUT label IDs
        input_to_output_class_mapping: Dict mapping OUTPUT class names to list of INPUT class names
        final_semantic_class_mapping: Dict mapping OUTPUT semantic class names to OUTPUT label IDs
        modality_index_mapping: Dict mapping modality names to their channel indices (e.g., {"T2": 0, "FLAIR": 1})
    
    Returns:
        True if case was processed, False otherwise
    """
    case_dir = Path(case_dir)
    output_images_path = Path(output_images_path)
    output_labels_path = Path(output_labels_path)
    
    # Extract case name from directory (sub-{case_name} -> {case_name})
    dir_name = case_dir.name
    if dir_name.startswith("sub-"):
        case_name = dir_name[4:]  # Remove "sub-" prefix
    else:
        case_name = dir_name
    
    # Read T2 image (primary modality, always 0000)
    t2_itk = sitk.ReadImage(str(t2_file))
    t2_array = sitk.GetArrayFromImage(t2_itk)
    t2_spacing = t2_itk.GetSpacing()
    t2_origin = t2_itk.GetOrigin()
    t2_direction = t2_itk.GetDirection()
    
    # Read complementary modality
    comp_itk = sitk.ReadImage(str(complementary_file))
    comp_spacing = comp_itk.GetSpacing()
    comp_origin = comp_itk.GetOrigin()
    comp_direction = comp_itk.GetDirection()
    
    # Read lesion mask and separate by label ID
    lesion_itk = sitk.ReadImage(str(lesion_mask_file))
    lesion_array = sitk.GetArrayFromImage(lesion_itk).astype(np.uint8)
    
    # Extract unique label IDs (excluding 0/background if present)
    unique_labels = np.unique(lesion_array)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    
    # Check if any foreground labels exist
    if len(unique_labels) == 0:
        warnings.warn(f"No foreground labels found in lesion mask for case {case_name}. Skipping case.")
        return False
    
    # Create output directories
    output_images_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)
    
    output_case_images_dir = output_images_path / case_name
    output_case_images_dir.mkdir(parents=True, exist_ok=True)
    
    output_case_labels_dir = output_labels_path / case_name
    output_case_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Write T2 image (channel index 0000)
    output_image_file = output_case_images_dir / f"{case_name}_0000.nii.gz"
    sitk.WriteImage(t2_itk, str(output_image_file))
    
    # Write complementary modality with consistent index
    complementary_name = complementary_file.name.replace('.nii.gz', '').split('_')[-1]  # e.g., "FLAIR" from "xxx_FLAIR.nii.gz"
    comp_index = modality_index_mapping[complementary_name]
    output_comp_file = output_case_images_dir / f"{case_name}_{comp_index:04d}.nii.gz"
    sitk.WriteImage(comp_itk, str(output_comp_file))
    
    # Create separate instances for each label ID and save them as semantic class segmentations
    # Each unique label ID becomes an instance/class
    annotator_id = 1  # Single annotator since no multiple annotators
    
    # Merge all lesion labels into a single binary mask
    merged_lesion_mask = np.zeros_like(lesion_array, dtype=np.uint8)
    for label_id in unique_labels:
        merged_lesion_mask |= (lesion_array == label_id).astype(np.uint8)
    
    # Create semantic class folder for single lesion class
    output_class_name = "lesion"
    class_annotator_dir = output_case_labels_dir / f"annotator_{annotator_id}" / f"semantic_class_{output_class_name}"
    class_annotator_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged mask as single binary segmentation
    output_file = class_annotator_dir / f"{case_name}_0001.nii.gz"
    merged_itk = sitk.GetImageFromArray(merged_lesion_mask)
    merged_itk.SetSpacing(t2_spacing)
    merged_itk.SetOrigin(t2_origin)
    merged_itk.SetDirection(t2_direction)
    sitk.WriteImage(merged_itk, str(output_file))

    # Save the corresponding background mask as the inverse of the lesion mask.
    background_mask = (merged_lesion_mask == 0).astype(np.uint8)
    background_dir = output_case_labels_dir / f"annotator_{annotator_id}" / "semantic_class_background"
    background_dir.mkdir(parents=True, exist_ok=True)
    background_file = background_dir / f"{case_name}_0001.nii.gz"
    background_itk = sitk.GetImageFromArray(background_mask)
    background_itk.SetSpacing(t2_spacing)
    background_itk.SetOrigin(t2_origin)
    background_itk.SetDirection(t2_direction)
    sitk.WriteImage(background_itk, str(background_file))
    
    # Record metadata
    metadata_file = output_case_images_dir / "metadata.json"
    metadata = {
        "case_name": case_name,
        "complementary_modality": complementary_name,
        "lesion_labels": [int(l) for l in unique_labels]
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True

def process_ms_multispine_dataset(input_dir, output_dir, dataset_id, semantic_class_mapping=None, num_processes=1, methodology_fraction=1.0):
    """
    Process MS-Multispine dataset with label-based instance separation.
    
    Args:
        input_dir: Path to input directory containing case subdirectories
        output_dir: Not used - output will be created under datasets/ using dataset_id
        dataset_id: Dataset ID number
        semantic_class_mapping: Dict mapping OUTPUT semantic class names to OUTPUT label IDs
                               Example: {"lesion": 1, "background": 0}
        num_processes: Number of worker processes
        methodology_fraction: Fraction of cases for training (rest go to test)
    """
    if semantic_class_mapping is None:
        raise Exception('Semantic class mapping is required')
    
    # Check if dataset ID is already in use
    existing_datasets = check_dataset_existence(datasets_path, dataset_id)
    assert len(existing_datasets) == 0, f"Target dataset id {dataset_id} is already taken. Conflicting dataset: {existing_datasets}"
    
    # Create target dataset directory
    target_dataset_name = f"Dataset{dataset_id:03d}_MSMultispine"
    target_folder = os.path.join(datasets_path, target_dataset_name)
    
    input_path = Path(input_dir)
    output_path = Path(target_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create imagesTr, labelsTr, imagesTs, labelsTs directories
    output_images_tr = output_path / "imagesTr"
    output_labels_tr = output_path / "labelsTr"
    output_images_ts = output_path / "imagesTs"
    output_labels_ts = output_path / "labelsTs"
    
    output_images_tr.mkdir(parents=True, exist_ok=True)
    output_labels_tr.mkdir(parents=True, exist_ok=True)
    output_images_ts.mkdir(parents=True, exist_ok=True)
    output_labels_ts.mkdir(parents=True, exist_ok=True)
    
    # Collect and validate all case directories
    all_case_dirs = [case_dir for case_dir in sorted(input_path.iterdir()) if case_dir.is_dir()]
    print(f"Found {len(all_case_dirs)} case directories")
    
    if not all_case_dirs:
        print(f"No case directories found in {input_dir}")
        return
    
    # Validate cases and collect unique complementary modality names
    print("Validating cases...")
    validated_cases = []
    unique_modalities = set()
    
    for case_dir in all_case_dirs:
        is_valid, t2_file, lesion_mask_file, complementary_file = validate_case(case_dir)
        if is_valid:
            validated_cases.append((case_dir, t2_file, lesion_mask_file, complementary_file))
            # Extract modality name from filename
            comp_name = complementary_file.name.replace('.nii.gz', '').split('_')[-1]  # e.g., "FLAIR" from "xxx_FLAIR.nii.gz"
            unique_modalities.add(comp_name)
    
    print(f"Found {len(validated_cases)} valid cases")
    print(f"Found {len(unique_modalities)} unique complementary modalities: {unique_modalities}")
    
    if not validated_cases:
        print("No valid cases found!")
        return
    
    # Create modality index mapping (T2=0, then complementary modalities starting from 1)
    modality_index_mapping = {"T2": 0}
    for idx, modality_name in enumerate(sorted(unique_modalities), start=1):
        modality_index_mapping[modality_name] = idx
    
    print(f"Modality index mapping: {modality_index_mapping}")
    
    # Process all valid cases
    print(f"Processing {len(validated_cases)} cases...")
    if num_processes == 1:
        results = [process_single_case(case_dir, t2_file, lesion_mask_file, complementary_file, 
                                       output_images_tr, output_labels_tr, 
                                       semantic_class_mapping, {}, semantic_class_mapping, modality_index_mapping)
                   for case_dir, t2_file, lesion_mask_file, complementary_file in validated_cases]
    else:
        with multiprocessing.Pool(num_processes) as p:
            results = p.starmap_async(
                process_single_case,
                [(case_dir, t2_file, lesion_mask_file, complementary_file,
                  output_images_tr, output_labels_tr,
                  semantic_class_mapping, {}, semantic_class_mapping, modality_index_mapping)
                 for case_dir, t2_file, lesion_mask_file, complementary_file in validated_cases]
            )
            results = results.get()
    
    # Extract just the case names (without sub- prefix) from processed cases
    processed_case_names = []
    for i, result in enumerate(results):
        if result:
            dir_name = validated_cases[i][0].name
            if dir_name.startswith("sub-"):
                case_name = dir_name[4:]
            else:
                case_name = dir_name
            processed_case_names.append(case_name)
    
    print(f"Successfully processed {len(processed_case_names)} cases")
    
    if not processed_case_names:
        print("No cases with valid foreground found!")
        return
    
    # Split into train/test
    num_train_cases = int(np.ceil(len(processed_case_names) * methodology_fraction))
    train_case_names = set(random.sample(processed_case_names, k=num_train_cases))
    test_case_names = [name for name in processed_case_names if name not in train_case_names]
    
    print(f"Split into {len(train_case_names)} training and {len(test_case_names)} test cases")
    
    # Move cases to appropriate splits (they're already in imagesTr/labelsTr, move to test if needed)
    for case_name in test_case_names:
        # Move imagesTr to imagesTs
        tr_case_images = output_images_tr / case_name
        ts_case_images = output_images_ts / case_name
        if tr_case_images.exists():
            shutil.move(str(tr_case_images), str(ts_case_images))
        
        # Move labelsTr to labelsTs
        tr_case_labels = output_labels_tr / case_name
        ts_case_labels = output_labels_ts / case_name
        if tr_case_labels.exists():
            shutil.move(str(tr_case_labels), str(ts_case_labels))
    
    train_processed = len(train_case_names)
    test_processed = len(test_case_names)
    
    # Generate dataset.json
    print("Generating dataset.json...")
    
    # Build channel_names from modality index mapping
    channel_names = {}
    for modality_name, idx in sorted(modality_index_mapping.items(), key=lambda x: x[1]):
        channel_names[modality_name] = str(idx)
    
    dataset_json = {
        "name": "MS-Multispine",
        "description": "Multispinal segmentation dataset from Multispine challenge",
        "tensorImageSize": "3D",
        "numTrain": train_processed,
        "numTest": test_processed,
        "channel_names": channel_names,
        "file_ext": ".nii.gz",
        "train": {},
        "test": {}
    }
    
    # Add semantic classes
    dataset_json["semantic_classes"] = {}
    for class_name, class_id in sorted(semantic_class_mapping.items()):
        dataset_json["semantic_classes"][class_name] = {"id": str(class_id), "optional": True, "semantic_type": "stuff"}
    
    # Add train cases
    for case_name in sorted(train_case_names):
        case_image_file = output_images_tr / case_name / f"{case_name}_0000.nii.gz"
        if case_image_file.exists():
            # Build labels dictionary with hierarchy:
            # labels -> annotator -> semantic class -> instance_1
            labels_dict = {}
            case_labels_dir = output_labels_tr / case_name
            if case_labels_dir.exists():
                for annotator_dir in sorted(case_labels_dir.iterdir()):
                    if annotator_dir.is_dir():
                        for semantic_class_dir in sorted(annotator_dir.iterdir()):
                            if semantic_class_dir.is_dir():
                                class_name = semantic_class_dir.name.replace("semantic_class_", "")
                                seg_file = semantic_class_dir / f"{case_name}_0001.nii.gz"
                                if seg_file.exists():
                                    annotator_name = annotator_dir.name
                                    if annotator_name not in labels_dict:
                                        labels_dict[annotator_name] = {}
                                    labels_dict[annotator_name][class_name] = {
                                        "instance_1": f"./labelsTr/{case_name}/{annotator_name}/{semantic_class_dir.name}/{seg_file.name}"
                                    }
            
            # Build images dict with only available modalities
            images_dict = {}
            for modality_name, idx in sorted(modality_index_mapping.items(), key=lambda x: x[1]):
                image_file = output_images_tr / case_name / f"{case_name}_{idx:04d}.nii.gz"
                if image_file.exists():
                    images_dict[modality_name] = f"./imagesTr/{case_name}/{case_name}_{idx:04d}.nii.gz"
            
            dataset_json["train"][case_name] = {
                "images": images_dict,
                "labels": labels_dict
            }
    
    # Add test cases
    for case_name in sorted(test_case_names):
        case_image_file = output_images_ts / case_name / f"{case_name}_0000.nii.gz"
        if case_image_file.exists():
            # Build labels dictionary with hierarchy:
            # labels -> annotator -> semantic class -> instance_1
            labels_dict = {}
            case_labels_dir = output_labels_ts / case_name
            if case_labels_dir.exists():
                for annotator_dir in sorted(case_labels_dir.iterdir()):
                    if annotator_dir.is_dir():
                        for semantic_class_dir in sorted(annotator_dir.iterdir()):
                            if semantic_class_dir.is_dir():
                                class_name = semantic_class_dir.name.replace("semantic_class_", "")
                                seg_file = semantic_class_dir / f"{case_name}_0001.nii.gz"
                                if seg_file.exists():
                                    annotator_name = annotator_dir.name
                                    if annotator_name not in labels_dict:
                                        labels_dict[annotator_name] = {}
                                    labels_dict[annotator_name][class_name] = {
                                        "instance_1": f"./labelsTs/{case_name}/{annotator_name}/{semantic_class_dir.name}/{seg_file.name}"
                                    }
            
            # Build images dict with only available modalities
            images_dict = {}
            for modality_name, idx in sorted(modality_index_mapping.items(), key=lambda x: x[1]):
                image_file = output_images_ts / case_name / f"{case_name}_{idx:04d}.nii.gz"
                if image_file.exists():
                    images_dict[modality_name] = f"./imagesTs/{case_name}/{case_name}_{idx:04d}.nii.gz"
            
            dataset_json["test"][case_name] = {
                "images": images_dict,
                "labels": labels_dict
            }
    
    # Write dataset.json
    dataset_json_path = output_path / "dataset.json"
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Dataset.json saved to {dataset_json_path}")


if __name__ == "__main__":
    input_directory = "/home/parhomesmaeili/Radiology_Datasets/MS_MultiSpine_dataset/derivatives/preprocessedAndRegistered"  # Update this path
    
    # Define semantic class mapping for output (after instance separation)
    semantic_class_mapping = {
        "lesion": 1,
        "background": 0,
    }
    
    dataset_id = 40  # Adjust as needed
    num_processes = 8
    methodology_fraction = 1.0  # 80% train, 20% test
    
    process_ms_multispine_dataset(
        input_directory, None, dataset_id, 
        semantic_class_mapping=semantic_class_mapping,
        num_processes=num_processes, 
        methodology_fraction=methodology_fraction
    )
    