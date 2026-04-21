import os
from ndindex import Tuple
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

def validate_case(dataset_path, case_name):
    """
    Validate if a case has the required files (image and label).

    Args:
        dataset_path: Path to the dataset directory
        case_name: Name of the case to validate
    
    Returns:
        Tuple    of (is_valid, image_file, label_file) or (False, None, None)
    """
    
    # folder structure: dataset 
                            # - imagesTr
                                # - case_name_0000.nii.gz
                            # - labelsTr
                                # - case_name.nii.gz
    im_path = Path(dataset_path/f"imagesTr/{case_name}_0000.nii.gz")
    label_path = Path(dataset_path/f"labelsTr/{case_name}.nii.gz")
    print(im_path, label_path)
    if not im_path.exists() or not label_path.exists():
        return False, None, None
    

    return True, im_path, label_path


def process_single_case(
    case_name, 
    image_file, 
    label_file, 
    output_images_path, 
    output_labels_path, 
    semantic_class_mapping,
    input_to_output_class_mapping, 
    final_semantic_class_mapping, 
    modality_index_mapping
    ):
    """
    Process a single case directory.
    
    Args:
        case_name: Name of the case
        image_file: Path to image file
        label_file: Path to label file
        output_images_path: Output directory for images
        output_labels_path: Output directory for labels
        semantic_class_mapping: Dict mapping input semantic class names to input label IDs
        input_to_output_class_mapping: Dict mapping OUTPUT class names to list of INPUT class names
        final_semantic_class_mapping: Dict mapping OUTPUT semantic class names to OUTPUT label IDs
        modality_index_mapping: Dict mapping modality names to their channel indices (e.g., {"T2": 0, "FLAIR": 1})
    
    Returns:
        True if case was processed, False otherwise
    """
    output_images_path = Path(output_images_path)
    output_labels_path = Path(output_labels_path)

    # Create output directories
    output_images_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)
    
    output_case_images_dir = output_images_path / case_name
    output_case_images_dir.mkdir(parents=True, exist_ok=True)
    
    output_case_labels_dir = output_labels_path / case_name
    output_case_labels_dir.mkdir(parents=True, exist_ok=True)


    
    # Read image (primary modality, always 0000)
    image_itk = sitk.ReadImage(str(image_file))
    image_array = sitk.GetArrayFromImage(image_itk)
    image_spacing = image_itk.GetSpacing()
    image_origin = image_itk.GetOrigin()
    image_direction = image_itk.GetDirection()
    
    # Read label mask and separate by label ID
    label_itk = sitk.ReadImage(str(label_file))
    label_array = sitk.GetArrayFromImage(label_itk).astype(np.uint8)
    label_spacing = label_itk.GetSpacing()
    label_origin = label_itk.GetOrigin()
    label_direction = label_itk.GetDirection()

    if not image_array.shape == label_array.shape:
        warnings.warn(f"Image and label shapes do not match for case {case_name}. Skipping case.")
        return False
    if not np.allclose(image_spacing, label_spacing):
        warnings.warn(f"Image and label spacings do not match for case {case_name}. Skipping case.")
        return False
    if not np.allclose(image_origin, label_origin):
        warnings.warn(f"Image and label origins do not match for case {case_name}. Skipping case.")
        return False
    if not np.allclose(image_direction, label_direction):
        warnings.warn(f"Image and label directions do not match for case {case_name}. Skipping case.")
        return False

    # Write image (channel index 0000) after checking consistency of metadata
    output_image_file = output_case_images_dir / f"{case_name}_0000.nii.gz"
    sitk.WriteImage(image_itk, str(output_image_file))
    
    # Extract unique label IDs (excluding 0/background if present)
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    
    # Check if any foreground labels exist in original mask.
    if len(unique_labels) == 0:
        warnings.warn(f"No foreground labels found in label mask for case {case_name}. Skipping case.")
        return False
    
    #Check if the any foreground labels exist in the output semantic class mapping.
    fg_available = False 
    for output_class_name, output_id in final_semantic_class_mapping.items():
        if output_id != 0:
            #Now lets collect the ids of the input classes that map to this and check if any of them are present in the
            #unique labels.
            input_class_names = input_to_output_class_mapping[output_class_name]
            input_class_ids = [semantic_class_mapping[input_class_name] for input_class_name in input_class_names]
            if any(input_class_id in unique_labels for input_class_id in input_class_ids):
                fg_available = True
                break

    if not fg_available:
        warnings.warn(f"No foreground labels found in label mask for case {case_name} after applying class mapping. Skipping case.")
        return False        


    
    annotator_id = 1  # Single annotator since no multiple annotators
    #Lets do the semantic class mapping (its just 0-> 0, 1->1 in this case but we will keep the same logic as before for readability)

    for output_class_name, input_class_names in input_to_output_class_mapping.items():
        # Merge all input classes that map to this output class
        class_seg = None
        for input_class_name in input_class_names:
            input_label_id = semantic_class_mapping[input_class_name]
            input_mask = (label_array == input_label_id).astype(np.uint8)
            class_seg = input_mask if class_seg is None else (class_seg | input_mask)
    
        # Create annotator and semantic class folders using output class name
        class_annotator_dir = output_case_labels_dir / f"annotator_{annotator_id}" / f"semantic_class_{output_class_name}"
        class_annotator_dir.mkdir(parents=True, exist_ok=True)
        # Save with instance ID = 1 (semantic seg has single instance per class) in .nii.gz format
        output_file = class_annotator_dir / f"{case_name}_0001.nii.gz"
        class_seg_itk = sitk.GetImageFromArray(class_seg)
        # Apply spacing and orientation from original segmentation file
        class_seg_itk.SetSpacing(label_spacing)
        class_seg_itk.SetOrigin(label_origin)
        class_seg_itk.SetDirection(label_direction)
        sitk.WriteImage(class_seg_itk, str(output_file))
    
    return True

def process_topcowMR_dataset(
    dataset_dir, 
    output_dir, 
    dataset_id, 
    semantic_class_mapping,
    input_to_output_class_mapping,
    final_semantic_class_mapping, 
    num_processes=1, methodology_fraction=1.0):
    """
    Process TopCOW-MR dataset with label-based instance separation.
    
    Args:
        dataset_dir: Path to dataset dir
        DEPRECATED - output_dir: Path to output directory where processed dataset will be saved
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
    target_dataset_name = f"Dataset{dataset_id:03d}_TopCowMR"
    target_folder = os.path.join(datasets_path, target_dataset_name)
    
    dataset_path = Path(dataset_dir)
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
    all_case_names = [case_file.name.replace('.nii.gz', '') for case_file in sorted(Path(f"{dataset_path}/labelsTr").iterdir()) if case_file.is_file()]
    print(f"Found {len(all_case_names)} case directories")
    
    if not all_case_names:
        print(f"No cases found in {dataset_dir}")
        return
    
    # Validate cases and collect unique complementary modality names
    print("Validating cases...")
    validated_cases = []
    unique_modalities = set()
    
    for case_name in all_case_names:
        is_valid, image_file, label_file = validate_case(dataset_path=dataset_path, case_name=case_name)
        if is_valid:
            validated_cases.append((case_name, image_file, label_file))

    print(f"Found {len(validated_cases)} valid cases")
    
    if not validated_cases:
        print("No valid cases found!")
        return
    
    # Create modality index mapping (MRA=0 only)
    modality_index_mapping = {"MRA": 0}
    for idx, modality_name in enumerate(sorted(unique_modalities), start=1):
        modality_index_mapping[modality_name] = idx
    
    print(f"Modality index mapping: {modality_index_mapping}")
    
    # Process all valid cases
    print(f"Processing {len(validated_cases)} cases...")
    if num_processes == 1:
        results = [process_single_case(case_name, image_file, label_file,
                                       output_images_tr, output_labels_tr, 
                                       semantic_class_mapping, input_to_output_class_mapping, final_semantic_class_mapping, modality_index_mapping)
                   for case_name, image_file, label_file in validated_cases]
    else:
        with multiprocessing.Pool(num_processes) as p:
            results = p.starmap_async(
                process_single_case,
                [(case_name, image_file, label_file,
                  output_images_tr, output_labels_tr,
                  semantic_class_mapping, input_to_output_class_mapping, final_semantic_class_mapping, modality_index_mapping)
                 for case_name, image_file, label_file in validated_cases]
            )
            results = results.get()
    
    # Extract just the case names from processed cases
    processed_case_names = []
    for i, result in enumerate(results):
        if result:
            case_name = validated_cases[i][0]
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
        "name": "TopCOW-MR",
        "description": "TopCOW-MR dataset for circle of willis segmentation",
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
    for class_name, class_id in sorted(final_semantic_class_mapping.items()):
        dataset_json["semantic_classes"][class_name] = {"id": str(class_id), "optional": True, "semantic_type": "stuff"}
    
    # Add train cases
    for case_name in sorted(train_case_names):
        case_image_file = output_images_tr / case_name / f"{case_name}_0000.nii.gz"
        if case_image_file.exists():
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
                                    if class_name not in labels_dict:
                                        labels_dict[class_name] = {}
                                    annotator_name = annotator_dir.name
                                    labels_dict[class_name][annotator_name] = f"./labelsTr/{case_name}/{annotator_name}/{semantic_class_dir.name}/{seg_file.name}"
            
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
                                    if class_name not in labels_dict:
                                        labels_dict[class_name] = {}
                                    annotator_name = annotator_dir.name
                                    labels_dict[class_name][annotator_name] = f"./labelsTs/{case_name}/{annotator_name}/{semantic_class_dir.name}/{seg_file.name}"
            
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
    input_directory = "/home/parhomesmaeili/Radiology_Datasets/topcow2024_mr"  # Update this path
    
    dataset_id = 33  # Adjust as needed
    num_processes = 8
    methodology_fraction = 1.0  # 80% train, 20% test
    

    # Define semantic class mapping for output
    semantic_class_mapping = {
        "BA": 1,
        "R-PCA": 2,
        "L-PCA": 3,
        "R-ICA": 4,
        "R-MCA": 5,
        "L-ICA": 6,
        "L-MCA": 7,
        "R-Pcom": 8,
        "L-Pcom": 9,
        "Acom": 10,
        "R-ACA": 11,
        "L-ACA": 12,
        "3rd-A2": 15,
        "background": 0
    }
    input_to_output_map = {
        "cow": ["BA", "R-PCA", "L-PCA", "R-ICA", "R-MCA", "L-ICA", "L-MCA", "R-Pcom", "L-Pcom", "Acom", "R-ACA", "L-ACA", "3rd-A2"],
        "background": ["background"],
    }
    final_map = {
        "cow": 1,
        "background": 0,
    }
    process_topcowMR_dataset(
        input_directory, None, dataset_id, 
        semantic_class_mapping=semantic_class_mapping,
        input_to_output_class_mapping=input_to_output_map,
        final_semantic_class_mapping=final_map,
        num_processes=num_processes, 
        methodology_fraction=methodology_fraction
    )
    