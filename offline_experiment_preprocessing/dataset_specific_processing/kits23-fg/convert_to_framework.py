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
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
)
import torch
import itk

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


def _log_case_event(level, stage, case_name, reason):
    print(f"[{level}][{stage}] case={case_name} reason={reason}")


def _assert_not_under_source_case(source_case_dir, candidate_path, case_name, tag):
    source_case_dir = Path(source_case_dir).resolve(strict=False)
    candidate_path = Path(candidate_path).resolve(strict=False)
    source_prefix = str(source_case_dir) + os.sep
    if str(candidate_path) == str(source_case_dir) or str(candidate_path).startswith(source_prefix):
        raise RuntimeError(
            f"Unsafe output path for case {case_name}: tag={tag}, path={candidate_path}. "
            f"Refusing to write inside source case directory {source_case_dir}."
        )

def save_nifti_images(output_dict, file_path_dict, case_name):
    output_files = {
        key: output_dict[key][0] for key in file_path_dict.keys() #This creates  adict with
        #name of the file as key, and image as value. 
    }
    standard_shape = None
    for image in output_files.values():
        if standard_shape is None:
            standard_shape = image.shape
        else:
            assert image.shape == standard_shape, f"Image shapes are not the same for case {case_name}, got {image.shape} and {standard_shape}. Please check the original image shapes for this case."


    affines = {
        key: output_dict[key].meta['affine'] for key in file_path_dict.keys()
    } 
    standard_affine = None
    for affine in affines.values():
        if standard_affine is None:
            standard_affine = affine
        else:
            assert torch.allclose(standard_affine, affine), f"Affines are not close for key {affine[0]}, got {affine[1]} and {standard_affine[1]}. Please check the original affines for this case: {case_name}."
    
    if isinstance(output_files['image'], torch.Tensor):
        for key, array in output_files.items(): 
            output_files[key] = array.numpy()

    if isinstance(standard_affine, torch.Tensor):
        standard_affine = standard_affine.numpy()

    if len(standard_shape) >= 2:
        for key, image in output_files.items():
            output_files[key] = image.transpose().copy() #Transposition operation is necessary to convert between the axis ordering in MONAI IO and ITK
    output_files['image'] = output_files['image'].astype(np.float32)

    for key, array in output_files.items():
        if key != 'image':
            output_files[key] = array.astype(np.uint8)

    if standard_affine is not None:
        
        convert_aff_mat = np.diag([-1, -1, 1, 1])
        if standard_affine.shape[0] == 3:
            raise NotImplementedError('We do not yet provide handling for 2D images')
            # if affine.shape[0] == 3:  # Handle RGB (2D Image)
                # convert_aff_mat = np.diag([-1, -1, 1])

        standard_affine = convert_aff_mat @ standard_affine

        dim = standard_affine.shape[0] - 1
        _origin_key = (slice(-1), -1)
        _m_key = (slice(-1), slice(-1))
        origin = standard_affine[_origin_key]
        spacing = np.linalg.norm(standard_affine[_m_key] @ np.eye(dim), axis=0)
        direction = standard_affine[_m_key] @ np.diag(1 / spacing)

        for key, array in output_files.items():
            itk_array = itk.image_from_array(array)
            itk_array.SetDirection(itk.matrix_from_array(direction))
            itk_array.SetSpacing(spacing)
            itk_array.SetOrigin(origin)

            itk.imwrite(itk_array, file_path_dict[key], compression=True)
    else:
        raise ValueError(f"Affine is None for case {case_name}, cannot save NIfTI files without affine information. Please check the original images for this case.")


def validate_case(case_dir, num_annotators):
    """
    Validate if a case has the required files and correct number of annotators.
    
    Args:
        case_dir: Path to case directory
        num_annotators: Expected number of annotators (None to skip check)
    
    Returns:
        True if case is valid, False otherwise
    """
    case_dir = Path(case_dir)
    
    imaging_file = case_dir / "imaging.nii.gz"
    segmentation_file = case_dir / "segmentation.nii.gz"
    instances_dir = case_dir / "instances"
    
    # Check required files and directories exist
    if not (imaging_file.exists() and segmentation_file.exists() and instances_dir.exists()):
        _log_case_event("SKIP", "validate_case", case_dir.name, "missing_required_files_or_instances_dir")
        return False
    
    # Count annotators
    annotators = set()
    
    for instance_file in instances_dir.iterdir():
        stem = instance_file.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        
        parts = stem.split('_annotation-')
        if len(parts) != 2:
            continue
        
        annotator_id = parts[1]
        annotators.add(annotator_id)
    
    # Skip if doesn't match expected number of annotators
    if num_annotators is not None and len(annotators) != num_annotators:
        _log_case_event(
            "SKIP",
            "validate_case",
            case_dir.name,
            f"annotator_count_mismatch expected={num_annotators} got={len(annotators)}",
        )
        return False
    elif len(annotators) < 2:
        _log_case_event("SKIP", "validate_case", case_dir.name, "annotator_count_below_two")
        return False
    
    return True


def process_single_case(case_dir, output_images_path, output_labels_path, semantic_class_mapping, input_to_output_class_mapping, final_semantic_class_mapping, num_annotators):
    """
    Process a single case directory.
    Note: Case should already be validated before calling this function.
    
    Args:
        case_dir: Path to case directory
        output_images_path: Output directory for images (imagesTr/imagesTs)
        output_labels_path: Output directory for labels (labelsTr/labelsTs)
        semantic_class_mapping: Dict mapping INPUT semantic class names to INPUT label IDs
        input_to_output_class_mapping: Dict mapping OUTPUT class names to list of INPUT class names
                                      Example: {"whole_kidney": ["kidney", "tumor"], "background": ["cyst"]}
        final_semantic_class_mapping: Dict mapping OUTPUT semantic class names to OUTPUT label IDs
                                     Example: {"whole_kidney": 1, "background": 0}
        num_annotators: Expected number of annotators (passed for reference)
    
    Returns:
        True if case was processed, False otherwise
    """
    case_dir = Path(case_dir)
    output_images_path = Path(output_images_path)
    output_labels_path = Path(output_labels_path)
    
    imaging_file = case_dir / "imaging.nii.gz"
    segmentation_file = case_dir / "segmentation.nii.gz"
    instances_dir = case_dir / "instances"
    
    # Check required files and directories exist
    if not (imaging_file.exists() and segmentation_file.exists() and instances_dir.exists()):
        _log_case_event("SKIP", "process_single_case", case_dir.name, "missing_required_files_or_instances_dir")
        return False
    
    # Group instance segmentations by semantic class and annotator
    # Format: {semantic_class}_instance-{instance_id}_annotation-{annotator_id}
    class_annotator_instances = defaultdict(lambda: defaultdict(list))
    annotators = set()
    semantic_classes = set()
    
    for instance_file in instances_dir.iterdir():
        # Parse filename: tumor_instance-1_annotation-2.nii.gz
        stem = instance_file.stem
        # Remove .nii if present (from .nii.gz)
        if stem.endswith('.nii'):
            stem = stem[:-4]
        
        # Split by annotation- to separate the annotator_id
        parts = stem.split('_annotation-')
        if len(parts) != 2:
            continue
        
        class_instance_part = parts[0]  # e.g., "tumor_instance-1"
        annotator_id = parts[1]  # e.g., "2"
        
        # Extract semantic class (everything before first underscore of instance)
        class_parts = class_instance_part.rsplit('_instance-', 1)
        if len(class_parts) != 2:
            continue
        
        semantic_class = class_parts[0]  # e.g., "tumor"
        
        class_annotator_instances[semantic_class][annotator_id].append(instance_file)
        annotators.add(annotator_id)
        semantic_classes.add(semantic_class)
    
    # Case should already be validated, but skip if no semantic classes found
    if not semantic_classes:
        raise Exception(f"No semantic classes found in instances for case {case_dir.name} despite passing validation. Please check the instance filenames and semantic_class_mapping.")
    
    # Initialize empty instance lists for any classes referenced in the mapping that don't have instances
    # (e.g., background class which is implicit, not explicit in instance files)
    for output_class_name, input_class_names in input_to_output_class_mapping.items():
        for input_class_name in input_class_names:
            if input_class_name not in class_annotator_instances:
                class_annotator_instances[input_class_name] = defaultdict(list)
            for ann_id in annotators:
                if ann_id not in class_annotator_instances[input_class_name]:
                    class_annotator_instances[input_class_name][ann_id] = []
    
    # Check if any of the output classes have at least one instance from any annotator
    has_foreground_instances = False
    for output_class_name, input_class_names in input_to_output_class_mapping.items():
        output_label_id = final_semantic_class_mapping[output_class_name]
        if output_label_id == 0:  # Skip background
            assert output_class_name == "background", f"Output class {output_class_name} has label ID 0 but is not named 'background'. Please check your final_semantic_class_mapping."
            continue
        for input_class_name in input_class_names:
            for ann_id in annotators:
                if class_annotator_instances[input_class_name][ann_id]:
                    has_foreground_instances = True
                    break
            if has_foreground_instances:
                break
        if has_foreground_instances:
            break
    if not has_foreground_instances:
        _log_case_event("SKIP", "process_single_case", case_dir.name, "no_foreground_instances_for_output_mapping")
        return False
    
    output_images_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)
    
    case_name = case_dir.name
    
    # Create case subdirectory in images directory
    output_case_images_dir = output_images_path / case_name
    output_case_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create path for output imaging file in case-specific images directory (with channel index 0000)
    output_image_file = output_case_images_dir / f"{case_name}_0000.nii.gz"
    _assert_not_under_source_case(case_dir, output_image_file, case_name, "output_image_file")
    
    # Create case subdirectory in labels for all semantic classes and annotators
    output_case_labels_dir = output_labels_path / case_name
    output_case_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Partition and copy consensus segmentation by semantic class
    if str(segmentation_file).endswith('.nii.gz'):
        consensus_itk = sitk.ReadImage(str(segmentation_file))
        consensus_data = sitk.GetArrayFromImage(consensus_itk).astype(np.uint8)
        # Preserve metadata from original segmentation
        seg_spacing = consensus_itk.GetSpacing()
        seg_origin = consensus_itk.GetOrigin()
        seg_direction = consensus_itk.GetDirection()
    else:
        raise ValueError(f"Unsupported file format for consensus segmentation: {segmentation_file}")
    
    # Cache image shape for background generation fallback
    image_shape = consensus_data.shape
    
    # Check if foreground mask is non-empty at image level
    # Merge all foreground (non-zero output label) classes to check if any voxels are foreground
    consensus_foreground = None
    for output_class_name, input_class_names in input_to_output_class_mapping.items():
        output_label_id = final_semantic_class_mapping[output_class_name]
        if output_label_id == 0:  # Skip background
            continue
        for input_class_name in input_class_names:
            input_label_id = semantic_class_mapping[input_class_name]
            class_seg = (consensus_data == input_label_id).astype(np.uint8)
            consensus_foreground = class_seg if consensus_foreground is None else (consensus_foreground | class_seg)
    
    # Skip if consensus foreground is all zeros
    if consensus_foreground is None or np.all(consensus_foreground == 0):
        warnings.warn(f'Consensus foreground is all zeros for case {case_name}, despite the fact that there are instance segmentations in the foreground classes. \n'
                      'This could arise from the fusion strategy, but please check the instance segmentations and consensus segmentation for this case to ensure they are correct and consistent with each other. Skipping this case as it has no foreground in the consensus segmentation after fusion.')
        _log_case_event("SKIP", "process_single_case", case_name, "empty_consensus_foreground_after_mapping")
        return False
    
    consensus_annotator_id = len(annotators) + 1
    
    #Now we crop the images, instance segmentations and consensus
    #according to the foreground region of the consensus segmentation. 

    #NOTE: we assumed a binary semseg task so the foreground region for cropping is only intended for that 
    # given set of labels. this means that we do not run into issues where we would have determiend
    # the fg according to one set of classes, but then cropped a fg determined by another set of 
    # classes. This would have created an inconsistency in how the image and segmentations would
    # have been cropped otherwise. 

    #First we need to process the consensus segmentation into the segmentation file for the specific
    #task at hand (i.e. binary sem seg for whole kidney, whole mass, whole tumour etc0)
    
    # Save consensus segmentation with output semantic applied (merged by output class name)
        #We save this in a temporary directory for now.
    
    temp_seg_dir = output_case_labels_dir
    temp_seg_dir.mkdir(parents=True, exist_ok=True)

    assert len(input_to_output_class_mapping) == 2, f"Expected exactly 2 output classes (foreground and background) in input_to_output_class_mapping, but got {len(input_to_output_class_mapping)}. Please check your mapping."
    for output_class_name, input_class_names in input_to_output_class_mapping.items():
        if output_class_name == 'background':
            continue  # Skip background class for consensus segmentation, as it will be implicitly defined as everything not in foreground classes

        # Merge all input classes that map to this output class
        class_seg = None
        for input_class_name in input_class_names:
            input_label_id = semantic_class_mapping[input_class_name]
            input_mask = (consensus_data == input_label_id).astype(np.uint8)
            class_seg = input_mask if class_seg is None else (class_seg | input_mask)
        
        # Create annotator and semantic class folders using output class name
        class_annotator_dir = temp_seg_dir / f"annotator_{consensus_annotator_id}" / f"semantic_class_{output_class_name}"
        class_annotator_dir.mkdir(parents=True, exist_ok=True)
        # Save with instance ID = 1 (semantic seg has single instance per class) in .nii.gz format
        fg_file = class_annotator_dir / f"{case_name}_0001.nii.gz"
        _assert_not_under_source_case(case_dir, fg_file, case_name, "consensus_foreground_file")
        class_seg_itk = sitk.GetImageFromArray(class_seg)
        # Apply spacing and orientation from original segmentation file
        class_seg_itk.SetSpacing(seg_spacing)
        class_seg_itk.SetOrigin(seg_origin)
        class_seg_itk.SetDirection(seg_direction)
        sitk.WriteImage(class_seg_itk, str(fg_file))
    
    foreground_file = fg_file
    
    #We need to map each of the annotator's instance level segmentations to a single annotator level
    #semantic segmentation file, so that we can apply the cropping using the consensus.

    #We will also save these intermediate files in a temporary directory which must be removed after processing.

    # Fuse instances per output class and annotator (using output mapping) for the foreground class
    fused_instances = dict() 

    for output_class_name, input_class_names in input_to_output_class_mapping.items():
        if output_class_name == 'background':
            continue  # Skip background class for instance fusion, as it will be implicitly defined as everything not in foreground classes

        for annotator_id in annotators:
            # Collect instances from all input classes that map to this output class
            all_instance_files = []
            for input_class_name in input_class_names:
                all_instance_files.extend(class_annotator_instances[input_class_name][annotator_id])
            
            if all_instance_files:
                # Fuse all instances for this output class and annotator
                fused = None
                for inst_file in all_instance_files:
                    if str(inst_file).endswith('.nii.gz') or str(inst_file).endswith('.nii'):
                        inst_itk = sitk.ReadImage(str(inst_file))
                        inst_data = sitk.GetArrayFromImage(inst_itk).astype(np.uint8)
                        if inst_data.shape != image_shape:
                            raise Exception(
                                f"Instance segmentation {inst_file.name} has inconsistent voxel shape compared to consensus segmentation. "
                                f"Instance shape={inst_data.shape}, consensus shape={image_shape}. "
                                f"Please check the instance segmentations and ensure they are aligned with the consensus segmentation."
                            )
                        # Enforce consistent header information with consensus segmentation
                        inst_spacing = inst_itk.GetSpacing()
                        inst_origin = inst_itk.GetOrigin()
                        inst_direction = inst_itk.GetDirection()
                        
                        if inst_spacing != seg_spacing or inst_origin != seg_origin or inst_direction != seg_direction:
                            raise Exception(f"Instance segmentation {inst_file.name} has inconsistent header information compared to consensus segmentation. "
                                          f"Instance: spacing={inst_spacing}, origin={inst_origin}, direction={inst_direction}. "
                                          f"Consensus: spacing={seg_spacing}, origin={seg_origin}, direction={seg_direction}. "
                                          f"Please check the instance segmentations and ensure they are aligned with the consensus segmentation.")
                    else:
                        raise ValueError(f"Unsupported file format for instance segmentation: {inst_file}")
                    fused = inst_data if fused is None else (fused | inst_data)
            else:
                if output_class_name != "background":
                    warnings.warn(f"No instances found for output class '{output_class_name}' and annotator '{annotator_id}' in case {case_name}. This will result in an empty segmentation for this class-annotator combination. Please check the instance segmentations and your input_to_output_class_mapping to ensure this is expected.")
                    if len(final_semantic_class_mapping) == 2:
                        raise Exception('The output semantic mapping is length 2, fg and bg, but there are no instances for the foreground. Should have already been flagged')
                # Empty image for missing output class-annotator combination
                fused = np.zeros(image_shape, dtype=np.uint8)
            
            # Create annotator and semantic class folders using output class name
            class_annotator_dir = temp_seg_dir / f"annotator_{annotator_id}" / f"semantic_class_{output_class_name}"
            class_annotator_dir.mkdir(parents=True, exist_ok=True)
            # Save with instance ID = 1 (semantic seg has single instance per class) in .nii.gz format
            output_file = class_annotator_dir / f"{case_name}_0001.nii.gz"
            _assert_not_under_source_case(case_dir, output_file, case_name, f"fused_instance_file_annotator_{annotator_id}")
            fused_itk = sitk.GetImageFromArray(fused)
            # Apply spacing and orientation from original segmentation file
            fused_itk.SetSpacing(seg_spacing)
            fused_itk.SetOrigin(seg_origin)
            fused_itk.SetDirection(seg_direction)
            sitk.WriteImage(fused_itk, str(output_file))
            
            #Here we store the path to the fused instance segmentation.
            fused_instances[annotator_id] = output_file 

    #NOTE: This process it not ideal, its just easier to reuse code


    #Now lets create the input dict for performing the foreground cropping:

    fg_crop_input_dict = {
        'image': imaging_file,
        'consensus_segmentation': foreground_file,
    }
    for annotator_id in fused_instances:
        fg_crop_input_dict[f'annotator_{annotator_id}_segmentation'] = fused_instances[annotator_id]
    
    #Now we will use the MONAI implementation for loading and cropping, then saving the final
    #cropped images in place. We can then move the temporary files to the permanent location outside
    #of this function (essentially we perform an in-place crop and overwrite the temporary files with the cropped versions, then move them to the final location after processing).)

    #We for now will assume our transforms are fixed configuration so that we don't need dynamic initialisation
    #according to non fixed parameters. 
    transforms = [
        LoadImaged(keys=list(fg_crop_input_dict.keys()), image_only=True, reader='ITKReader'),
        EnsureChannelFirstd(keys=list(fg_crop_input_dict.keys())),
        CropForegroundd(keys=list(fg_crop_input_dict.keys()), source_key='consensus_segmentation', margin=30, allow_smaller=True)
    ]

    #We now execute the transform.
    cropped_output = Compose(transforms)(fg_crop_input_dict)
    #Save the output images 

    output_paths = {
        'image': output_image_file,
        'consensus_segmentation': foreground_file
    }
    for annotator_id in fused_instances:
        output_paths[f'annotator_{annotator_id}_segmentation'] = fused_instances[annotator_id]
    for key, path in output_paths.items():
        _assert_not_under_source_case(case_dir, path, case_name, f"save_target_{key}")
    save_nifti_images(cropped_output, output_paths, case_name)
    
    #We now need to save the background segmentations for each segmentation
    for key, output_file in output_paths.items():
        if key == 'image':
            continue  # Skip image, only want to create background for segmentations
        # Load the cropped segmentation
        seg_itk = sitk.ReadImage(str(output_file))
        seg_data = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
        # Create background by inverting the foreground (assuming binary segmentation after cropping)
        background_data = (seg_data == 0).astype(np.uint8)
        # Save background segmentation as a sibling semantic class directory under the same annotator
        bg_output_file = output_file.parent.parent / "semantic_class_background" / output_file.name
        _assert_not_under_source_case(case_dir, bg_output_file, case_name, f"background_file_from_{key}")
        bg_output_file.parent.mkdir(parents=True, exist_ok=True)
        bg_itk = sitk.GetImageFromArray(background_data)
        # Apply spacing and orientation from original segmentation file
        bg_itk.SetSpacing(seg_spacing)
        bg_itk.SetOrigin(seg_origin)
        bg_itk.SetDirection(seg_direction)
        sitk.WriteImage(bg_itk, str(bg_output_file))

    return True


def process_single_case_safe(case_dir, output_images_path, output_labels_path, semantic_class_mapping, input_to_output_class_mapping, final_semantic_class_mapping, num_annotators):
    case_name = Path(case_dir).name
    try:
        return process_single_case(
            case_dir,
            output_images_path,
            output_labels_path,
            semantic_class_mapping,
            input_to_output_class_mapping,
            final_semantic_class_mapping,
            num_annotators,
        )
    except Exception as e:
        _log_case_event("ERROR", "process_single_case", case_name, f"exception={type(e).__name__}: {e}")
        return False

def process_multi_annotator_segmentations(input_dir, output_dir, dataset_id, semantic_class_mapping=None, input_to_output_class_mapping=None, final_semantic_class_mapping=None, num_annotators=None, num_processes=1, methodology_fraction=1.0):
    """
    Process multi-annotator segmentations with multiprocessing support.
    
    Args:
        input_dir: Path to input directory
        output_dir: Not used - output will be created under datasets/ using dataset_id
        dataset_id: Dataset ID number (will create Dataset{dataset_id:03d}_Kits23)
        semantic_class_mapping: Dict mapping INPUT semantic class names to INPUT label IDs
                               Includes "background": 0
                               Example: {"background": 0, "kidney": 1, "tumor": 2, "cyst": 3}
        input_to_output_class_mapping: Dict mapping OUTPUT class names to list of INPUT class names.
                                       If None, one-to-one mapping is created.
                                       Example: {"whole_kidney": ["kidney", "tumor"], "background": ["cyst"]}
        final_semantic_class_mapping: Dict mapping OUTPUT semantic class names to OUTPUT label IDs.
                                     Must be provided when input_to_output_class_mapping is provided.
                                     Example: {"whole_kidney": 1, "background": 0}
        num_annotators: Expected number of annotators. If provided, only keeps cases with exactly this many annotators.
        num_processes: Number of worker processes for parallelization
        methodology_fraction: Fraction of cases to use for training (0.0 to 1.0). Rest go to test.
                             Default 1.0 means all cases go to train.
    """
    if semantic_class_mapping is None:
        raise Exception('We need the input semantic class mapping to process the dataset')
    
    # Setup input-to-output class mapping and final mapping
    if input_to_output_class_mapping is None:
        # Default: one-to-one mapping (each input class becomes its own output class)
        input_to_output_class_mapping = {class_name: [class_name] for class_name in semantic_class_mapping.keys()}
    
    if final_semantic_class_mapping is None:
        # Default: one-to-one mapping with original label IDs
        final_semantic_class_mapping = {class_name: label_id for class_name, label_id in semantic_class_mapping.items()}
    
    # Validate that final mapping is not empty
    if not final_semantic_class_mapping:
        raise ValueError("Final semantic mapping is empty. Please check your configuration.")
    
    # Create target dataset directory with naming convention: Dataset{id:03d}_{name}
    target_dataset_name = f"Dataset{dataset_id:03d}_Kits23"
    target_folder = os.path.join(datasets_path, target_dataset_name)
    target_folder_path = Path(target_folder)
    
    # Skip conflict check if folder already exists (likely a resume)
    if not target_folder_path.exists():
        existing_datasets = check_dataset_existence(datasets_path, dataset_id)
        assert len(existing_datasets) == 0, f"Target dataset id {dataset_id} is already taken, please consider changing " \
                                            f"it. Conflicting dataset: {existing_datasets}"
    else:
        print(f"Dataset folder already exists - resuming dataset {dataset_id}...")
    
    # Now safe to create the output directory
    input_path = Path(input_dir)
    output_path = target_folder_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create/load checkpoint file
    checkpoint_file = output_path / "processing_checkpoint.json"
    if not checkpoint_file.exists():
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed_cases': []}, f)
    
    # Create temporary directory for all processed cases
    temp_output_path = output_path / "_temp_processing"
    temp_images = temp_output_path / "images"
    temp_labels = temp_output_path / "labels"
    temp_images.mkdir(parents=True, exist_ok=True)
    temp_labels.mkdir(parents=True, exist_ok=True)
    
    # Create imagesTr, labelsTr, imagesTs, labelsTs directories (will populate later)
    output_images_tr = output_path / "imagesTr"
    output_labels_tr = output_path / "labelsTr"
    output_images_ts = output_path / "imagesTs"
    output_labels_ts = output_path / "labelsTs"
    
    output_images_tr.mkdir(parents=True, exist_ok=True)
    output_labels_tr.mkdir(parents=True, exist_ok=True)
    output_images_ts.mkdir(parents=True, exist_ok=True)
    output_labels_ts.mkdir(parents=True, exist_ok=True)
    
    # Collect all case directories
    all_case_dirs = [case_dir for case_dir in sorted(input_path.iterdir()) if case_dir.is_dir()]
    
    if not all_case_dirs:
        print(f"No case directories found in {input_dir}")
        return
    
    # Validate cases - filter to only those with correct number of annotators
    print(f"Validating {len(all_case_dirs)} cases...")
    potential_case_dirs = [case_dir for case_dir in all_case_dirs if validate_case(case_dir, num_annotators)]
    print(f"Found {len(potential_case_dirs)} potential cases with {num_annotators} annotators")
    
    if not potential_case_dirs:
        print("No potential cases found!")
        return
    
    # Load checkpoint if resuming
    processed_case_names = []
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_case_names = checkpoint_data.get('processed_cases', [])
            print(f"Resuming from checkpoint: {len(processed_case_names)} cases already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Starting fresh.")
            processed_case_names = []
    
    # Identify cases that still need processing
    processed_set = set(processed_case_names)
    cases_to_process = [case_dir for case_dir in potential_case_dirs if case_dir.name not in processed_set]
    
    if cases_to_process:
        print(f"Processing {len(cases_to_process)} remaining cases to temporary folder...")
        if num_processes == 1:
            for i, case_dir in enumerate(cases_to_process):
                print(f"  [{i+1}/{len(cases_to_process)}] Processing {case_dir.name}...")
                result = process_single_case_safe(
                    case_dir,
                    temp_images,
                    temp_labels,
                    semantic_class_mapping,
                    input_to_output_class_mapping,
                    final_semantic_class_mapping,
                    num_annotators,
                )
                if result:
                    processed_case_names.append(case_dir.name)
                # Save checkpoint after each case
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed_cases': processed_case_names}, f)
        else:
            with multiprocessing.Pool(num_processes) as p:
                results = p.starmap_async(
                    process_single_case_safe,
                    [(case_dir, temp_images, temp_labels, semantic_class_mapping, input_to_output_class_mapping, final_semantic_class_mapping, num_annotators) for case_dir in cases_to_process]
                )
                results_list = results.get(timeout=None)
                newly_processed = [cases_to_process[i].name for i, result in enumerate(results_list) if result]
                processed_case_names.extend(newly_processed)
            # Save checkpoint after batch processing
            with open(checkpoint_file, 'w') as f:
                json.dump({'processed_cases': processed_case_names}, f)
    else:
        print("All cases already processed!")
    
    print(f"Successfully processed {len(processed_case_names)} cases with non-empty foreground")
    
    if not processed_case_names:
        print("No cases with non-empty foreground found!")
        shutil.rmtree(temp_output_path)
        return
    
    # Split valid cases based on methodology_fraction
    num_train_cases = int(np.ceil(len(processed_case_names) * methodology_fraction))
    train_case_names = random.sample(processed_case_names, k=num_train_cases)
    train_case_set = set(train_case_names)
    test_case_names = [name for name in processed_case_names if name not in train_case_set]
    
    print(f"Splitting into {len(train_case_names)} training and {len(test_case_names)} test cases")
    
    # Move training cases from temp to train folders
    for case_name in train_case_names:
        temp_case_images_dir = temp_images / case_name
        temp_case_labels = temp_labels / case_name
        
        if temp_case_images_dir.exists():
            shutil.move(str(temp_case_images_dir), str(output_images_tr / case_name))
        if temp_case_labels.exists():
            shutil.move(str(temp_case_labels), str(output_labels_tr / case_name))
    
    # Move test cases from temp to test folders
    for case_name in test_case_names:
        temp_case_images_dir = temp_images / case_name
        temp_case_labels = temp_labels / case_name
        
        if temp_case_images_dir.exists():
            shutil.move(str(temp_case_images_dir), str(output_images_ts / case_name))
        if temp_case_labels.exists():
            shutil.move(str(temp_case_labels), str(output_labels_ts / case_name))
    
    # Clean up temp folder
    shutil.rmtree(temp_output_path)
    
    train_processed = len(train_case_names)
    test_processed = len(test_case_names)
    
    # Generate dataset.json
    print("Generating dataset.json...")
    dataset_json = {
        "name": "KITS23",
        "description": "Kidney segmentation dataset with multi-annotator labels",
        "reference": "https://kits-challenge.org/",
        "licence": "CC-BY 4.0",
        "converted_by": "Parhom Esmaeili",
        "tensorImageSize": "3D",
        "numTrain": train_processed,
        "numTest": test_processed,
        "annotators": {},
        "semantic_classes": {},
        "channel_names": {
            "CT": "0"
        },
        "file_ext": ".nii.gz",
        "train": {},
        "test": {}
    }
    
    # Add annotators from processed cases (extract from original input directories)
    all_annotators = set()
    for case_name in processed_case_names:
        for potential_case in potential_case_dirs:
            if potential_case.name == case_name:
                instances_dir = potential_case / "instances"
                if instances_dir.exists():
                    for instance_file in instances_dir.iterdir():
                        stem = instance_file.stem
                        if stem.endswith('.nii'):
                            stem = stem[:-4]
                        # Extract annotator ID from filename
                        if '_annotation-' in stem:
                            annotator_id = stem.split('_annotation-')[-1]
                            all_annotators.add(annotator_id)
                break
    
    for annotator_id in sorted(all_annotators):
        dataset_json["annotators"][f"annotator_{annotator_id}"] = {
            "annotator_id": annotator_id,
            "annotation_protocol": "KITS23 official protocol"
        }
    
    # Add semantic classes from final_semantic_class_mapping
    for output_class_name, output_label_id in sorted(final_semantic_class_mapping.items()):
        dataset_json["semantic_classes"][output_class_name] = {
            "id": str(output_label_id),
            "optional": True,
            "semantic_type": "stuff"
        }
    
    # Add training cases
    for case_name in train_case_names:
        if (output_images_tr / case_name / f"{case_name}_0000.nii.gz").exists():
            # Build labels dictionary with hierarchy:
            # labels -> annotator -> semantic class -> instance_1
            labels_dict = {}
            case_labels_dir = output_labels_tr / case_name
            if case_labels_dir.exists():
                for annotator_dir in sorted(case_labels_dir.iterdir()):
                    if annotator_dir.is_dir():
                        annotator_name = annotator_dir.name
                        if annotator_name not in labels_dict:
                            labels_dict[annotator_name] = {}
                        for semantic_class_dir in sorted(annotator_dir.iterdir()):
                            if semantic_class_dir.is_dir():
                                semantic_class_name = semantic_class_dir.name.replace("semantic_class_", "")
                                seg_file = semantic_class_dir / f"{case_name}_0001.nii.gz"
                                if seg_file.exists():
                                    labels_dict[annotator_name][semantic_class_name] = {
                                        "instance_1": f"./labelsTr/{case_name}/{annotator_dir.name}/{semantic_class_dir.name}/{seg_file.name}"
                                    }
            
            dataset_json["train"][case_name] = {
                "images": {
                    "CT": f"./imagesTr/{case_name}/{case_name}_0000.nii.gz"
                },
                "labels": labels_dict
            }
    
    # Add test cases
    for case_name in test_case_names:
        if (output_images_ts / case_name / f"{case_name}_0000.nii.gz").exists():
            # Build labels dictionary with hierarchy:
            # labels -> annotator -> semantic class -> instance_1
            labels_dict = {}
            case_labels_dir = output_labels_ts / case_name
            if case_labels_dir.exists():
                for annotator_dir in sorted(case_labels_dir.iterdir()):
                    if annotator_dir.is_dir():
                        annotator_name = annotator_dir.name
                        if annotator_name not in labels_dict:
                            labels_dict[annotator_name] = {}
                        for semantic_class_dir in sorted(annotator_dir.iterdir()):
                            if semantic_class_dir.is_dir():
                                semantic_class_name = semantic_class_dir.name.replace("semantic_class_", "")
                                seg_file = semantic_class_dir / f"{case_name}_0001.nii.gz"
                                if seg_file.exists():
                                    labels_dict[annotator_name][semantic_class_name] = {
                                        "instance_1": f"./labelsTs/{case_name}/{annotator_dir.name}/{semantic_class_dir.name}/{seg_file.name}"
                                    }
            
            dataset_json["test"][case_name] = {
                "images": {
                    "CT": f"./imagesTs/{case_name}/{case_name}_0000.nii.gz"
                },
                "labels": labels_dict
            }
    
    # Write dataset.json
    dataset_json_path = output_path / "dataset.json"
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Dataset.json saved to {dataset_json_path}")


if __name__ == "__main__":
    input_directory = "/home/parhomesmaeili/Radiology_Datasets/kits23/dataset"
    
    # Define semantic class to label ID mapping (INPUT - from the data)
    semantic_class_mapping = {
        "background": 0,
        "kidney": 1,
        "tumor": 2,
        "cyst": 3,
    }
    
    # Expected number of annotators
    num_annotators = 3
    
    # Fraction of cases to use for training
    methodology_fraction = 1.0
    
    # Number of worker processes (set to 1 for safer single-process mode, increase if you have many CPU cores and want parallelization)
    num_processes = 1
    
    # # ===== Example 2: Binary kidney dataset (kidneys, tumors, cysts foreground, others background) =====
    # # Merges all foregrounds into one class. This is the most inclusive definition of foreground, and should have the most cases with non-empty foreground, as any instance of kidney, tumor or cyst will count towards the foreground.
    # dataset_id = 23
    # input_to_output_map = {
    #     "whole_kidney": ["kidney", "cyst", "tumor"],
    #     "background": ["background"],
    # }
    # final_map = {
    #     "whole_kidney": 1,
    #     "background": 0,
    # }
    
    # process_multi_annotator_segmentations(
    #     input_directory, None, dataset_id, 
    #     semantic_class_mapping,
    #     input_to_output_class_mapping=input_to_output_map,
    #     final_semantic_class_mapping=final_map,
    #     num_annotators=num_annotators, 
    #     num_processes=num_processes, 
    #     methodology_fraction=methodology_fraction
    # )

    # # # ===== Example 2: Binary tumor dataset (tumor foreground, others background) =====
    # # # Maps kidney and cyst into background, keeps tumor separate
    # dataset_id = 27
    # input_to_output_map = {
    #     "tumor": ["tumor"],
    #     "background": ["background", "kidney", "cyst"],
    # }
    # final_map = {
    #     "tumor": 1,
    #     "background": 0,
    # }
    
    # process_multi_annotator_segmentations(
    #     input_directory, None, dataset_id, 
    #     semantic_class_mapping,
    #     input_to_output_class_mapping=input_to_output_map,
    #     final_semantic_class_mapping=final_map,
    #     num_annotators=num_annotators, 
    #     num_processes=num_processes, 
    #     methodology_fraction=methodology_fraction
    # )

    # ===== Example 3: Binary mass dataset (tumor and cyst foreground, others background) =====
    # Maps kidney to background, keeps mass separate
    dataset_id = 31
    input_to_output_map = {
        "mass": ["tumor", "cyst"],
        "background": ["background", "kidney"],
    }
    final_map = {
        "mass": 1,
        "background": 0,
    }
    
    process_multi_annotator_segmentations(
        input_directory, None, dataset_id, 
        semantic_class_mapping,
        input_to_output_class_mapping=input_to_output_map,
        final_semantic_class_mapping=final_map,
        num_annotators=num_annotators, 
        num_processes=num_processes, 
        methodology_fraction=methodology_fraction
    )



    # # ===== Example 4: Debugging - Binary cyst dataset (cyst foreground, others background) =====
    # # Maps cyst into fg this is good for checkign that we filter out cases with empty fg because cyst is not always
    # # present in the instances, so should have fewer cases than the tumor foreground version.
    # dataset_id = 999
    # input_to_output_map = {
    #     "cyst": ["cyst"],
    #     "background": ["background", "kidney", "tumor"],
    # }
    # final_map = {
    #     "cyst": 1,
    #     "background": 0,
    # }
    
    # process_multi_annotator_segmentations(
    #     input_directory, None, dataset_id, 
    #     semantic_class_mapping,
    #     input_to_output_class_mapping=input_to_output_map,
    #     final_semantic_class_mapping=final_map,
    #     num_annotators=num_annotators, 
    #     num_processes=num_processes, 
    #     methodology_fraction=methodology_fraction
    # )
    