#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import numpy as np
import SimpleITK as sitk

codebase_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, codebase_dir)

from src.general_utils.dict_utils import extract_config, extractor, dict_deep_equals, has_path
from src.data.utils import init_task_cases


def resolve_experiment_config(
    dataset_name, experiment_conf_id, experiment_basename, run_num,
    data_root, configs_root, checkpoint_root,
):
    input_dataset_dir = os.path.join(data_root, 'datasets', dataset_name)
    exp_conf_dir = os.path.join(configs_root, dataset_name)
    exp_manifest_path = os.path.join(exp_conf_dir, 'experiment_manifest.json')
    task_conf_path = os.path.join(exp_conf_dir, 'task_configs.txt')
    prompter_manifest_path = os.path.join(exp_conf_dir, 'prompter_manifest.json')
    prompt_conf_path = os.path.join(exp_conf_dir, 'prompts_configs.txt')
    metric_conf_path = os.path.join(exp_conf_dir, 'metrics_configs.txt')

    exp_config = extract_config(exp_manifest_path, f'exp_config_{experiment_conf_id}')

    orig_task_configs = extract_config(task_conf_path, None)
    orig_prompter_manifest = extract_config(prompter_manifest_path, None)
    orig_metric_configs = extract_config(metric_conf_path, None)

    task_id = exp_config['task']['task_id']
    metric_id = exp_config['metrics']['metrics_config_id']
    prompter_id = exp_config['prompter']['prompter_id']

    assert dict_deep_equals(
        extractor(orig_task_configs, (task_id,)),
        exp_config['task']['config'],
    )
    assert dict_deep_equals(
        extractor(orig_metric_configs, (metric_id,)),
        exp_config['metrics']['config'],
    )
    assert dict_deep_equals(
        extractor(orig_prompter_manifest, (prompter_id,)),
        exp_config['prompter']['config'],
    )

    task_configs = exp_config['task']['config']
    prompter_configs = exp_config['prompter']['config']
    metric_configs = exp_config['metrics']['config']

    assert has_path(task_configs, ('data_sampling', 'sample_group_category'))
    assert has_path(task_configs, ('data_sampling', 'image_conf'))
    assert has_path(task_configs, ('infer_info',))
    assert has_path(task_configs, ('seg_problem',))
    assert has_path(task_configs, ('data_transforms', 'semantic_class_mapping'))

    assert has_path(prompter_configs, ('annotation_conf',))

    assert has_path(metric_configs, ('metrics',))
    assert has_path(metric_configs, ('data_sampling', 'annotation_conf'))

    try:
        spacing_config = extract_config(
            os.path.join(exp_conf_dir, 'spacing_config.json'), None
        )
    except Exception:
        spacing_config = None

    dataset_level_data_schema = {
        'dataset_name': dataset_name,
        'dataset_image_channels': extract_config(
            os.path.join(data_root, 'datasets', dataset_name, 'dataset.json'),
            'channel_names',
        ),
        'task_channels': extractor(
            task_configs, ('data_sampling', 'image_conf', 'image_channel')
        ),
        'spacing_info': spacing_config,
    }

    experiment_name = f'{experiment_basename}_{run_num}'
    checkpoint_path = os.path.join(
        checkpoint_root, experiment_name + '.pkl'
    )

    return {
        'input_dataset_dir': input_dataset_dir,
        'task_configs': task_configs,
        'prompter_configs': prompter_configs,
        'metric_configs': metric_configs,
        'dataset_level_data_schema': dataset_level_data_schema,
        'checkpoint_path': checkpoint_path,
    }


def monai_to_itk(meta_img, dtype=np.float32):
    """Convert a MONAI MetaTensor to a SimpleITK image.

    Matches the convention in kits23-fg/convert_to_framework.py:save_nifti_images():
    transpose axes, flip x/y in affine, extract spacing/origin/direction.
    """
    array = meta_img.numpy()
    affine = meta_img.affine.numpy().copy()

    if array.shape[0] == 1:
        array = array[0]
    array = np.transpose(array).copy().astype(dtype)

    convert_aff = np.diag([-1, -1, 1, 1])
    affine = convert_aff @ affine

    dim = affine.shape[0] - 1
    m_key = (slice(-1), slice(-1))
    origin = affine[slice(-1), -1]
    spacing = np.linalg.norm(affine[m_key] @ np.eye(dim), axis=0)
    direction = affine[m_key] @ np.diag(1.0 / spacing)

    sitk_img = sitk.GetImageFromArray(array)
    sitk_img.SetSpacing(spacing.tolist())
    sitk_img.SetOrigin(origin.tolist())
    sitk_img.SetDirection(direction.flatten().tolist())
    return sitk_img


def save_triplet(case_dict, case_staging_dir):
    """Save image, eval_label, and reference_label from a Dataset case dict.

    The Dataset is produced by init_task_cases() and applies the full MONAI
    transform pipeline (LoadImaged -> EnsureChannelFirstd -> Orientationd ->
    MergeImChannels / MergeSegmentations). We convert each MetaTensor to
    SimpleITK and write as nifti.
    """
    os.makedirs(case_staging_dir, exist_ok=True)

    img_path = os.path.join(case_staging_dir, "image.nii.gz")
    sitk_img = monai_to_itk(case_dict['image'], dtype=np.float32)
    sitk.WriteImage(sitk_img, img_path, useCompression=True)

    eval_path = os.path.join(case_staging_dir, "eval_label.nii.gz")
    sitk_eval = monai_to_itk(case_dict['eval_label'], dtype=np.uint8)
    sitk.WriteImage(sitk_eval, eval_path, useCompression=True)

    ref_path = os.path.join(case_staging_dir, "reference_label.nii.gz")
    sitk_ref = monai_to_itk(case_dict['reference_label'], dtype=np.uint8)
    sitk.WriteImage(sitk_ref, ref_path, useCompression=True)

    return img_path


def build_dataset_level_schema(dataset_level_data_schema, semantic_id_dict, full_image_cache, input_dataset_dir):
    # full_image_cache from init_task_cases:
    #   {case_id: {"images": {ch_name: rel_path, ...}, "labels": None}}
    # rel paths come from dataset.json — wrap with input_dataset_dir to make absolute.
    # 'labels' is always None (removed by init_task_cases), so we guard with isinstance.
    schema = {
        'data_schema': dataset_level_data_schema,
        'segmentation_task_schema': {'semantic_id_dict': semantic_id_dict},
        'full_image_cache': {
            case_id: {
                k_1: {
                    k_2: os.path.abspath(os.path.join(input_dataset_dir, v_2))
                    for k_2, v_2 in v_1.items()
                } if isinstance(v_1, dict) else v_1
                for k_1, v_1 in case_cache.items()
            }
            for case_id, case_cache in full_image_cache.items()
        },
    }
    return schema


def main():
    parser = argparse.ArgumentParser(
        description='Export napari config for CLoPA adapted model inference'
    )
    parser.add_argument('--dataset_name', type=str, default='Dataset005_Prostate',
                        help='Name of the dataset subfolder under datasets/')
    parser.add_argument('--experiment_conf_id', type=int, default=6,
                        help='Experiment config ID from the experiment_manifest.json')
    parser.add_argument('--experiment_basename', type=str, required=True,
                        help='Experiment basename for .pkl lookup '
                             '(e.g. post_refactor_experiment6)')
    parser.add_argument('--run_num', type=str, default='run1',
                        help='Run number for .pkl filename')
    parser.add_argument('--output', type=str, required=True,
                        help='Root directory for all output files. An experiment-specific '
                             'subdirectory <dataset>/<basename>_<run>/ is created inside, '
                             'containing config.json and (if --preprocess) per-case '
                             'image/label triplets.')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory containing datasets/')
    parser.add_argument('--configs_root', type=str, default=None,
                        help='Root directory containing exp_configs/')
    parser.add_argument('--continue_exec_root', type=str, default=None,
                        help='Root directory containing .pkl checkpoint files')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run MONAI dataloader transforms (RAS orientation, '
                             'channel/label merging) and save processed triplet '
                             '(image, eval_label, reference_label) per case')
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = codebase_dir
    if args.configs_root is None:
        args.configs_root = os.path.join(codebase_dir, 'exp_configs')
    if args.continue_exec_root is None:
        args.continue_exec_root = os.path.join(
            codebase_dir, 'continue_execution_files'
        )

    experiment_name = f'{args.experiment_basename}_{args.run_num}'
    experiment_dir = os.path.join(args.output, args.dataset_name, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    exp_config = resolve_experiment_config(
        dataset_name=args.dataset_name,
        experiment_conf_id=args.experiment_conf_id,
        experiment_basename=args.experiment_basename,
        run_num=args.run_num,
        data_root=args.data_root,
        configs_root=args.configs_root,
        checkpoint_root=args.continue_exec_root,
    )

    semantic_id_dict, full_image_cache, dataloader = init_task_cases(
        dataset_dir=exp_config['input_dataset_dir'],
        exp_task_configs=exp_config['task_configs'],
        metric_configs=exp_config['metric_configs'],
        prompter_configs=exp_config['prompter_configs'],
        shuffle_bool=False,
        random_seed=None,
        last_completed_case=None,
        last_completed_idx=None,
    )

    if args.preprocess:
        for case_dict in dataloader:
            case_id = case_dict['case_name']
            case_dir = os.path.join(experiment_dir, case_id)
            img_path = save_triplet(case_dict, case_dir)
            if case_id in full_image_cache:
                full_image_cache[case_id]['images'] = {"merged": img_path}

    dataset_level_schema = build_dataset_level_schema(
        dataset_level_data_schema=exp_config['dataset_level_data_schema'],
        semantic_id_dict=semantic_id_dict,
        full_image_cache=full_image_cache,
        input_dataset_dir=exp_config['input_dataset_dir'],
    )

    pkl_path = exp_config['checkpoint_path']
    default_episode_number = None

    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            checkpoint = pickle.load(f)

        algo_state = checkpoint.get('algorithm_state', {})
        meta_state = algo_state.get('meta_algorithm_state', {})
        adaptation_number = meta_state.get('adaptation_number')
        if adaptation_number is not None:
            default_episode_number = adaptation_number - 1

    output = {
        'dataset_level_schema': dataset_level_schema,
        'checkpoint_path': pkl_path,
        'default_episode_number': default_episode_number,
    }

    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Config written to {config_path}')
    if default_episode_number is not None:
        print(f'Default episode: {default_episode_number}')
    else:
        print('Default episode: null (widget will pick latest)')


if __name__ == '__main__':
    main()
