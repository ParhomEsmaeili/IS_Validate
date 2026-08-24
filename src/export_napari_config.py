#!/usr/bin/env python3
"""
export-napari-config: bundle an experiment configuration for initialising a
continually adapted method in an interactive front-end.

Given a dataset, experiment config ID, a required --sample_group_category, and
an optional adapted-model checkpoint, this script:

  1. Resolves the experiment config from the validation framework's config
     registry (task configs, prompter configs, metric configs). The case set
     browsable in the front-end comes from --sample_group_category, not from
     the experiment's own sample_group_category — those are the cases the
     experiment was actually trained/adapted on, which the front-end must not
     expose.

  2. Builds a dataset-level schema containing:
       - dataset metadata (channels, spacing)
       - semantic class mapping (e.g. background=0, whole_prostate=1)
       - full image cache with absolute paths to every case in
         --sample_group_category

  3. Checks for an existing algorithm-state checkpoint (.pkl) to determine
     the default adaptation episode number, and cross-checks
     --sample_group_category's cases against the checkpoint's own recorded
     adaptation-eligible case pool — the export fails if any overlap.

  4. Writes everything to config.json with the following fields:

       dataset_level_schema:
         data_schema:
           dataset_name           Name of the dataset
           dataset_image_channels Channel name to index mapping
           task_channels          Selected channels for this task
           spacing_info           Median spacing and anisotropy metadata
         segmentation_task_schema:
           semantic_id_dict       Class name to integer ID mapping
         full_image_cache         Case ID -> absolute paths to image files

       checkpoint_path            Path to the algorithm-state .pkl (or null)
       default_episode_number     Adaptation episode to load (null = latest)

     This bundles the full experiment configuration and checkpoint reference,
     allowing the same task to be reproduced in the front-end for interactive
     experimentation with the continually adapted method.

  With --preprocess, also runs the MONAI transform pipeline
  (LoadImaged -> Orientationd -> channel/label merging) on every case
  and writes per-case nifti triplets (image, eval_label, reference_label)
  so the front-end receives data in a consistent format (RAS orientation,
  merged channels/labels). The image cache is updated to point to these
  preprocessed files instead of the originals.
"""
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
    data_root, configs_root, checkpoint_root, sample_group_category,
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
    )[0]
    assert dict_deep_equals(
        extractor(orig_metric_configs, (metric_id,)),
        exp_config['metrics']['config'],
    )[0]
    assert dict_deep_equals(
        extractor(orig_prompter_manifest, (prompter_id,)),
        exp_config['prompter']['config'],
    )[0]

    task_configs = exp_config['task']['config']
    prompter_configs = exp_config['prompter']['config']
    metric_configs = exp_config['metrics']['config']

    assert has_path(task_configs, ('data_sampling', 'sample_group_category'))
    assert has_path(task_configs, ('data_sampling', 'image_conf'))
    assert has_path(task_configs, ('infer_info',))
    assert has_path(task_configs, ('seg_problem',))
    assert has_path(task_configs, ('data_transforms', 'semantic_class_mapping'))

    # task_configs['data_sampling']['sample_group_category'] as read from the experiment
    # manifest is whatever case set the actual experiment/adaptation run trained on. The
    # front-end needs a case set for interactive testing, which is a genuinely different
    # thing — the cases it browses must be ones the checkpoint has never adapted on — so
    # it's a required, separately-specified input here, not read from the manifest.
    task_configs['data_sampling']['sample_group_category'] = sample_group_category

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


def build_dataset_level_schema(dataset_level_data_schema, semantic_id_dict, full_image_cache, input_dataset_dir, case_present_channels):
    # full_image_cache from init_task_cases (or, with --preprocess, already rewritten
    # to {"merged": path} per case):
    #   {case_id: {"images": {ch_name: rel_path, ...}, "labels": None}}
    # rel paths come from dataset.json — wrap with input_dataset_dir to make absolute.
    # os.path.join discards input_dataset_dir if the path is already absolute (e.g. the
    # --preprocess merged path), so this is safe either way.
    # 'labels' is always None (removed by init_task_cases), so we guard with isinstance.
    # 'images' has already been filtered down to existing channels, and cases with none
    # of the task channels present have already been excluded from full_image_cache, by
    # filter_case_images_and_task_channels() inside init_task_cases() — every case
    # remaining here has an entry in case_present_channels.
    schema = {
        'data_schema': dataset_level_data_schema,
        'segmentation_task_schema': {'semantic_id_dict': semantic_id_dict},
        'full_image_cache': {
            case_id: {
                **{
                    k_1: {
                        k_2: os.path.abspath(os.path.join(input_dataset_dir, v_2))
                        for k_2, v_2 in v_1.items()
                    } if isinstance(v_1, dict) else v_1
                    for k_1, v_1 in case_cache.items()
                },
                'task_channels': case_present_channels[case_id],
            }
            for case_id, case_cache in full_image_cache.items()
        },
    }
    return schema


def main():
    parser = argparse.ArgumentParser(
        description='Bundle experiment config, dataset schema, and checkpoint reference '
                    'for initialising a continually adapted method in an interactive front-end'
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
    parser.add_argument('--sample_group_category', type=str, nargs='+', required=True,
                        help='Case set to expose in the front-end, as <category> '
                             '[fold ...] (e.g. "kfold_5_train fold_4", or a single '
                             '"all_..." category with no folds). Checked against the '
                             'checkpoint\'s own recorded case pool — the export fails '
                             'if any case here was ever eligible for adaptation.')
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
        sample_group_category=args.sample_group_category,
    )

    semantic_id_dict, full_image_cache, task_channels, case_present_channels, fully_missing_case_ids, partially_missing_case_ids, dataloader = init_task_cases(
        dataset_dir=exp_config['input_dataset_dir'],
        exp_task_configs=exp_config['task_configs'],
        metric_configs=exp_config['metric_configs'],
        prompter_configs=exp_config['prompter_configs'],
        shuffle_bool=False,
        random_seed=None,
        last_completed_case=None,
        last_completed_idx=None,
    )
    # Overwrite with the exact value init_task_cases actually used to compute
    # case_present_channels — previously extracted a second, independent time in
    # resolve_experiment_config() (same underlying config, no drift risk, but two
    # computations of the same thing regardless).
    exp_config['dataset_level_data_schema']['task_channels'] = task_channels
    # Unlike run.py, the front-end export tolerates cases missing all task channels — they're
    # simply excluded from the browsable set, not a fatal error (full_image_cache above is
    # already filtered down accordingly by init_task_cases).
    if fully_missing_case_ids:
        print(f'Excluding {len(fully_missing_case_ids)} case(s) missing all task channels: {fully_missing_case_ids}')

    # Cross-check against the checkpoint's own recorded case pool — the set of cases
    # that was actually eligible for adaptation, as recorded in the checkpoint itself
    # (not re-derived from the experiment manifest, which could drift from what the
    # checkpoint actually saw). This is the real leakage guard: --sample_group_category
    # must produce a case set the checkpoint has never had a chance to adapt on.
    pkl_path = exp_config['checkpoint_path']
    checkpoint = None
    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            checkpoint = pickle.load(f)

        checkpoint_case_pool = set(
            checkpoint.get('algorithm_state', {})
            .get('meta_algorithm_state', {})
            .get('dataset_level_schema', {})
            .get('full_image_cache', {})
            .keys()
        )
        leaked_case_ids = set(full_image_cache.keys()) & checkpoint_case_pool
        if leaked_case_ids:
            raise ValueError(
                f'--sample_group_category {args.sample_group_category} includes '
                f'{len(leaked_case_ids)} case(s) that were eligible for adaptation in '
                f'checkpoint {pkl_path}: {sorted(leaked_case_ids)[:10]}'
                f'{"..." if len(leaked_case_ids) > 10 else ""}. '
                'Pick a case set disjoint from the checkpoint\'s adaptation pool.'
            )

    if args.preprocess:
        for case_dict in dataloader:
            case_id = case_dict['case_name']
            if case_id in fully_missing_case_ids:
                continue
            case_dir = os.path.join(experiment_dir, case_id)
            img_path = save_triplet(case_dict, case_dir)
            if case_id in full_image_cache:
                full_image_cache[case_id]['images'] = {"merged": img_path}

    dataset_level_schema = build_dataset_level_schema(
        dataset_level_data_schema=exp_config['dataset_level_data_schema'],
        semantic_id_dict=semantic_id_dict,
        full_image_cache=full_image_cache,
        input_dataset_dir=exp_config['input_dataset_dir'],
        case_present_channels=case_present_channels,
    )

    default_episode_number = None
    if checkpoint is not None:
        algo_state = checkpoint.get('algorithm_state', {})
        meta_state = algo_state.get('meta_algorithm_state', {})
        adaptation_number = meta_state.get('adaptation_number')
        if adaptation_number is not None:
            default_episode_number = adaptation_number - 1

    output = {
        'dataset_level_schema': dataset_level_schema,
        'checkpoint_path': pkl_path,
        'default_episode_number': default_episode_number,
        'sample_group_category': args.sample_group_category,
    }

    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Config written to {config_path}')
    print(f'Sample group category: {args.sample_group_category} ({len(full_image_cache)} case(s))')
    if default_episode_number is not None:
        print(f'Default episode: {default_episode_number}')
    else:
        print('Default episode: null (widget will pick latest)')


if __name__ == '__main__':
    main()
