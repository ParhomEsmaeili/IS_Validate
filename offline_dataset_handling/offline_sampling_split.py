import json
import argparse
import os
import re
import random

# This is a basic utility script for sampling cases, it is not capable of handling complexities such as determining the
# annotator in a multi-annotator setting, or the different instances in a multi-instance setting etc. It is only designed
# to sample cases from a dataset structure defined in a dataset.json file, which is expected to be present in the dataset directory.

# class SingleLineEncoder(json.JSONEncoder):
#     """
#     Custom encoder to print all list values (except 'meta') under all sample strategies as single-line arrays.
#     """
#     def encode(self, obj):
#         s = super().encode(obj)
#         if 'sampling' in obj:
#             for strat_key, strat_dict in obj['sampling'].items():
#                 for key, value in strat_dict.items():
#                     if key != "meta" and isinstance(value, list):
#                         arr_json = json.dumps(value, ensure_ascii=False)
#                         s = re.sub(
#                             rf'("{key}": )\[[\s\S]*?\]',
#                             r'\1' + arr_json,
#                             s
#                         )
#         return s

def load_dataset_structure(dataset_dir):
    dataset_json_path = os.path.join(dataset_dir, "dataset.json")
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
    return data

def get_case_ids(data, split):
    if split not in data:
        raise ValueError(f"Split '{split}' not found in dataset.json")
    return list(data[split].keys())

def sample_cases(case_ids, strategy_config):
    """
    strategy_config: dict with at least a 'strategy_type' key.
    Supported strategy types:
    - 'all': returns {'all_cases': [...]}
    - 'kfold': returns {'fold_0': [...], 'fold_1': [...], ...}
    - (future strategies can add more keys as needed)

    Each strategy type can have additional parameters, e.g.:
    - 'kfold': requires 'k_folds' key to specify number of folds, which will be used to split the case_ids into k folds.
    - 'all': does not require additional parameters.
    - 'shuffle_off': bool, whether the shuffle is off before splitting (default: False)
    - 'shuffle_seed': int, optional, for reproducibility

    returns a dictionary with the relevant partitioning:
    e.g., for 'all' strategy:
    {'all_cases': ['case1', 'case2', ...]}
    or for 'kfold' strategy:
    {'fold_0': ['case1', 'case2', ...], 'fold_1': ['case3', 'case4', ...], ...}
    """
    strategy_type = strategy_config.get('strategy_type')
    shuffle_off = strategy_config.get('shuffle_off', False)
    shuffle_seed = strategy_config.get('shuffle_seed', None)

    case_ids_proc = case_ids[:]
    if not shuffle_off:  # Shuffle only if shuffle_off is False
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(case_ids_proc)

    if strategy_type == 'all':
        return {'all_cases': case_ids_proc}
    elif strategy_type == 'kfold':
        k_folds = strategy_config.get('k_folds', 5)
        fold_size = len(case_ids_proc) // k_folds
        folds = [case_ids_proc[i*fold_size:(i+1)*fold_size] for i in range(k_folds)]
        remainder = len(case_ids_proc) % k_folds
        for i in range(remainder):
            folds[i].append(case_ids_proc[k_folds*fold_size + i])
        return {f'fold_{i}': folds[i] for i in range(k_folds)}
    else:
        raise ValueError(f"Unknown sampling strategy_type: {strategy_type}")

def update_sampling_json(dataset_dir, strategy_config, selection, meta):
    split_json_path = os.path.join(dataset_dir, "dataset_split.json")
    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            out_data = json.load(f)
    else:
        out_data = {}

    if 'sampling' not in out_data:
        out_data['sampling'] = {}
    print(f'original out data {out_data}')
    strategy_type = strategy_config.get('strategy_type')
    if strategy_type == 'kfold':
        sampling_split_key = f'kfold_{meta["k_folds"]}_{meta["split"]}'
        out_data['sampling'][sampling_split_key] = {
            'meta': meta,
        }
        out_data['sampling'][sampling_split_key].update(selection)
    elif strategy_type == 'all':
        sampling_split_key = f'all_{meta["split"]}'
        print(sampling_split_key)
        out_data['sampling'][sampling_split_key] = {
            'meta': meta
        }
        out_data['sampling'][sampling_split_key].update(selection)
    else:
        raise ValueError(f"Unknown sampling strategy_type: {strategy_type}, please check your input parameters.")

    json_str = json.dumps(
        out_data, indent=2, ensure_ascii=False
    )
    json_str = single_line_lists(json_str)
    json_str = add_blank_lines_between_fields(json_str)
    # print(json_str)
    with open(split_json_path, 'w') as f:
        f.write(json_str)
    print(f"Sampling selection '{sampling_split_key}' saved to {split_json_path}")

def single_line_lists(json_str):
    # This regex matches lists that are not under "meta"
    # It looks for: "key": [ ... ] (not preceded by "meta":)
    pattern = re.compile(r'(\"(?!meta)[^\"]+\": )\[\s*([^\[\]]*?)\s*\]', re.MULTILINE)
    def replacer(match):
        key, values = match.groups()
        # Remove newlines and extra spaces in the list
        values = re.sub(r'\\n|\\r|\\s+', ' ', values)
        values = re.sub(r',\s+', ', ', values)
        return f'{key}[{values.strip()}]'
    return pattern.sub(replacer, json_str)

def add_blank_lines_between_fields(json_str):
    # Add a blank line between every field in every dictionary, at any indentation level
    # Matches:   "key": ...\n  "next_key":
    # and replaces with:   "key": ...\n\n  "next_key":
    # Handles both single-line and multi-line values
    return re.sub(
        r'(\n\s*"[^"]+": [\s\S]*?)(,)(\n\s*"[^"]+": )',
        r'\1,\n\3',
        json_str
    )

def main():
    parser = argparse.ArgumentParser(description="Offline dataset sampling utility (all/kfold, modular for train/test, structured config).")
    parser.add_argument('--dataset_dir', type=str, required=False, help='Path to dataset folder containing dataset.json',
        default='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset001_BrainTumour')
    parser.add_argument('--strategy_type', type=str, required=False, choices=['all', 'kfold'], help='Sampling strategy type', default='all')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for kfold strategy')
    parser.add_argument('--split', type=str, required=False, choices=['train', 'test'], help='Which split to sample from', default='test')
    parser.add_argument('--shuffle_off', action='store_true', default=False, help='Turn off shuffle for cases before splitting (default: False, it will shuffle)')
    parser.add_argument('--shuffle_seed', type=int, default=None, help='Random seed for shuffling (optional)')
    args = parser.parse_args()

    # Build strategy_config dictionary
    strategy_config = {
        'strategy_type': args.strategy_type,
        'shuffle_off': args.shuffle_off,
        'shuffle_seed': args.shuffle_seed
    }
    if args.strategy_type == 'kfold':
        strategy_config['k_folds'] = args.k_folds

    data = load_dataset_structure(args.dataset_dir)
    case_ids = get_case_ids(data, args.split)
    selection = sample_cases(
        case_ids,
        strategy_config
    )
   
    meta = {
        'strategy_type': args.strategy_type,
        'k_folds': args.k_folds if args.strategy_type == 'kfold' else None,
        'split': args.split,
        'shuffle_off': args.shuffle_off,
        'shuffle_seed': args.shuffle_seed
    }

    update_sampling_json(args.dataset_dir, strategy_config, selection, meta)

if __name__ == '__main__':
    main()