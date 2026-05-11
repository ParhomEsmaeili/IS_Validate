"""
Task definition for IS-Validation is as follows:
Each task is defined by:
    - A dataset (implicitly here).
    - Image data config: Acquisitions [modality, series],  patch config (voxel count, spacing).
        For the sake of performing offline/self-supervised training we integrate the patch config into
        the dataset curation process.. so this must be well documented. 
    - Segmentation problem definition (semantic classes, merging rules, etc., problem type -> for now
    we only will support semantic seg. so this will need future expansion).
    - Infer modes: This outlines the inference modes for this task, typically this will be for
    holdout inference, but could potentially be used in the process of continual adaptation (i.e.,
    providing an algorithm with prompting sequences for also incorporate into its training data)
    - 

Validate task manifest structure, specifically sample_group_categories.

Structure Assertions:
- sample_group_category must be a list with 2 elements
- First element: category type ('all_train', 'all_test', 'kfold_5_train')
- Second element: category value (string or list of strings)
- For 'all_train'/'all_test': second element must be 'all_cases'
- For 'kfold_5_train': second element must be fold specification
  - If string: must be 'fold_N' format
  - If list: must be list of 'fold_X' strings with no duplicates

Ordering Assertions:
- All tasks organised in blocks of 12
- Within each block of 12:
  1-2: all_train, all_test
  3-7: Individual folds (fold_0, fold_1, fold_2, fold_3, fold_4)
  8-12: Compound folds (5 variants, each leaving out one fold)
- Within each block: all fields except sample_group_category must be identical

Additional structure:


Uniqueness Assertions:
- No two tasks within the same dataset can have identical configs (including sample_group_category)
- Within each block of 12, all sample_group_categories must be unique, but remaining fields must be
identical (except for the expected variations in sample_group_category)


"""

import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.general_utils.dict_utils import dict_deep_equals


class TaskManifestValidator:
    def __init__(self, exp_configs_path: str = None):
        if exp_configs_path is None:
            exp_configs_path = '/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs'
        self.exp_configs_path = exp_configs_path
        self.errors = []
        self.warnings = []

    def validate_sample_group_category(self, sgc: Any, dataset: str, task_id: str) -> bool:
        """
        Validate sample_group_category structure.
        Returns True if valid, False otherwise.
        """
        # Must be a list
        if not isinstance(sgc, list):
            self.errors.append(f"{dataset}/{task_id}: sample_group_category is not a list: {type(sgc)}")
            return False
        
        # Must have exactly 2 elements
        if len(sgc) != 2:
            self.errors.append(f"{dataset}/{task_id}: sample_group_category must have 2 elements, got {len(sgc)}")
            return False
        
        category_type, category_value = sgc
        
        # First element must be a string
        if not isinstance(category_type, str):
            self.errors.append(f"{dataset}/{task_id}: category_type must be string, got {type(category_type)}")
            return False
        
        # Validate based on category type
        if category_type == 'all_train':
            if category_value != 'all_cases':
                self.errors.append(f"{dataset}/{task_id}: all_train value must be 'all_cases', got '{category_value}'")
                return False
        elif category_type == 'all_test':
            if category_value != 'all_cases':
                self.errors.append(f"{dataset}/{task_id}: all_test value must be 'all_cases', got '{category_value}'")
                return False
        elif category_type == 'kfold_5_train':
            # category_value can be string (e.g., 'fold_0') or list of strings
            if isinstance(category_value, str):
                # Must be 'fold_N' format
                if not category_value.startswith('fold_'):
                    self.errors.append(f"{dataset}/{task_id}: fold value must start with 'fold_', got '{category_value}'")
                    return False
            elif isinstance(category_value, list):
                # Must be list of fold strings
                if not all(isinstance(f, str) and f.startswith('fold_') for f in category_value):
                    self.errors.append(f"{dataset}/{task_id}: compound fold must be list of 'fold_X' strings, got {category_value}")
                    return False
                # Check for duplicates
                if len(category_value) != len(set(category_value)):
                    self.errors.append(f"{dataset}/{task_id}: compound fold has duplicate values: {category_value}")
                    return False
            else:
                self.errors.append(f"{dataset}/{task_id}: fold value must be string or list, got {type(category_value)}")
                return False
        else:
            self.errors.append(f"{dataset}/{task_id}: unknown category_type '{category_type}'")
            return False
        
        return True

    def validate_dataset(self, tasks: Dict, dataset: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Basic structural smoke test for a dataset's tasks.
        Checks for required fields and correct SGC format (list with 2 elements, correct types).
        Returns (is_valid, stats_dict).
        """
        valid = True
        stats = {
            'total_tasks': len(tasks),
            'structure_errors': 0,
        }
        
        for task_id, task in tasks.items():
            try:
                # Check required fields
                if 'data_sampling' not in task:
                    self.errors.append(f"{dataset}/{task_id}: missing 'data_sampling'")
                    valid = False
                    continue
                
                data_sampling = task['data_sampling']
                
                # Check required data_sampling fields
                if 'sample_group_category' not in data_sampling:
                    self.errors.append(f"{dataset}/{task_id}: missing 'data_sampling.sample_group_category'")
                    valid = False
                    stats['structure_errors'] += 1
                else:
                    # Validate sample_group_category structure (list with 2 elements, correct types)
                    sgc = data_sampling['sample_group_category']
                    if not self.validate_sample_group_category(sgc, dataset, task_id):
                        valid = False
                        stats['structure_errors'] += 1
                
                if 'image_conf' not in data_sampling:
                    self.errors.append(f"{dataset}/{task_id}: missing 'data_sampling.image_conf'")
                    valid = False
                    stats['structure_errors'] += 1
                
                # Check required data_transforms field at task level
                if 'data_transforms' not in task:
                    self.errors.append(f"{dataset}/{task_id}: missing 'data_transforms'")
                    valid = False
                    stats['structure_errors'] += 1
                else:
                    # Check for semantic_class_mapping inside data_transforms
                    if 'semantic_class_mapping' not in task['data_transforms']:
                        self.errors.append(f"{dataset}/{task_id}: missing 'data_transforms.semantic_class_mapping'")
                        valid = False
                        stats['structure_errors'] += 1
                
                # Check required infer_info field
                if 'infer_info' not in task:
                    self.errors.append(f"{dataset}/{task_id}: missing 'infer_info'")
                    valid = False
                    stats['structure_errors'] += 1
                    continue
                
                infer_info = task['infer_info']
                required_infer_fields = ['infer_init', 'infer_edit_bool', 'sim_empty_fg_automatic']
                for field in required_infer_fields:
                    if field not in infer_info:
                        self.errors.append(f"{dataset}/{task_id}: missing 'infer_info.{field}'")
                        valid = False
                        stats['structure_errors'] += 1
                
            except Exception as e:
                self.errors.append(f"{dataset}/{task_id}: EXCEPTION during validation - {e}")
                valid = False
        
        return valid, stats

    def validate_task_block(self, block_num: int, block_task_ids: List[str], tasks: Dict, dataset: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a single block of 12 tasks.
        
        A block groups a single config (image_conf, data_transforms, seg_problem, infer_info, etc.)
        with 12 different sample_group_category variations:
        1-2: all_train, all_test
        3-7: fold_0, fold_1, fold_2, fold_3, fold_4
        8-12: compound folds (leave-one-out variants)
        
        All fields except sample_group_category must be identical within a block.
        """
        block_info = {
            'block_num': block_num,
            'task_ids': block_task_ids,
            'valid': True,
            'errors': [],
            'sgc_sequence': [],
            'consistency': True,
        }
        
        if len(block_task_ids) != 12:
            block_info['valid'] = False
            block_info['errors'].append(f"Block must have 12 tasks, got {len(block_task_ids)}")
            return False, block_info
        
        # Get sample_group_categories
        sgc_sequence = []
        block_tasks = []
        for task_id in block_task_ids:
            task = tasks[task_id]
            sgc = task['data_sampling']['sample_group_category']
            sgc_sequence.append(sgc)
            block_tasks.append(task)
        
        block_info['sgc_sequence'] = [str(sgc) for sgc in sgc_sequence]
        
        # Expected pattern
        expected_pattern = [
            ('all_train', 'all_cases'),
            ('all_test', 'all_cases'),
            ('kfold_5_train', 'fold_0'),
            ('kfold_5_train', 'fold_1'),
            ('kfold_5_train', 'fold_2'),
            ('kfold_5_train', 'fold_3'),
            ('kfold_5_train', 'fold_4'),
        ]
        
        # Check first 7 tasks
        for idx in range(7):
            sgc = sgc_sequence[idx]
            exp_type, exp_val = expected_pattern[idx]
            
            if sgc[0] != exp_type or sgc[1] != exp_val:
                block_info['valid'] = False
                block_info['errors'].append(
                    f"Position {idx+1}: expected ({exp_type}, {exp_val}), got ({sgc[0]}, {sgc[1]})"
                )
        
        # Check last 5 are compound folds
        for idx in range(7, 12):
            sgc = sgc_sequence[idx]
            if sgc[0] != 'kfold_5_train' or not isinstance(sgc[1], list):
                block_info['valid'] = False
                block_info['errors'].append(
                    f"Position {idx+1}: expected compound fold (kfold_5_train, list), " +
                    f"got ({sgc[0]}, {type(sgc[1]).__name__})"
                )
            else:
                # Validate leave-one-out structure
                # Compound fold index 0-4 should leave out fold_0-4 respectively
                fold_index_to_omit = (idx - 7) + 2  # Maps to expected_pattern index (2-6)
                fold_to_omit = expected_pattern[fold_index_to_omit][1]  # e.g., 'fold_0'
                
                # Expected set: all folds except the omitted one
                expected_folds_set = {f'fold_{i}' for i in range(5) if f'fold_{i}' != fold_to_omit}
                actual_folds_set = set(sgc[1])
                
                if actual_folds_set != expected_folds_set:
                    block_info['valid'] = False
                    block_info['errors'].append(
                        f"Position {idx+1}: expected compound fold leaving out {fold_to_omit}, " +
                        f"got {sgc[1]}"
                    )
        
        # Check consistency: all fields except sample_group_category should be identical
        template_task = copy.deepcopy(block_tasks[0])
        template_sgc = template_task['data_sampling'].pop('sample_group_category')
        
        for idx, task in enumerate(block_tasks[1:], 1):
            check_task = copy.deepcopy(task)
            check_sgc = check_task['data_sampling'].pop('sample_group_category')
            
            if check_task != template_task:
                block_info['consistency'] = False
                block_info['errors'].append(
                    f"Task {idx+1} ({block_task_ids[idx]}) differs from task 1 in fields other than sample_group_category"
                )
            
            # Assert that SGCs are different (within-block uniqueness ensures this, but sanity check)
            if template_sgc == check_sgc:
                block_info['valid'] = False
                block_info['errors'].append(
                    f"Task {idx+1} ({block_task_ids[idx]}): identical sample_group_category to task 1"
                )
        
        # Check within-block uniqueness: all sample_group_categories must be unique
        seen_sgc = {}
        for idx, sgc in enumerate(sgc_sequence):
            # Convert to comparable format (tuples for lists, strings otherwise)
            if isinstance(sgc[1], list):
                sgc_tuple = (sgc[0], tuple(sorted(sgc[1])))
            else:
                sgc_tuple = (sgc[0], sgc[1])
            
            if sgc_tuple in seen_sgc:
                block_info['valid'] = False
                block_info['errors'].append(
                    f"Position {idx+1} ({block_task_ids[idx]}): duplicate sample_group_category {sgc} " +
                    f"(already seen at position {seen_sgc[sgc_tuple]+1})"
                )
            else:
                seen_sgc[sgc_tuple] = idx
        
        if not block_info['errors']:
            block_info['valid'] = True
        
        return block_info['valid'] and block_info['consistency'], block_info

    def validate_task_ordering(self, dataset: str, tasks: Dict) -> Tuple[bool, List[Dict]]:
        """
        Validate that all tasks are organized into blocks of 12.
        Each block should have the expected structure and consistency.
        Returns (is_valid, block_info_list).
        """
        # Sort task IDs numerically
        sorted_task_ids = sorted(tasks.keys(), key=lambda x: int(x.split('_')[2]))
        
        if len(sorted_task_ids) % 12 != 0:
            return False, [{'error': f"Total tasks {len(sorted_task_ids)} not divisible by 12"}]
        
        # Split into blocks of 12
        blocks = []
        ordering_valid = True
        
        for block_num in range(len(sorted_task_ids) // 12):
            start_idx = block_num * 12
            end_idx = start_idx + 12
            block_task_ids = sorted_task_ids[start_idx:end_idx]
            
            valid, block_info = self.validate_task_block(block_num, block_task_ids, tasks, dataset)
            blocks.append(block_info)
            if not valid:
                ordering_valid = False
        
        return ordering_valid, blocks

    def validate_all(self) -> bool:
        """
        Validate all datasets.
        Returns True if all datasets are valid.
        Checks in order:
        1. Load all files
        2. Structural validation (required fields, SGC format)
        3. Task ordering and block structure
        4. Within-dataset uniqueness: no fully identical task configs within same dataset
        """
        datasets = sorted([d for d in os.listdir(self.exp_configs_path)
                          if os.path.isdir(os.path.join(self.exp_configs_path, d)) 
                          and d.startswith('Dataset')])
        
        all_valid = True
        dataset_results = {}
        all_datasets_tasks = {}  # Load all files first
        
        # Step 1: Load all dataset files
        for dataset in datasets:
            filepath = os.path.join(self.exp_configs_path, dataset, 'task_configs.txt')
            if not os.path.exists(filepath):
                self.errors.append(f"{dataset}: NO task_configs.txt FILE")
                all_valid = False
                continue
            
            try:
                with open(filepath, 'r') as f:
                    tasks = json.loads(f.read())
                    all_datasets_tasks[dataset] = tasks
            except json.JSONDecodeError as e:
                self.errors.append(f"{dataset}: JSON PARSE ERROR - {e}")
                all_valid = False
            except Exception as e:
                self.errors.append(f"{dataset}: ERROR reading file - {e}")
                all_valid = False
        
        # Step 2, 3, 4: Validate structure, ordering, and within-dataset uniqueness
        for dataset, tasks in all_datasets_tasks.items():
            valid, stats = self.validate_dataset(tasks, dataset)
            
            # Validate ordering and block structure
            ordering_valid, ordering_blocks = self.validate_task_ordering(dataset, tasks)
            
            # Check within-dataset uniqueness: no fully identical task configs
            dataset_valid = valid and ordering_valid
            registry_tasks = {}  # Maps task_id -> task config
            for task_id, task in tasks.items():
                # Check against all previously seen tasks in this dataset
                for prev_task_id, prev_task in registry_tasks.items():
                    is_equal, differences = dict_deep_equals(task, prev_task)
                    if is_equal:
                        self.errors.append(
                            f"{dataset}/{task_id}: identical task config exists in {prev_task_id}"
                        )
                        dataset_valid = False
                
                # Add this task to registry
                registry_tasks[task_id] = task
            
            dataset_results[dataset] = (dataset_valid, stats, ordering_blocks)
            if not dataset_valid:
                all_valid = False
        
        return all_valid, dataset_results

    def print_report(self, all_valid: bool, dataset_results: Dict):
        """Print validation report."""
        print("\n" + "="*80)
        print("TASK MANIFEST VALIDATION REPORT")
        print("="*80)
        
        # Summary by dataset
        print("\nDATASET VALIDATION STATUS:")
        print("-" * 80)
        
        for dataset in sorted(dataset_results.keys()):
            result = dataset_results[dataset]
            # Handle both old (valid, stats) and new (valid, stats, blocks) formats
            if len(result) == 3:
                valid, stats, blocks = result
            else:
                valid, stats = result
                blocks = []
            
            status = "✓ PASS" if valid else "✗ FAIL"
            
            print(f"\n{status} {dataset}")
            if stats:
                print(f"  Tasks: {stats['total_tasks']}")
                if stats.get('structure_errors', 0) > 0:
                    print(f"  ✗ Structure errors: {stats['structure_errors']}")
            
            # Show block ordering info
            if blocks:
                for block in blocks:
                    if 'error' in block:
                        print(f"  ✗ {block['error']}")
                    else:
                        block_status = "✓" if block['valid'] else "✗"
                        task_range = f"{block['task_ids'][0]}-{block['task_ids'][-1]}"
                        print(f"  {block_status} Block {block['block_num']}: {task_range}")
                        
                        if not block['consistency']:
                            print(f"      ⚠ Consistency check failed")
                        
                        if block['errors']:
                            for error in block['errors']:
                                print(f"      ✗ {error}")
        
        # Error report
        if self.errors:
            print("\n" + "="*80)
            print(f"ERRORS ({len(self.errors)}):")
            print("-" * 80)
            for error in self.errors:
                print(f"  ✗ {error}")
        
        # Summary
        print("\n" + "="*80)
        if all_valid and not self.errors:
            print("✅ ALL DATASETS VALID - ALL ASSERTIONS PASSED")
        else:
            print(f"❌ VALIDATION FAILED - {len(self.errors)} errors found")
        print("="*80 + "\n")


def main():
    validator = TaskManifestValidator()
    all_valid, dataset_results = validator.validate_all()
    validator.print_report(all_valid, dataset_results)
    
    return 0 if all_valid and not validator.errors else 1


if __name__ == '__main__':
    exit(main())
