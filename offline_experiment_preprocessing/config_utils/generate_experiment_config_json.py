import argparse
import copy
import json
import os
from itertools import product
from pathlib import Path
import sys
import warnings

warnings.formatwarning = lambda msg, category, filename, lineno, line=None: f"{category.__name__}: {msg}\n"

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)
from src.general_utils.dict_utils import extract_config, dict_deep_equals


EMPTY_PROMPT_NAMES = {"", "null", "none", "nulltype"}
AUTOMATIC_INIT_LABEL = "Automatic Init"

def _parse_id_list(values):
	out = []
	for v in values:
		out.extend([x.strip() for x in v.split(",") if x.strip()])
	return out

def _load_experiments(path):
	p = Path(path)
	if not p.exists():
		return {}

	raw = p.read_text(encoding="utf-8").strip()
	if not raw:
		return {}

	data = json.loads(raw)
	if isinstance(data, dict):
		return data
	raise ValueError("Existing experiments JSON must be a dict")


def _exp_key(exp):
	task_id = (
		exp.get("task", {}).get("task_id")
		if isinstance(exp.get("task"), dict)
		else exp.get("task_config_id")
	)
	prompter_id = (
		exp.get("prompter", {}).get("prompter_id")
		if isinstance(exp.get("prompter"), dict)
		else exp.get("prompter_id")
	)
	metrics_config_id = (
		exp.get("metrics", {}).get("metrics_config_id")
		if isinstance(exp.get("metrics"), dict)
		else exp.get("metrics_config_id")
	)
	return (task_id, prompter_id, metrics_config_id)


def _format_exp_ids(task_id, prompter_id, metrics_id):
	return f"(task_id={task_id}, prompter_id={prompter_id}, metrics_config_id={metrics_id})"


def _is_automatic_init(infer_init):
	return isinstance(infer_init, str) and infer_init.strip() == AUTOMATIC_INIT_LABEL


def _is_empty_prompt_conf(prompt_conf):
	if not isinstance(prompt_conf, dict):
		return True

	name = prompt_conf.get("name")
	if name is None:
		return True
	if isinstance(name, str) and name.strip().lower() in EMPTY_PROMPT_NAMES:
		return True

	config = prompt_conf.get("config")
	return config in (None, {}, [])


def _validate_early_termination_criterion(metrics_cfg):
	errors = []
	criterion = metrics_cfg.get("early_termination_criterion")
	if not isinstance(criterion, dict):
		errors.append("metrics config must include early_termination_criterion as a dict")
		return errors

	if "metric" not in criterion:
		errors.append("early_termination_criterion must include 'metric'")
	if "threshold" not in criterion:
		errors.append("early_termination_criterion must include 'threshold'")

	if criterion.get("metric") != "Dice":
		errors.append("early_termination_criterion.metric must be 'Dice'")

	threshold = criterion.get("threshold")
	if not isinstance(threshold, (int, float)):
		errors.append("early_termination_criterion.threshold must be numeric")
	elif float(threshold) != 1.0:
		errors.append("early_termination_criterion.threshold must be 1.0")

	return errors


def validate_experiment_compatibility(task_cfg, prompter_cfg, metrics_cfg):
	"""Return a list of compatibility errors for a task/prompter/metrics triple."""
	errors = []

	infer_info = task_cfg.get("infer_info")
	if not isinstance(infer_info, dict):
		errors.append("task config is missing infer_info")
		infer_info = {}

	infer_init = infer_info.get("infer_init")
	infer_edit_bool = infer_info.get("infer_edit_bool")
	if infer_init is None:
		errors.append("task infer_info is missing infer_init")
	if infer_edit_bool is None:
		errors.append("task infer_info is missing infer_edit_bool")
	elif not isinstance(infer_edit_bool, bool):
		errors.append("task infer_info.infer_edit_bool must be a boolean")

	if isinstance(infer_edit_bool, bool):
		infer_edit_nums = prompter_cfg.get("infer_edit_nums")
		if not isinstance(infer_edit_nums, int):
			errors.append("prompter infer_edit_nums must be an integer")
		elif infer_edit_bool and infer_edit_nums == 0:
			errors.append("editing-enabled task cannot use a prompter with infer_edit_nums == 0")
		elif not infer_edit_bool and infer_edit_nums != 0:
			errors.append("init-only task cannot use a prompter with infer_edit_nums != 0")

	init_prompt_conf = prompter_cfg.get("init_prompt_conf")
	edit_prompt_conf = prompter_cfg.get("edit_prompt_conf")
	init_prompt_empty = _is_empty_prompt_conf(init_prompt_conf)
	edit_prompt_empty = _is_empty_prompt_conf(edit_prompt_conf)

	if _is_automatic_init(infer_init):
		if not init_prompt_empty:
			errors.append("Automatic Init task requires an empty/null init prompter")
	else:
		if init_prompt_empty:
			errors.append("non-Automatic Init task requires a non-empty init prompter")

	if isinstance(infer_edit_bool, bool):
		if infer_edit_bool and edit_prompt_empty:
			errors.append("editing-enabled task requires a non-empty edit prompter")
		if not infer_edit_bool and not edit_prompt_empty:
			errors.append("init-only task requires an empty/null edit prompter")

	# Check 1: If init-only, edit prompter must be empty
	if not infer_edit_bool and not edit_prompt_empty:
		errors.append("init-only task must have empty/null edit prompter")

	# Check 2: If automatic init, init prompter must be empty
	if _is_automatic_init(infer_init) and not init_prompt_empty:
		errors.append("automatic init task must have empty/null init prompter")

	metrics_sampling = metrics_cfg.get("data_sampling", {}) if isinstance(metrics_cfg, dict) else {}
	metrics_annotation_conf = metrics_sampling.get("annotation_conf") if isinstance(metrics_sampling, dict) else None
	task_annotation_conf = task_cfg.get("data_sampling", {}).get("annotation_conf") if isinstance(task_cfg.get("data_sampling"), dict) else None
	if task_annotation_conf is not None and metrics_annotation_conf is not None:
		match, diffs = dict_deep_equals(task_annotation_conf, metrics_annotation_conf)
		if not match:
			errors.append(f"task annotation_conf and metrics annotation_conf differ: {diffs}")

	errors.extend(_validate_early_termination_criterion(metrics_cfg))
	return errors


def validate_manifest_consistency(
	experiments: dict,
	task_map: dict,
	prompter_map: dict,
	metrics_map: dict
	):
	"""
	Pre-check: validate that the manifest entries match the current source
	definitions and satisfy the task/prompter/metrics compatibility rules.

	Returns:
		tuple: (all_consistent: bool, divergences: list[dict])
	"""

	divergences = []

	for exp in experiments.values():
		task_id = exp.get("task", {}).get("task_id")
		prompter_id = exp.get("prompter", {}).get("prompter_id")
		metrics_id = exp.get("metrics", {}).get("metrics_config_id")
		exp_label = _format_exp_ids(task_id, prompter_id, metrics_id)

		task_cfg = task_map.get(task_id, {})
		prompter_cfg = prompter_map.get(prompter_id, {})
		metrics_cfg = metrics_map.get(metrics_id, {})

		if task_id not in task_map:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "task_error": f"{exp_label}: task config id not found in task configs"})
		if prompter_id not in prompter_map:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "prompter_error": f"{exp_label}: prompter config id not found in prompter manifest"})
		if metrics_id not in metrics_map:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "metrics_error": f"{exp_label}: metrics config id not found in metrics configs"})

		exp_task_cfg = exp.get("task", {}).get("config", {})
		task_match, task_diffs = dict_deep_equals(exp_task_cfg, task_cfg)
		if not task_match:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "task_diffs": f"{exp_label}: {task_diffs}"})

		exp_prompter_cfg = exp.get("prompter", {}).get("config", {})
		prompter_match, prompter_diffs = dict_deep_equals(exp_prompter_cfg, prompter_cfg)
		if not prompter_match:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "prompter_diffs": f"{exp_label}: {prompter_diffs}"})

		exp_metrics_cfg = exp.get("metrics", {}).get("config", {})
		metrics_match, metrics_diffs = dict_deep_equals(exp_metrics_cfg, metrics_cfg)
		if not metrics_match:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "metrics_diffs": f"{exp_label}: {metrics_diffs}"})

		compat_errors = validate_experiment_compatibility(task_cfg, prompter_cfg, metrics_cfg)
		if compat_errors:
			divergences.append({"exp_id": (task_id, prompter_id, metrics_id), "compatibility_errors": [f"{exp_label}: {msg}" for msg in compat_errors]})

	return len(divergences) == 0, divergences


def add_new_experiments(
	experiments: dict,
	task_ids: list,
	prompter_ids: list,
	metrics_config_ids: list,
	task_map: dict,
	prompter_map: dict,
	metrics_map: dict
	):
	'''
	Add all combinations of IDs to experiments, avoiding duplicates.
	
	Returns:
		int: Number of experiments added
	'''
	combinations = list(product(task_ids, prompter_ids, metrics_config_ids))
	existing = {_exp_key(e) for e in experiments.values()}
	added = 0
	for task_id, prompter_id, metrics_id in combinations:
		key = (task_id, prompter_id, metrics_id)
		if key in existing:
			warnings.warn(
				f"Experiment with Task ID '{task_id}', Prompter ID '{prompter_id}', and Metrics Config ID '{metrics_id}' already exists. Skipping duplicate."
			)
			continue

		task_cfg = copy.deepcopy(task_map[task_id])
		prompter_cfg = copy.deepcopy(prompter_map[prompter_id])
		metrics_cfg = copy.deepcopy(metrics_map[metrics_id])
		compat_errors = validate_experiment_compatibility(task_cfg, prompter_cfg, metrics_cfg)
		if compat_errors:
			raise ValueError(
				f"Invalid experiment combination {_format_exp_ids(task_id, prompter_id, metrics_id)}: "
				+ "; ".join(compat_errors)
			)

		experiments[f"exp_config_{len(experiments) + 1}"] = {
			"task": {"task_id": task_id, "config": task_cfg},
			"prompter": {"prompter_id": prompter_id, "config": prompter_cfg},
			"metrics": {"metrics_config_id": metrics_id, "config": metrics_cfg},
		}
		existing.add(key)
		added += 1

	return added, experiments


def main():
	parser = argparse.ArgumentParser(
		description="Append experiment snapshots to a dataset experiments JSON"
	)
	parser.add_argument(
		"--task-config-id",
		default=["task_id_3", "task_id_4", "task_id_5", "task_id_6", "task_id_7"],
		required=True,
		nargs="+",
		help="Task config ID(s). Supports repeated args and/or comma-separated values.",
	)
	parser.add_argument(
		"--prompter-id", 
		required=True,
		default=["prototype"],
		nargs="+",
		help="Prompter config ID"
		)
	parser.add_argument(
		"--metrics-config-id", 
		required=True,
		default=["prototype"],
		nargs="+",
		help="Metrics config ID"
		)
	parser.add_argument(
		"--task-configs-file", 
		required=True, 
		default="exp_configs/Dataset001_BrainTumour/task_configs.txt",
		help="Path to task configs txt/json file"
		)
	parser.add_argument(
		"--prompter-configs-file", 
		required=True, 
		default="exp_configs/Dataset001_BrainTumour/prompter_manifest.json",
		help="Path to prompter configs txt/json file")
	parser.add_argument(
		"--metrics-configs-file", 
		required=True, 
		default="exp_configs/Dataset001_BrainTumour/metrics_configs.txt",
		help="Path to metrics configs txt/json file")
	parser.add_argument(
		"--experiments-json", 
		required=True, 
		default="exp_configs/Dataset001_BrainTumour/experiment_manifest.json",
		help="Path to existing experiments JSON (created if missing)")
	parser.add_argument(
		"--output-json", 
		required=False,
		default="exp_configs/Dataset001_BrainTumour/experiment_manifest.json",
		help="Output path (defaults to --experiments-json)")

	args = parser.parse_args()

	task_ids = _parse_id_list(args.task_config_id)
	prompter_ids = _parse_id_list(args.prompter_id)
	metrics_config_ids = _parse_id_list(args.metrics_config_id)
	if not task_ids:
		raise ValueError("At least one task_config_id is required")
	if not prompter_ids:
		raise ValueError("At least one prompter_id is required")
	if not metrics_config_ids:
		raise ValueError("At least one metrics_config_id is required")

	# Load all config maps (one per file)
	task_map = extract_config(os.path.join(base_dir, args.task_configs_file))
	prompter_map = extract_config(os.path.join(base_dir, args.prompter_configs_file))
	metrics_map = extract_config(os.path.join(base_dir, args.metrics_configs_file))

	experiments = _load_experiments(os.path.join(base_dir, args.experiments_json))
	
	# Validate existing experiments for config consistency
	consistent, divergences = validate_manifest_consistency(
		experiments, 
		task_map, 
		prompter_map, 
		metrics_map
	)
	if not consistent:
		print("WARNING: Config divergences detected in existing experiments:")
		for div in divergences:
			exp_id = div['exp_id']
			diverged_configs = [k for k in div.keys() if k != 'exp_id']
			print(f"  {exp_id}: {', '.join(diverged_configs)}")
	
	# Add new experiments for all ID combinations
	added, experiments = add_new_experiments(
		experiments, task_ids, prompter_ids, metrics_config_ids,
		task_map, prompter_map, metrics_map
	)

	output_path = Path(os.path.join(base_dir, args.output_json)) if args.output_json else Path(os.path.join(base_dir, args.experiments_json))
	output_path.write_text(json.dumps(experiments, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"Done. Added {added} experiment(s). Total experiments: {len(experiments)}")

if __name__ == "__main__":
	main()
