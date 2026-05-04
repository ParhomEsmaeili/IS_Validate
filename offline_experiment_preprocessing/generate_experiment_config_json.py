import argparse
import ast
import copy
import json
from pathlib import Path
from src.general_utils.dict_utils import extractor, extract_config, dict_deep_equals

def _parse_id_list(values):
	out = []
	for v in values:
		out.extend([x.strip() for x in v.split(",") if x.strip()])
	return out

def _load_experiments_by_dataset(path, dataset):
	p = Path(path)
	if not p.exists():
		return {dataset: []}

	raw = p.read_text(encoding="utf-8").strip()
	if not raw:
		return {dataset: []}

	data = json.loads(raw)
	if isinstance(data, dict):
		if dataset not in data or not isinstance(data.get(dataset), list):
			data[dataset] = []
		return data
	if isinstance(data, list):
		return {dataset: data}
	raise ValueError("Existing experiments JSON must be a dict or list")


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

def check_pre_existence(
		experiments: dict[str, dict],
		task_config: dict, 
		prompter_config: dict, 
		metrics_config: dict
	):
	'''
	Function which checks whether an experiment with the same task, prompter and metrics configuration
	already exists in the current manifest. This is to avoid duplicates when appending new experiments.
	'''

	
def main():
	parser = argparse.ArgumentParser(
		description="Append experiment snapshots to a dataset experiments JSON"
	)
	parser.add_argument(
		"--dataset",
		default="Dataset001_BrainTumor", 
		required=False, #True, 
		help="Dataset name/key"
		)
	parser.add_argument(
		"--task-config-id",
		default=["task_id_3", "task_id_4", "task_id_5", "task_id_6", "task_id_7"],
		required=False, #True,
		nargs="+",
		help="Task config ID(s). Supports repeated args and/or comma-separated values.",
	)
	parser.add_argument(
		"--prompter-id", 
		required=False, #True,
		nargs="+",
		help="Prompter config ID"
		)
	parser.add_argument(
		"--metrics-config-id", 
		required=False, #True,
		nargs="+",
		help="Metrics config ID"
		)
	parser.add_argument(
		"--task-configs-file", 
		required=False,#True, 
		help="Path to task configs txt/json file"
		)
	parser.add_argument(
		"--prompter-configs-file", 
		required=False, #True, 
		help="Path to prompter configs txt/json file")
	parser.add_argument(
		"--metrics-configs-file", 
		required=False, #True, 
		help="Path to metrics configs txt/json file")
	parser.add_argument(
		"--experiments-json", 
		required=False, #True, 
		help="Path to existing experiments JSON (created if missing)")
	parser.add_argument(
		"--output-json", 
		default=None, 
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

	task_map = extract_config(args.task_configs_file)
	prompter_map = extract_config(args.prompter_configs_file)
	metrics_map = extract_config(args.metrics_configs_file)

	missing_tasks = [t for t in task_ids if t not in task_map]
	if missing_tasks:
		raise KeyError(f"Missing task config IDs: {missing_tasks}")
	
	by_dataset = _load_experiments_by_dataset(args.experiments_json, args.dataset)
	experiments = by_dataset[args.dataset]

	existing = {_exp_key(e) for e in experiments}
	added = 0
	for task_id in task_ids:
		key = (task_id, args.prompter_id, args.metrics_config_id)
		if key in existing:
			continue

		experiments.append(
			{
				"task": {"task_id": task_id, "config": copy.deepcopy(task_map[task_id])},
				"prompter": {
					"prompter_id": args.prompter_id,
					"config": copy.deepcopy(prompter_map[args.prompter_id]),
				},
				"metrics": {
					"metrics_config_id": args.metrics_config_id,
					"config": copy.deepcopy(metrics_map[args.metrics_config_id]),
				},
			}
		)
		existing.add(key)
		added += 1

	output_path = Path(args.output_json or args.experiments_json)
	output_path.write_text(json.dumps(by_dataset, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"Done. Added {added} experiment(s). Total for dataset '{args.dataset}': {len(experiments)}")


if __name__ == "__main__":
	main()
