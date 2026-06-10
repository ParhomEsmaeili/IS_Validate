"""Generate deterministic prompter IDs from prompting and sampling configs.

Config variable meanings:
- init_prompt_conf_name: Prompt configuration id used for initialisation prompts.
- edit_prompt_conf_name: Prompt configuration id used for iterative edit prompts.
- infer_edit_nums: Maximum number of edit interactions during inference.
- use_mem_inf_edit: Whether interaction memory is used when generating edit prompts.
- im_conf_remove_init: Whether the initialisation interaction is removed from memory.
- im_conf_mem_len: Interaction-memory retention length after each edit iteration.
	- Allowed values: -1, or any positive integer.
	- -1 means keep full interaction memory (no truncation).
	- N > 0 means keep only the most recent N interaction states (rolling window).
	- 0 is invalid because it would drop all memory including the current state.
	- This only has practical effect when use_mem_inf_edit is True.
- annotation_conf: Stored as one config dictionary with two list fields:
	- annotator: list of annotator ids.
	- instance_id: list of annotation instance ids.

Important behavior constraints:
- annotator and instance_id are not expanded as Cartesian-product combinations.
- a new prompter config is assigned the next available prompter_N index based on
	any existing prompter manifest.
- new config insertion is blocked if it exactly matches an existing manifest entry.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.general_utils.dict_utils import extract_config, dict_deep_equals


DEFAULT_REGISTRY_NAME = "prompts_configs.txt"
PROMPTER_ID_PATTERN = re.compile(r"^prompter_(\d+)$")


def _slugify(value: str) -> str:
	value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
	value = re.sub(r"-+", "-", value).strip("-._")
	return value or "unknown"


def _parse_bool(value: str) -> bool:
	normalized = value.strip().lower()
	if normalized in {"1", "true", "t", "yes", "y", "on"}:
		return True
	if normalized in {"0", "false", "f", "no", "n", "off"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _load_json_if_possible(text: str) -> Any:
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		return None

def load_existing_prompter_manifest(manifest_path: Path | None) -> dict[str, dict[str, Any]]:
	if manifest_path is None or not manifest_path.exists():
		return {}

	parsed = extract_config(str(manifest_path), None)
	if not isinstance(parsed, dict):
		raise ValueError(
			f"Existing prompter manifest must be a JSON object keyed by prompter id: {manifest_path}"
		)

	manifest: dict[str, dict[str, Any]] = {}
	for key, value in parsed.items():
		if not isinstance(value, dict):
			raise ValueError(
				f"Manifest entry '{key}' must be a JSON object, got {type(value).__name__}"
			)
		manifest[str(key)] = value

	return manifest


def parse_annotation_conf(value: str) -> dict[str, list[str]]:
	parsed = _load_json_if_possible(value)
	if parsed is None:
		path = Path(value)
		if not path.exists():
			raise ValueError(
				"annotation_conf must be a JSON string or a path to a JSON file"
			)
		parsed = json.loads(path.read_text(encoding="utf-8"))

	if not isinstance(parsed, dict):
		raise ValueError("annotation_conf must decode to a JSON object")

	annotators = parsed.get("annotator", [])
	instance_ids = parsed.get("instance_id", [])
	if not isinstance(annotators, list) or not isinstance(instance_ids, list):
		raise ValueError("annotation_conf.annotator and annotation_conf.instance_id must be lists")
    
	annotators = [str(item) for item in annotators]
	instance_ids = [str(item) for item in instance_ids]
	if not annotators or not instance_ids:
		raise ValueError("annotation_conf must include at least one annotator and one instance_id")

	return {"annotator": annotators, "instance_id": instance_ids}


def _next_prompter_index(existing_ids: set[str]) -> int:
	max_index = -1
	for prompter_id in existing_ids:
		match = PROMPTER_ID_PATTERN.match(prompter_id)
		if match is None:
			continue
		max_index = max(max_index, int(match.group(1)))
	return max_index + 1


@dataclass(frozen=True)
class PrompterGenerationConfig:
	init_prompt_conf_name: str
	edit_prompt_conf_name: str
	infer_edit_nums: int
	use_mem_inf_edit: bool
	im_conf_remove_init: bool
	im_conf_mem_len: int
	annotation_conf: dict[str, list[str]]


def generate_prompter_ids(
	config: PrompterGenerationConfig,
	prompts_registry: dict[str, dict] | None = None,
	existing_manifest: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
	if config.im_conf_mem_len == 0:
		raise ValueError("im_conf_mem_len must be -1 or a positive integer")
	if config.edit_prompt_conf_name is not None:
		if config.infer_edit_nums < 1:
			raise ValueError("infer_edit_nums must be >= 1")
	else:
		if config.infer_edit_nums != 0:
			raise ValueError("infer_edit_nums must be 0 when edit_prompt_conf_name is None")
	if prompts_registry is not None:
		missing = [name for name in (config.init_prompt_conf_name, config.edit_prompt_conf_name) if name is not None and name not in prompts_registry]
		if missing:
			raise KeyError(f"Unknown prompt conf name(s): {missing}. Available: {sorted(prompts_registry.keys())}")

	new_config = {
		"init_prompt_conf": {"name": config.init_prompt_conf_name, "config": prompts_registry[config.init_prompt_conf_name] if config.init_prompt_conf_name != None else None},
		"edit_prompt_conf": {"name": config.edit_prompt_conf_name, "config": prompts_registry[config.edit_prompt_conf_name] if config.edit_prompt_conf_name != None else None},
		"infer_edit_nums": config.infer_edit_nums,
		"use_mem_inf_edit": config.use_mem_inf_edit,
		"im_conf_remove_init": config.im_conf_remove_init,
		"im_conf_mem_len": config.im_conf_mem_len,
		"annotation_conf": config.annotation_conf,
	}

	results: dict[str, dict[str, Any]] = {}
	existing_manifest = existing_manifest or {}
	for existing_id, existing_config in existing_manifest.items():
		equal, _ = dict_deep_equals(existing_config, new_config)
		if equal:
			raise ValueError(
				f"New prompter config already exists in manifest as '{existing_id}'."
			)
	next_idx = _next_prompter_index(set(existing_manifest.keys()))
	results[f"prompter_{next_idx}"] = new_config

	return results


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--prompts-configs",
		type=Path,
		default=Path(__file__).with_name(DEFAULT_REGISTRY_NAME),
		help="Path to prompts_configs.txt registry.",
	)
	parser.add_argument(
		"--init-prompt-conf-name", 
		default=None,
		type=str,
		help="Prompt configuration name for initialisation prompts. Must exist in the prompts registry."
		)
	parser.add_argument(
		"--edit-prompt-conf-name", 
		default=None,
		type=str,
		help="Prompt configuration name for editing prompts. Must exist in the prompts registry."
		)
	parser.add_argument(
		"--infer-edit-nums", 
		type=int, 
		required=True
		)
	parser.add_argument(
		"--use-mem-inf-edit", 
		type=_parse_bool, 
		required=True
		)
	parser.add_argument(
		"--im-conf-remove-init", 
		type=_parse_bool, 
		required=True
		)
	parser.add_argument(
		"--im-conf-mem-len",
		type=int,
		required=True,
		help=
			"Interaction-memory length. Use -1 for full memory, or N>0 to keep only "
			"the latest N interactions. 0 is invalid."
		,
	)
	parser.add_argument(
		"--annotation-conf",
		required=True,
		help='JSON string or path to JSON file, e.g. {"annotator":["annotator_1"],"instance_id":["instance_1"]}',
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Path to save the merged prompter manifest. If not provided, only prints to stdout.",
	)
	parser.add_argument(
		"--compact",
		action="store_true",
		help="Write compact JSON (pretty-printed output is the default).",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Deprecated compatibility flag; output is pretty-printed by default.",
	)
	return parser


def main() -> int:
	args = build_parser().parse_args()
	
	#We are going to actually pull the configs for each of the named configurations and store the
	# actual dictionaries rather than just the names. Makes it more self-contained and robust to
	# changes in the prompts registry.
	prompts_registry = extract_config(args.prompts_configs)
	# Normalise "null" string to None for optional prompt conf names
	init_name = None if args.init_prompt_conf_name in (None, "null", "None") else args.init_prompt_conf_name
	edit_name = None if args.edit_prompt_conf_name in (None, "null", "None") else args.edit_prompt_conf_name
	config = PrompterGenerationConfig(
		init_prompt_conf_name=init_name,
		edit_prompt_conf_name=edit_name,
		infer_edit_nums=args.infer_edit_nums,
		use_mem_inf_edit=args.use_mem_inf_edit,
		im_conf_remove_init=args.im_conf_remove_init,
		im_conf_mem_len=args.im_conf_mem_len,
		annotation_conf=parse_annotation_conf(args.annotation_conf),
	)
	existing_manifest = load_existing_prompter_manifest(args.output)
	result = generate_prompter_ids(
		config,
		prompts_registry=prompts_registry,
		existing_manifest=existing_manifest,
	)
	existing_manifest.update(result)
	
	# Output JSON to stdout (pretty by default to match existing manifests)
	output_json = json.dumps(
		existing_manifest,
		indent=None if args.compact else "\t",
		ensure_ascii=False,
	)
	print(output_json)
	
	# Save to file if output path specified
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(output_json, encoding="utf-8")
	
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
