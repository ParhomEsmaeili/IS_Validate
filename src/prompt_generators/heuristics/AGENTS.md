# Heuristics Module — Agent Reference

## File layout

```
heuristics/
├── prompt_mixtures.py          # Cascade classes that orchestrate prompt generation
├── prompt_bases.py             # Base classes for point/scribble/bbox/lasso
├── heuristic_prompt_utils/
│   └── bbox.py, point.py etc                #Utilities for prompt generation, e.g. bbox generation: component extraction + sampling + augmentation
└── spatial_utils/
    ├── component_extraction.py # Connected component analysis, selection (top-k, etc.)
    ├── update_binary_mask.py   # Region mask modification by prompt coordinates
    ├── distance_maps.py        # Distance transform utilities
    └── boundary_selection.py   # Boundary point selection
```

---

## Cascade Architecture

### Class hierarchy (`prompt_mixtures.py`)

```
BaseMixture
  └── BasicValidOnlyMixture
        ├── SimplifiedPrototypePseudoMixture  (active)
        └── RandomPromptTypeAgent             (placeholder, NotImplementedError)
```

### The 4-level toggle cascade

Entry point: `__call__` → cascade order:

1. **`togg_class_level`** — Iterates classes in `semantic_id_dict`. For each class: extracts GT + error region, calls `togg_inter_prompt_level`, merges results with class-int labels.

2. **`togg_inter_prompt_level`** — Selects sampling region (GT for init, error for edit), manages cross-ptype interactions.

   Before generating a new prompt type, zeroes out the exact voxel coordinates of all previously placed prompts from the candidate region via `update_error_region` → `update_binary_mask_freeform` (sets mask[idx] = False). This is voxel-level sampling-without-replacement — entire subregions are not removed, just the individual voxels that already have prompts.

   - **Prototype**: Iterates priority groups (`[['bboxes'], ['scribbles', 'points']]`). Passes the depleted region to `togg_intra_prompt_level`.
   - **Simplified**: Only 1 valid ptype allowed — takes `valid_ptypes[0]` directly, no iteration or region updating needed (nothing to subtract).

3. **`togg_intra_prompt_level`** — Shuffles heuristic list. For each heuristic: copies region, calls `update_error_region` to remove placed points (sampling w/o replacement), calls `togg_intra_heur_level`.

4. **`togg_intra_heur_level`** — Looks up `heur_fnc` + `params` from config, calls `heur_fnc(samp_region, params)`, returns list of prompts.

### Setup steps (in `__call__`, before cascade)

- **`init_sample_regions_no_components`** — Splits GT/error into `{class_label: bool_mask}` per class. Returns `{'gt': dict, 'error_regions': dict | None}`.
- **`init_prompts`** — Initialises output dicts (empty lists for valid ptypes, None for others).
- **`rm_intra_prompt_spat_repeats`** — Post-cascade dedup within each prompt type.
- **`output_processor`** — Final format: filter empties → device → discrete dtype.

### Dead / unused code

- **`init_sample_regions_components`** — Per-class per-component splitting. Raises `NotImplementedError` (never functional).
- **`sort_components`** — Sort by size or random. Never called.
- **`BasicMistakesMixture`** — Commented-out skeleton at line 802.
- **`RandomPromptTypeAgent`** — Full class shell, `NotImplementedError` in `__init__`.

### Key insight

Component selection (connected components, top-k, etc.) is **not** handled by the cascade. The bbox heuristic (`heuristic_prompt_utils/bbox.py`) handles its own component extraction internally — it receives a raw binary mask from the cascade and calls `extract_connected_components` → `select_component` itself. This is by design: keep component logic self-contained in the heuristic that needs it, not in the generic cascade.

### Toggle dict

The `toggling_dict` controls branching at each cascade level via mode strings.

**`class_level`** modes:
- `None` (default) — iterate all classes
- `{'mode': 'basic_skip', 'skip_classes': ['background']}` — skip specified classes

Other levels (`inter_prompt_level`, `intra_prompt_level`) always use `None` defaults.

---

## Future discussion (Phase 2 refactor)

Consider replacing `bbox_from_binary_mask` (flat function in registry) with a proper class inheriting from `BboxBase`. The function is an orchestrator (component extraction, slice selection, dimensionality branching, augmentation) — not a leaf utility like the point heuristics. A class would:
- Keep bbox-specific orchestration self-contained
- Follow the intended `BboxBase` pattern from `prompt_bases.py`
- Decide whether `togg_intra_heur_level` dispatches via protocol (`callable`, `hasattr(heuristic, 'generate')`, etc.) or stays as-is
