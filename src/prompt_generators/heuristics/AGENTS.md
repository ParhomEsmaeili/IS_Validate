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
        ├── PrototypePseudoMixture           (original prototype)
        ├── SimplifiedPrototypePseudoMixture  (simplified, currently in use)
        └── RandomPromptTypeAgent             (placeholder, NotImplementedError)
```

### The 4-level toggle cascade

Entry point: `__call__` → cascade order:

1. **`togg_class_level`** — Iterates classes in `semantic_id_dict`. For each class: extracts GT + error region, calls `togg_inter_prompt_level`, merges results with class-int labels.

2. **`togg_inter_prompt_level`** — Selects sampling region (GT for init, error for edit).
   - **Prototype**: Iterates `prompt_level_order` priority groups shuffling for diversity; updates region by subtracting placed prompts.
   - **Simplified**: Only 1 valid ptype allowed — takes `valid_ptypes[0]` directly, no iteration or region updating.

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

The `toggling_dict` always uses `None` defaults — no actual toggling or branching is implemented. The 4-level structure is aspirational scaffolding for future mixture models. Currently the cascade is effectively a flat heuristic dispatch with sampling-without-replacement bookkeeping.
