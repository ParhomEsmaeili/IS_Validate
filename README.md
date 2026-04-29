# IS-Validation-Framework

# Contract Specification

> ⚠️ **Work in Progress** — Implementation of and adjustment into these contract formats is actively under development. Definitions are subject to change as the working group progresses through its action items.

---

## Overview

The contract is organised around a fundamental structural distinction: what happens at the level of a **single sample** (per-sample contract) and what **persists across samples** (dataset-level contract).

A first-class design principle applies throughout: all fields are either **required** (the minimal set without which the contract is meaningless) or **optional** (additive enrichments that evolve with the field and depend on model capacity or use case).

---

## Table of Contents

- [1. Context & Schema Contract](#1-context--schema-contract)
- [2. Data Referencing Contract](#2-data-referencing-contract)
- [3. Prompting Contract](#3-prompting-contract)
- [4. Output Contract](#4-output-contract)

---

## 1. Context & Schema Contract

> ⚠️ *Implementation WIP*

The context contract defines what is accessible beyond the current sample. Per-sample interactions are self-contained; anything crossing the sample boundary is dataset-level context. The evaluation vs. deployment distinction reduces to what annotations are available at the dataset level.

### Context Levels

| Level | Description |
|---|---|
| **Level 1 — Sample** | A single image, together with as many rounds of prompting as needed. New information is only passed through the API (including the final annotation at completion). |
| **Level 2 — Dataset** | Anything beyond the current sample — prior cases, the full image dataset, accumulated annotations. Can be accessed at any point during a run. |

### Schema Types

| Schema | Description |
|---|---|
| **Dataset-level schema** | Stored alongside the image dataset. Describes properties consistent across the dataset as a whole (e.g. imaging modalities, expected dimensionality, domain). Fixed and known upfront. Algorithms may use it to guide preprocessing decisions. |
| **Sample-level schema** | Included with every request. Declares case-specific properties for this sample — which modalities are available, and where any expected property is absent, why (never acquired, deliberately excluded, corrupted). Carries no label information; safe to pass during evaluation. |

### Context Contract Fields

| Field | Format | Status | Description |
|---|---|---|---|
| `image_cache` | `json` | Optional (unrestricted) | Dataset-level. Unrestricted access to the full image dataset across all cases, which could be used for training; contains relative paths to image files. Requires that stateless API calls provide non-augmented image data. **Note:** For adaptive methods, this cache must be limited to the samples which can be used for training, to prevent data leakage. |
| `dataset_level_schema` | `json` | Optional (unrestricted) | Dataset-level schema: dataset information (e.g. modalities, image channel correspondences, resolution, number of samples) stored alongside image data and accessible without being passed in each request. Fields must correspond to the `image_cache` available. |
| `image` | Array (default) or pointer to file | **Required** | The sample-level image itself. |
| `sample_level_schema` | `json` | Optional | Sample-level information about the image (e.g. modality-to-channel correspondence). |
| `annotations` | `json` or array | Optional (potentially restrictive) | How annotations are made available. In evaluation, released one at a time through the API. In deployment, the algorithm has access to a real history of prior annotations from the clinical site. This is the only meaningful distinction between evaluation and deployment mode. |

---

## 2. Data Referencing Contract

> ⚠️ *Implementation WIP*

A key difference between evaluation and deployment concerns **sample identification**:

- **Evaluation** — The algorithm uses its own internal identifier for each sample. This keeps the algorithm's working store self-contained and prevents any back-door alignment with the framework's reference annotations.
- **Deployment** — The real sample name can be passed back through the contract so that generated segmentations can be paired directly with the original data on the user's mount.

### Architectural Note

| Scope | Approach |
|---|---|
| Per-sample data references | Can be handled statelessly (image passed with each request). |
| Dataset-level data access | Realistically requires persistent storage — shared memory or local mounting — rather than passing the full cache through a stateless API call on every request. |

---

## 3. Prompting Contract

> ⚠️ *Implementation WIP*

Every prompt, regardless of type, carries a minimal required payload and an open auxiliary field. The contract is deliberately malleable to accommodate evolving prompt types (VLMs, text vs. visual prompts, etc.). Multiple prompts may be passed together in a single JSON request.

### Prompt Context — All Types

| Field | Format | Status | Description |
|---|---|---|---|
| `semantic_id_dictionary` | `json` | **Required** | Maps correspondences between the prompt and semantic/instance-level integer codes for the output segmentation. |
| `segmentation_task_type` | `json` | **Required** | Declares what kind of segmentation is expected: `semantic` (classify every voxel by class), `instance` (identify individual objects), or `panoptic` (both). Must be declared upfront — cannot be inferred from prompts alone. |

### Base Prompt Structure — All Types

| Field | Format | Status | Description |
|---|---|---|---|
| `prompt` | `json` | **Required** | The prompt itself (spatial or semantic). Deliberately open-set. Rasterised masks are excluded — the algorithm manages these internally. |
| `type` | `str` | **Required** | Category of prompt: `spatial` (a location or region in the image) or `text` (a word, phrase, or semantic description). |
| `subtype` | `str` | **Required** | Specific type within the category. For spatial: `point`, `scribble`, `lasso`, `bounding_box`, or `other`. For text: `free_text` or `other`. |
| `semantic_id_key` | `json` | Optional | Dictionary of keys identifying the semantic integer code and instance ID associated with a prompt. Required for spatial prompts. |
| `aux` | `json` | Optional | Optional additional information specific to the prompt type (e.g. brush size for a scribble, tolerance for a lasso, ontology reference for a class label). Ignored by the algorithm if not understood — the base contract remains stable. |

### Prompt Taxonomy

The fundamental split is between **spatial** and **text** prompts. Hybrid prompts (e.g. a spatial click + class label) are represented as a **prompt sequence**, not a new type.

#### Spatial Prompts
Anchored to a location in the image. Coordinate information is always required.

| Subtype | Description |
|---|---|
| `point` | A single 3D coordinate. |
| `scribble` | A sequence of 2D coordinates forming an open path. Optional extra: brush size. |
| `bbox` | Two 2D coordinates defining the corners of a bounding box. |
| `lasso` | A closed loop of floating-point 2D coordinates. The algorithm handles inside/outside interpretation. |

#### Semantic / Text Prompts
Based on language or semantic meaning rather than image coordinates.

| Subtype | Description |
|---|---|
| `free_text` | Any free-form text string. Optional extras: language, persistent context (e.g. information/context provided by clinical guidelines). |

> **Note:** All prompt representations must be expressible as lightweight structured data (JSON). Rasterised pixel masks are deliberately excluded from the contract due to overhead concerns.

---

## 4. Output Contract

> ⚠️ *Implementation WIP*

The caller declares what outputs it needs; the algorithm returns what it is capable of providing. Some outputs are always required; others are optional and only returned if both requested and supported by the model.

| Output | Format | Status | Notes |
|---|---|---|---|
| **Segmentation mask** | Two arrays | **Required** | Always returned. **(1)** A semantic array giving a class label for every voxel. **(2)** An instance array giving a unique instance number for every voxel, distinct within each class. For classes where only one instance exists (e.g. background, whole organ), the instance value is `1`. Together, these arrays unambiguously identify every region regardless of task type. Output must match the input image in resolution and spatial orientation. |
| **Probability map** | Array | Optional (if requested + model supports) | Currently assumes a multi-channel representation — one probability value per class per voxel. How this should be structured to expand beyond semantic segmentation is to be resolved by the consortium. |
| **Uncertainty map** | Array | Optional (if requested + model supports) | A validation framework or UI accepts this output but does not compute it — it must come from the app itself. May feed into active learning and evaluation pipelines (e.g. model-informed prompting). Presumably a single-channel array with voxel-level uncertainties. Structure is to be resolved by HAIG consortium. |

---

