from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import re
# NIfTI via SimpleITK
try:
    import SimpleITK as sitk  # type: ignore
except ImportError as exc:
    raise ImportError(
        "SimpleITK is required for .nii/.nii.gz masks. Install with: pip install SimpleITK"
    ) from exc

"""
Compute inter-rater Dice agreement for KiTS23-style converted framework data.

Assumed folder structure (default):
<dataset_root>/<dataset_name>/labelsTr/<annotator_mask_file>

Example:
python inter_rater_threshold.py /path/to/kits23_converted Dataset011_Kits23 --class-labels whole_kidney --save-json results.json
"""

def load_mask(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.int8)


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)


def collect_case_dirs(labels_dir: Path) -> List[Path]:
    return sorted([p for p in labels_dir.iterdir() if p.is_dir()])


def infer_annotator_name(path_obj: Path) -> str:
    """Infer annotator name from path, preferring explicit annotator tokens."""
    joined = str(path_obj)
    m = re.search(r"(annotator[_-]?\w+)", joined, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return path_obj.name.lower()


def select_annotator_dirs(
    annotator_dirs: Sequence[Path],
    include_annotators: Sequence[str] | None,
    exclude_annotators: Sequence[str] | None,
) -> Tuple[List[Path], List[str]]:
    include_set = {a.lower() for a in include_annotators} if include_annotators else None
    exclude_set = {a.lower() for a in exclude_annotators} if exclude_annotators else set()

    selected_dirs: List[Path] = []
    annotator_names: List[str] = []

    for p in annotator_dirs:
        ann = infer_annotator_name(p)
        if include_set is not None and ann not in include_set:
            continue
        if ann in exclude_set:
            continue
        selected_dirs.append(p)
        annotator_names.append(ann)

    return selected_dirs, annotator_names


def collect_case_class_masks(
    case_dir: Path,
    class_label: str,
    include_annotators: Sequence[str] | None,
    exclude_annotators: Sequence[str] | None,
) -> Dict[str, Path]:
    """
    Collect one semantic mask per selected annotator for a given case/class.

    Expected layout:
    labelsTr/<case>/annotator_<id>/semantic_class_<class>/<case>_0001.nii.gz
    """
    annotator_dirs = sorted(
        [p for p in case_dir.iterdir() if p.is_dir() and p.name.lower().startswith("annotator_")]
    )
    annotator_dirs, annotator_names = select_annotator_dirs(
        annotator_dirs,
        include_annotators=include_annotators,
        exclude_annotators=exclude_annotators,
    )

    case_name = case_dir.name
    out: Dict[str, Path] = {}
    for ann_dir, ann_name in zip(annotator_dirs, annotator_names):
        mask_path = ann_dir / f"semantic_class_{class_label}" / f"{case_name}_0001.nii.gz"
        if mask_path.is_file():
            out[ann_name] = mask_path
    return out


def infer_class_labels_from_structure(labels_dir: Path) -> List[str]:
    """Infer class labels from semantic_class_<label> directories."""
    labels = set()
    for case_dir in collect_case_dirs(labels_dir):
        annotator_dirs = [p for p in case_dir.iterdir() if p.is_dir() and p.name.lower().startswith("annotator_")]
        for ann_dir in annotator_dirs:
            for sem_dir in ann_dir.iterdir():
                if sem_dir.is_dir() and sem_dir.name.startswith("semantic_class_"):
                    labels.add(sem_dir.name.replace("semantic_class_", "", 1))

    inferred = sorted(labels)
    return [lb for lb in inferred if lb.lower() != "background"]


def resolve_class_labels(labels_dir: Path, selected_class_labels: Sequence[str] | None) -> List[str]:
    available = infer_class_labels_from_structure(labels_dir)
    if not available:
        raise ValueError(f"No semantic_class_<label> folders found in {labels_dir}")

    if selected_class_labels is None:
        return available

    for label in selected_class_labels:
        if label.lower() == "background":
            continue
        if label not in available:
            raise ValueError(
                f"Requested class label '{label}' not found in semantic class folders. "
                f"Available labels: {sorted(available + ['background'])}"
            )
    return list(selected_class_labels)


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def resolve_default_save_path(dataset_root: Path, dataset_name: str) -> Path:
    """Resolve default path: exp_configs/<dataset_name>/inter_rater_info.json."""
    for ancestor in [dataset_root, *dataset_root.parents]:
        candidate = ancestor / "exp_configs"
        if candidate.is_dir():
            return candidate / dataset_name / "inter_rater_info.json"

    # Fallback if exp_configs is not found while walking upwards.
    return dataset_root.parent / "exp_configs" / dataset_name / "inter_rater_info.json"


def compute_inter_rater(
    dataset_dirs: Sequence[Path],
    labels_subdir: str,
    class_labels: Sequence[str],
    include_annotators: Sequence[str] | None = None,
    exclude_annotators: Sequence[str] | None = None,
) -> Dict[str, object]:
    if not dataset_dirs:
        raise ValueError("No dataset directories provided for inter-rater computation.")

    per_sample = {}
    per_class_all_samples: Dict[str, List[float]] = {label: [] for label in class_labels}
    skipped_due_to_filter = 0

    for ds_dir in dataset_dirs:
        labels_dir = ds_dir / labels_subdir
        case_dirs = collect_case_dirs(labels_dir)
        if not case_dirs:
            continue

        for idx, case_dir in enumerate(case_dirs):
            print(f"Processing case number {idx + 1}/{len(case_dirs)}:")
            case_name = case_dir.name
            class_means = {}
            class_pair_details = {}
            case_annotators = set()

            for class_label in class_labels:
                annotator_to_mask = collect_case_class_masks(
                    case_dir,
                    class_label,
                    include_annotators=include_annotators,
                    exclude_annotators=exclude_annotators,
                )
                annotator_names = sorted(annotator_to_mask.keys())
                if len(annotator_names) < 2:
                    continue

                case_annotators.update(annotator_names)
                arrays = [load_mask(annotator_to_mask[ann]) > 0 for ann in annotator_names]

                shapes = {a.shape for a in arrays}
                if len(shapes) != 1:
                    raise ValueError(
                        f"Shape mismatch in case {case_name}, class {class_label}: {sorted(str(s) for s in shapes)}"
                    )

                pair_scores = []
                pair_detail = {}
                for i, j in itertools.combinations(range(len(arrays)), 2):
                    score = dice_binary(arrays[i], arrays[j])
                    pair_scores.append(score)
                    pair_detail[f"{annotator_names[i]}__vs__{annotator_names[j]}"] = score

                cmean = mean(pair_scores)
                class_means[class_label] = cmean
                class_pair_details[class_label] = pair_detail
                per_class_all_samples[class_label].append(cmean)

            if not class_means:
                skipped_due_to_filter += 1
                continue

            per_sample[case_name] = {
                "dataset_id": ds_dir.name,
                "annotators": sorted(case_annotators),
                "class_mean_dice": class_means,
                "pairwise_dice_by_class": class_pair_details,
            }

    dataset_class_means = {cid: mean(scores) for cid, scores in per_class_all_samples.items()}

    return {
        "n_samples": len(per_sample),
        "n_samples_skipped_after_annotator_filter": skipped_due_to_filter,
        "class_labels": list(class_labels),
        "include_annotators": list(include_annotators) if include_annotators else None,
        "exclude_annotators": list(exclude_annotators) if exclude_annotators else [],
        "dataset_class_mean_dice": dataset_class_means,
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average inter-rater Dice per sample and across dataset."
    )
    parser.add_argument(
        "--dataset_root", 
        type=Path, 
        default='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets',
        help="Root directory containing <dataset_name>/labelsTr/"
        )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Dataset011_Kits23",
        help="Dataset folder name under dataset_root (e.g. Dataset011_Kits23).",
    )
    parser.add_argument(
        "--labels-subdir",
        default="labelsTr",
        help="Name of labels subfolder inside each dataset-id folder (default: labelsTr)",
    )
    parser.add_argument(
        "--class-labels",
        nargs="+",
        default=['whole_kidney'],
        help=(
            "Semantic class labels to evaluate (must match semantic_class_<label> folders). "
            "Default: all non-background class labels inferred from labelsTr structure."
        ),
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help=(
            "Optional path to save detailed JSON report. "
            "Default: exp_configs/<dataset_name>/inter_rater_info.json"
        ),
    )
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=None,
        help=(
            "Optional explicit annotator names to include (case-insensitive). "
            "Example: --annotators annotator_1 annotator_2 annotator_3"
        ),
    )
    parser.add_argument(
        "--exclude-annotators",
        nargs="+",
        default=['annotator_4'],
        help=(
            "Optional annotator names to exclude (case-insensitive). "
            "Useful for dropping pseudo annotators."
        ),
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {dataset_root}")

    dataset_dir = dataset_root / args.dataset_name
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    labels_dir = dataset_dir / args.labels_subdir
    if not labels_dir.is_dir():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    dataset_dirs = [dataset_dir]

    class_labels = resolve_class_labels(labels_dir, args.class_labels)

    report = compute_inter_rater(
        dataset_dirs,
        args.labels_subdir,
        class_labels,
        include_annotators=args.annotators,
        exclude_annotators=args.exclude_annotators,
    )

    print(f"Samples evaluated: {report['n_samples']}")
    print(f"Samples skipped after annotator filter: {report['n_samples_skipped_after_annotator_filter']}")
    print(f"Class labels: {report['class_labels']}")
    print("Dataset per-class mean Dice:")
    for class_label, score in report["dataset_class_mean_dice"].items():
        print(f"  class {class_label}: {score:.4f}")

    save_path = args.save_json or resolve_default_save_path(dataset_root, dataset_dir.name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {save_path}")


if __name__ == "__main__":
    main()