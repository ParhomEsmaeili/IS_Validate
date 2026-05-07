
"""
Plot NSD tolerance comparison:
- Prescribed tolerances vs nnU-Net-derived tolerances
- X-axis: dataset
- Y-axis: tolerance value

Supports loading from JSON/text config files with configurable paths.
"""

import argparse
import json
from pathlib import Path
from typing import Any
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.general_utils.dict_utils import extract_config, extractor


def load_tolerance_config(
    config_path: str,
    tolerance_path: str,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """
    Load tolerance values from a JSON or text config file.
    
    Args:
        config_path: Path to the JSON/text config file
        tolerance_path: Tuple to direct where to extract the tolerance value from in the manifest.
        dataset_name: Dataset name to use. If None, extracted from filename.
    
    Returns:
        DataFrame with columns ["dataset", "tolerance"]
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {config_path}")

    # Load JSON file
    with open(p, "r") as f:
        data = json.load(f)

    # Extract tolerance value
    try:
        tolerance = extractor(data, tolerance_path)
        tolerance = float(tolerance)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to extract tolerance from {config_path} using path '{tolerance_path}': {e}")

    # Determine dataset name
    if dataset_name is None:
        # Use filename without extension
        dataset_name = p.stem

    return pd.DataFrame({
        "dataset": [dataset_name],
        "tolerance": [tolerance]
    })


def build_comparison(
    prescribed_dfs: dict[str, list[pd.DataFrame]],
    nnunet_dfs: dict[str, list[pd.DataFrame]],
    dataset_correspondence: dict[str, str],
) -> pd.DataFrame:
    """
    Merge prescribed and nnU-Net DataFrames based on dataset correspondence.
    
    Args:
        prescribed_dfs: Dict mapping prescribed dataset names to lists of DataFrames
        nnunet_dfs: Dict mapping nnU-Net dataset names to lists of DataFrames
        dataset_correspondence: Dict mapping prescribed dataset names to nnU-Net dataset names
    
    Returns:
        Merged DataFrame with columns ["dataset", "tolerance_prescribed", "tolerance_nnunet"]
    """
    # Initialise the merged DataFrame
    merged = pd.DataFrame()

    # Merge each prescribed DataFrame with the corresponding nnU-Net DataFrame
    for prescribed_dataset, nnunet_dataset in dataset_correspondence.items():
        prescribed_df_list = prescribed_dfs.get(prescribed_dataset)
        nnunet_df_list = nnunet_dfs.get(nnunet_dataset)

        assert prescribed_df_list is not None and nnunet_df_list is not None, \
            f"Missing DataFrame for prescribed dataset '{prescribed_dataset}' or nnU-Net dataset '{nnunet_dataset}'. Please check your config files and paths."
        
        # Combine the list of DataFrames (typically length 1)
        prescribed_df = pd.concat(prescribed_df_list, ignore_index=True) if prescribed_df_list else None
        nnunet_df = pd.concat(nnunet_df_list, ignore_index=True) if nnunet_df_list else None

        if prescribed_df is not None and nnunet_df is not None:
            temp_merged = prescribed_df.merge(
                nnunet_df,
                on="dataset",
                how="outer",
                suffixes=("_prescribed", "_nnunet"),
            )
            merged = pd.concat([merged, temp_merged], ignore_index=True)

    return merged


def get_next_config_id() -> str:
    """
    Auto-generate the next config ID based on existing directories in comparison_plots/.
    
    Returns:
        Next config ID as a string (e.g., "1", "2", "3", ...)
    """
    tolerance_extraction_dir = Path(__file__).parent
    comparison_plots_dir = tolerance_extraction_dir / "comparison_plots"
    
    # If directory doesn't exist, start with ID 1
    if not comparison_plots_dir.exists():
        return "1"
    
    # Count subdirectories
    existing_dirs = [d for d in comparison_plots_dir.iterdir() if d.is_dir()]
    num_dirs = len(existing_dirs)
    
    return str(num_dirs + 1)


def plot_comparison(
    df: pd.DataFrame,
    title: str = "NSD Tolerance Comparison",
    output: str | None = None,
    show: bool = True,
) -> None:
    """
    Create and save a bar chart comparing prescribed vs nnU-Net tolerances.
    
    Args:
        df: Comparison DataFrame with columns ["dataset", "tolerance_prescribed", "tolerance_nnunet"]
        title: Plot title
        output: Path to save the plot. If None, plot is not saved.
        show: Whether to display the plot window.
    """
    if df.empty:
        raise ValueError("No data to plot after filtering/merging.")

    datasets = df["dataset"].tolist()
    print(f"Datasets for x-axis: {datasets}")
    
    x = np.arange(len(datasets))
    width = 0.38

    prescribed_vals = df["tolerance_prescribed"].to_numpy(dtype=float)
    nnunet_vals = df["tolerance_nnunet"].to_numpy(dtype=float)

    # Increase figure size based on number of datasets
    fig_width = max(12, len(datasets) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(x - width / 2, prescribed_vals, width, label="Prescribed NSD tolerance")
    ax.bar(x + width / 2, nnunet_vals, width, label="nnU-Net-derived tolerance")

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Tolerance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved plot: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_comparison_with_manifest(
    comparison_df: pd.DataFrame,
    prescribed_config: str,
    prescribed_path: str,
    nnunet_config: str,
    nnunet_path: str,
    config_id: str,
    title: str,
    show: bool,
) -> None:
    """
    Save comparison plot and manifest to comparison_plots/{config_id}/ directory.
    
    Args:
        comparison_df: Comparison DataFrame
        prescribed_config: Path to prescribed config file
        prescribed_path: Path expression for prescribed tolerance
        nnunet_config: Path to nnU-Net config file
        nnunet_path: Path expression for nnU-Net tolerance
        config_id: Configuration ID (subdirectory name)
        title: Plot title
        show: Whether to display the plot
    """
    tolerance_extraction_dir = Path(__file__).parent
    comparison_plots_dir = tolerance_extraction_dir / "comparison_plots"
    config_dir = comparison_plots_dir / config_id
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "config_id": config_id,
        "sources": {
            "prescribed": {
                "config_path": str(Path(prescribed_config).resolve()),
                "tolerance_path": prescribed_path,
            },
            "nnunet": {
                "config_path": str(Path(nnunet_config).resolve()),
                "tolerance_path": nnunet_path,
            }
        }
    }

    # Save manifest
    manifest_path = config_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

    # Save plot
    plot_path = config_dir / f"{config_id}_tolerance_comparison.png"
    plot_comparison(
        comparison_df,
        title=title,
        output=str(plot_path),
        show=show,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare prescribed NSD tolerances with nnU-Net-derived tolerances from JSON/text configs."
    )
    parser.add_argument(
        "--prescribed_config_basedir",
        required=False, #True,
        default="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs",
        help="Path to prescribed tolerance config file (JSON/text).",
    )
    parser.add_argument(
        "--prescribed_path",
        required=False, #True,
        nargs="+",
        default=["prototype_annotator_4", "metrics", "NSD", "tolerance_mm"],
        #["prototype", "metrics", "NSD", "tolerance_mm"],
        help="Iterable path to extract tolerance value from the config. We assume it is fixed across the set of datasets configured.",
    )
    parser.add_argument(
        "--nnunet_config_basedir",
        required=False, #True,
        default="/home/parhomesmaeili/Helmholtz Group/nnUNet/nnUNet_surface_distance/kfold_5_train",
        help="Path to nnU-Net-derived tolerance config file (JSON/text).",
    )
    parser.add_argument(
        "--nnunet_label_path_basedir",
        required=False, #True,
        default="/home/parhomesmaeili/Helmholtz Group/nnUNet/nnUNet_raw",
        help="Path to the dataset json file which contains the config labels for which we have derived tolerances.",
    )
    parser.add_argument(
        "--prescribed_datasets",
        nargs="+",
        default=[
            # "Dataset001_BrainTumour", 
            # "Dataset003_Liver", 
            # "Dataset004_Hippocampus",
            # "Dataset005_Prostate",
            # "Dataset006_Lung",
            # "Dataset007_Pancreas",
            # "Dataset008_HepaticVessel",
            # "Dataset010_Colon"
            "Dataset011_Kits23",
            "Dataset015_Kits23",
            "Dataset019_Kits23",
            "Dataset023_Kits23",
            "Dataset027_Kits23",
            "Dataset031_Kits23"
            ],
        help="Optional list of datasets to include in the comparison"
    )
    parser.add_argument(
        "--nnunet_datasets",
        nargs="+",
        default=[
            # "Dataset001_BrainTumour", 
            # "Dataset003_Liver", 
            # "Dataset004_Hippocampus",
            # "Dataset005_Prostate",
            # "Dataset006_Lung",
            # "Dataset007_Pancreas",
            # "Dataset008_HepaticVessel",
            # "Dataset010_Colon"
            "Dataset011_Kits23",
            "Dataset015_Kits23",
            "Dataset019_Kits23",
            "Dataset023_Kits23",
            "Dataset027_Kits23",
            "Dataset031_Kits23"
            ],
        help="Optional list of datasets to include in the comparison (should correspond to the datasets for which we " \
        "have derived nnU-Net tolerances). Needs to be a 1-to-1 map but need not be the same, as we have different naming conventions"
    )
    parser.add_argument(
        "--title",
        default="NSD Tolerance Comparison",
        help="Plot title.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window.",
    )
    args = parser.parse_args()
    
    assert len(args.prescribed_datasets) == len(args.nnunet_datasets), "The number of prescribed datasets must match the number of nnU-Net datasets for a 1-to-1 comparison."
    #We assume that currently we only have a single tolerance for the validation side, even if we had multiple classes.
    #(Which we do not). Given that kits had different tolerances we have split these into separate datasets (also because
    #we have different quantities of cases with the foreground available for each class).
    foreground_classes = {
        dataset: [
            class_label for class_label in extract_config(os.path.join(args.nnunet_label_path_basedir, dataset, 'dataset.json'), 'labels')
            if class_label != 'background'
        ]
        for dataset in args.prescribed_datasets
    }

    nnu_net_config_paths = {
        nnunet_dataset: [(f"symmetric_surface_distance_95.0_{fg_class}", "median") for fg_class in foreground_classes[nnunet_dataset]]
        for nnunet_dataset in args.nnunet_datasets
    }
    prescribed_config_paths = {
        prescribed_dataset: [tuple(args.prescribed_path)] for prescribed_dataset in args.prescribed_datasets
    }
    
    nnunet_dfs = dict()
    for nnunet_dataset, config_paths in nnu_net_config_paths.items():
        nnunet_dfs[nnunet_dataset] = [
            load_tolerance_config(
                os.path.join(args.nnunet_config_basedir, nnunet_dataset, 'surface_distance_summary.json'), 
                conf_path,
                dataset_name=nnunet_dataset
            ) 
                for conf_path in config_paths
            ]
        assert len(nnunet_dfs[nnunet_dataset]) == 1, "Currently only support one tolerance per dataset for nnU-Net-derived tolerances. Please check your config files and paths."
    
    # Load tolerance values from config files
    prescribed_dfs = dict()
    for prescribed_dataset, prescribed_paths in prescribed_config_paths.items():
        prescribed_dfs[prescribed_dataset] = [
            load_tolerance_config(
                os.path.join(args.prescribed_config_basedir, prescribed_dataset, 'metrics_configs.txt'),
                prescribed_path,
                dataset_name=prescribed_dataset
            )
            for prescribed_path in prescribed_paths
        ]
        assert len(prescribed_dfs[prescribed_dataset]) == 1, "Currently only support one tolerance per dataset for prescribed-derived tolerances. Please check your config files and paths."
    
    # Create dataset correspondence mapping
    dataset_correspondence = dict(zip(args.prescribed_datasets, args.nnunet_datasets))
    
    comparison_df = build_comparison(prescribed_dfs, nnunet_dfs, dataset_correspondence)

    # Optional: report missing values in either source
    missing_prescribed = comparison_df["tolerance_prescribed"].isna().sum()
    missing_nnunet = comparison_df["tolerance_nnunet"].isna().sum()
    if missing_prescribed or missing_nnunet:
        print(
            f"Warning: missing values -> prescribed: {missing_prescribed}, nnunet: {missing_nnunet}"
        )

    # Auto-generate config ID based on existing directories
    config_id = f"config_{get_next_config_id()}"
    print(f"Auto-generated config ID: {config_id}")

    # Save comparison with manifest
    save_comparison_with_manifest(
        comparison_df,
        prescribed_config=args.prescribed_config_basedir,
        prescribed_path=".".join(args.prescribed_path),
        nnunet_config=args.nnunet_config_basedir,
        nnunet_path="symmetric_surface_distance_95.0",
        config_id=config_id,
        title=args.title,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()