"""
Insert nnU-Net-derived NSD tolerances into metrics_configs.txt files.

This script reads the existing metrics configs (e.g., prototype_annotator_4) and creates
a new config entry with "_nnunet" appended to the name (e.g., prototype_annotator_4_nnunet)
that contains the nnU-Net-derived NSD tolerance value.
"""

import argparse
import json
from pathlib import Path
from typing import Any
import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.general_utils.dict_utils import extract_config, extractor


def load_nnunet_tolerance(
    nnunet_config_basedir: str,
    nnunet_label_path_basedir: str,
    dataset_name: str,
) -> float:
    """
    Load nnU-Net-derived NSD tolerance from surface_distance_summary.json.
    
    Args:
        nnunet_config_basedir: Base directory for nnU-Net configs
        nnunet_label_path_basedir: Base directory for nnU-Net dataset.json files
        dataset_name: Dataset name (e.g., "Dataset011_Kits23")
    
    Returns:
        Tolerance value as float
    """
    # Load foreground class from dataset.json
    dataset_json_path = os.path.join(nnunet_label_path_basedir, dataset_name, 'dataset.json')
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")
    
    labels = extract_config(dataset_json_path, 'labels')
    foreground_classes = [label for label in labels if label != 'background']
    
    if not foreground_classes:
        raise ValueError(f"No foreground classes found in {dataset_json_path}")
    
    # Use first foreground class
    fg_class = foreground_classes[0]
    
    config_path = os.path.join(nnunet_config_basedir, dataset_name, 'surface_distance_summary.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"nnU-Net config not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = json.load(f)
    
    # Extract median symmetric_surface_distance_95.0_{class} tolerance
    tolerance_key = f"symmetric_surface_distance_95.0_{fg_class}"
    try:
        tolerance = extractor(data, (tolerance_key, "median"))
        tolerance = float(tolerance)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to extract tolerance for key '{tolerance_key}' from {config_path}: {e}")
    
    return tolerance


def insert_nnunet_tolerances(
    metrics_config_path: str,
    nnunet_tolerance: float,
) -> dict:
    """
    Read metrics config, create new nnunet entry with updated tolerance.
    
    Args:
        metrics_config_path: Path to metrics_configs.txt
        nnunet_tolerance: nnU-Net-derived tolerance value
    
    Returns:
        Updated config dictionary
    """
    if not os.path.exists(metrics_config_path):
        raise FileNotFoundError(f"Metrics config not found: {metrics_config_path}")
    
    with open(metrics_config_path, "r") as f:
        config = json.load(f)
    
    # Find the first non-nnunet config entry (e.g., prototype_annotator_4)
    source_config_name = None
    for key in config.keys():
        if "_nnunet" not in key:
            source_config_name = key
            break
    
    if source_config_name is None:
        raise ValueError(f"No non-nnunet config found in {metrics_config_path}")
    
    # Create new nnunet config by copying the source
    source_config = config[source_config_name]
    nnunet_config_name = f"{source_config_name}_nnunet"
    
    # Deep copy the source config
    nnunet_config = copy.deepcopy(source_config)
    
    # Update the NSD tolerance - replace tolerance_sf with tolerance_mm if needed
    if "metrics" in nnunet_config and "NSD" in nnunet_config["metrics"]:
        nsd_config = nnunet_config["metrics"]["NSD"]
        if "tolerance_mm" in nsd_config:
            # Field already uses tolerance_mm, just update the value
            nsd_config["tolerance_mm"] = nnunet_tolerance
        elif "tolerance_sf" in nsd_config:
            # Replace tolerance_sf with tolerance_mm
            del nsd_config["tolerance_sf"]
            nsd_config["tolerance_mm"] = nnunet_tolerance
        else:
            # Neither field exists, add tolerance_mm
            nsd_config["tolerance_mm"] = nnunet_tolerance
    else:
        raise ValueError(f"NSD metrics not found in {source_config_name} of {metrics_config_path}")
    
    # Add or update the nnunet config
    config[nnunet_config_name] = nnunet_config
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Insert nnU-Net-derived NSD tolerances into metrics_configs.txt files."
    )
    parser.add_argument(
        "--prescribed_config_basedir",
        required=False,
        default="/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs",
        help="Base directory containing dataset config folders.",
    )
    parser.add_argument(
        "--nnunet_config_basedir",
        required=False,
        default="/home/parhomesmaeili/Helmholtz Group/nnUNet/nnUNet_surface_distance/kfold_5_train",
        help="Base directory containing nnU-Net-derived tolerance configs.",
    )
    parser.add_argument(
        "--nnunet_label_path_basedir",
        required=False,
        default="/home/parhomesmaeili/Helmholtz Group/nnUNet/nnUNet_raw",
        help="Base directory containing nnU-Net dataset.json files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            # "Dataset011_Kits23",
            # "Dataset015_Kits23",
            # "Dataset019_Kits23",
            # "Dataset023_Kits23",
            # "Dataset027_Kits23",
            # "Dataset031_Kits23"
            "Dataset001_BrainTumour",
            "Dataset003_Liver",
            "Dataset004_Hippocampus",
            "Dataset005_Prostate",
            "Dataset006_Lung",
            "Dataset007_Pancreas",
            "Dataset008_HepaticVessel",
            "Dataset010_Colon",
            "Dataset040_MSMultispine",
            "Dataset041_Parse",
            "Dataset042_TopCowCT"
            "Dataset043_TopCowMR",
            "Dataset044_TopBrainCT",
            "Dataset045_TopBrainMR"
        ],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without saving.",
    )
    args = parser.parse_args()
    
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}")
        print('='*60)
        
        try:
            # Load nnU-Net tolerance
            nnunet_tolerance = load_nnunet_tolerance(
                args.nnunet_config_basedir,
                args.nnunet_label_path_basedir,
                dataset
            )
            print(f"nnU-Net tolerance: {nnunet_tolerance}")
            
            # Paths
            metrics_config_path = os.path.join(
                args.prescribed_config_basedir, dataset, 'metrics_configs.txt'
            )
            
            # Insert nnunet tolerances
            updated_config = insert_nnunet_tolerances(metrics_config_path, nnunet_tolerance)
            
            # Print summary
            print(f"Config entries in {dataset}:")
            for config_name in updated_config.keys():
                if "metrics" in updated_config[config_name] and "NSD" in updated_config[config_name]["metrics"]:
                    nsd = updated_config[config_name]["metrics"]["NSD"]
                    # Check which tolerance field exists
                    if "tolerance_mm" in nsd:
                        print(f"  - {config_name}: tolerance_mm = {nsd['tolerance_mm']}")
                    elif "tolerance_sf" in nsd:
                        print(f"  - {config_name}: tolerance_sf = {nsd['tolerance_sf']}")
                    else:
                        print(f"  - {config_name}: no tolerance field found")
            
            # Save
            if not args.dry_run:
                with open(metrics_config_path, "w") as f:
                    json.dump(updated_config, f, indent='\t')
                print(f"✓ Saved: {metrics_config_path}")
            else:
                print("[DRY RUN - No changes saved]")
        
        except Exception as e:
            print(f"✗ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
