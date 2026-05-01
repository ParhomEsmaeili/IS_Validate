#!/usr/bin/env python3
"""
Script to pull and summarize nnUNet surface distance metrics from JSON files.
Computes dataset-wide statistics for surface distance metrics.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from src.general_utils.dict_utils import extractor

def load_surface_distance_jsons(directory: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load all surface distance JSON files from the specified directory.
    
    Args:
        directory: Path to directory containing surface distance JSON files
        
    Returns:
        Dictionary containing surface distance metrics: Structured as:
        {
            case_id:{
                'class_label': {
                    'metric_1': value,
                    'metric_2': value,
                    ...}
                }
        }
    """
    json_file = os.path.join(directory, 'surface_distances.json')
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def extract_numeric_value(value: Any) -> float:
    """
    Extract numeric value from potentially nested or formatted data.
    
    Args:
        value: The value to extract from
        
    Returns:
        Float value, or None if not numeric
    """
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def compute_field_statistics(
        metrics_collection: Dict[str, Dict[str, Dict[str, Any]]], 
        field_name: str,
        class_label: str) -> Dict[str, float]:
    """
    Compute statistics for a specific field across all metrics.
    
    Args:
        metrics_collection: Dictionary of metric dictionaries
        field_name: Name of the field to compute statistics for
        class_label: Label of the class for which to compute statistics
        
    Returns:
        Dictionary containing min, max, mean, median, and stdev
    """
    values = []
    dict_paths = [(case_id, class_label, field_name) for case_id in metrics_collection.keys()]
    for path in dict_paths:
        value = extractor(metrics_collection, path)
        numeric_value = extract_numeric_value(value)
        if numeric_value is not None:
            values.append(numeric_value)
    if not values:
        raise ValueError(f"No valid numeric values found for field '{field_name}' and class '{class_label}'")
    
    stats = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }
    
    if len(values) > 1:
        stats["stdev"] = statistics.stdev(values)
        stats["lq"] = statistics.quantiles(values, n=4)[0]  # 25th percentile
        stats["uq"] = statistics.quantiles(values, n=4)[2]  # 75th percentile
        stats["iqr"] = stats["uq"] - stats["lq"]
    
    return stats


def summarize_surface_distance_metrics(metrics_collection: Dict[str, Dict[str,  Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics across all relevant surface distance fields.
    
    Args:
        metrics_collection: Dictionary of metric dictionaries from all samples
        
    Returns:
        Dictionary mapping field names to their computed statistics
    """
    if not metrics_collection:
        print("No metrics found to summarize")
        return {}
    
    # Identify all relevant fields from the first entry
    all_fields = set()
    all_classes = set()
    for case_metric in metrics_collection.values():
        all_classes.update(case_metric.keys())
        for class_metrics in case_metric.values():
            all_fields.update(class_metrics.keys())
            
    print(f"Identified {len(all_fields)} unique metric fields across dataset")
    print(f"All identified fields: {', '.join(sorted(all_fields))}")
    print(f"All identified classes: {', '.join(sorted(all_classes))}")
    
    summary = {}
    for field in sorted(all_fields):
        for class_label in all_classes:
            field_name = f"{field}_{class_label}"
            field_stats = compute_field_statistics(metrics_collection, field, class_label)
            if field_stats:
                summary[field_name] = field_stats
    
    return summary


def print_summary_report(summary: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted summary report of the statistics.
    
    Args:
        summary: Dictionary of computed statistics
    """
    print("\n" + "="*80)
    print("Surface Distance Metrics Summary Report")
    print("="*80 + "\n")
    
    for field_name, stats in summary.items():
        print(f"\n{field_name}:")
        print(f"  Count:     {stats.get('count', 'N/A')}")
        print(f"  Min:       {stats.get('min', 'N/A'):.6f}")
        print(f"  Max:       {stats.get('max', 'N/A'):.6f}")
        print(f"  Mean:      {stats.get('mean', 'N/A'):.6f}")
        print(f"  Median:    {stats.get('median', 'N/A'):.6f}")
        if 'stdev' in stats:
            print(f"  StDev:     {stats['stdev']:.6f}")
        if 'lq' in stats:
            print(f"  25th Percentile: {stats['lq']:.6f}")
        if 'uq' in stats:
            print(f"  75th Percentile: {stats['uq']:.6f}")
        if 'iqr' in stats:
            print(f"  IQR: {stats['iqr']:.6f}")


def save_summary_to_json(summary: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Save summary statistics to a JSON file.
    
    Args:
        summary: Dictionary of computed statistics
        output_path: Path where to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize nnUNet surface distance metrics across a dataset"
    )
    parser.add_argument(
        "--input_dir",
        help="Directory containing surface distance JSON files"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found: {args.input_dir}")
        return
    
    print(f"Loading JSON files from: {args.input_dir}")
    metrics_collection = load_surface_distance_jsons(args.input_dir)
    print(f"Loaded {len(metrics_collection)} metric files")
    
    summary = summarize_surface_distance_metrics(metrics_collection)
    print_summary_report(summary)
    
    output_dir = os.path.dirname(args.input_dir)
    output_path = os.path.join(output_dir, "surface_distance_summary.json")    
    save_summary_to_json(summary, output_path)


if __name__ == "__main__":
    main()
