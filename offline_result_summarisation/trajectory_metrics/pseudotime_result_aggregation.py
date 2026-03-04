"""
Script to average a set of tables (CSVs) across runs.
Usage:
    python pseudotime_result_aggregation.py --tables table1.csv table2.csv ... --output averaged.csv
"""
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average a set of tables (CSVs) across runs.")
    parser.add_argument('--algorithm_result_roots', type=str, nargs='+', required=True, help='Paths to CSV tables to average')
    parser.add_argument('--output_result_root', type=str, required=True, help='Root directory for output results')
    args = parser.parse_args()

    dfs = [pd.read_csv(path) for path in args.algorithm_result_roots]
    # Check all tables have the same shape
    shapes = [df.shape for df in dfs]
    if len(set(shapes)) != 1:
        raise ValueError(f"All tables must have the same shape. Got: {shapes}")

    # Average numeric columns
    avg_df = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
    # For non-numeric columns, just take from the first table
    for col in dfs[0].columns:
        if not pd.api.types.is_numeric_dtype(dfs[0][col]):
            avg_df[col] = dfs[0][col]
    # Reorder columns to match input
    avg_df = avg_df[dfs[0].columns]

    output_dir = os.path.join(args.output_result_root, "pseudotime_metrics")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_pseudotime_metrics.csv")
    avg_df.to_csv(output_path, index=False)
    print(f"Averaged table saved to {output_path}")

    # Save config JSON at the end
    import json
    config = {
        'input_algorithm_result_roots': args.algorithm_result_roots,
        'output_path': output_path
    }
    config_path = os.path.join(output_dir, 'pseudotime_aggregation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")