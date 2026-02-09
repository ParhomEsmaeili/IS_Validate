'''
This is a script which aggregates standard metrics across multiple runs for a given algorithm and task. 
'''
import numpy as np
import pandas as pd
import os
import sys
import json
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import warnings 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate standard metrics across multiple runs for a given algorithm and task.")
    parser.add_argument('--algorithm_result_roots', type=str, nargs='+', required=True)
    parser.add_argument('--output_result_root', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True, help='Filename of the CSV file containing the metrics.')
    parser.add_argument('--metric', type=str, required=True, help='Reference metric for the threshold')
    parser.add_argument('--infer_info', type=str, required=True, help='json string containing a dictionary describing the inference for determinining the reference columns')
    # parser.add_argument('--output_base_folder', type=str, required=True, help='Output folder for metrics')
    args = parser.parse_args()
    os.makedirs(args.output_result_root, exist_ok=True)
    # print(args.algorithm_result_roots)
    # print(args.metric)
    # print(args.filename)
    infer_info = json.loads(args.infer_info)
    init = infer_info.get('init')
    edit_interaction_max = infer_info.get('edit') 

    if init is None:
        raise ValueError("Inference info must contain 'init' key for initial metric column name.")
    if edit_interaction_max is None:
        raise ValueError("Inference info must contain 'edit' key for maximum number of edit interactions.")

    #now we read the csv files in the nnunet folder, and the reference algorithm as we want to calculate the 
    # metric for.
    algo_metrics_dfs = dict() 

    for idx,result_root in enumerate(args.algorithm_result_roots):
        column_headers = ['Case_Name', init] + [f'Interactive Edit Iter {i + 1}' for i in range(edit_interaction_max)]
        # print(result_root)
        filepath=os.path.join(result_root, args.metric, args.filename)
        algo_metrics_dfs[f'run_{idx}'] = pd.read_csv(filepath, skiprows=1, names=column_headers) 
        #NOTE: This is because the first row contained
        #a column with a header describing the case ID which was messing with the pandas read function. 

    #We will now aggregate the metrics across the different runs through iteration-based averaging. 
    #looping through each item in list and appending to empty data frame
    df = None 
    for run, result in algo_metrics_dfs.items():
        # print(result)
        if df is None:
            df = result
            num_cases = len(result)
            print(f"Number of cases found: {num_cases}")
        else:
            df = pd.concat([df, result], ignore_index=True)
            if len(result) != num_cases:
                raise ValueError(f"Number of cases in run '{run}' ({len(result)}) does not match expected number of cases ({num_cases}) \n "
                " from other runs.")
    # group by case name (string column), average only numeric columns
    # Assume first column is 'Case_Name', rest are numeric
    data_mean = df.groupby('Case_Name', as_index=False).mean(numeric_only=True)
    os.makedirs(os.path.join(args.output_result_root, 'metrics', args.metric), exist_ok=True)
    output_file = os.path.join(args.output_result_root, 'metrics', args.metric, args.filename)
    data_mean.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")

    #Now we write the config used for this particular processing method in order to contextualise the result.
    config_output = {
        'algo_results_paths': args.algorithm_result_roots,
        'infer_info': infer_info,
    }
    config_output_file = os.path.join(args.output_result_root, 'metrics', args.metric, 'aggregation_config.json')
    with open(config_output_file, 'w') as f:
        json.dump(config_output, f, indent=4)
    print(f"Configuration saved to {config_output_file}")