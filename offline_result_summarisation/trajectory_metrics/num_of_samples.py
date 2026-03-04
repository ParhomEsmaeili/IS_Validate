import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
from scipy.stats import norm, t, laplace, beta
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import warnings 
from typing import Dict, List


TRANSLATE_NNUNET_STATISTIC_TO_PSEUDOTIME_NAME = {
    'quantile': {"0.25":"lq", "0.5":"median", "0.75":"uq"},
    'gaussian': {"0":"mean"},
}
std_bounded_statistics = ('gaussian', 'student', 'laplace', 'beta')
quantile_statistics = ('quantile',)

def load_metrics_config(json_dict:str) -> Dict[str, List[str]]:
    cfg = json.loads(json_dict)
    if not isinstance(cfg, dict):
        raise ValueError("metrics config must be a JSON object mapping metric -> [metric information]")
    return cfg

def extract_nnunet_thresholds(nnunet_statistic, nnunet_bounds, nnunet_metrics_dfs, metrics):
    for statistic_id, fit in enumerate(nnunet_statistic):
        if fit in std_bounded_statistics and (float(nnunet_bounds[statistic_id]) < 0):
            raise ValueError(f"Invalid nnUNet bound used")
        if fit in quantile_statistics and (float(nnunet_bounds[statistic_id]) < 0 or float(nnunet_bounds[statistic_id]) > 1):
            raise ValueError(f"Invalid nnUNet bound used")


        nnunet_thresholds = dict()

        for metric in metrics:
            nnunet_df = nnunet_metrics_dfs[metric]

            #We calculate the threshold for the nnUNet metric based on the specified statistic and standard deviation bound.
            if fit == 'gaussian':
                # raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                # Fit a Gaussian distribution to the nnUNet metric values
                mu, sigma = norm.fit(nnunet_df['Automatic Init'])
                mean = mu
                std = sigma

                mean = nnunet_df['Automatic Init'].mean()
                std = nnunet_df['Automatic Init'].std()
                threshold = mean - float(nnunet_bounds[statistic_id]) * std

            elif fit == 'student':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')

                # Fit a Student's t-distribution to the nnUNet metric values
                # mean = nnunet_df['Automatic Init'].mean()
                # std = nnunet_df['Automatic Init'].std()
                df, loc, scale = t.fit(nnunet_df['Automatic Init'])
                mean = t.mean(df, loc=loc, scale=scale)
                std = t.std(df, loc=loc, scale=scale)
                threshold = mean - float(args.nnunet_bound[statistic_id]) * std

            elif fit == 'laplace':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')

                # Fit a Laplace distribution to the nnUNet metric values
                loc, scale = laplace.fit(nnunet_df['Automatic Init'])
                mean = laplace.mean(loc=loc, scale=scale)
                std = laplace.std(loc=loc, scale=scale) 
                threshold = mean - args.nnunet_bound[statistic_id] * std

            elif fit == 'beta':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                #Fit a beta distribution to the nnUNet metric values as it is bounded between 0 and 1
                #and flexible to asymmetric distributions. 
                
                #We clip the values to the open interval (0, 1) to avoid issues with beta fitting.
                #We use a very very small epsilon.
                eps = 1e-6
                data = np.clip(data, eps, 1 - eps)
                a, b, loc, scale = beta.fit(data, floc=0, fscale=1) 
                mean = beta.mean(a, b, loc=loc, scale=scale)
                std = beta.std(a, b, loc=loc, scale=scale)
                threshold = mean - args.nnunet_bound[statistic_id] * std

            elif fit == 'quantile':
                #We use a quantile based thresholding for this... more ad-hoc.
                threshold = nnunet_df['Automatic Init'].quantile(float(nnunet_bounds[statistic_id]))
                # threshold = percentile
                line_name = f'Q = {nnunet_bounds[statistic_id]}'
            else:
                NotImplementedError(f"Unknown fit type: {fit}. Supported types are 'gaussian', 'student', 'laplace', 'beta', 'quantile'.")

            if threshold < 0:
                threshold = 0
                warnings.warn(f"Threshold for {metric} is negative for metrics which are bounded above 0. Setting to 0.")
            nnunet_thresholds[metric] = threshold
    return nnunet_thresholds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate number of samples")
    parser.add_argument('--algo_results_path', type=str, required=True)
    parser.add_argument('--output_result_root', type=str, required=True)
    
    parser.add_argument('--metrics_config', type=str, required=True, help='JSON string of metrics config')
    # parser.add_argument('--nnunet_fraction', type=float, nargs='+', default=[0.9,1], help='Fraction of nnUNet performance for thresholding against')
    parser.add_argument('--reference_root', type=str, required=True)
    parser.add_argument('--reference_filename', type=str, required=True)
    parser.add_argument('--nnunet_statistic', required=True, nargs='+', help='Statistical fit for selecting threshold.')
    parser.add_argument('--nnunet_bound', required=True, nargs='+', help='Bound for the nnUNet metric thresholding wrt statistic.')
    
    # parser.add_argument('--output_base_folder', type=str, required=True, help='Output folder for metrics')
    args = parser.parse_args()
    os.makedirs(args.output_result_root, exist_ok=True)
    
    metrics_config = load_metrics_config(args.metrics_config)
    # Path to all_pseudotime_metrics.csv (update as needed)
    pseudotime_metrics_path = os.path.join(args.algo_results_path, 'all_pseudotime_metrics.csv')
    if os.path.exists(pseudotime_metrics_path):
        df = pd.read_csv(pseudotime_metrics_path)
        # nnUNet reference thresholds (replace with actual values or load from config)
        nnunet_df_column_headers = ['Case_Name', 'Automatic Init']
        nnunet_dfs = {
            metric: pd.read_csv(os.path.join(args.reference_root, f"{metric}", args.reference_filename), skiprows=1, names=nnunet_df_column_headers) 
            for metric in metrics_config.keys()
        }
        nnunet_thresholds = extract_nnunet_thresholds(
            args.nnunet_statistic, 
            args.nnunet_bound, 
            nnunet_dfs,
            metrics_config.keys()
        )
        # Find the first index where metric under consideration in pseudotime hit or exceed nnUNet threshold
        cross_indices = dict()
        for metric,config in metrics_config.items():
            if metric not in nnunet_thresholds:
                raise ValueError(f"Threshold for metric '{metric}' not found in extracted nnUNet thresholds.")
            #Lets construct the name of the col from the metric config.
            if len(args.nnunet_statistic) == 1:
                statistic_dict = TRANSLATE_NNUNET_STATISTIC_TO_PSEUDOTIME_NAME[args.nnunet_statistic[0]]
                statistic_name = statistic_dict[args.nnunet_bound[0]]
                col_name = f'{metric}_{statistic_name}'
                if len(config) == 1:
                    subcomp_name = config[0]
                    col_name = f'{col_name} {subcomp_name}' #super hacky code. probabbly wont generalise beyond the per-itermetrics.
                elif len(config) == 0:
                    pass
                else:
                    raise NotImplementedError("Currently only supports at most one subcomponent per metric.")
                
            else:
                raise NotImplementedError("Currently only supports one nnUNet statistic for thresholding. Please provide exactly one statistic in --nnunet_statistic.")
            
            pseudo_metric_vals = df[col_name].values
            print(pseudo_metric_vals)
            cross_idx = np.argmax(pseudo_metric_vals >= nnunet_thresholds[metric]) if np.any(pseudo_metric_vals >= nnunet_thresholds[metric]) else np.nan  # If never crosses, set to length of array
            print(f"Threshold ({nnunet_thresholds[metric]}) crossed at pseudotime index: {cross_idx}")
            cross_indices[metric] = cross_idx

        #convert the dict into a pandas df.
        df_cross_indices = pd.DataFrame.from_dict(cross_indices, orient='index', columns=['cross_index'])
        df_cross_indices /= len(df)  # Normalize by the number of pseudotime points to get a fraction
        threshold_cross_output = os.path.join(args.output_result_root, 'pseudotime_threshold_crossing.csv')
        df_cross_indices.to_csv(threshold_cross_output, index=True, na_rep='nan')
        print(f"Pseudotime threshold crossing indices saved to {threshold_cross_output}")
    else:
        raise FileNotFoundError(f"Pseudotime metrics file not found at {pseudotime_metrics_path}. Please check the path and ensure the file exists.")