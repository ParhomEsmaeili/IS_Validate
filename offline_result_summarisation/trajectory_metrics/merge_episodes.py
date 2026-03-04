from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import argparse
import json
import os
import re 

# Metrics that can have multiple iterations (borrowed from table_generator)
ITERABLE_METRICS = {
    'Dice_median',
    'Dice_mean',
    'NSD_median',
    'NSD_mean'
}

#we are just going to extract the name of the statistic.
cross_check_with_nnunet_metric_regex = {
    '^Dice_(mean|median) (Interactive.*)': lambda m: f"{m.group(1).capitalize()}",
    '^Dice_(mean|median) (Init.)': lambda m: f"{m.group(1).capitalize()}",
    '^NSD_(mean|median) (Init.)': lambda m: f"{m.group(1).capitalize()}",
    '^NSD_(mean|median) (Interactive.*)': lambda m: f"{m.group(1).capitalize()}",
    # '^Normalised_(Mean|Median)_NOI': lambda m: f"{m.group(1).capitalize()}",
}   

# MAP_METRIC_TO_SUBPATH = {
#     'Dice_auc_mean': os.path.join('AUC', 'Dice_auc_mean.csv'),
#     'NSD_auc_mean': os.path.join('AUC', 'NSD_auc_mean.csv'),
#     'Dice_median': os.path.join('Dice', 'Dice_median.csv'),
#     'NSD_median': os.path.join('NSD', 'NSD_median.csv'),
#     'Normalised_Mean_NOI': os.path.join('NOI', 'Normalised_Mean_NOI.csv'),
#     'Failure_Cases_Fraction': os.path.join('Failure_Cases', 'Failure_Cases_Fraction.csv')
# }

def load_metrics_config(json_dict: str) -> Dict[str, Any]:
    """Parse metrics config JSON string."""
    cfg = json.loads(json_dict)
    if not isinstance(cfg, dict):
        raise ValueError("metrics config must be a JSON object mapping metric -> [metric information]")
    return cfg


def gather_metrics_from_folder(folder_path: str, metrics_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Searches a given folder for the metrics files specified in metrics_config, then extracts the relevant metrics.
    Returns a dictionary of extracted metrics flattened into dictionaries with single layer depth.
    
    This is the same extraction logic used in table_generator.py.
    """
    metric_dict: Dict[str, Any] = {}
    
    for metric_file, metric_info in metrics_config.items():
        file_path = os.path.join(folder_path, metric_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metric file not found: {file_path}")
        
        table = pd.read_csv(file_path)

        for metric_name, extraction_info in metric_info.items():
            if extraction_info is None:
                # Single row extraction - verify and extract
                if table.shape[0] != 1:
                    raise ValueError(
                        f"Expected a single row in {metric_file} for metric {metric_name}, "
                        f"but found {table.shape[0]} rows. "
                        "Please amend the metrics config to reflect the correct extraction strategy."
                    )
                metric_dict[metric_name] = table.at[0, metric_name]
            else:
                # Multi-row extraction using search strings
                if 'rows' in extraction_info:
                    for search_str in extraction_info['rows']:
                        mask = table.apply(
                            lambda row: row.astype(str).str.contains(search_str, na=False)
                        ).any(axis=1)
                        indices = table.index[mask].tolist()
                        
                        if len(indices) != 1:
                            raise ValueError(
                                f"{len(indices)} rows found matching '{search_str}' "
                                f"in {metric_file} for metric {metric_name}."
                            )
                        
                        if metric_name in ITERABLE_METRICS:
                            if indices[0] == 0:
                                metric_dict[f'{metric_name} Init.'] = table.at[indices[0], metric_name]
                            else:
                                metric_dict[f'{metric_name} Interactive Edit Iter {indices[0]}'] = table.at[indices[0], metric_name]
                        else:
                            metric_dict[metric_name] = table.at[indices[0], metric_name]
                else:
                    raise NotImplementedError(
                        "Unsupported extraction information provided in metrics config. "
                        "Only 'rows' is a supported configuration for multi-row extraction."
                    )
    
    return metric_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Merge episode metrics across directories')
    parser.add_argument('--adaptive', action='store_true', help='Flag to indicate if the run is adaptive')
    parser.add_argument('--metric_roots', nargs='+', required=True, help='List of paths to metric root directories')
    parser.add_argument('--pretrained_name', type=str, required=True, help='Name of the pretrained model for non-adaptive runs')
    parser.add_argument('--output_result_root', type=str, required=True, help='Path to output the merged results')
    parser.add_argument('--metrics_config', type=str, required=True)
    parser.add_argument('--samples_counts', nargs='+', type=int, help='List of cumulative sample counts for each episode, required for adaptive runs')
    parser.add_argument('--total_samples', type=int, required=True, help='Total number of samples (may differ from sum of episode lengths)')
    parser.add_argument('--nnunet_statistic', required=True, help='Statistic name for selecting threshold for nnUNet comparison. This is used for a final sanity check to ensure that the correct per-iteration metric is being used for comparison to nnUNet performance.')
    # Add more arguments here
    return parser.parse_args()

def merge_episodes(
    metrics: Dict[int, float],
    episode_samples_dict: Dict[int, int],
    total_samples: int
) -> np.ndarray:
    """
    Merge metrics by spreading values across samples allocated by episode number.
    
    Args:
        metrics: Dictionary mapping episode number to metric value
        episode_samples_dict: Dictionary mapping episode number to CUMULATIVE sample count
                         (i.e., total samples up to and including that episode)
    
    Returns:
        Numpy array of merged metrics spread across samples
    """
    assert len(metrics) == len(episode_samples_dict) + 1, "Metrics must have one more (pretrained) entry than episode_samples_dict"
    merged = []
    # sorted_episodes = sorted(metrics.keys())
    
    for episode_number, num_episode_samples in episode_samples_dict.items():
        metric_value = metrics[episode_number]
        cumulative_samples = num_episode_samples
        # print(cumulative_samples)
        # Compute per-episode sample count from cumulative values
        if episode_number == 0:
            num_samples = cumulative_samples - 1 #We subtract 1 here because we want our expected performance at
            #N samples to be represent the performance of the model at that point. So if its trained on N, then
            #we pad to N-1. So that ~N is the new performance.
            print(num_samples)
        else:
            prev_episode = list(episode_samples_dict.keys())[episode_number - 1]
            
            num_samples = cumulative_samples - episode_samples_dict[prev_episode] 
            print(num_samples)
            #We don't need to subtract 1 here, as the lag from the initial phase propagates through for the pseudotime
            #performance.
        
        # Spread the current metric value across all samples in this episode
        episode_values = np.full(num_samples, metric_value)
        merged.extend(episode_values)

        if episode_number == len(episode_samples_dict) - 1: #zero indexed, so we check ep number is len - 1
            break 
        #We break here so that we can handle the final episode separately, as we want to ensure that we don't have an 
        # off-by-one error with the total sample count. 
        # We will handle the final episode after the loop, and we will pad to total samples if necessary.
        
    # print(episode_number)
    # print(len(metrics))
    # print(len(episode_samples_dict))
    assert episode_number + 1 == len(metrics) - 1, "Episode number mismatch between metrics and episode_samples_dict"
    #We assert that the episode number (+1 due to zero index) must be one fewer than the number of metrics. This is because we start from
    #a pretrained state, so the num samplse always represent the length of samplesbetween current episode and prior.

    #Lets now assert that merged length cannot be larger than total samples
    if len(merged) >= total_samples:
        raise ValueError(f"Merged length {len(merged)} cannot be geq than total samples {total_samples}. Please check the episode_samples_dict for consistency.")
        #We have a time lag, at MOST the update for the final episode falls directly on the last sample. so we 
        #would have to pad that 1 sample with the final episode's value, but if we are already at or above total samples, then we have a problem with the episode_samples_dict.
    
    # If the total number of samples is greater than the sum of episode samples, pad with last episode's value
    if len(merged) < total_samples:
        merged.extend([metrics[episode_number+1]] * (total_samples - len(merged)))
    
    return np.array(merged)


# Example usage
if __name__ == "__main__":
    args = parse_args()
    
    print(f"Adaptive flag: {args.adaptive}")
    print(f"Pretrained name: {args.pretrained_name}")
    print(f"Output result root: {args.output_result_root}")
    print(f"Metrics config: {args.metrics_config}")
    if args.samples_counts:
        print(f"Sample counts: {args.samples_counts}")
    
    # Load metrics config
    metrics_config = load_metrics_config(args.metrics_config)

    if args.adaptive and not args.samples_counts:
        raise ValueError("For adaptive runs, --samples_counts must be provided to specify the cumulative sample counts for each episode.")

    if args.adaptive:
        # Extract metrics from each metric root
        all_episode_metrics: Dict[int, Dict[str, Any]] = {}
        for episode_idx, root in enumerate(args.metric_roots):
            print(f"Processing episode {episode_idx}: {root}")
            episode_metrics = gather_metrics_from_folder(root, metrics_config)
            all_episode_metrics[episode_idx] = episode_metrics
            print(f"  Extracted metrics: {list(episode_metrics.keys())}")
        
        # Build episode_samples mapping if sample counts provided
        if args.samples_counts:
            episode_samples = {i: count for i, count in enumerate(args.samples_counts)}
            print(f"Episode samples mapping: {episode_samples}")
        else:
            raise ValueError("Sample counts must be provided for adaptive runs to determine episode sample allocations.")
        # For each metric, merge across episodes
        output_pd = pd.DataFrame()
        if all_episode_metrics:
            first_episode_metrics = all_episode_metrics[0]
            for metric_name in first_episode_metrics.keys():
                # Build per-episode values for this metric
                metric_per_episode = {
                    ep: all_episode_metrics[ep].get(metric_name)
                    for ep in all_episode_metrics.keys()
                }
                print(f"Metric '{metric_name}' per episode: {metric_per_episode}")
                
                merged = merge_episodes(metric_per_episode, episode_samples, total_samples=args.total_samples)
                print(f"  Merged array shape: {merged.shape}")
                table = pd.DataFrame({f'{metric_name}': merged})
                if output_pd.empty:
                    output_pd = table
                else:
                    output_pd = pd.concat([output_pd, table], axis=1)
            os.makedirs(os.path.join(args.output_result_root, 'pseudotime_metrics'), exist_ok=True)
            output_path = os.path.join(args.output_result_root, 'pseudotime_metrics', 'all_pseudotime_metrics.csv')
            output_pd.to_csv(output_path, index=False)
            print(f"Saved merged trajectory for all metrics to {output_path}")
        else:
            raise ValueError("No episode metrics were extracted. Please check the metric roots and config.")
        
        #Lets do a final check to ensure that we are using the correct per-iter metrics for expected performance and
        #downstream use to compare to nnunet performace (i.e., lets assure that its the same as the statistic used for nnunet NOI thresholding)
        for metric in first_episode_metrics.keys():
            for pat, statistic_name in cross_check_with_nnunet_metric_regex.items():
                m = re.match(pat, metric)
                if m:
                    if callable(statistic_name):
                        equivalent_name = statistic_name(m)
                    else:
                        equivalent_name = statistic_name
                    print(f"Checking metric '{equivalent_name}' against expected nnUNet statistic '{args.nnunet_statistic}' for consistency.")
                    assert equivalent_name == args.nnunet_statistic, f"Expected nnUNet statistic '{equivalent_name}' for metric '{metric}'. Please check the input files and ensure consistency."

    else:
        #Creating a pseudo trajectory for non-adaptive runs is straightforward, we just need to repeat the metric value across all samples.
        assert len(args.metric_roots) == 1, "For non-adaptive runs, only one metric root should be provided."
        assert args.pretrained_name in args.metric_roots[0], "For non-adaptive runs, the provided metric root should match the pretrained name."
        print(f"Processing non-adaptive run with pretrained model: {args.pretrained_name}")
        non_adaptive_metrics = gather_metrics_from_folder(args.metric_roots[0], metrics_config)
        print(f"Extracted metrics: {list(non_adaptive_metrics.keys())}")
        # create a table across metrics with a pseudo-trajectory by repeating the value across total_samples
        final_table = pd.DataFrame()
        for metric_name, metric_value in non_adaptive_metrics.items():
            merged = np.full(args.total_samples, metric_value)
            #Now we need to save this to a pandas array
            table = pd.DataFrame({f'{metric_name}': merged})
            final_table = pd.concat([final_table, table], axis=1)
        os.makedirs(os.path.join(args.output_result_root, 'pseudotime_metrics'), exist_ok=True)
        output_path = os.path.join(args.output_result_root, 'pseudotime_metrics', 'all_pseudotime_metrics.csv')
        final_table.to_csv(output_path, index=False)
        print(f"Saved merged trajectory for all metrics to {output_path}")

        #Lets do a final check to ensure that we are using the correct per-iter metrics for expected performance and
        #downstream use to compare to nnunet performace (i.e., lets assure that its the same as the statistic used for nnunet NOI thresholding)
        for metric in non_adaptive_metrics.keys():
            for pat, statistic_name in cross_check_with_nnunet_metric_regex.items():
                m = re.match(pat, metric)
                if m:
                    if callable(statistic_name):
                        equivalent_name = statistic_name(m)
                    else:
                        equivalent_name = statistic_name
                    print(f"Checking metric '{equivalent_name}' against expected nnUNet statistic '{args.nnunet_statistic}' for consistency.")
                    assert equivalent_name == args.nnunet_statistic, f"Expected nnUNet statistic '{equivalent_name}' for metric '{metric}'. Please check the input files and ensure consistency."
