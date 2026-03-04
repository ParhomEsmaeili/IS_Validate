#This is a script which uses the per-task and per-metric wilcoxon signed rank test results + the info as to which was better
# on that metric to generate overall algorithm rankings in the MSD style.
# This is a script which generates a set of bools which indicate whether a given pair of datasets
# have statistically significant differences in performance for a given task. For NOW, we will assume
# binary sem.seg tasks only so we do not need to account for different regions formed by merging
# different labels. It will do this for all of the metrics requested.

#The output should look like a set of confusion matrices, essentially. Each cell (i,j) indicates whether algo i is
#statistically different (NOT BETTER, just different) from algo j. We will use this in tandem with the
#other scripts to generate final rankings.

#We follow the MSD challenge approach here: medicaldecathlon.com/files/MSD-Ranking-scheme.pdf
# They use the Wilcoxon signed-rank test to compare different methods.
import os
import re 
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy.stats import wilcoxon, fisher_exact
import argparse
import json

ALGO_AXIS_OF_COMPLEXITY = {
    "Image Voxel Count Variation": [
        "Dataset004_Hippocampus",
        "Dataset001_BrainTumour",
        "Dataset003_Liver",
        ],
    "Target Size Variation": [
        "Dataset006_Lung",
        "Dataset007_Pancreas",
        "Dataset003_Liver"
    ],
    "Image Anisotropy Variation": [
        'Dataset005_Prostate',
        'Dataset001_BrainTumour',
    ],
    "Target Shape Complexity Variation": [
        'Dataset005_Prostate',
        'Dataset001_BrainTumour',
        'Dataset008_HepaticVessel'
    ],
    "Anatomical Site Variation": [
        'Dataset003_Liver',
        'Dataset010_Colon'
    ],
    "Dataset Size Variation": [
        'Dataset005_Prostate',
        'Dataset003_Liver',
        'Dataset007_Pancreas'
    ]
}
        
MAPPING_SUBCATEGORIES = {
    'AUC': 'nAUC',
    'Init.': 'Init.',
    'Interactive Edit Iter': 'Interactive Edit Iter',
    'NOI': 'nNoI',
    'NoF': 'NoF'
}

def sort_by_substring_order(input_list: List[str], priority_list: List[str]) -> List[str]:
    def sort_key(input_list):
        for i, substr in enumerate(priority_list):
            if substr.lower() in input_list.lower():
                return i
        return len(input_list)  # put unmatched at the end
    return sorted(input_list, key=sort_key)

def compute_per_metric_rankings(per_metric_dfs: Dict[str, pd.DataFrame], apps: List[str]) -> pd.DataFrame:
    #Here, for each metric, we compute the ranking of algorithms based on the per-metric dfs. This uses
    #the number of times an algorithm is significantly better than other algorithms as the ranking criterion.

    per_metric_rankings = dict()
    #Lets first assert that any cells with "tie" in the who-is-better table also have False in the stat sig table.
    for metric, dfs in per_metric_dfs.items():
        stat_sig_df = dfs['stat_sig']
        who_is_better_df = dfs['who_is_better']
        assert stat_sig_df.shape == who_is_better_df.shape, f"Shape mismatch between stat sig and who-is-better for metric {metric}."
        
        
        #Here we are going to cross-check the who-is-better table and the stat sig table to ensure that any cells marked as "tie" in the who-is-better table correspond to False in the stat sig table. This is a sanity check to ensure consistency between the two tables before 
        # we proceed with the ranking computation. We will flag if there are any cells where there is a tie in the metric, but
        # the stat sig test indicates a significant difference, as this would be an inconsistency that needs to be investigated.

        #I.e., it means we have an assumption that needs to be revisited. E.g., when we used fisher-exact for NoF, 
        #it was not looking at whether marginal rate differed. Only on a pair-wise disparity in failure/not. So a difference
        #on failed samples could lead to a significant result, even if the overall marginal failure rates were not different. 
        # This is an edge case that we should be aware of and investigate if it arises.
        mask = who_is_better_df.astype(str).apply(lambda col: col.str.contains('tie', regex=False, na=False))
        # Get locations (row, col) tuples
        locations = list(zip(*np.where(mask)))
        # Get the values
        values = stat_sig_df[mask].stack()
        # print(dfs)
        if not all(values == False):
            print(f"Locations with tie in who-is-better table for metric {metric}: {locations}")
            print(f"Apps at those locations: {[(who_is_better_df.iloc[row, col], who_is_better_df.columns[col]) for row, col in locations]}")
            print(f"Corresponding stat sig values at those locations: {values}")
        
            raise Warning(f"Tie found in who-is-better table but corresponding stat sig is True for metric {metric} at locations {locations}. Check the printed details for debugging. \n"
                          "This may indicate a bizarre inconsistency in the data, or an edge case where the who-is-better table has 'tie' for a pair of algorithms but the stat sig test is significant. This should be investigated further to understand the root cause.")
            
        
        # assert all(values == False), f"Tie found in who-is-better table but corresponding stat sig is True for metric {metric} at locations {locations}."

        #Now compute rankings.

        #We filter the who is better according to the stat sig. 
            
        #It performs a cross-comparison of the stat sig test and the who-is-better test to determine this.
        result = who_is_better_df.where(stat_sig_df.astype(bool) & (who_is_better_df != 'tie'), other='ignore') 
        # 'ignore' where not significant, OR where a tie exists. This means only cells where there is a significant difference
        #AND a winner for that metric will be counted. Theoretically, there should not be any cells where this happens if 
        #We selected the statistical significance tests correctly, but this is a safety check to ensure we are only counting significant wins.
    
        #Now, for each algorithm, count how many times it is better (and significantly better) than others, 
        # by counting the number of occurences of its own name in its column.
        counts = {}
        for app in apps:
            better_count = (result[app] == app).sum()  # Count occurrences of apps name within its own column.
            counts[app] = better_count
        grouped_counts = {i:[k for k,v in counts.items() if v == i] for i in range(len(apps))}
        filtered_grouped_counts = {k:v for k,v in grouped_counts.items() if len(v) > 0}
        #Now assign rankings based on counts.
        sorted_counts = sorted(filtered_grouped_counts.keys(), reverse=True)  # Higher counts get
        #If there are ties, we assign the same rank. 
        # rankings = [filtered_grouped_counts[i] for i in sorted_counts]
        final_ranking = {}
        for rank, sublist in enumerate([filtered_grouped_counts[i] for i in sorted_counts]):
            for name in sublist:
                final_ranking[name] = rank + 1 #Ranks start from 1
        pd_ranking = pd.Series(final_ranking)
        pd_ranking = pd_ranking.reindex(apps)  #Ensure order is same as apps list.
        per_metric_rankings[metric] = pd_ranking 
    per_metric_rankings_df = pd.DataFrame(per_metric_rankings)
    return per_metric_rankings_df

def group_and_rank_per_axis_of_complexity_rankings(per_dataset_rankings: pd.DataFrame, apps: List[str]) -> pd.DataFrame:
    #We want to group the dataset rankings according to the axes of complexity.. not averaging, just displaying them together.
    axis_grouped_rankings = {}
    for axis, datasets in ALGO_AXIS_OF_COMPLEXITY.items():
        per_axis_rankings = dict()
        for dataset in datasets:
            assert dataset in per_dataset_rankings, f"Dataset {dataset} in axis {axis} not found in per-dataset rankings."
            per_axis_rankings[dataset] = per_dataset_rankings[dataset]
        #First we group the datasets according to the axes of complexity.
        axis_grouped_rankings[axis] = pd.DataFrame(per_axis_rankings)

    #Now lets compute rankings for each axis of complexity by averaging the rankings across the datasets within that axis, and then ranking the algorithms based on that average.
    axis_rankings = dict()
    for axis, df in axis_grouped_rankings.items():
        axis_rankings[axis] = df.mean(axis=1).rank(method='min', ascending=True).reindex(apps).astype(int)
    axis_rankings_df = pd.DataFrame(axis_rankings)
    return axis_grouped_rankings, axis_rankings_df

def compute_dataset_ranking(per_metric_rankings: pd.DataFrame, apps: List[str]) -> pd.DataFrame:
    return per_metric_rankings.mean(axis=1).rank(method='min', ascending=True).reindex(apps).astype(int)
def compute_overall_ranking(per_dataset_rankings: pd.DataFrame, apps: List[str]) -> pd.DataFrame:
    overall_pd = pd.DataFrame(per_dataset_rankings)
    return overall_pd.mean(axis=1).rank(method='min', ascending=True).reindex(apps).astype(int)

def build_per_metric_dfs(
    folder: str,
    metrics_list: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Create a DataFrame where rows are algorithms (by folder name) and columns are
    metric_type:metric_name (or just metric_name), in config order.
    
    inputs: 
    folders: dict mapping algorithm name to folder path
    metrics_list: dict mapping metric file names to configuration information for extraction.
    """
 
    stat_sig_table = gather_stat_sig_bools(folder, metrics_list)
    better_name_table = gather_who_is_better(folder, metrics_list)

    per_metric_dfs = {
        metric: {'stat_sig': stat_sig_table[metric], 'who_is_better': better_name_table[metric]} for metric in metrics_list
    }
    return per_metric_dfs

def gather_stat_sig_bools(folder_path: str, metrics: List[str]) -> Dict[str, Any]:
    """
    Gather all the boolean confusion matrices denoting statistical significance, for all of the metrics requested.
    """
    stat_sig_dict: Dict[str, Any] = {}

    for metric in metrics:
        metric_file = os.path.join(folder_path, f'{metric}_significance.csv') 
        assert os.path.isfile(metric_file), f"Metric file {metric_file} not found in folder {folder_path}."

        #First read the file.
        table = pd.read_csv(metric_file)
        stat_sig_dict[metric] = table
        
    return stat_sig_dict

def gather_who_is_better(folder_path: str, metrics: List[str]) -> Dict[str, Any]:
    """
    Gather all the confusion matrices denoting which algo is better, for all of the metrics requested.
    """
    who_is_better_dict: Dict[str, Any] = {}

    for metric in metrics:
        metric_file = os.path.join(folder_path, f'{metric}_bettername.csv') 
        assert os.path.isfile(metric_file), f"Metric file {metric_file} not found in folder {folder_path}."

        #First read the file.
        table = pd.read_csv(metric_file)
        who_is_better_dict[metric] = table
        
    return who_is_better_dict


def parse_args():

    parser = argparse.ArgumentParser(description="Generate summary table (Excel + LaTeX) from algorithm result folders.")
    parser.add_argument("--algorithm_names", nargs="+", required=True, help="Names of algorithms to compare.")
    #We explicitly pass algorithm names to avoid relying on folder names.
    parser.add_argument("--ref_root", required=True, help="Root path where algorithm result folders are stored.")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metrics to include in the ranking.")
    parser.add_argument("--task_names", nargs="+", required=True, help="Subpath under each significance ranking folder where info is stored..")
    #NOTE: The subpath is assumed to be the same for all algorithms! 
    parser.add_argument("--output_root", required=True, help="Output root path where rankings will be saved.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    #Temporarily debugging variables: 
    # dataset_names=[
    #     "Dataset001_BrainTumour", 
    #     "Dataset003_Liver", 
    #     "Dataset004_Hippocampus", 
    #     "Dataset005_Prostate", 
    #     "Dataset006_Lung", 
    #     "Dataset007_Pancreas", 
    #     "Dataset008_HepaticVessel",
    #     "Dataset010_Colon"
    # ]

    # args.algorithm_names = ["nnintv1", "sammed3dv1", "segvolv1", "sammed2dv1", "sam2v1"] #["nnintv1", "adadesign1"]
    # args.algorithm_names = ["nnintv1", "adadesign1"]
    # # args.ref_root="/home/parhomesmaeili/IS-Validation-Framework/Results_StatSig/holdoutset" #designset"
    # args.ref_root="/home/parhomesmaeili/IS-Validation-Framework/Results_StatSig/designset"
    # # args.output_root="/home/parhomesmaeili/IS-Validation-Framework/Results_Ranking/holdoutset" #designset"
    # args.output_root="/home/parhomesmaeili/IS-Validation-Framework/Results_Ranking/designset"
    # args.metrics=[
    #     'NOI',
    #     'NoF',
    #     'Dice_AUC',
    #     'NSD_AUC',
    #     'Dice Interactive Init',
    #     'Dice Interactive Edit Iter 100',
    #     'NSD Interactive Init',
    #     'NSD Interactive Edit Iter 100'
    # ]
    # experiment_subpaths = []
    # for d_name in dataset_names:
    #     experiment_subpaths.append(f"{d_name}")
    # args.experiment_subpath = experiment_subpaths

    os.makedirs(args.output_root, exist_ok=True)
    
    dataset_wise_dfs = dict()
    for subpath in args.task_names:
        folder = os.path.join(args.ref_root, subpath)
        if os.name == 'posix':
            dataset_wise_dfs[subpath.split('/')[0]] = build_per_metric_dfs(folder, metrics_list=args.metrics)
        else:
            raise NotImplementedError("Windows OS is not currently supported for table generation.")
    
    dataset_rankings = dict()
    for dataset, per_metric_dfs in dataset_wise_dfs.items():
        print('==============================')
        print(f"Processing dataset: {dataset}")
        print('==============================')
        per_metric_rankings = compute_per_metric_rankings(per_metric_dfs, args.algorithm_names)
        dataset_ranking = compute_dataset_ranking(per_metric_rankings, args.algorithm_names)
        
        dataset_rankings[dataset] = dataset_ranking

        #save the per metric rankings within the dataset 
        per_metric_path = os.path.join(args.output_root, dataset, "per_metric_algorithm_rankings.csv")
        os.makedirs(os.path.dirname(per_metric_path), exist_ok=True)
        per_metric_rankings.to_csv(per_metric_path)
        #Save the overall dataset ranking to output file.
        per_dataset_path = os.path.join(args.output_root, dataset, "dataset_ranking.csv")
        os.makedirs(os.path.dirname(per_dataset_path), exist_ok=True)
        dataset_ranking.to_csv(per_dataset_path)

    #Now compute axis of complexity rankings across datasets.
    grouped_axis_rankings, axis_ranking = group_and_rank_per_axis_of_complexity_rankings(
        per_dataset_rankings=dataset_rankings,
        apps=args.algorithm_names
    )
    grouped_axis_rankings_paths = {k: os.path.join(args.output_root, 'axis_of_complexity', f"{k}_grouped.csv") for k in grouped_axis_rankings.keys()}
    axis_of_complexity_path = os.path.join(args.output_root, 'axis_of_complexity', "axis_ranking.csv")
    os.makedirs(os.path.dirname(axis_of_complexity_path), exist_ok=True)
    for k, v in grouped_axis_rankings_paths.items():
        grouped_axis_rankings[k].to_csv(v)
    axis_ranking.to_csv(axis_of_complexity_path)

    #Now compute overall rankings across datasets.
    overall_rankings = compute_overall_ranking(dataset_rankings, args.algorithm_names)
    overall_path = os.path.join(args.output_root, 'overall', "overall_algorithm_ranking_across_datasets.csv")
    os.makedirs(os.path.dirname(overall_path), exist_ok=True)
    overall_rankings.to_csv(overall_path)