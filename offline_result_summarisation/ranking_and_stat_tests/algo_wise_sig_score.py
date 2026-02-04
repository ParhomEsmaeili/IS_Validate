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
RESULT_SEARCHSTRING = {
    'Dice': 'Dice',
    'NSD': 'NSD',
    'NoI': ['NOI', 'Failure_Cases_Fraction']
}
MAPPING_SUBCATEGORIES = {
    'auc': 'nAUC',
    'Init.': 'Init.',
    'Interactive Edit Iter': 'Interactive Edit Iter',
    'NOI': 'nNoI',
    'Failure_Cases_Fraction': 'NoF'
}

ITERABLE_METRICS = {
    'Dice',
    'NSD'
}
# COLUMN_ORDERING = [
#     'Init.',
#     'Interactive Edit Iter',
#     'nAUC',
#     'nNoI',
#     'NoF'
# ]
# TABLE_MAP = {
#     'Dice_median': '',
#     'NSD_median': '',
#     'Dice_auc_median': 'nAUC',
#     'NSD_auc_median': 'nAUC',
#     'Normalised_Median_NOI': 'nNoI',
#     'Interactive Edit Iter': 'Iter.',
#     'Failure_Cases_Fraction': 'NoF'
# }

METRIC_TO_TEST = {
    'Dice Interactive Init': 'wilcoxon',
    'Dice Interactive Edit': 'wilcoxon', #Doesn't need to be the full string, but a substring match.
    'NSD Interactive Init': 'wilcoxon',
    'NSD Interactive Edit': 'wilcoxon',
    'Dice_AUC': 'wilcoxon',
    'NSD_AUC': 'wilcoxon',
    'NOI': 'wilcoxon',
    'NoF': 'fisher_exact', #Not a continuous metric, so we use a different test.
}

MEASURE_OF_BETTERNESS = {
    'Dice Interactive Init': 'mean_higher',
    'Dice Interactive Edit': 'mean_higher',
    'NSD Interactive Init': 'mean_higher',
    'NSD Interactive Edit': 'mean_higher',
    'Dice_AUC': 'mean_higher',
    'NSD_AUC': 'mean_higher',
    'NOI': 'mean_lower',
    'NoF': 'raw_count_lower'  #Lower is better
}
def sort_by_substring_order(input_list: List[str], priority_list: List[str]) -> List[str]:
    def sort_key(input_list):
        for i, substr in enumerate(priority_list):
            if substr.lower() in input_list.lower():
                return i
        return len(input_list)  # put unmatched at the end
    return sorted(input_list, key=sort_key)

def load_metrics_config(json_dict:str) -> Dict[str, List[str]]:
    cfg = json.loads(json_dict)
    if not isinstance(cfg, dict):
        raise ValueError("metrics config must be a JSON object mapping metric -> [metric information]")
    return cfg


def compute_algo_wise_significance_score(output_path, df, apps, metric):
    #Pull the appropriate test for the given metric.
    test_type = [re.search(metric_key, metric).group() for metric_key in METRIC_TO_TEST.keys() if re.search(metric_key, metric)]
    assert len(test_type) == 1, f"Could not uniquely identify test type for metric {metric}."
    test_type = METRIC_TO_TEST[test_type[0]]

    results = pd.DataFrame(index=apps, columns=apps)
    results = pd.DataFrame(index=apps, columns=apps)
    results.index.name = 'app_name'
    results.columns.name = 'app_name'
    for i, app1 in enumerate(apps):
        for j, app2 in enumerate(apps):
            if i >= j:
                continue  # Avoid redundant calculations

            # Extract metric values for both algorithms
            values1 = df[app1]
            values2 = df[app2]
            #FOR NOW, lets not drop any nans. we will need a separate script for handling early-termination

            # Ensure both have the same length by aligning on indices
            common_indices = values1.index.intersection(values2.index)
            values1 = values1.loc[common_indices]
            values2 = values2.loc[common_indices]
            assert common_indices.size == df.shape[0], f"Algorithms {app1} and {app2} do not have metrics for all cases."
            if len(values1) == 0 or len(values2) == 0:
                raise ValueError(f"No common data points between {app1} and {app2} for metric {metric}. What we are we doing here?!")

            if test_type == 'wilcoxon':
                # Perform Wilcoxon signed-rank test
                stat, p_value = wilcoxon(values1, values2)
            elif test_type == 'fisher_exact':
                #We have a contingency table, or at least columns of bools already.
                contingency_table = pd.crosstab([values1], [values2], rownames=[f'{app1}'], colnames=[f'{app2}'], dropna=False)
                _, p_value = fisher_exact(contingency_table)
            else:
                raise NotImplementedError(f"Test type {test_type} not implemented.")

            if np.isnan(p_value):
                print('absurdly high p-value encountered! Must be a warning, for sanity check \n' \
                'our apps are {} and {}, and the metric is {} for task {} \n'.format(app1, app2, metric, os.path.dirname(output_path)))
            # Determine significance (alpha = 0.05)
            significant = p_value < 0.05

            results.at[app1, app2] = significant
            results.at[app2, app1] = significant

    # Fill diagonal with False (an algorithm is not significantly different from itself)
    np.fill_diagonal(results.values, False)

    # Save results to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)


def compute_algo_wise_better_name(output_path, df, apps, metric):
    measure = [re.search(metric_key, metric).group() for metric_key in MEASURE_OF_BETTERNESS.keys() if re.search(metric_key, metric)]
    assert len(measure) == 1, f"Could not uniquely identify test type for metric {metric}."
    measure = MEASURE_OF_BETTERNESS[measure[0]]

    #This is a func which builds a confusion matrix indicating which algorithm is better than which other algorithm.
    results = pd.DataFrame(index=apps, columns=apps, dtype=str)
    results.index.name = 'app_name'
    results.columns.name = 'app_name'
    for i, app1 in enumerate(apps):
        for j, app2 in enumerate(apps):
            if i >= j:
                continue  # Avoid redundant calculations

            # Extract metric values for both algorithms
            values1 = df[app1]
            values2 = df[app2]

            # Determine which algorithm is better
            if measure == 'mean_higher':
                if values1.mean() == values2.mean():
                    better = None
                else:
                    better = values1.mean() >= values2.mean()
            elif measure == 'mean_lower':
                if values1.mean() == values2.mean():
                    better = None
                else:
                    better = values1.mean() <= values2.mean()
            elif measure == 'raw_count_lower': #We count it as better if the number of bools that are True is lower.
                if values1.sum() == values2.sum():
                    better = None
                else:
                    if metric == 'NoF':
                        better = (values1).sum() <= (values2).sum() 
                    else:
                        raise NotImplementedError(f"Betterment measure {measure} not implemented for metric {metric}.")
            else:
                raise NotImplementedError(f"Betterment measure {measure} not implemented.")
            if better == None:
                better_name = 'tie'
            else:
                if better:
                    better_name = app1
                else:
                    better_name = app2
            results.at[app1, app2] = better_name
            results.at[app2, app1] = better_name
    
    # Fill diagonal with nan (an algorithm is not better than itself)
    np.fill_diagonal(results.values, 'nan')
    # Save results to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)

def build_per_metric_dfs(
    folders: dict[str],
    task_name: str,
    metrics_config: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Create a DataFrame where rows are algorithms (by folder name) and columns are
    metric_type:metric_name (or just metric_name), in config order.
    
    inputs: 
    folders: dict mapping algorithm name to folder path
    metrics_config: dict mapping metric file names to configuration information for extraction.
    """

    metrics_storage = dict() 
    for folder, folder_path in folders.items():
        metrics_storage[folder] = gather_metrics_from_folder(folder_path, metrics_config)
        
    #Now using the extracted metrics to build a table with all of the relevant information placed together.

    #Lets piece together on a metric by metric basis?

    #Lets extract the full list of actual metric names now.
    full_list = [x for xs in [list(i['confs'].keys()) for i in metrics_config.values()] for x in xs]
    #Within each broad category we might have a
    per_metric_dfs = dict()
    for metric in full_list:
        # Get case names from first algorithm
        first_algo = next(iter(folders.keys()))
        assert metric in metrics_storage[first_algo], f"Metric {metric} not found in extracted metrics for algorithm {first_algo}."
        
        # Initialize DataFrame with case index
        case_names = metrics_storage[first_algo][metric].index
        per_metric_dfs[metric] = pd.DataFrame(index=case_names)
        
        for algo in folders.keys():
            assert metric in metrics_storage[algo], f"Metric {metric} not found in extracted metrics for algorithm {algo}."
            per_metric_dfs[metric][algo] = metrics_storage[algo][metric].iloc[:,1]

    return per_metric_dfs


def gather_metrics_from_folder(folder_path: str, metrics_config: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Searches a given folder for the metrics files specificied in metrics_config, then extracts the relevant metrics.
    Returns a dictionary of extracted metrics flattened into dictionaries with single layer depth.

    """
    metric_dict: Dict[str, Any] = {}

    for metric, metric_info in metrics_config.items():
        metric_file = os.path.join(folder_path, metric_info['subpath']) 
        assert os.path.isfile(metric_file), f"Metric file {metric_file} not found in folder {folder_path}."

        #First read the file.
        table = pd.read_csv(metric_file)
        extraction_info = metric_info['confs']

        for metric_name, info in extraction_info.items():
            if info == None:
                raise NotImplementedError("Extraction information must be provided for each metric in the config.")
            else:
                #In this case, we have some extraction information to use, e.g. cols we need to extract.
                if 'cols' in info:
                    #We need to extract specific rows based on the iteration information provided. We will assume that 
                    #the first column contains the iteration information, but won't actually use this. Instead we will use
                    #string matching. 
                    indices = []
                    #could vectorise but cba.
                    for search_str in info['cols']:
                        mask = table.columns == search_str
                        tmp_idx = table.columns[mask].tolist()
                        if len(tmp_idx) != 1:
                            raise ValueError(f"{len({tmp_idx})} cols found matching '{search_str}' in {metric_file} for metric {metric_name}. \n")
                        indices.extend(table.columns[mask].tolist())
                        #Sanity check!
                    assert indices, f"No columns found for metric {metric_name} in file {metric_file}."
                    metric_dict[metric_name] = table.loc[:, ['Case_Name'] + indices]
                else:
                    raise NotImplementedError("Unsupported extraction information provided in metrics config. Only iters is a supported \n" \
                    "configuration for multi-row extraction at the moment.")
        
    return metric_dict


def parse_args():

    parser = argparse.ArgumentParser(description="Generate summary table (Excel + LaTeX) from algorithm result folders.")
    parser.add_argument("--algorithm_names", nargs="+", required=True, help="Names of algorithms to compare.")
    parser.add_argument("--metrics_root", type=str, required=True, help="Root path to the folder which contains the metrics across all the algorithms.")
    
    #We explicitly pass algorithm names to avoid relying on folder names.
    parser.add_argument("--experiment_subpath", nargs="+", required=True, help="Subpath under each algorithm folder where metrics are stored.")
    #NOTE: The subpath is assumed to be the same for all algorithms! 
    parser.add_argument("--metrics_config", required=True, help="JSON file mapping metrics to information required for pulling.")
    parser.add_argument("--output_root", required=True, help="Output root path for Excel files.")
    
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
    # args.metrics_root="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/holdoutset" #designset"
    # args.metrics_root="/home/parhomesmaeili/IS-Validation-Framework/Results_Summary/designset"
    # args.output_root="/home/parhomesmaeili/IS-Validation-Framework/Results_StatSig/holdoutset" #designset"
    # args.output_root="/home/parhomesmaeili/IS-Validation-Framework/Results_StatSig/designset"
    # args.metrics_config={
    #     'NOI': {
    #         'subpath': 'metrics/NOI/casewise_noi.csv',
    #         'confs': {
    #             "NOI": {"cols": ["NOI_Dice_thr_quantile_Q = 0.5"]},
    #             "NoF": {"cols": ["Fail_Dice_thr_quantile_Q = 0.5"]}
    #             }
    #     },
    #     "AUC": {
    #         'subpath': 'metrics/AUC/casewise_aucs.csv',
    #         'confs': {
    #             "Dice_AUC": {"cols": ["Dice_auc_scores"]},
    #             "NSD_AUC": {"cols": ["NSD_auc_scores"]}
    #         }
    #     },
    #     "Dice": {
    #         'subpath': 'metrics/Dice/cross_class_scores.csv',
    #         'confs': {
    #             'Dice Interactive Init': {"cols": ["Interactive Init"]},
    #             "Dice Interactive Edit Iter 100": {"cols": ["Interactive Edit Iter 100"]}
    #             },
    #     }, 
    #     "NSD": {
    #         'subpath': 'metrics/NSD/cross_class_scores.csv',
    #         'confs': {
    #             'NSD Interactive Init': {"cols":["Interactive Init"]},
    #             'NSD Interactive Edit Iter 100': {"cols": ["Interactive Edit Iter 100"]}
    #         }
    #     }
    # }
    # experiment_subpaths = []
    # for d_name in dataset_names:
    #     experiment_subpaths.append(f"{d_name}/pointsonly/run-aggregated")
    # args.experiment_subpath = experiment_subpaths

    cfg = load_metrics_config(args.metrics_config)
    # cfg = args.metrics_config
    dataset_wise_dfs = dict()
    for experiment_subpath in args.experiment_subpath:
        folders = {alg_name: os.path.join(args.metrics_root, alg_name, experiment_subpath) for alg_name in args.algorithm_names}
        if os.name == 'posix':
            dataset_wise_dfs[experiment_subpath.split('/')[0]] = build_per_metric_dfs(folders, experiment_subpath.split('/')[0], cfg)
        else:
            raise NotImplementedError("Windows OS is not currently supported for table generation.")
    for dataset, per_metric_dfs in dataset_wise_dfs.items():
        for metric, df in per_metric_dfs.items():
            output_relpath_sig = os.path.join(dataset, f"{metric}_significance.csv")
            os.makedirs(os.path.dirname(os.path.join(args.output_root, output_relpath_sig)), exist_ok=True)
            compute_algo_wise_significance_score(output_path=os.path.join(args.output_root, output_relpath_sig), df=df, apps=args.algorithm_names, metric=metric)
            output_relpath_better = os.path.join(dataset, f"{metric}_bettername.csv")
            os.makedirs(os.path.dirname(os.path.join(args.output_root, output_relpath_better)), exist_ok=True)
            compute_algo_wise_better_name(output_path=os.path.join(args.output_root, output_relpath_better), df=df, apps=args.algorithm_names, metric=metric)