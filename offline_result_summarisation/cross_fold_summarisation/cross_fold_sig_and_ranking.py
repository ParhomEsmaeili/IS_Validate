import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import argparse


SNAPSHOT_METRIC_TO_TEST = {
    'Dice Init': 'wilcoxon',
    'Dice Interactive Edit': 'wilcoxon',
    'NSD Init': 'wilcoxon',
    'NSD Interactive Edit': 'wilcoxon',
    'Dice_AUC': 'wilcoxon',
    'NSD_AUC': 'wilcoxon',
    'NOI': 'wilcoxon',
    'NoF': 'mcnemar',
}


def require_env(var):
    val = os.environ.get(var)
    if val is None:
        raise RuntimeError(f"Environment variable {var} must be set.")
    return val


SNAPSHOT_MEASURES_OF_BETTERNESS = {
    'Dice Init': require_env('EXP_DICE_ITERATION_STATISTIC') + '_higher',
    'Dice Interactive Edit': require_env('EXP_DICE_ITERATION_STATISTIC') + '_higher',
    'NSD Init': require_env('EXP_NSD_ITERATION_STATISTIC') + '_higher',
    'NSD Interactive Edit': require_env('EXP_NSD_ITERATION_STATISTIC') + '_higher',
    'Dice_AUC': require_env('EXP_DICE_AUC_STATISTIC') + '_higher',
    'NSD_AUC': require_env('EXP_NSD_AUC_STATISTIC') + '_higher',
    'NOI': require_env('EXP_NOI_STATISTIC') + '_lower',
    'NoF': 'raw_count_lower',
}

TRAJECTORY_MEASURES_OF_BETTERNESS = {
    'Dice_AUC': require_env('EXP_DICE_AUC_STATISTIC') + '_higher',
    'NSD_AUC': require_env('EXP_NSD_AUC_STATISTIC') + '_higher',
    'NOI': require_env('EXP_NOI_STATISTIC') + '_lower',
    'NoF': 'raw_count_lower',
}


def parse_betterness_measure(measure_str: str):
    parts = measure_str.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse betterness measure: {measure_str}")
    statistic_name, direction = parts
    if direction not in ('higher', 'lower'):
        raise ValueError(f"Unknown direction in betterness measure: {measure_str}")
    return statistic_name, direction


def compute_fold_representative(values: pd.Series, statistic_name: str) -> float:
    if statistic_name == 'mean':
        return values.mean()
    elif statistic_name == 'median':
        return values.median()
    elif statistic_name == 'raw_count':
        return values.sum()
    else:
        raise NotImplementedError(f"Unknown fold representative statistic: {statistic_name}")


def is_better(mean1: float, mean2: float, direction: str) -> bool:
    if mean1 == mean2:
        return None
    if direction == 'higher':
        return mean1 > mean2
    elif direction == 'lower':
        return mean1 < mean2
    else:
        raise ValueError(f"Unknown direction: {direction}")


def load_metrics_config(json_dict: str) -> Dict[str, Any]:
    cfg = json.loads(json_dict)
    if not isinstance(cfg, dict):
        raise ValueError("metrics config must be a JSON object")
    return cfg


def gather_snapshot_per_case_from_folder(folder_path: str, metrics_config: Dict[str, Any]) -> Dict[str, Any]:
    metric_dict = {}
    for metric, metric_info in metrics_config.items():
        metric_file = os.path.join(folder_path, metric_info['subpath'])
        if not os.path.isfile(metric_file):
            raise FileNotFoundError(f"Metric file {metric_file} not found.")
        table = pd.read_csv(metric_file)
        extraction_info = metric_info['confs']
        for metric_name, info in extraction_info.items():
            if 'cols' in info:
                indices = []
                for search_str in info['cols']:
                    mask = table.columns == search_str
                    tmp_idx = table.columns[mask].tolist()
                    if len(tmp_idx) != 1:
                        raise ValueError(
                            f"{len(tmp_idx)} cols found matching '{search_str}' in "
                            f"{metric_file} for metric {metric_name}."
                        )
                    indices.extend(tmp_idx)
                if not indices:
                    raise ValueError(f"No columns found for metric {metric_name} in {metric_file}.")
                metric_dict[metric_name] = table.loc[:, ['Case_Name'] + indices]
            else:
                raise NotImplementedError("Only 'cols' extraction is supported.")
    return metric_dict


def gather_snapshot_metrics_across_folds(
    folds: List[str],
    metrics_root: str,
    algorithm_name: str,
    dataset_name: str,
    prompter: str,
    metrics_config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    all_per_case = {}
    for metric_name in [x for xs in [list(i['confs'].keys()) for i in metrics_config.values()] for x in xs]:
        all_per_case[metric_name] = []
    for fold in folds:
        experiment_subpath = f"{dataset_name}/{prompter}/run-aggregated"
        folder = os.path.join(metrics_root, fold, algorithm_name, experiment_subpath)
        if not os.path.isdir(folder):
            warnings.warn(f"Folder not found for fold {fold}, algo {algorithm_name}, dataset {dataset_name}: {folder}")
            continue
        fold_metrics = gather_snapshot_per_case_from_folder(folder, metrics_config)
        for metric_name, df in fold_metrics.items():
            all_per_case[metric_name].append(df)
    result = {}
    for metric_name, dfs in all_per_case.items():
        if not dfs:
            raise FileNotFoundError(f"No data found for metric {metric_name} across any fold.")
        result[metric_name] = pd.concat(dfs, ignore_index=True)
    return result


def compute_snapshot_algo_wise_significance(
    output_path: str, df: pd.DataFrame, apps: List[str], metric: str
):
    test_type_keys = [k for k in SNAPSHOT_METRIC_TO_TEST.keys() if re.search(k, metric)]
    if len(test_type_keys) != 1:
        raise ValueError(f"Could not uniquely identify test type for metric {metric}.")
    test_type = SNAPSHOT_METRIC_TO_TEST[test_type_keys[0]]
    results = pd.DataFrame(index=apps, columns=apps)
    results.index.name = 'app_name'
    results.columns.name = 'app_name'
    for i, app1 in enumerate(apps):
        for j, app2 in enumerate(apps):
            if i >= j:
                continue
            values1 = df[app1]
            values2 = df[app2]
            common_indices = values1.index.intersection(values2.index)
            values1 = values1.loc[common_indices]
            values2 = values2.loc[common_indices]
            if len(values1) == 0 or len(values2) == 0:
                raise ValueError(f"No common data between {app1} and {app2} for metric {metric}.")
            if test_type == 'wilcoxon':
                stat, p_value = wilcoxon(values1, values2)
            elif test_type == 'mcnemar':
                both_fail = ((values1 == True) & (values2 == True)).sum()
                both_pass = ((values1 == False) & (values2 == False)).sum()
                only_v1_fails = ((values1 == True) & (values2 == False)).sum()
                only_v2_fails = ((values1 == False) & (values2 == True)).sum()
                contingency_table = np.array([
                    [both_pass, only_v2_fails],
                    [only_v1_fails, both_fail],
                ])
                result = mcnemar(contingency_table, exact=True)
                p_value = result.pvalue
            else:
                raise NotImplementedError(f"Test type {test_type} not implemented.")
            if np.isnan(p_value):
                print(f"Warning: NaN p-value for {app1} vs {app2} on {metric}")
            significant = p_value < 0.05
            results.at[app1, app2] = significant
            results.at[app2, app1] = significant
    np.fill_diagonal(results.values, False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)


def compute_snapshot_algo_wise_betterness(
    fold_casewise_data: Dict[str, Dict[str, pd.DataFrame]],
    apps: List[str],
    metric: str,
    output_path: str,
):
    measure_keys = [k for k in SNAPSHOT_MEASURES_OF_BETTERNESS.keys() if re.search(k, metric)]
    if len(measure_keys) != 1:
        raise ValueError(f"Could not identify betterness measure for metric {metric}.")
    measure = SNAPSHOT_MEASURES_OF_BETTERNESS[measure_keys[0]]
    statistic_name, direction = parse_betterness_measure(measure)

    results = pd.DataFrame(index=apps, columns=apps, dtype=str)
    results.index.name = 'app_name'
    results.columns.name = 'app_name'

    for i, app1 in enumerate(apps):
        for j, app2 in enumerate(apps):
            if i >= j:
                continue
            fold_reps_1 = []
            fold_reps_2 = []
            for fold, fold_data in fold_casewise_data.items():
                if metric not in fold_data or app1 not in fold_data[metric].columns or app2 not in fold_data[metric].columns:
                    continue
                vals1 = fold_data[metric][app1].dropna()
                vals2 = fold_data[metric][app2].dropna()
                if len(vals1) == 0 or len(vals2) == 0:
                    continue
                rep1 = compute_fold_representative(vals1, statistic_name)
                rep2 = compute_fold_representative(vals2, statistic_name)
                fold_reps_1.append(rep1)
                fold_reps_2.append(rep2)

            if len(fold_reps_1) == 0:
                raise ValueError(f"No fold data for {app1} vs {app2} on metric {metric}.")

            cross_fold_mean_1 = np.mean(fold_reps_1)
            cross_fold_mean_2 = np.mean(fold_reps_2)

            better = is_better(cross_fold_mean_1, cross_fold_mean_2, direction)
            if better is None:
                better_name = 'tie'
            elif better:
                better_name = app1
            else:
                better_name = app2
            results.at[app1, app2] = better_name
            results.at[app2, app1] = better_name

    np.fill_diagonal(results.values, 'nan')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)


def compute_snapshot_cross_fold_variability(
    fold_casewise_data: Dict[str, Dict[str, pd.DataFrame]],
    apps: List[str],
    output_path: str,
):
    metric_to_statistic = {}
    for metric_key, measure in SNAPSHOT_MEASURES_OF_BETTERNESS.items():
        stat_name, _ = parse_betterness_measure(measure)
        metric_to_statistic[metric_key] = stat_name

    records = []
    first_fold_data = next(iter(fold_casewise_data.values()))
    for metric in first_fold_data.keys():
        matched = [k for k in metric_to_statistic.keys() if re.search(k, metric)]
        if not matched:
            continue
        stat_name = metric_to_statistic[matched[0]]
        for app in apps:
            fold_reps = []
            for fold, fold_data in fold_casewise_data.items():
                if metric not in fold_data or app not in fold_data[metric].columns:
                    continue
                vals = fold_data[metric][app].dropna()
                if len(vals) == 0:
                    continue
                rep = compute_fold_representative(vals, stat_name)
                fold_reps.append(rep)
            if len(fold_reps) > 0:
                records.append({
                    'metric': metric,
                    'algorithm': app,
                    'cross_fold_mean': np.mean(fold_reps),
                    'cross_fold_std': np.std(fold_reps, ddof=1) if len(fold_reps) > 1 else 0.0,
                    'num_folds': len(fold_reps),
                })
    summary_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def gather_trajectory_aucs_across_folds(
    folds: List[str],
    pseudotime_root: str,
    algorithm_name: str,
    dataset_name: str,
    prompter: str,
    metrics_config: Dict[str, Any],
) -> Dict[str, List[float]]:
    all_metric_names = []
    for metric_info in metrics_config.values():
        for metric_name in metric_info.get('confs', {}).keys():
            all_metric_names.append(metric_name)
    aucs_by_metric = {m: [] for m in all_metric_names}

    for fold in folds:
        auc_path = os.path.join(
            pseudotime_root, fold, algorithm_name, dataset_name, prompter,
            'run-aggregated', 'pseudotime_metrics', 'all_trajectory_aucs.csv'
        )
        if not os.path.isfile(auc_path):
            warnings.warn(f"Trajectory AUC file not found: {auc_path}")
            for m in all_metric_names:
                aucs_by_metric[m].append(np.nan)
            continue
        auc_table = pd.read_csv(auc_path)
        for metric_name in all_metric_names:
            col_pattern = metric_name.replace('/', '_').replace(' ', '_')
            matching_cols = [c for c in auc_table.columns if col_pattern.lower() in c.lower()]
            if len(matching_cols) != 1:
                matching_cols = [c for c in auc_table.columns if metric_name in c]
            if len(matching_cols) != 1:
                warnings.warn(f"Could not find AUC column for metric {metric_name} in {auc_path}. Found: {matching_cols}")
                aucs_by_metric[metric_name].append(np.nan)
            else:
                aucs_by_metric[metric_name].append(auc_table[matching_cols[0]].iloc[0])

    for m in all_metric_names:
        if len(aucs_by_metric[m]) != len(folds):
            raise ValueError(
                f"Expected {len(folds)} AUC values for metric {m}, got {len(aucs_by_metric[m])}."
            )
    return aucs_by_metric


def compute_trajectory_algo_wise_significance(
    output_path: str,
    aucs_by_metric: Dict[str, List[float]],
    apps: List[str],
    metric: str,
):
    results = pd.DataFrame(index=apps, columns=apps)
    results.index.name = 'app_name'
    results.columns.name = 'app_name'
    for i, app1 in enumerate(apps):
        for j, app2 in enumerate(apps):
            if i >= j:
                continue
            v1 = np.array(aucs_by_metric[app1].get(metric, [np.nan] * len(next(iter(aucs_by_metric.values()))[metric])))
            v2 = np.array(aucs_by_metric[app2].get(metric, [np.nan] * len(next(iter(aucs_by_metric.values()))[metric])))
            valid = ~(np.isnan(v1) | np.isnan(v2))
            v1 = v1[valid]
            v2 = v2[valid]
            if len(v1) < 2:
                print(f"Warning: Too few valid fold AUC pairs for {app1} vs {app2} on {metric}. Setting pval=1.0")
                p_value = 1.0
            else:
                stat, p_value = wilcoxon(v1, v2)
            significant = p_value < 0.05
            results.at[app1, app2] = significant
            results.at[app2, app1] = significant
    np.fill_diagonal(results.values, False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)


def compute_trajectory_algo_wise_betterness(
    aucs_by_metric: Dict[str, List[float]],
    apps: List[str],
    metric: str,
    output_path: str,
):
    measure_keys = [k for k in TRAJECTORY_MEASURES_OF_BETTERNESS.keys() if re.search(k, metric)]
    if len(measure_keys) != 1:
        raise ValueError(f"Could not identify betterness measure for metric {metric}.")
    measure = TRAJECTORY_MEASURES_OF_BETTERNESS[measure_keys[0]]
    statistic_name, direction = parse_betterness_measure(measure)

    results = pd.DataFrame(index=apps, columns=apps, dtype=str)
    results.index.name = 'app_name'
    results.columns.name = 'app_name'
    for i, app1 in enumerate(apps):
        for j, app2 in enumerate(apps):
            if i >= j:
                continue
            v1 = aucs_by_metric[app1].get(metric, [])
            v2 = aucs_by_metric[app2].get(metric, [])
            v1 = np.array([x for x in v1 if not np.isnan(x)])
            v2 = np.array([x for x in v2 if not np.isnan(x)])
            if len(v1) == 0 or len(v2) == 0:
                better_name = 'tie'
            else:
                mean1 = np.mean(v1)
                mean2 = np.mean(v2)
                better = is_better(mean1, mean2, direction)
                if better is None:
                    better_name = 'tie'
                elif better:
                    better_name = app1
                else:
                    better_name = app2
            results.at[app1, app2] = better_name
            results.at[app2, app1] = better_name
    np.fill_diagonal(results.values, 'nan')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path)


def compute_trajectory_cross_fold_variability(
    aucs_by_metric: Dict[str, Dict[str, List[float]]],
    apps: List[str],
    output_path: str,
):
    records = []
    sample_app = next(iter(aucs_by_metric.values()))
    for metric in sample_app.keys():
        for app in apps:
            vals = [x for x in aucs_by_metric[app].get(metric, []) if not np.isnan(x)]
            if len(vals) > 0:
                records.append({
                    'metric': metric,
                    'algorithm': app,
                    'cross_fold_mean': np.mean(vals),
                    'cross_fold_std': np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                    'num_folds': len(vals),
                })
    summary_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def build_snapshot_cross_fold_dfs(
    folds: List[str],
    metrics_root: str,
    apps: List[str],
    dataset_name: str,
    prompter: str,
    metrics_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    metric_names = [x for xs in [list(i['confs'].keys()) for i in metrics_config.values()] for x in xs]
    per_algo_by_fold = {app: {} for app in apps}
    for app in apps:
        per_algo_by_fold[app] = gather_snapshot_metrics_across_folds(
            folds, metrics_root, app, dataset_name, prompter, metrics_config
        )
    combined = {m: pd.DataFrame() for m in metric_names}
    for metric in metric_names:
        combined[metric] = pd.DataFrame({app: per_algo_by_fold[app][metric].iloc[:, 1] for app in apps})
    return combined


def build_snapshot_fold_casewise_data(
    folds: List[str],
    metrics_root: str,
    apps: List[str],
    dataset_name: str,
    prompter: str,
    metrics_config: Dict[str, Any],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    fold_data = {}
    for fold in folds:
        fold_metric_dfs = {}
        for app in apps:
            experiment_subpath = f"{dataset_name}/{prompter}/run-aggregated"
            folder = os.path.join(metrics_root, fold, app, experiment_subpath)
            if not os.path.isdir(folder):
                continue
            app_metrics = gather_snapshot_per_case_from_folder(folder, metrics_config)
            for metric_name, df in app_metrics.items():
                if metric_name not in fold_metric_dfs:
                    fold_metric_dfs[metric_name] = pd.DataFrame()
                fold_metric_dfs[metric_name][app] = df.iloc[:, 1]
        fold_data[fold] = fold_metric_dfs
    return fold_data


def build_trajectory_aucs_per_algo(
    folds: List[str],
    pseudotime_root: str,
    apps: List[str],
    dataset_name: str,
    prompter: str,
    metrics_config: Dict[str, Any],
) -> Dict[str, Dict[str, List[float]]]:
    aucs_by_app = {}
    for app in apps:
        aucs_by_app[app] = gather_trajectory_aucs_across_folds(
            folds, pseudotime_root, app, dataset_name, prompter, metrics_config
        )
    return aucs_by_app


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-fold significance scoring and ranking.")
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--algorithm_names", nargs="+", required=True)
    parser.add_argument("--metrics_root", type=str, required=True)
    parser.add_argument("--pseudotime_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prompter", type=str, default="pointsonly")
    parser.add_argument("--snapshot_metrics_config", type=str, default=None,
                        help="JSON config for snapshot metrics (same format as BASIC_SIG_METRICS_CONFIG)")
    parser.add_argument("--trajectory_metrics_config", type=str, default=None,
                        help="JSON config for trajectory metrics (same format as TRAJ_SIG_METRICS_CONFIG)")
    parser.add_argument("--ranking_metrics", nargs="+", default=None,
                        help="List of metrics to include in rankings (subset of all metrics)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    if not os.path.exists(os.path.join(args.output_root, 'apps.json')):
        with open(os.path.join(args.output_root, 'apps.json'), 'w') as f:
            json.dump(args.algorithm_names, f, indent=2)

    snapshot_config = load_metrics_config(args.snapshot_metrics_config) if args.snapshot_metrics_config else None
    trajectory_config = load_metrics_config(args.trajectory_metrics_config) if args.trajectory_metrics_config else None
    all_metric_names = set()
    if snapshot_config:
        snapshot_metric_names = [x for xs in [list(i['confs'].keys()) for i in snapshot_config.values()] for x in xs]
        all_metric_names.update(snapshot_metric_names)
    if trajectory_config:
        trajectory_metric_names = [x for xs in [list(i['confs'].keys()) for i in trajectory_config.values()] for x in xs]
        all_metric_names.update(trajectory_metric_names)

    print(f"Folds: {args.folds}")
    print(f"Datasets: {args.dataset_names}")
    print(f"Algorithms: {args.algorithm_names}")
    print(f"Snapshot metrics: {list(snapshot_metric_names) if snapshot_config else 'N/A'}")
    print(f"Trajectory metrics: {list(trajectory_metric_names) if trajectory_config else 'N/A'}")

    for dataset_name in args.dataset_names:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset_output_root = os.path.join(args.output_root, dataset_name)
        os.makedirs(dataset_output_root, exist_ok=True)

        if snapshot_config:
            print(f"  Computing snapshot significance...")
            combined_dfs = build_snapshot_cross_fold_dfs(
                args.folds, args.metrics_root, args.algorithm_names,
                dataset_name, args.prompter, snapshot_config
            )
            for metric, df in combined_dfs.items():
                sig_path = os.path.join(dataset_output_root, f"{metric}_significance.csv")
                compute_snapshot_algo_wise_significance(sig_path, df, args.algorithm_names, metric)

            print(f"  Computing snapshot betterness...")
            fold_casewise = build_snapshot_fold_casewise_data(
                args.folds, args.metrics_root, args.algorithm_names,
                dataset_name, args.prompter, snapshot_config
            )
            for metric in combined_dfs.keys():
                bet_path = os.path.join(dataset_output_root, f"{metric}_bettername.csv")
                compute_snapshot_algo_wise_betterness(fold_casewise, args.algorithm_names, metric, bet_path)

            print(f"  Computing snapshot cross-fold variability...")
            var_path = os.path.join(dataset_output_root, 'cross_fold_summary_snapshot.csv')
            compute_snapshot_cross_fold_variability(fold_casewise, args.algorithm_names, var_path)

        if trajectory_config:
            print(f"  Computing trajectory significance...")
            aucs_by_algo = build_trajectory_aucs_per_algo(
                args.folds, args.pseudotime_root, args.algorithm_names,
                dataset_name, args.prompter, trajectory_config
            )
            for metric in trajectory_metric_names:
                sig_path = os.path.join(dataset_output_root, f"{metric}_significance.csv")
                compute_trajectory_algo_wise_significance(sig_path, aucs_by_algo, args.algorithm_names, metric)

            print(f"  Computing trajectory betterness...")
            for metric in trajectory_metric_names:
                bet_path = os.path.join(dataset_output_root, f"{metric}_bettername.csv")
                compute_trajectory_algo_wise_betterness(aucs_by_algo, args.algorithm_names, metric, bet_path)

            print(f"  Computing trajectory cross-fold variability...")
            var_path = os.path.join(dataset_output_root, 'cross_fold_summary_trajectory.csv')
            compute_trajectory_cross_fold_variability(aucs_by_algo, args.algorithm_names, var_path)

    print("\nCross-fold significance and betterness computation complete.")
