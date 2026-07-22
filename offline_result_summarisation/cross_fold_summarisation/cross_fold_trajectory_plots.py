import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.stats import norm

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

TRANSLATE_NNUNET_STATISTIC_TO_NAME = {
    'quantile_0.25': 'LQ',
    'quantile_0.5': 'Median',
    'quantile_0.75': 'UQ',
    'gaussian_0': 'Mean',
}

TRANSLATE_NNUNET_STATISTIC_TO_PSEUDOTIME_NAME = {
    'quantile': {"0.25": "lq", "0.5": "median", "0.75": "uq"},
    'gaussian': {"0": "mean"},
}

ALGORITHM_MAPPING = {
    'sam2v1': 'SAM2',
    'sammed2dv1': 'SAM-Med2D',
    'sammed3dv1': 'SAM-Med3D',
    'segvolv1': 'SegVol',
    'nnintv1': 'nnInteractive',
    'adaptiveISv1': 'AdaptiveIS',
}

translate_metric_name_to_plot_label = {
    'Dice_auc': 'Dice AUC',
    'NSD_auc': 'NSD AUC',
    'Dice_mean': 'Dice Score',
    'Dice_mean Init.': 'Dice Score Init.',
    'Dice_mean Interactive Edit': 'Dice Score at Edit Termination',
    'NSD_mean': 'NSD Score',
    'NSD_mean Init.': 'NSD Score Init.',
    'NSD_mean Interactive Edit': 'NSD Score at Edit Termination',
    'Normalised_Mean_NOI': 'Norm. NoI',
    'Failure_Cases_Fraction': 'Failure Fraction',
}

std_bounded_statistics = ('gaussian', 'student', 'laplace', 'beta')
quantile_statistics = ('quantile',)


def extract_nnunet_threshold(nnunet_statistic, nnunet_bound, nnunet_df, metric):
    if nnunet_statistic == 'gaussian':
        mean = nnunet_df['Automatic Init'].mean()
        std = nnunet_df['Automatic Init'].std()
        threshold = mean - float(nnunet_bound) * std
    elif nnunet_statistic == 'quantile':
        threshold = nnunet_df['Automatic Init'].quantile(float(nnunet_bound))
    else:
        raise NotImplementedError(f"Unknown nnUNet statistic: {nnunet_statistic}")
    if threshold < 0:
        threshold = 0
        warnings.warn(f"Threshold for {metric} is negative. Setting to 0.")
    return threshold


def read_fold_trajectories(
    folds: List[str],
    pseudotime_root: str,
    app: str,
    dataset_name: str,
    prompter: str,
    metric_col: str,
) -> Tuple[List[np.ndarray], List[int]]:
    trajectories = []
    lengths = []
    for fold in folds:
        path = os.path.join(
            pseudotime_root, fold, app, dataset_name, prompter,
            'run-aggregated', 'pseudotime_metrics', 'all_pseudotime_metrics.csv'
        )
        if not os.path.isfile(path):
            warnings.warn(f"Trajectory file not found: {path}")
            continue
        df = pd.read_csv(path)
        if metric_col not in df.columns:
            warnings.warn(f"Column {metric_col} not found in {path}. Available: {list(df.columns)}")
            continue
        trajectories.append(df[metric_col].values)
        lengths.append(len(df[metric_col]))
    return trajectories, lengths


def normalise_and_interpolate(
    trajectories: List[np.ndarray],
    n_grid: int = 1000,
) -> np.ndarray:
    if not trajectories:
        raise ValueError("No trajectories to normalise and interpolate.")
    grid = np.linspace(0, 1, n_grid)
    interp_trajs = []
    for traj in trajectories:
        n = len(traj)
        old_grid = np.linspace(0, 1, n)
        f = interp1d(old_grid, traj, kind='linear', bounds_error=False, fill_value=(traj[0], traj[-1]))
        interp_trajs.append(f(grid))
    return np.array(interp_trajs), grid


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-fold trajectory plots and NoS.")
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--algorithm_names", nargs="+", required=True)
    parser.add_argument("--pseudotime_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prompter", type=str, default="pointsonly")
    parser.add_argument("--nnunet_root", type=str, required=True)
    parser.add_argument("--nnunet_reference_filename", type=str, default="cross_class_scores.csv")
    parser.add_argument("--nnunet_statistic", type=str, nargs='+', required=True)
    parser.add_argument("--nnunet_bound", type=str, nargs='+', required=True)
    parser.add_argument("--metrics_config", type=str, required=True,
                        help="JSON: metric_name -> list of subcomponent names to plot")
    parser.add_argument("--nos_epoch", type=str, default="Interactive Edit Iter 100",
                        help="Epoch to use for NoS computation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metrics_cfg = json.loads(args.metrics_config)
    nnunet_stat = args.nnunet_statistic[0]
    nnunet_bd = args.nnunet_bound[0]

    os.makedirs(args.output_root, exist_ok=True)

    all_metrics_to_plot = {}
    for metric_name, subcomponents in metrics_cfg.items():
        stats_list = []
        stat_key = f"{nnunet_stat}_{nnunet_bd}"
        stat_name = TRANSLATE_NNUNET_STATISTIC_TO_NAME.get(stat_key, 'Mean')
        if subcomponents is None:
            col_name = f"{metric_name}"
            all_metrics_to_plot[col_name] = (metric_name, None)
        else:
            for sub in subcomponents:
                col_name = f"{metric_name} {sub}"
                all_metrics_to_plot[col_name] = (metric_name, sub)

    nnunet_reference_root = os.path.join(args.nnunet_root)
    nnunet_column_headers = ['Case_Name', 'Automatic Init']

    for dataset_name in args.dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_nnunet_path = os.path.join(nnunet_reference_root, dataset_name, 'nnUNet_metrics')

        for app in args.algorithm_names:
            print(f"  Algorithm: {app}")

            for col_name, (base_metric, subcomp) in all_metrics_to_plot.items():
                trajectories, lengths = read_fold_trajectories(
                    args.folds, args.pseudotime_root, app, dataset_name,
                    args.prompter, col_name
                )
                if len(trajectories) < 2:
                    print(f"    Skipping {col_name}: only {len(trajectories)} valid trajectories (<2)")
                    continue

                # Normalise and interpolate
                interp_array, grid = normalise_and_interpolate(trajectories, n_grid=1000)
                mean_traj = np.mean(interp_array, axis=0)
                std_traj = np.std(interp_array, axis=0, ddof=1)

                # Compute nnUNet threshold for NoS
                nnunet_df = pd.read_csv(
                    os.path.join(dataset_nnunet_path, base_metric, args.nnunet_reference_filename),
                    skiprows=1, names=nnunet_column_headers
                )
                threshold = extract_nnunet_threshold(nnunet_stat, nnunet_bd, nnunet_df, base_metric)

                # NoS on averaged trajectory
                nos_val = np.nan
                crossing_idx = np.argmax(mean_traj >= threshold) if np.any(mean_traj >= threshold) else -1
                if crossing_idx >= 0:
                    nos_val = crossing_idx / (len(mean_traj) - 1)
                failure_count = sum(
                    1 for traj in trajectories
                    if not np.any(traj >= threshold)
                )

                # Save per-fold NoS for failure fraction reporting
                per_fold_nos = []
                for traj in trajectories:
                    idx = np.argmax(traj >= threshold) if np.any(traj >= threshold) else -1
                    per_fold_nos.append(idx / (len(traj) - 1) if idx >= 0 else np.nan)
                failure_fraction = np.mean([1 if np.isnan(x) else 0 for x in per_fold_nos])

                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                display_name = ALGORITHM_MAPPING.get(app, app)
                plot_label = translate_metric_name_to_plot_label.get(col_name, col_name)

                grid_pct = grid * 100
                ax.plot(grid_pct, mean_traj, label=f'{display_name} (mean)', color='blue', linewidth=2)
                ax.fill_between(
                    grid_pct,
                    mean_traj - std_traj,
                    mean_traj + std_traj,
                    alpha=0.2, color='blue', label='±1 std'
                )
                # Thin grey lines for individual folds
                for traj in interp_array:
                    ax.plot(grid_pct, traj, color='grey', alpha=0.15, linewidth=0.5)

                # nnUNet threshold line
                ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1,
                           label=f'nnUNet threshold ({TRANSLATE_NNUNET_STATISTIC_TO_NAME.get(f"{nnunet_stat}_{nnunet_bd}", "Mean")})')

                # NoS marker
                if not np.isnan(nos_val):
                    ax.axvline(x=nos_val * 100, color='green', linestyle=':', linewidth=1.5,
                               label=f'NoS = {nos_val:.2%}')

                ax.set_xlabel('Proportion of Pseudotime (%)')
                ax.set_ylabel(plot_label)
                ax.set_title(f'{display_name} — {dataset_name}\n{failure_fraction:.0%} of folds never crossed')
                ax.legend(loc='best')
                ax.set_xlim(0, 100)
                ax.grid(True, alpha=0.3)

                os.makedirs(os.path.join(args.output_root, 'plots'), exist_ok=True)
                plot_path = os.path.join(
                    args.output_root, 'plots',
                    f'{app}_{dataset_name}_{col_name.replace(" ", "_")}_crossfold.png'
                )
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved plot: {plot_path}")

                # Save numerical summary
                os.makedirs(os.path.join(args.output_root, 'numerical'), exist_ok=True)
                num_path = os.path.join(
                    args.output_root, 'numerical',
                    f'{app}_{dataset_name}_{col_name.replace(" ", "_")}_crossfold_numerical.csv'
                )
                summary_records = {
                    'app': app,
                    'dataset': dataset_name,
                    'metric': col_name,
                    'cross_fold_mean': mean_traj.mean(),
                    'cross_fold_std': std_traj.mean(),
                    'nos': nos_val,
                    'failure_fraction': failure_fraction,
                    'num_valid_folds': len(trajectories),
                    'nnunet_threshold': threshold,
                }
                pd.DataFrame([summary_records]).to_csv(num_path, index=False)

    print("\nCross-fold trajectory plotting complete.")
