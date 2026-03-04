import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os
import sys
import json
import argparse
from typing import Dict, List
import numpy as np
import copy 
import re
import warnings
from scipy.stats import norm, t, laplace, beta

DATASET_MAPPING = {
    'Dataset001_BrainTumour': 'Brain Tumour Core',
    'Dataset002_Heart': 'Heart',
    'Dataset003_Liver': 'Whole Liver',
    'Dataset004_Hippocampus': 'Whole Hippocampus',
    'Dataset005_Prostate': 'Whole Prostate',
    'Dataset006_Lung': 'Lung Lesion',
    'Dataset007_Pancreas': 'Whole Pancreas',
    'Dataset008_HepaticVessel': 'Hepatic Vessels',
    'Dataset009_Spleen': 'Spleen',
    'Dataset010_Colon': 'Colon'
}

def map_algorithm_name(short_name):
    """Map shortened algorithm names to full display names"""
    # Numeric adatest mapping
    NUMERIC_MAP = {
        "2": "CLoPA-Inst",
        "5": "CLoPA-ConvNorm",
        "6": "CLoPA-ConvNorm",
        # Add more mappings as needed
    }

    patterns = [
        (r'adatest(\d+)-episode(\w+)', None),
        (r'adatest(\d+)', None),
        (r'adadesign(\d+)', None),
    ]

    for pattern, replacement in patterns:
        m = re.match(pattern, short_name)
        if m:
            if pattern.startswith('adatest'):
                num = m.group(1)
                mapped = NUMERIC_MAP.get(num, f'adatest{num}')
                # Handle episodic variant
                if '-episode' in short_name:
                    episode = m.group(2)
                    return f'{mapped}-Episode{episode}' if episode != 'final' else f'{mapped}'
                else:
                    return mapped
            elif pattern.startswith('adadesign'):
                num = m.group(1)
                mapped = NUMERIC_MAP.get(num, f'adadesign{num}')
                return mapped
            else:
                return re.sub(pattern, replacement, short_name)

    return short_name  # Return original if no match
ALGORITHM_MAPPING = {
    'sam2v1': 'SAM2',
    'sammed2dv1': 'SAM-Med2D',
    'sammed3dv1': 'SAM-Med3D',
    'segvolv1': 'SegVol',
    'nnintv1': 'nnInteractive',
    'adaptiveISv1': 'AdaptiveIS'
}



w = 5


smoothing_params = {
    'Dataset001_BrainTumour': 2/(w+1),
    'Dataset003_Liver': 2/(w+1),
    'Dataset004_Hippocampus': 2/(w+1),
    'Dataset005_Prostate': 2/(w+1),
    'Dataset006_Lung': 2/(w+1),
    'Dataset007_Pancreas': 2/(w+1),
    'Dataset008_HepaticVessel': 2/(w+1),
    'Dataset010_Colon': 2/(w+1),
}

translate_metric_name_to_plot_label = {
    '^Dice_auc_(mean|median)': lambda m: f'{m.group(1).capitalize()} Dice AUC',
    '^Dice_(mean|median) (Interactive.*)': lambda m: f'{m.group(1).capitalize()} Dice Score at Editing Termination',
    '^Dice_(mean|median) (Init.)': lambda m: f'{m.group(1).capitalize()} Dice Score Initialisation',
    '^NSD_auc_(mean|median)': lambda m: f'{m.group(1).capitalize()} NSD AUC',
    '^NSD_(mean|median) (Interactive.*)': lambda m: f'{m.group(1).capitalize()} NSD at Editing Termination',
    '^NSD_(mean|median) (Init.)': lambda m: f'{m.group(1).capitalize()} NSD Initialisation',
    '^Normalised_(Mean|Median)_NOI': lambda m: f'{m.group(1).capitalize()} Normalised NoI',
    'Failure_Cases_Fraction': 'Number of Failures Percentage'
}
#we are just going to extract the name of the statistic.
cross_check_with_nnunet_metric_regex = {
    '^Dice_(mean|median) (Interactive.*)': lambda m: f"{m.group(1).capitalize()}",
    '^Dice_(mean|median) (Init.)': lambda m: f"{m.group(1).capitalize()}",
    '^NSD_(mean|median) (Init.)': lambda m: f"{m.group(1).capitalize()}",
    '^NSD_(mean|median) (Interactive.*)': lambda m: f"{m.group(1).capitalize()}",
    # '^Normalised_(Mean|Median)_NOI': lambda m: f"{m.group(1).capitalize()}",
}    

translate_subcol = {
    'Interactive Init': 'Init.',
}

std_bounded_statistics = ('gaussian', 'student', 'laplace', 'beta')
quantile_statistics = ('quantile',)

TRANSLATE_NNUNET_STATISTIC_TO_PSEUDOTIME_NAME = {
    'quantile': {"0.25":"LQ", "0.5":"Median", "0.75":"UQ"},
    'gaussian': {"0":"Mean"},
}
nnunet_permitted_metrics = ('Dice', 'NSD')

trajectory_to_nnunet_match_naming = {
    '^Dice_(mean|median) (Interactive.*)': lambda m: f"Dice {m.group(1).capitalize()}",
    '^Dice_(mean|median) Init.': lambda m: f"Dice {m.group(1).capitalize()}",
    '^NSD_(mean|median) (Interactive.*)': lambda m: f"NSD {m.group(1).capitalize()}",
    '^NSD_(mean|median) (Init.)': lambda m: f"NSD {m.group(1).capitalize()}"
}

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

                # mean = nnunet_df['Automatic Init'].mean()
                # std = nnunet_df['Automatic Init'].std()
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
            stat = TRANSLATE_NNUNET_STATISTIC_TO_PSEUDOTIME_NAME.get(fit, {}).get(str(nnunet_bounds[statistic_id]), f"{fit}_{nnunet_bounds[statistic_id]}")
            nnunet_thresholds[f"{metric} {stat}"] = threshold
    return nnunet_thresholds

def plot_pseudotime_aucs(input_metrics, processed_pseudotime_aucs, trajectory_pseudotime_auc, output_folder, nnunet_metrics, nos_values, nos_epoch, smoothing_param=None):
    # raw pseudotime aucs are app-separated dataframes, where each dataframe has columns for each metric and rows for each pseudotime point.

    
    #For each base metric and each app we have a nested dict for the raw pseudotime aucs.
    #Then we plot the pseudotime AUC trajectory for each metric, where the x-axis is the pseudotime and the y-axis is the AUC value.
    #with all requested apps on the same plot for comparison.

    # Create a plot for each metric, with subplots for each dataset
    for metric in input_metrics:
        assert metric in processed_pseudotime_aucs, f"Expected metric '{metric}' not found in the processed pseudotime AUCs."
        datasets = list(processed_pseudotime_aucs[metric].keys())
        # Use plt.subplots and manually adjust the 3rd subplot position to minimize gap after 2nd column
        n = len(datasets)
        fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 7), squeeze=False)
        plt.subplots_adjust(wspace=0.05)  # Small gap globally
        axes = axes[0]
        if n > 2:
            # Get the right edge of the 2nd axis
            pos2 = axes[1].get_position()
            for idx in range(2, n):
                pos = axes[idx].get_position()
                # Move left edge to right edge of previous axis minus a tiny epsilon
                new_x0 = pos2.x1 - 0.005
                axes[idx].set_position([new_x0, pos.y0, pos.width, pos.height])
        # Now plot as before, but use axes list
        # Compute the minimum value across all subplots for this metric
        min_val = float('inf')
        min_vals_per_subplot = []
        for dataset in datasets:
            subplot_min = float('inf')
            for app in processed_pseudotime_aucs[metric][dataset].columns:
                v = processed_pseudotime_aucs[metric][dataset][app].min()
                min_val = min(min_val, v)
                subplot_min = min(subplot_min, v)
            min_vals_per_subplot.append(subplot_min)
        # Set y-limits and y-ticks independently for each subplot
        y_max = 1
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            # Make all tick labels bold
            ax.tick_params(axis='both', labelsize='medium')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            subplot_min = min_vals_per_subplot[idx]
            subplot_y_min = max(0, np.floor((subplot_min - 0.05) / 0.05) * 0.05)
            subplot_y_max = y_max
            subplot_y_ticks = np.arange(subplot_y_min, subplot_y_max + 0.001, 0.05)
            ax.set_ylim(subplot_y_min, subplot_y_max)
            ax.set_yticks(subplot_y_ticks)
            ax.set_yticklabels([f"{ytick:.2f}" for ytick in subplot_y_ticks], fontweight='bold')
            trajectory_handles = []
            trajectory_labels = []
            vline_handles = []
            vline_labels = []
            nnunet_thresh_handles = []
            nnunet_thresh_labels = []
            for app in processed_pseudotime_aucs[metric][dataset].columns:
                col_name = f'{metric}_trajectory_auc'
                mapped = ALGORITHM_MAPPING.get(app, app)
                app_print = map_algorithm_name(mapped)
                if col_name in trajectory_pseudotime_auc[dataset][app].columns:
                    traj_auc = trajectory_pseudotime_auc[dataset][app][col_name].iloc[0]
                    label = f'{app_print} (Trajectory-AUC: {traj_auc:.3f})'
                else:
                    raise ValueError(f"Expected column '{col_name}' not found in the pseudotime trajectory CSV file for app '{app}' in dataset '{dataset}'.")
                x_vals = np.arange(1, len(processed_pseudotime_aucs[metric][dataset][app]) + 1)
                app_line, = ax.plot(x_vals, processed_pseudotime_aucs[metric][dataset][app], label=label)
                trajectory_handles.append(app_line)
                trajectory_labels.append(label)
                app_colour = app_line.get_color()

                if nos_epoch in metric:
                    if app not in nos_values[dataset]:
                        raise ValueError(f"Expected app '{app}' not found in the nos_values dictionary for dataset '{dataset}'.")
                    nos_reference = copy.deepcopy(metric).split('_')[0]
                    assert nos_reference in ('Dice', 'NSD'), f"Expected metric reference '{nos_reference}' to be either 'Dice' or 'NSD'. Please check the metric naming convention."
                    temp_df = nos_values[dataset][app]
                    nos_val = temp_df.loc[nos_reference, 'cross_index']
                    if np.isnan(nos_val):
                        print(f"Warning: NaN value encountered for NoS for app '{app}' and metric '{metric}' in dataset '{dataset}'")
                    else:
                        nos_val = round(nos_val * len(processed_pseudotime_aucs[metric][dataset][app])) + 1
                        vline = ax.axvline(x=nos_val, color=app_colour, linestyle=':', linewidth=2)
                        vline_handles.append(mlines.Line2D([], [], color=app_colour, linestyle=':'))
                        vline_labels.append(f'{app_print} NoS: {nos_val}')
            ax.set_xlim(left=1)
            max_x = max([len(processed_pseudotime_aucs[metric][dataset][app]) for app in processed_pseudotime_aucs[metric][dataset].columns])
            n_ticks = 8
            step = max(1, int(np.ceil(max_x / n_ticks)))
            xticks = np.arange(1, max_x + 1, step)
            if xticks[-1] != max_x:
                xticks = np.append(xticks, max_x)
            ax.set_xticks(xticks)

            for pat, nnunet_metric in trajectory_to_nnunet_match_naming.items():
                m = re.match(pat, metric)
                if m:
                    if callable(nnunet_metric):
                        nnunet_metric_name = nnunet_metric(m)
                    else:
                        nnunet_metric_name = nnunet_metric
                    if nnunet_metric_name in nnunet_metrics[dataset]:
                        nnunet_threshold = nnunet_metrics[dataset][nnunet_metric_name]
                        x_vals = processed_pseudotime_aucs[metric][dataset].index + 1
                        nnunet_thresh_plt = ax.hlines(y=nnunet_threshold, xmin=x_vals[0], xmax=x_vals[-1], color='r', linestyle='--')
                        nnunet_thresh_handles.append(mlines.Line2D([], [], color='r', linestyle='--',label=f'nnUNet {nnunet_metric_name} Performance'))
                        nnunet_thresh_labels.append(f'nnUNet Performance: {nnunet_threshold:.3f}')
                    else:
                        raise ValueError(f"Expected nnUNet metric '{nnunet_metric_name}' not found in the provided nnUNet metrics for dataset '{dataset}'. Please check the input files and ensure consistency.")

            m = False
            for pat, plot_label in translate_metric_name_to_plot_label.items():
                m = re.match(pat, metric)
                if m:
                    translated_label = plot_label(m) if callable(plot_label) else plot_label
                    label = translated_label
                    assert isinstance(label, str), f"Label is not a string after translation: {label} (type: {type(label)})"
                    break
            if not m:
                label = copy.deepcopy(metric)
            ax.set_title(DATASET_MAPPING.get(dataset, dataset), fontweight='bold')
            ax.set_xlabel('Data Samples Received', fontweight='bold')
            ticks = ax.get_xticks()
            if 1 not in ticks:
                new_ticks = np.insert(ticks, 0, 1.0)
                ax.set_xticks(new_ticks)
            # Only set y-label and y-ticks for the leftmost subplot
            if idx == 0:
                # Leftmost: nearest 0.1 below minima
                subplot_min = min_vals_per_subplot[idx]
                subplot_y_min = max(0, np.floor((subplot_min - 0.1) / 0.1) * 0.1)
                subplot_y_max = 1
                if subplot_y_min == 0:
                    subplot_y_ticks = np.arange(0, subplot_y_max + 0.001, 0.2)
                else:
                    subplot_y_ticks = np.arange(subplot_y_min, subplot_y_max + 0.001, 0.15)
                ax.set_ylim(subplot_y_min, subplot_y_max)
                ax.set_yticks(subplot_y_ticks)
                ax.set_ylabel(label, fontweight='bold')
                ax.set_yticklabels([f"{ytick:.2f}" for ytick in subplot_y_ticks], fontweight='bold')
            # elif idx == 1:
            #     ax.set_ylabel("")
            #     # Show y-tick labels for the second subplot
            #     ax.set_yticklabels([f"{ytick:.2f}" for ytick in ax.get_yticks()], fontweight='bold')
            # else:
            #     ax.set_ylabel("")
            #     ax.set_yticklabels([])
            #     ax.tick_params(axis='y', length=0)
            else:
                ax.set_ylabel("")
                # Show y-tick labels for the second subplot
                ax.set_yticklabels([f"{ytick:.2f}" for ytick in ax.get_yticks()], fontweight='bold')
            
            ax.grid()
            if idx == 0:
                first_legend = ax.legend(
                    handles=trajectory_handles,
                    labels=trajectory_labels,
                    loc='center right',
                    title='Trajectory Metrics',
                    fontsize=11.5,  # Match axis label size
                    title_fontsize=11.5  # Match axis label size
                )
            elif idx > 0:
                first_legend = ax.legend(
                    handles=trajectory_handles,
                    labels=trajectory_labels,
                    loc='upper right',
                    title='Trajectory Metrics',
                    fontsize=11.5,  # Match axis label size
                    title_fontsize=11.5  # Match axis label size
                )
            ax.add_artist(first_legend)
            combined_handles = nnunet_thresh_handles + vline_handles
            combined_labels = nnunet_thresh_labels + vline_labels
            renderer = fig.canvas.get_renderer()
            bbox = first_legend.get_window_extent(renderer=renderer)
            bbox_ax = bbox.transformed(ax.transAxes.inverted())
            # Place the second legend directly below the first legend
            if nos_epoch in metric:
                anchor_point = (bbox_ax.x1, bbox_ax.y0)
                second_legend = ax.legend(
                    handles=combined_handles,
                    labels=combined_labels,
                    bbox_to_anchor=anchor_point,
                    loc='upper right',
                    title='Thresholds & Number of Samples (NoS)',
                    fontsize=11.5,  # Match axis label size
                    title_fontsize=11.5  # Match axis label size
                )
                ax.add_artist(second_legend)
        # No need to hide unused axes; gridspec handles this
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'pseudotime_{label}_trajectory_subfigs.eps'))
        plt.close()
    # Now lets do the same but with EMA smoothed trajectories for better visualisation of trends.
    for metric in input_metrics:
        assert metric in processed_pseudotime_aucs, f"Expected metric '{metric}' not found in the processed pseudotime AUCs."
        datasets = list(processed_pseudotime_aucs[metric].keys())
        n = len(datasets)
        fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 7), squeeze=False)
        plt.subplots_adjust(wspace=0.05)
        axes = axes[0]
        if n > 2:
            pos2 = axes[1].get_position()
            for idx in range(2, n):
                pos = axes[idx].get_position()
                new_x0 = pos2.x1 - 0.005
                axes[idx].set_position([new_x0, pos.y0, pos.width, pos.height])
        min_val = float('inf')
        min_vals_per_subplot = []
        for dataset in datasets:
            subplot_min = float('inf')
            for app in processed_pseudotime_aucs[metric][dataset].columns:
                # Use EMA smoothing with correct alpha per dataset
                smoothed = processed_pseudotime_aucs[metric][dataset][app].ewm(alpha=smoothing_param[dataset]).mean()
                v = smoothed.min()
                min_val = min(min_val, v)
                subplot_min = min(subplot_min, v)
            min_vals_per_subplot.append(subplot_min)
        y_max = 1
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            ax.tick_params(axis='both', labelsize='medium')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            subplot_min = min_vals_per_subplot[idx]
            subplot_y_min = max(0, np.floor((subplot_min - 0.05) / 0.05) * 0.05)
            subplot_y_max = y_max
            subplot_y_ticks = np.arange(subplot_y_min, subplot_y_max + 0.001, 0.05)
            ax.set_ylim(subplot_y_min, subplot_y_max)
            ax.set_yticks(subplot_y_ticks)
            ax.set_yticklabels([f"{ytick:.2f}" for ytick in subplot_y_ticks], fontweight='bold')
            trajectory_handles = []
            trajectory_labels = []
            vline_handles = []
            vline_labels = []
            nnunet_thresh_handles = []
            nnunet_thresh_labels = []
            for app in processed_pseudotime_aucs[metric][dataset].columns:
                col_name = f'{metric}_trajectory_auc'
                mapped = ALGORITHM_MAPPING.get(app, app)
                app_print = map_algorithm_name(mapped)
                if col_name in trajectory_pseudotime_auc[dataset][app].columns:
                    traj_auc = trajectory_pseudotime_auc[dataset][app][col_name].iloc[0]
                    label = f'{app_print} (Trajectory-AUC: {traj_auc:.3f})'
                else:
                    raise ValueError(f"Expected column '{col_name}' not found in the pseudotime trajectory CSV file for app '{app}' in dataset '{dataset}'.")
                x_vals = np.arange(1, len(processed_pseudotime_aucs[metric][dataset][app]) + 1)
                smoothed = processed_pseudotime_aucs[metric][dataset][app].ewm(alpha=smoothing_param[dataset]).mean()
                app_line, = ax.plot(x_vals, smoothed, label=label)
                trajectory_handles.append(app_line)
                trajectory_labels.append(label)
                app_colour = app_line.get_color()

                if nos_epoch in metric:
                    if app not in nos_values[dataset]:
                        raise ValueError(f"Expected app '{app}' not found in the nos_values dictionary for dataset '{dataset}'.")
                    nos_reference = copy.deepcopy(metric).split('_')[0]
                    assert nos_reference in ('Dice', 'NSD'), f"Expected metric reference '{nos_reference}' to be either 'Dice' or 'NSD'. Please check the metric naming convention."
                    temp_df = nos_values[dataset][app]
                    nos_val = temp_df.loc[nos_reference, 'cross_index']
                    if np.isnan(nos_val):
                        print(f"Warning: NaN value encountered for NoS for app '{app}' and metric '{metric}' in dataset '{dataset}'")
                    else:
                        nos_val = round(nos_val * len(processed_pseudotime_aucs[metric][dataset][app])) + 1
                        vline = ax.axvline(x=nos_val, color=app_colour, linestyle=':', linewidth=2)
                        vline_handles.append(mlines.Line2D([], [], color=app_colour, linestyle=':'))
                        vline_labels.append(f'{app_print} NoS: {nos_val}')
            ax.set_xlim(left=1)
            max_x = max([len(processed_pseudotime_aucs[metric][dataset][app]) for app in processed_pseudotime_aucs[metric][dataset].columns])
            n_ticks = 8
            step = max(1, int(np.ceil(max_x / n_ticks)))
            xticks = np.arange(1, max_x + 1, step)
            if xticks[-1] != max_x:
                xticks = np.append(xticks, max_x)
            ax.set_xticks(xticks)

            for pat, nnunet_metric in trajectory_to_nnunet_match_naming.items():
                m = re.match(pat, metric)
                if m:
                    if callable(nnunet_metric):
                        nnunet_metric_name = nnunet_metric(m)
                    else:
                        nnunet_metric_name = nnunet_metric
                    if nnunet_metric_name in nnunet_metrics[dataset]:
                        nnunet_threshold = nnunet_metrics[dataset][nnunet_metric_name]
                        x_vals = processed_pseudotime_aucs[metric][dataset].index + 1
                        nnunet_thresh_plt = ax.hlines(y=nnunet_threshold, xmin=x_vals[0], xmax=x_vals[-1], color='r', linestyle='--')
                        nnunet_thresh_handles.append(mlines.Line2D([], [], color='r', linestyle='--',label=f'nnUNet {nnunet_metric_name} Performance'))
                        nnunet_thresh_labels.append(f'nnUNet Performance: {nnunet_threshold:.3f}')
                    else:
                        raise ValueError(f"Expected nnUNet metric '{nnunet_metric_name}' not found in the provided nnUNet metrics for dataset '{dataset}'. Please check the input files and ensure consistency.")

            m = False
            for pat, plot_label in translate_metric_name_to_plot_label.items():
                m = re.match(pat, metric)
                if m:
                    translated_label = plot_label(m) if callable(plot_label) else plot_label
                    label = translated_label
                    assert isinstance(label, str), f"Label is not a string after translation: {label} (type: {type(label)})"
                    break
            if not m:
                label = copy.deepcopy(metric)
            ax.set_title(DATASET_MAPPING.get(dataset, dataset), fontweight='bold')
            ax.set_xlabel('Data Samples Received', fontweight='bold')
            ticks = ax.get_xticks()
            if 1 not in ticks:
                new_ticks = np.insert(ticks, 0, 1.0)
                ax.set_xticks(new_ticks)
            if idx == 0:
                subplot_min = min_vals_per_subplot[idx]
                subplot_y_min = max(0, np.floor((subplot_min - 0.1) / 0.1) * 0.1)
                subplot_y_max = 1
                if subplot_y_min == 0:
                    subplot_y_ticks = np.arange(0, subplot_y_max + 0.001, 0.2)
                else:
                    subplot_y_ticks = np.arange(subplot_y_min, subplot_y_max + 0.001, 0.15)
                ax.set_ylim(subplot_y_min, subplot_y_max)
                ax.set_yticks(subplot_y_ticks)
                ax.set_ylabel(label, fontweight='bold')
                ax.set_yticklabels([f"{ytick:.2f}" for ytick in subplot_y_ticks], fontweight='bold')
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([f"{ytick:.2f}" for ytick in ax.get_yticks()], fontweight='bold')
            ax.grid()
            if idx == 0:
                first_legend = ax.legend(
                    handles=trajectory_handles,
                    labels=trajectory_labels,
                    loc='center right',
                    title='Trajectory Metrics',
                    fontsize=11.5,
                    title_fontsize=11.5
                )
            elif idx > 0:
                first_legend = ax.legend(
                    handles=trajectory_handles,
                    labels=trajectory_labels,
                    loc='upper right',
                    title='Trajectory Metrics',
                    fontsize=11.5,
                    title_fontsize=11.5
                )
            ax.add_artist(first_legend)
            combined_handles = nnunet_thresh_handles + vline_handles
            combined_labels = nnunet_thresh_labels + vline_labels
            renderer = fig.canvas.get_renderer()
            bbox = first_legend.get_window_extent(renderer=renderer)
            bbox_ax = bbox.transformed(ax.transAxes.inverted())
            if nos_epoch in metric:
                anchor_point = (bbox_ax.x1, bbox_ax.y0)
                second_legend = ax.legend(
                    handles=combined_handles,
                    labels=combined_labels,
                    bbox_to_anchor=anchor_point,
                    loc='upper right',
                    title='Thresholds & Number of Samples (NoS)',
                    fontsize=11.5,
                    title_fontsize=11.5
                )
                ax.add_artist(second_legend)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'pseudotime_{label}_trajectory_ema_subfigs.eps'))
        plt.close()

def set_parse():
    parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
    parser.add_argument('--algorithm_results_root', type=str, required=True, help='Root path to the algorithm results')
    parser.add_argument('--output_folder_root', type=str, required=True, help='Root path to the output folder for results summarisation')
    parser.add_argument('--experiment_names', nargs='+', required=True, help='Names of the experiment for which the metrics are being summarised')
    parser.add_argument('--apps', nargs='+', required=True, help='List of applications for which the metrics are being summarised')
    parser.add_argument('--linegraph_filename', type=str, required=True, help='Filename of the CSV file containing the metrics.')
    parser.add_argument('--trajectory_filename', type=str, required=True, help='Filename of the CSV file containing trajectory AUCs.')
    parser.add_argument('--metrics_config', type=str, required=True, help='JSON string of metrics config')
    
    parser.add_argument('--nnunet_statistic', nargs='+', required=True, help='Statistic to use for nnUNet threshold calculation. Supported options are "gaussian", "student", "laplace", "beta", "quantile".')
    parser.add_argument('--nnunet_bound', nargs='+', required=True, help='Bound to use for nnUNet threshold calculation. Interpretation depends on the chosen statistic. For "gaussian", "student", "laplace", "beta", this is the number of standard deviations below the mean. For "quantile", this is the quantile to use (between 0 and 1).')
    parser.add_argument('--nnunet_reference_filename', type=str, required=True, help='Filename of the CSV file containing the nnUNet metric values for threshold calculation.')
    parser.add_argument('--nnunet_result_roots', nargs='+', required=True, help='Root path to the nnUNet results.')
    
    parser.add_argument('--nos_epoch', type=str, required=True, help='Number of samples epoch to use for labelling the plots. This is just for labelling and does not affect the actual data used for plotting.')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = set_parse()
   
    input_results_root=  args.algorithm_results_root
    output_folder_root = args.output_folder_root
    apps = args.apps
    experiment_names = args.experiment_names
    # print(experiment_name)

    input_folders = {
        experiment_name.split('/')[0]: {app: os.path.join(input_results_root, app, experiment_name) for app in apps}
        for experiment_name in experiment_names
    }
    output_folder = os.path.join(output_folder_root, 'subfigure_plots')
    filename = args.linegraph_filename
    trajectory_filename = args.trajectory_filename

    os.makedirs(output_folder, exist_ok=True) 
    #Lets dump the set of apps for the given config.
    if not os.path.exists(os.path.join(output_folder, 'apps.json')):
        with open(os.path.join(output_folder, 'apps.json'), 'w') as f:
            json.dump(apps, f, indent=2)
    else:
        #ASSERT that the apps in the existing file match the current set of apps to avoid confusion.
        with open(os.path.join(output_folder, 'apps.json'), 'r') as f:
            existing_apps = json.load(f)
        assert set(existing_apps) == set(apps), f"Existing apps in {os.path.join(output_folder, 'apps.json')} do not match the current set of apps. Existing: {existing_apps} Current: {apps}. Please check the input files and ensure consistency to avoid confusion."    
    # input_metrics = ('Dice', 'NSD')
    metrics_config = load_metrics_config(args.metrics_config)
    #Read the raw pseudotime metrics from the CSV file.
    raw_pseudotime_metrics = {
        dataset: {
        app:pd.read_csv(os.path.join(input_folders[dataset][app], 'pseudotime_metrics', filename))
        for app in apps
        }
        for dataset in input_folders.keys()
    }
    trajectory_pseudotime_aucs = {
        dataset: {
        app:pd.read_csv(os.path.join(input_folders[dataset][app], 'pseudotime_metrics', trajectory_filename))
        for app in apps
    }
        for dataset in input_folders.keys()
    }

    # Assert that each dataset name is present in the directory listing of the corresponding nnunet_result_root
    for idx, dataset in enumerate(input_folders.keys()):
        assert dataset in args.nnunet_result_roots[idx], f"Dataset {dataset} not found in nnUNet results root {args.nnunet_result_roots[idx]}"

    nnunet_thresholds = {
        dataset: extract_nnunet_thresholds(
            args.nnunet_statistic, 
            args.nnunet_bound, 
            {nnunet_metric: pd.read_csv(os.path.join(args.nnunet_result_roots[idx], f"{nnunet_metric}", args.nnunet_reference_filename), skiprows=1, names=['Case_Name', 'Automatic Init']) for nnunet_metric in nnunet_permitted_metrics},
            nnunet_permitted_metrics
        )
        for idx, dataset in enumerate(input_folders.keys())
    }
    nos_values = {
        dataset: {
        app: pd.read_csv(os.path.join(input_folders[dataset][app], 'pseudotime_metrics', 'pseudotime_threshold_crossing.csv'), usecols=[0,1]) for app in apps
        }
        for dataset in input_folders.keys()
    }
    for dataset, dicts in nos_values.items():
        for app, df in dicts.items():
            if 'Unnamed: 0' in df.columns:
                df.set_index('Unnamed: 0', inplace=True)
                df.index.name = None

    #Lets rearrange into metric separated dataframes for easier plotting, where each dataframe has columns for each app and rows for each pseudotime point.
    processed_pseudotime_dfs = {}
    #First lets unroll all of the metric names into separate metrics so its easier to loop through for plotting.
    for metric, subcols in metrics_config.items():
        if subcols is not None:
            for subcol in subcols:
                if subcol in translate_subcol:
                    subcol = translate_subcol[subcol]
                new_metric_name = f"{metric} {subcol}"
                processed_pseudotime_dfs[new_metric_name] = None #initialise with None for now, we will fill it in the next loop.
        else:
            processed_pseudotime_dfs[metric] = None #initialise with None for now, we will fill it in the next loop.
    
    for metric in processed_pseudotime_dfs.keys():
        cross_dataset_metric_df = dict()
        for dataset in raw_pseudotime_metrics.keys():
            metric_df = pd.DataFrame()
            for app in apps:
                assert f'{metric}' in raw_pseudotime_metrics[dataset][app].columns, f"Expected column '{metric}' not found in the input CSV file for app '{app}' in dataset '{dataset}'."
                metric_df[app] = raw_pseudotime_metrics[dataset][app][metric]
            cross_dataset_metric_df[dataset] = metric_df
        processed_pseudotime_dfs[metric] = cross_dataset_metric_df
    #Plot the pseudotime AUC trajectories for each metric.
    smoothing_param = {dataset: smoothing_params.get(dataset) for dataset in raw_pseudotime_metrics.keys()}
    # print(smoothing_param)

    #We are going to do a check here, that asserts that our metrics are consistent with nnunet thresholding statistic.
    nnunet_statistic_name = TRANSLATE_NNUNET_STATISTIC_TO_PSEUDOTIME_NAME.get(args.nnunet_statistic[0], {}).get(str(args.nnunet_bound[0]), f"{args.nnunet_statistic[0]}_{args.nnunet_bound[0]}")

    for metric in processed_pseudotime_dfs.keys():
        for pat, statistic_name in cross_check_with_nnunet_metric_regex.items():
            m = re.match(pat, metric)
            if m:
                if callable(statistic_name):
                    equivalent_name = statistic_name(m)
                else:
                    equivalent_name = statistic_name
                assert equivalent_name == nnunet_statistic_name, f"Expected nnUNet statistic '{equivalent_name}' for metric '{metric}'. Please check the input files and ensure consistency."
    # print(processed_pseudotime_dfs)
    plot_pseudotime_aucs(processed_pseudotime_dfs.keys(), processed_pseudotime_dfs, trajectory_pseudotime_aucs, output_folder, nnunet_thresholds, nos_values, nos_epoch=args.nos_epoch, smoothing_param=smoothing_param)