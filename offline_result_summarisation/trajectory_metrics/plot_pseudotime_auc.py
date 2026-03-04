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

    # Create a plot for each metric
    for metric in input_metrics:
        assert metric in processed_pseudotime_aucs, f"Expected metric '{metric}' not found in the processed pseudotime AUCs."
        fig, ax = plt.subplots(figsize=(10, 6))
        trajectory_handles = []
        trajectory_labels = []
        vline_handles = []
        vline_labels = []
        nnunet_thresh_handles = []
        nnunet_thresh_labels = []
        for app in processed_pseudotime_aucs[metric].columns:
            # Get trajectory AUC for legend label
            col_name = f'{metric}_trajectory_auc'
            mapped = ALGORITHM_MAPPING.get(app, app)
            app_print = map_algorithm_name(mapped)
            if col_name in trajectory_pseudotime_auc[app].columns:
                traj_auc = trajectory_pseudotime_auc[app][col_name].iloc[0]
                label = f'{app_print} (Trajectory-AUC: {traj_auc:.3f})'
            else:
                raise ValueError(f"Expected column '{col_name}' not found in the pseudotime trajectory CSV file for app '{app}'.")
            x_vals = np.arange(1, len(processed_pseudotime_aucs[metric][app]) + 1)
            # If app_print is a string, wrap in list for mapping, then extract the mapped value
            
            app_line, = ax.plot(x_vals, processed_pseudotime_aucs[metric][app], label=label)
            trajectory_handles.append(app_line)
            trajectory_labels.append(label)
            app_colour = app_line.get_color()

            if nos_epoch in metric:
                #If epoch name is in metric then print a vertical line for num_of_samples metric for each app.
                if app not in nos_values:
                    raise ValueError(f"Expected app '{app}' not found in the nos_values dictionary.")
                
                # print(nos_values[app])
                nos_reference=copy.deepcopy(metric)
                nos_reference = nos_reference.split('_')[0]
                # print(nos_reference)
                assert nos_reference in ('Dice', 'NSD'), f"Expected metric reference '{nos_reference}' to be either 'Dice' or 'NSD'. Please check the metric naming convention."
                temp_df = nos_values[app]
                nos_val = temp_df.loc[nos_reference, 'cross_index'] 
                if np.isnan(nos_val):
                    print(f"Warning: NaN value encountered for NoS for app '{app}' and metric '{metric}'")
                else:
                    nos_val = round(nos_val * len(processed_pseudotime_aucs[metric][app])) + 1 # Use .loc for correct row/col access
                    vline = ax.axvline(x=nos_val, color=app_colour, linestyle=':', linewidth=2)
                    vline_handles.append(mlines.Line2D([], [], color=app_colour, linestyle=':'))
                    vline_labels.append(f'{app_print} NoS: {nos_val}')
        # Ensure x-axis starts at 1 and ticks are not overcrowded
        ax.set_xlim(left=1)
        max_x = max([len(processed_pseudotime_aucs[metric][app]) for app in processed_pseudotime_aucs[metric].columns])
        n_ticks = 8  # target number of ticks
        step = max(1, int(np.ceil(max_x / n_ticks)))
        xticks = np.arange(1, max_x + 1, step)
        if xticks[-1] != max_x:
            xticks = np.append(xticks, max_x)
        ax.set_xticks(xticks)
            
        #Lets check if the metric is in the nnunet regex match dict. if so we will add a horizontal line for 
        #that given base metric.
        
        for pat, nnunet_metric in trajectory_to_nnunet_match_naming.items():
            m = re.match(pat, metric)
            if m:
                # print(nnunet_metrics)
                if callable(nnunet_metric):
                    nnunet_metric_name = nnunet_metric(m)
                else:
                    nnunet_metric_name = nnunet_metric

                if nnunet_metric_name in nnunet_metrics:
                    nnunet_threshold = nnunet_metrics[nnunet_metric_name]
                    x_vals = processed_pseudotime_aucs[metric].index + 1
                    nnunet_thresh_plt = ax.hlines(y=nnunet_threshold, xmin=x_vals[0], xmax=x_vals[-1], color='r', linestyle='--')
                    nnunet_thresh_handles.append(mlines.Line2D([], [], color='r', linestyle='--',label=f'nnUNet {nnunet_metric_name} Performance'))
                    nnunet_thresh_labels.append(f'nnUNet Performance: {nnunet_threshold:.3f}')
                else:
                    raise ValueError(f"Expected nnUNet metric '{nnunet_metric_name}' not found in the provided nnUNet metrics. Please check the input files and ensure consistency.")

        # Use regex to match metric name and translate to plot label
        m = False
        for pat, plot_label in translate_metric_name_to_plot_label.items():
            # print(f"Checking pattern '{pat}' against metric '{metric}'")
            m = re.match(pat, metric)
            if m:
                translated_label = plot_label(m) if callable(plot_label) else plot_label
                # print(f"Pattern '{pat}' matched metric '{metric}'. Using plot label: '{translated_label}'")
                label = translated_label
                # print(f"Translated metric '{metric}' to plot label '{label}' using regex pattern '{pat}'")
                # print(f"DEBUG: type(label) after translation: {type(label)} value: {label}")
                assert isinstance(label, str), f"Label is not a string after translation: {label} (type: {type(label)})"
                break
        if not m:
            label = copy.deepcopy(metric)
            # print(f"No regex pattern matched metric '{metric}'. Using original metric name as label.")
        ax.set_title(f'{label} Trajectory Over Data Samples Received')
        # Set x-axis label and force a tick at 1
        ax.set_xlabel('Data Samples Received')
        ticks = ax.get_xticks()
        if 1 not in ticks:
            # Insert 1 at the beginning and set new ticks
            new_ticks = np.insert(ticks, 0, 1.0)
            ax.set_xticks(new_ticks)
        ax.set_ylabel(label)
        ax.grid()

        # First legend: trajectory metrics
        first_legend = ax.legend(handles=trajectory_handles, labels=trajectory_labels, loc='best', title='Trajectory Metrics')
        ax.add_artist(first_legend)

        # Second legend: thresholds and NoS vlines
        combined_handles = nnunet_thresh_handles + vline_handles
        combined_labels = nnunet_thresh_labels + vline_labels

        # Calculate bbox just below the first legend, but keep x fixed (e.g., at 0)
        renderer = plt.gcf().canvas.get_renderer()
        bbox = first_legend.get_window_extent(renderer=renderer)
        bbox_ax = bbox.transformed(ax.transAxes.inverted())
        # Anchor directly below the first legend's horizontal position
        # Align left sides: use bbox_ax.x0 for both, and place second legend just below first
        x_anchor = bbox_ax.x0 
        y_anchor = bbox_ax.y1 - 0.15
        if nos_epoch in metric:
            second_legend = ax.legend(handles=combined_handles, labels=combined_labels,
                     bbox_to_anchor=(x_anchor, y_anchor), loc='upper left', title='Thresholds & NoS')
            ax.add_artist(second_legend)

        plt.savefig(os.path.join(output_folder, f'pseudotime_{label}_trajectory.eps'))
        plt.close()
        #Clear the ax
        ax.cla()
    #Now lets do the same but with EMA smoothed trajectories for better visualisation of trends.
    # print(input_metrics)
    # for metric in input_metrics:
    #     # print(f"DEBUG: type(metric) at start of smoothed loop: {type(metric)} value: {metric}")
    #     assert isinstance(metric, str), f"Metric is not a string at start of smoothed loop: {metric} (type: {type(metric)})"
    #     assert metric in processed_pseudotime_aucs, f"Expected metric '{metric}' not found in the processed pseudotime AUCs."
    #     plt.figure(figsize=(10, 6))
    #     for app in processed_pseudotime_aucs[metric].columns:
    #         smoothed_values = processed_pseudotime_aucs[metric][app].ewm(alpha=smoothing_param).mean()
    #         # Get trajectory AUC for legend label
    #         col_name = f'{metric}_trajectory_auc'
    #         mapped = ALGORITHM_MAPPING.get(app, app)
    #         app_print = map_algorithm_name(mapped)
    #         if col_name in trajectory_pseudotime_auc[app].columns:
    #             traj_auc = trajectory_pseudotime_auc[app][col_name].iloc[0]
    #             label = f'{app_print} (Trajectory-AUC: {traj_auc:.3f})'
    #         else:
    #             raise ValueError(f"Expected column '{col_name}' not found in the pseudotime trajectory CSV file for app '{app}'.")
    #         x_vals = np.arange(1, len(processed_pseudotime_aucs[metric][app]) + 1)
    #         plt.plot(x_vals, smoothed_values, label=label)
    #     # Ensure x-axis starts at 1 and ticks are not overcrowded
    #     plt.xlim(left=1)
    #     max_x = max([len(processed_pseudotime_aucs[metric][app]) for app in processed_pseudotime_aucs[metric].columns])
    #     n_ticks = 8  # target number of ticks
    #     step = max(1, int(np.ceil(max_x / n_ticks)))
    #     xticks = np.arange(1, max_x + 1, step)
    #     if xticks[-1] != max_x:
    #         xticks = np.append(xticks, max_x)
    #     plt.xticks(xticks)
        
    #     for pat, nnunet_metric in trajectory_to_nnunet_match_naming.items():
    #         m = re.match(pat, metric)
    #         if m:
    #             # print(nnunet_metrics)
    #             if callable(nnunet_metric):
    #                 nnunet_metric_name = nnunet_metric(m)
    #             else:
    #                 nnunet_metric_name = nnunet_metric

    #             if nnunet_metric_name in nnunet_metrics:
    #                 nnunet_threshold = nnunet_metrics[nnunet_metric_name]
    #                 plt.axhline(y=nnunet_threshold, color='r', linestyle='--', label=f'nnUNet {nnunet_metric_name} Threshold')
    #             else:
    #                 raise ValueError(f"Expected nnUNet metric '{nnunet_metric_name}' not found in the provided nnUNet metrics. Please check the input files and ensure consistency.")
            

    #     m = False
    #     for pat, plot_label in translate_metric_name_to_plot_label.items():
    #         # print(f"Checking pattern '{pat}' against metric '{metric}'")
    #         m = re.match(pat, metric)
    #         if m:
    #             translated_label = plot_label(m) if callable(plot_label) else plot_label
    #             # print(f"Pattern '{pat}' matched metric '{metric}'. Using plot label: '{translated_label}'")
    #             label = translated_label
    #             # print(f"Translated metric '{metric}' to plot label '{label}' using regex pattern '{pat}'")
    #             # print(f"DEBUG: type(label) after translation: {type(label)} value: {label}")
    #             assert isinstance(label, str), f"Label is not a string after translation: {label} (type: {type(label)})"
    #             break
    #     if not m:
    #         label = copy.deepcopy(metric)
    #         # print(f"No regex pattern matched metric '{metric}'. Using original metric name as label.")
    #     plt.ylabel(f'{label}')
    #     plt.title(f'Smoothed Trajectory for {label} with EMA smoothing (α={smoothing_param})')
    #     # Set x-axis label and force a tick at 1
    #     plt.xlabel('Data Samples Received')
    #     ax = plt.gca()
    #     ticks = ax.get_xticks()
    #     if 1 not in ticks:
    #         # Insert 1 at the beginning and set new ticks
    #         new_ticks = np.insert(ticks, 0, 1.0)
    #         ax.set_xticks(new_ticks)
    #     plt.grid()
    #     plt.legend()
    #     plt.savefig(os.path.join(output_folder, f'pseudotime_{label}_smoothed_trajectory.eps'))
    #     plt.close()

def set_parse():
    parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
    parser.add_argument('--algorithm_results_root', type=str, required=True, help='Root path to the algorithm results')
    parser.add_argument('--output_folder_root', type=str, required=True, help='Root path to the output folder for results summarisation')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment for which the metrics are being summarised')
    parser.add_argument('--apps', nargs='+', required=True, help='List of applications for which the metrics are being summarised')
    parser.add_argument('--linegraph_filename', type=str, required=True, help='Filename of the CSV file containing the metrics.')
    parser.add_argument('--trajectory_filename', type=str, required=True, help='Filename of the CSV file containing trajectory AUCs.')
    parser.add_argument('--metrics_config', type=str, required=True, help='JSON string of metrics config')
    
    parser.add_argument('--nnunet_statistic', nargs='+', required=True, help='Statistic to use for nnUNet threshold calculation. Supported options are "gaussian", "student", "laplace", "beta", "quantile".')
    parser.add_argument('--nnunet_bound', nargs='+', required=True, help='Bound to use for nnUNet threshold calculation. Interpretation depends on the chosen statistic. For "gaussian", "student", "laplace", "beta", this is the number of standard deviations below the mean. For "quantile", this is the quantile to use (between 0 and 1).')
    parser.add_argument('--nnunet_reference_filename', type=str, required=True, help='Filename of the CSV file containing the nnUNet metric values for threshold calculation.')
    parser.add_argument('--nnunet_result_root', type=str, required=True, help='Root path to the nnUNet results.')
    
    parser.add_argument('--nos_epoch', type=str, required=True, help='Number of samples epoch to use for labelling the plots. This is just for labelling and does not affect the actual data used for plotting.')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = set_parse()
   
    input_results_root=  args.algorithm_results_root
    output_folder_root = args.output_folder_root
    apps = args.apps
    experiment_name = args.experiment_name
    # print(experiment_name)

    input_folders = {app: os.path.join(input_results_root, app, experiment_name) for app in apps}
    output_folder = os.path.join(output_folder_root, 'pseudotime_metrics')
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
        app:pd.read_csv(os.path.join(input_folders[app], 'pseudotime_metrics', filename))
        for app in apps
    }
    trajectory_pseudotime_aucs = {
        app:pd.read_csv(os.path.join(input_folders[app], 'pseudotime_metrics', trajectory_filename))
        for app in apps
    }

    nnunet_thresholds = extract_nnunet_thresholds(
        args.nnunet_statistic, 
        args.nnunet_bound, 
        {nnunet_metric: pd.read_csv(os.path.join(args.nnunet_result_root, f"{nnunet_metric}", args.nnunet_reference_filename), skiprows=1, names=['Case_Name', 'Automatic Init']) for nnunet_metric in nnunet_permitted_metrics},
        nnunet_permitted_metrics
    )
    nos_values = {
        app: pd.read_csv(os.path.join(input_folders[app], 'pseudotime_metrics', 'pseudotime_threshold_crossing.csv'), usecols=[0,1]) for app in apps
    }
    for app, df in nos_values.items():
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
        metric_df = pd.DataFrame()
        for app in apps:
            assert f'{metric}' in raw_pseudotime_metrics[app].columns, f"Expected column '{metric}' not found in the input CSV file for app '{app}'."
            metric_df[app] = raw_pseudotime_metrics[app][metric]
        processed_pseudotime_dfs[metric] = metric_df
    #Plot the pseudotime AUC trajectories for each metric.
    smoothing_param = smoothing_params.get(experiment_name.split('/')[0])
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
    plot_pseudotime_aucs(processed_pseudotime_dfs.keys(), processed_pseudotime_dfs, trajectory_pseudotime_aucs, output_folder, nnunet_thresholds, nos_values, nos_epoch=args.nos_epoch, smoothing_param=smoothing_param)