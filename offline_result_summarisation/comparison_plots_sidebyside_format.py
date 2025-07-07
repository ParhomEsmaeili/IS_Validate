import os 
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import argparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
matplotlib.style.use('ggplot')
import pandas as pd
import seaborn as sns 

def plot_evolution_metrics(metrics_storage_dict, output_folder, input_metrics, summary_statistics=('Mean','Median')):
    """
    Plots the evolution of specified metrics.

    Args:
        metrics_storage_dict: Dictionary containing app names and their corresponding paths to the folder
        where the metrics are stored. 
        output_folder (str): Path to the folder where the plots will be saved.
        input_metrics (tuple): Tuple of metric names to plot.
    """
    # Build metric_dict for all metrics and all apps
    metric_dict = {metric: {} for metric in input_metrics}
    for metric in input_metrics:
        for app_name, metric_folder in metrics_storage_dict.items():
            metric_file = os.path.join(metric_folder, metric, 'cross_class_scores.csv')
            if not os.path.exists(metric_file):
                raise FileNotFoundError(f"Metric file not found: {metric_file}")
            column_headers = ['Case_Name', 'Interactive Init'] + [f'Interactive Edit Iter {i + 1}' for i in range(100)]
            df = pd.read_csv(metric_file, skiprows=1, names=column_headers)
            columns_to_summarise = [col for col in df.columns if col != 'Case_Name']
            summary = pd.DataFrame({
                f'{metric}_Mean': df[columns_to_summarise].mean(),
                f'{metric}_Median': df[columns_to_summarise].median(),
            })
            metric_dict[metric][app_name] = summary

    # Now plot
    for summary_stat in summary_statistics:
        all_data = []
        for metric in input_metrics:
            for app_name, summary_df in metric_dict[metric].items():
                y = summary_df[f'{metric}_{summary_stat}'].values
                for iteration, value in enumerate(y):
                    all_data.append({
                        'Iteration': iteration,
                        'Score': value,
                        'App': app_name,
                        'Metric': metric
                    })
        plot_df_long = pd.DataFrame(all_data)

        plt.figure(figsize=(10, 6))
        dashes = {'Dice': '', 'NSD': (3, 3)}

        custom_palette = {
            "SAMMed2D": "#1f77b4",   # blue
            #"SAM2": "#800080",       # orange
            "SAM2": "#ff7f0e",  
            # "SAMMed3D": "#2ca02c",   # green
            "SAMMed3D": "#7ED957",
            "SegVol": "#d62728",  
            }  # purple for SAM2

        sns.lineplot(
            data=plot_df_long,
            x='Iteration',
            y='Score',
            hue='App',
            style='Metric',
            dashes=dashes,
            marker=None,
            palette=custom_palette
        )
        plt.title(f'Evolution of {summary_stat} Dice/NSD', fontsize=18, fontweight='bold')
        plt.xlabel('Iteration', fontsize=15, fontweight='bold', color='black')
        plt.ylabel(f'{summary_stat} Score', fontsize=15, fontweight='bold', color='black')
        plt.xticks(fontsize=13, fontweight='bold', color='black')
        plt.yticks(fontsize=13, fontweight='bold', color='black')
        plt.tight_layout()

        ax = plt.gca()
        # Remove the default legend
        ax.legend_.remove()

        # 1. App legend (color)
        app_handles = []
        for app in app_names:
            app_handles.append(
                mlines.Line2D([], [], color=custom_palette.get(app, None), label=app, linewidth=3)
            )
        legend1 = ax.legend(
            handles=app_handles,
            title="App",
            title_fontproperties=fm.FontProperties(weight='bold', size=13),
            fontsize=12,
            bbox_to_anchor=(1.05, 1),  # Top right
            loc='upper left',
            borderaxespad=0.,
            labelspacing=1.2,
        )
        ax.add_artist(legend1)

        # 2. Metric legend (line style)
        metric_handles = [
            mlines.Line2D([], [], color='black', linestyle='-' if m == 'Dice' else (0, (3, 3)), label=m, linewidth=3)
            for m in input_metrics
        ]
        legend2 = ax.legend(
            handles=metric_handles,
            title="Metric",
            title_fontproperties=fm.FontProperties(weight='bold', size=13),
            fontsize=12,
            bbox_to_anchor=(1.05, 0.55),  # Move this further down (try 0.55 or lower)
            loc='upper left',
            borderaxespad=0.,
            labelspacing=1.2,
        )

        output_path = os.path.join(output_folder, f'{summary_stat}_Dice_NSD_evolution.eps')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


def plot_evolution_metrics_side_by_side(
    datasets, 
    datetimes_list, 
    app_names, 
    parent_dir, 
    input_metrics, 
    output_folder, 
    summary_statistics=('Mean', 'Median')
):
    """
    Plots the evolution of specified metrics for multiple datasets side by side.
    """
    for summary_stat in summary_statistics:
        fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 6), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        for idx, dataset_name in enumerate(datasets):
            datetimes = datetimes_list[idx]
            metric_storage_dict = {
                app_name: os.path.join(parent_dir, 'results', dataset_name, datetime, 'metrics')
                for app_name, datetime in zip(app_names, datetimes)
            }
            # Build metric_dict for all metrics and all apps
            metric_dict = {metric: {} for metric in input_metrics}
            for metric in input_metrics:
                for app_name, metric_folder in metric_storage_dict.items():
                    metric_file = os.path.join(metric_folder, metric, 'cross_class_scores.csv')
                    if not os.path.exists(metric_file):
                        raise FileNotFoundError(f"Metric file not found: {metric_file}")
                    column_headers = ['Case_Name', 'Interactive Init'] + [f'Interactive Edit Iter {i + 1}' for i in range(100)]
                    df = pd.read_csv(metric_file, skiprows=1, names=column_headers)
                    columns_to_summarise = [col for col in df.columns if col != 'Case_Name']
                    summary = pd.DataFrame({
                        f'{metric}_Mean': df[columns_to_summarise].mean(),
                        f'{metric}_Median': df[columns_to_summarise].median(),
                    })
                    metric_dict[metric][app_name] = summary

            all_data = []
            for metric in input_metrics:
                for app_name, summary_df in metric_dict[metric].items():
                    y = summary_df[f'{metric}_{summary_stat}'].values
                    for iteration, value in enumerate(y):
                        all_data.append({
                            'Iteration': iteration,
                            'Score': value,
                            'App': app_name,
                            'Metric': metric
                        })
            plot_df_long = pd.DataFrame(all_data)

            dashes = {'Dice': '', 'NSD': (3, 3)}
            custom_palette = {
                "SAMMed2D": "#1f77b4",
                "SAM2": "#ff7f0e",
                "SAMMed3D": "#7ED957",
                "SegVol": "#d62728",
            }
            ax = axes[idx]
            sns.lineplot(
                data=plot_df_long,
                x='Iteration',
                y='Score',
                hue='App',
                style='Metric',
                dashes=dashes,
                marker=None,
                palette=custom_palette,
                ax=ax
            )

            #We hardcode some nicer title names: 
            if dataset_name == 'Dataset001_BrainTumour':
                title_name = 'Brain Tumour Core'
            elif dataset_name == 'Dataset004_Hippocampus':
                title_name = 'Whole Hippocampus'
            elif dataset_name == 'Dataset005_Prostate':
                title_name = 'Whole Prostate'
            elif dataset_name == 'Dataset006_Lung':
                title_name = 'Lung Lesion' 
            elif dataset_name == 'Dataset007_Pancreas':
                title_name = 'Whole Pancreas'
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
            ax.set_title(title_name, fontsize=16, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=13, fontweight='bold', color='black')
            ax.set_ylim(0, 1)  # Share vertical axis from 0 to 1

            # Set ticks and labels
            ax.tick_params(axis='both', labelsize=12, width=2.5)
            if idx == 0:
                ax.set_ylabel(f'{summary_stat} Score', fontsize=13, fontweight='bold', color='black')
                ax.tick_params(axis='y', left=True, labelleft=True)
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', left=False, labelleft=False)

            # Make tick labels bold and black (must be after tick_params)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')

            ax.legend_.remove()

        # Add a single legend for all apps and metrics inside the right-most subplot
        handles = [
            mlines.Line2D([], [], color=custom_palette.get(app, None), label=app, linewidth=3)
            for app in app_names
        ]
        metric_handles = [
            mlines.Line2D([], [], color='black', linestyle='-' if m == 'Dice' else (0, (3, 3)), label=m, linewidth=3)
            for m in input_metrics
        ]
        # axes[0].legend(
        #     handles + metric_handles,
        #     [h.get_label() for h in handles + metric_handles],
        #     title="App / Metric",
        #     title_fontproperties=fm.FontProperties(weight='bold', size=12),  # smaller title
        #     fontsize=12,  # smaller legend text
        #     loc='upper left',
        #     bbox_to_anchor=(0.025, 0.975),  # inside the axes, top left
        #     borderaxespad=0.3,
        #     labelspacing=0.7,  # less vertical space between entries
        #     frameon=True
        # )
        # axes[0].legend(
        #     handles + metric_handles,
        #     [h.get_label() for h in handles + metric_handles],
        #     title="App / Metric",
        #     title_fontproperties=fm.FontProperties(weight='bold', size=12),  # smaller title
        #     fontsize=12,  # smaller legend text
        #     loc='lower right',
        #     bbox_to_anchor=(0.95, 0.05),  # bottom right of the first subplot
        #     borderaxespad=0.,
        #     labelspacing=1.2,  # less vertical space between entries
        #     frameon=True,
        #     ncol=len(handles + metric_handles) // 2
        # )
        axes[0].legend(
            handles + metric_handles,
            [h.get_label() for h in handles + metric_handles],
            title="App / Metric",
            title_fontproperties=fm.FontProperties(weight='bold', size=12),  # smaller title
            fontsize=12,  # smaller legend text
            loc='center right',
            bbox_to_anchor=(0.95, 0.4),  # center right of the first subplot
            borderaxespad=0.,
            labelspacing=1.2,  # less vertical space between entries
            frameon=True,
            ncol=2
        )

        plt.tight_layout()
        output_path = os.path.join(output_folder, f'{summary_stat}_side_by_side_evolution.eps')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


# def set_parse():
#     parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
#     parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
#     parser.add_argument('--datetimes', nargs='+', type=str, required=True, help='Datetime string for the results folder')
#     parser.add_argument('--app_names', nargs='+', type=str, required=True, help='Names of the applications to compare')
#     return parser.parse_args()

def set_parse():
    parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
    parser.add_argument('--dataset_names', nargs='+', type=str, required=True, help='Names of the datasets')
    parser.add_argument('--datetimes_list', nargs='+', type=str, action='append', required=True,
                        help='List of datetimes for each dataset (one quoted string per dataset)')
    parser.add_argument('--app_names', nargs='+', type=str, required=True, help='Names of the applications to compare')
    return parser.parse_args()

if __name__ == "__main__":

    input_metrics = ('Dice', 'NSD')
    args = set_parse()
    datasets = args.dataset_names
    datetimes_list = args.datetimes_list
    print(datetimes_list)
    app_names = args.app_names
    # print(datetimes)
    # app_names = ("SAMMed2D", "SAM2", "SAMMed3D", "SegVol")


    #We will put in a check, which is to go through the logfile and check if the app name is present in the log file.
    #Otherwise we have made an error and we should not proceed with the plotting. 

    for dataset_name, datetimes in zip(datasets, datetimes_list):

        assert len(datetimes) == len(app_names), "The number of datetimes must match the number of app names."

        for datetime, app_name in zip(datetimes, app_names):
            logfile_path = os.path.join(parent_dir, 'results', dataset_name, datetime, f'experiment_{datetime}_logs.log')
            if not os.path.exists(logfile_path):
                raise FileNotFoundError(f"Log file not found for {app_name} at {logfile_path}")
            
            found=False 
            with open(logfile_path, 'r') as f:
                for line in f:
                    if f'app_name: "Sample_{app_name}' in line:
                        found = True
                        break
            if not found:
                raise ValueError(f"App name {app_name} not found in log file for {datetime} at {logfile_path}")



    #Now we convert the list of datetimes and app names into a paired dictionary so that we can access it.
    metric_storage_dict = {app_name: os.path.join(parent_dir, 'results', dataset_name, datetime, 'metrics') for app_name, datetime in zip(app_names,datetimes)} #My brain not working at this point so lets just hack 
    #it together. time isn't going to be an issue here.


    input_folder =  os.path.join(parent_dir, 'results', dataset_name, datetime, 'metrics')
    # Replace with your input file path
    output_folder = os.path.join(parent_dir, 'results_summary', 'miccai_plots_merge', "_".join(datasets))

    os.makedirs(output_folder, exist_ok=True) 
    

    # plot_evolution_metrics(
    #     metric_storage_dict,
    #     output_folder,
    #     input_metrics,
    #     ) #,
    #     # output_interval_metric_types,
    #     # peak_metric_types)

    plot_evolution_metrics_side_by_side(
        datasets=datasets,
        datetimes_list=datetimes_list,
        app_names=app_names,
        parent_dir=parent_dir,
        input_metrics=input_metrics,
        output_folder=output_folder,
        summary_statistics=('Mean', 'Median')
    )
