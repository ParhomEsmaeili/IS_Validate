import numpy as np
import csv
import pandas as pd 
import os 
import sys
import json
from scipy import integrate 
import argparse
# Add the parent directory to know how to find the saved metrics, very hacky.
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def summarise_standard_metrics(
        input_folder, 
        output_folder, 
        aggregated_run_roots,
        filename,
        ): 
    """
    Summarises standard metrics from a CSV file and writes the results to a new CSV file.
    
    Parameters:
    input_folder (str): Path to the folder containing the folders with CSV files with the metrics.
    output_folder (str): Path to the output folder where summarised metrics will be saved.
    output_filename (str): Name of the output CSV file. 
    filename (str): Filename of the CSV file containing the metrics
    aggregated_run_roots (list): List of original run roots that were used to generate the aggregated results. This is used for sanity checking and variability estimation purposes.
    
    """



    if aggregated_run_roots is not None: #THIS IS ONLY FOR SANITY CHECKING!!!!!!!!!!!!1
        assert len(aggregated_run_roots) > 0, "Aggregated run roots list is empty. Please provide valid paths to the original runs that were used to generate the aggregated results for sanity checking and variability estimation purposes."
    
        #Now lets extract the reference metrics for the original runs that were used to generate the aggregated results, and calculate the trajectory AUC for each of these runs to sanity check that the trajectory AUC for the aggregated run is within the variability bounds of the trajectory AUCs for the original runs.

        reference_metrics_dfs = dict()
        #Here we will sanity check:
        for idx,path in enumerate(aggregated_run_roots):

            input_file = os.path.join(path, filename)
            
            reference_metrics_dfs[idx] = pd.read_csv(input_file) 

        # sanity_check_trajectory_auc_means = dict()
        trajectory_auc_variability_metrics = dict()
        #Lets iterate by column by metrics and calculate the trajectory AUC for each of the original runs.
        
        for col_name in reference_metrics_dfs[0].keys():
            # col_name is the column name, col_data is a pandas Series
        # for metric in input_metrics:
            trajectory_auc_values = []
            #Lets grab the column, and create a pandas dataframe across the different runs for easier processing.
            for key, df in reference_metrics_dfs.items():
                assert f'{col_name}' in df.columns, f"Expected column '{col_name}' not found in the input CSV file for original run at path '{aggregated_run_roots[key]}'."
                trajectory_auc_values.append(integrate.trapezoid(df[f'{col_name}']) / (len(df[f'{col_name}']) - 1))
            trajectory_auc_mean = np.mean(trajectory_auc_values)
            trajectory_auc_min = np.min(trajectory_auc_values)
            trajectory_auc_max = np.max(trajectory_auc_values)
            trajectory_auc_std = np.std(trajectory_auc_values)
    

            #Now lets store this.
            trajectory_auc_variability_metrics[col_name] = {
                'mean': trajectory_auc_mean,
                'min': trajectory_auc_min,
                'max': trajectory_auc_max,
                'std': trajectory_auc_std,
            }
            # print(f"Sanity check - mean of trajectory AUCs for metric '{metric}' across original runs: {trajectory_auc_mean}")
    
    
    metrics_dfs=pd.read_csv(os.path.join(input_folder, filename)) #PRESUMMARISED! 
    #Now we need to calculate the normalised trajectory AUC metric.
    trajectory_auc = dict()
    # trajectory_auc_summaries = dict()
    for col_name in metrics_dfs.columns:
        col = metrics_dfs[col_name]
        auc_score = integrate.trapezoid(col) / (len(col) - 1)
        # for idx, row in enumerate(df[columns_to_summarise].iterrows()):
        #     scores=row[1].values
        #     #Calculate the normalised AUC.

        #     #We we will use the trapezoidal rule to calculate this. Why is this reasonable? Because we don't want to
        #     #over-smooth or overshoot any representation of the area under the curve. There is no interpretation for
        #     #scores between the iterations, so piece-wise linear interpolation is a reasonable approach (as opposed to
        #     # using polynomial interpolation or something to that effect). 

        #     #We essentially want to penalise any cases where the score is not monotonically increasing.

        #     normalised_auc = integrate.trapezoid(scores)/(len(scores) - 1)  
        #     #Normalised AUC calculation
        #     auc_scores.append({'pseudotime': df.iloc[idx]['pseudotime'], f'{metric}_auc_scores': normalised_auc})
        
        # aucs[metric] = pd.DataFrame(
        #     auc_scores
        # )
        # auc_values = np.array([item[f'{metric}_auc_scores'] for item in auc_scores])
        # auc_summary = pd.DataFrame({
        #     f'pseudotime_{metric}_auc_mean': [auc_values.mean()],
        #     f'pseudotime_{metric}_auc_median': [np.median(auc_values)],
        #     f'pseudotime_{metric}_auc_std': [np.std(auc_values)],
        #     f'pseudotime_{metric}_auc_lq': [np.quantile(auc_values, 0.25)],
        #     f'pseudotime_{metric}_auc_uq': [np.quantile(auc_values, 0.75)],
        # })
        # auc_summaries[metric] = auc_summary
        trajectory_auc[f'{col_name}_trajectory_auc'] = pd.DataFrame({
            f'{col_name}_trajectory_auc': [auc_score]
        })

        #Lets compare these values to the trajectory AUC values for the original runs that were used to generate the aggregated results for sanity checking and variability estimation purposes.
        if aggregated_run_roots is not None:
            agg_value = trajectory_auc[f'{col_name}_trajectory_auc'][f'{col_name}_trajectory_auc'].iloc[0]
            mean_value = trajectory_auc_variability_metrics[col_name]['mean']
            assert np.isclose(agg_value, mean_value, atol=1e-6), f"Sanity check failed for metric '{col_name}': Trajectory AUC for aggregated run ({agg_value}) does not match mean trajectory AUC across original runs ({mean_value}). Please check the input files and calculations for consistency."

            #We will write the variability metrics to a json file for easier access later on.
            #Lets turn it into one dataframe for easier saving to csv.
            variability_df = pd.DataFrame({
                f'{col_name}_trajectory_auc_mean': [trajectory_auc_variability_metrics[col_name]['mean']],
                f'{col_name}_trajectory_auc_min': [trajectory_auc_variability_metrics[col_name]['min']],
                f'{col_name}_trajectory_auc_max': [trajectory_auc_variability_metrics[col_name]['max']],
                f'{col_name}_trajectory_auc_std': [trajectory_auc_variability_metrics[col_name]['std']],
            })
            os.makedirs(os.path.join(output_folder, 'variability'), exist_ok=True)
            variability_df.to_csv(os.path.join(output_folder, 'variability',f'{col_name}_trajectory_auc_variability.csv'))
    
    #Now we will write the trajectory AUC summaries to a csv file.
    all_trajectory_aucs = pd.concat([trajectory_auc[col_name] for col_name in trajectory_auc.keys()], axis=1)
    all_trajectory_aucs.to_csv(os.path.join(output_folder,'all_trajectory_aucs.csv'))

    
def set_parse():
    parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
    parser.add_argument('--aggregated_experiment_roots', nargs='+', required=False, help='List of original experiment names that were used to generate the aggregated results. This is used for sanity checking and variability estimation purposes.')
    parser.add_argument('--algorithm_results_root', type=str, required=True, help='Root path to the algorithm results')
    parser.add_argument('--output_folder_root', type=str, required=True, help='Root path to the output folder for results summarisation')
    parser.add_argument('--filename', type=str, required=True, help='Filename of the CSV file containing the metrics.')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = set_parse()
   
    input_results_root=  args.algorithm_results_root
    aggregated_experiment_roots = args.aggregated_experiment_roots
    if aggregated_experiment_roots is not None:
        assert all(os.path.exists(exp_root) for exp_root in aggregated_experiment_roots), "One or more of the provided aggregated experiment roots do not exist. Please provide valid paths for sanity checking and variability estimation purposes."
    output_folder = args.output_folder_root
    filename = args.filename

    os.makedirs(output_folder, exist_ok=True) 
    summarise_standard_metrics(
        input_results_root,
        output_folder, 
        aggregated_experiment_roots,
        filename)
