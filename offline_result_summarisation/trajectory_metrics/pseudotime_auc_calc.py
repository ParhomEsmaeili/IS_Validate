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
        filename,
        ): 
    """
    Summarises standard metrics from a CSV file and writes the results to a new CSV file.
    
    Parameters:
    input_folder (str): Path to the folder containing the folders with CSV files with the metrics.
    output_folder (str): Path to the output folder where summarised metrics will be saved.
    filename (str): Filename of the CSV file containing the metrics
    """
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

    #Now we will write the trajectory AUC summaries to a csv file.
    all_trajectory_aucs = pd.concat([trajectory_auc[col_name] for col_name in trajectory_auc.keys()], axis=1)
    all_trajectory_aucs.to_csv(os.path.join(output_folder,'all_trajectory_aucs.csv'))

    
def set_parse():
    parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
    parser.add_argument('--algorithm_results_root', type=str, required=True, help='Root path to the algorithm results')
    parser.add_argument('--output_folder_root', type=str, required=True, help='Root path to the output folder for results summarisation')
    parser.add_argument('--filename', type=str, required=True, help='Filename of the CSV file containing the metrics.')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = set_parse()
   
    input_results_root = args.algorithm_results_root
    output_folder = args.output_folder_root
    filename = args.filename

    os.makedirs(output_folder, exist_ok=True) 
    summarise_standard_metrics(
        input_results_root,
        output_folder, 
        filename)
