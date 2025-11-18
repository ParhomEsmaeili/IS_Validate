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
        input_metrics,
        infer_info,
        filename,
        interval_metric_epochs): 
    """
    Summarises standard metrics from a CSV file and writes the results to a new CSV file.
    
    Parameters:
    input_folder (str): Path to the folder containing the folders with CSV files with the metrics.
    output_folder (str): Path to the output folder where summarised metrics will be saved.
    output_filename (str): Name of the output CSV file. 
    input_metrics (tuple): Tuple of metrics to be summarised.
    filename (str): Filename of the CSV file containing the metrics.
    interval_metric_epochs (tuple): Tuple of epochs for interval based metrics.
    # output_interval_metric_types (dict): Dictionary defining the types of interval metrics to be summarised.
    # peak_metric_types (dict): Dictionary defining the types of "peak" metrics to be summarised.

    """
    init = infer_info.get('init')
    edit_interaction_max = infer_info.get('edit') 
    # print(output_folder)
    # Read the input CSV file
    metrics_dfs = dict()
    for metric in input_metrics:
        #NOTE: Hardcoded the file name itself for now as we have only had the capability of running with binary semantic
        #segmentation tasks so far. 
        input_file = os.path.join(input_folder, metric, filename)
        # print(input_file)
        column_headers = ['Case_Name', init] + [f'Interactive Edit Iter {i + 1}' for i in range(edit_interaction_max)]
        metrics_dfs[metric] = pd.read_csv(input_file, skiprows=1, names=column_headers) #NOTE: This is because the first row contained
        #a column with a header describing the case ID which was messing with the pandas read function. 

    #Now we do the summarisation of the metrics. 
    metric_summaries = dict()
    for metric in input_metrics:
        df = metrics_dfs[metric]
        columns_to_summarise = [col for col in df.columns if col != 'Case_Name']  # Exclude 'Case_Name

        summary = pd.DataFrame({
            f'{metric}_mean': df[columns_to_summarise].mean(),
            f'{metric}_median': df[columns_to_summarise].median(),
            f'{metric}_std': df[columns_to_summarise].std(),
            f'{metric}_lq': df[columns_to_summarise].quantile(0.25),
            f'{metric}_uq': df[columns_to_summarise].quantile(0.75)
        })
        metric_summaries[metric] = summary 
    
    #Now we need to calculate the peak metrics. 
    peak_summaries = dict()
    for metric in input_metrics:
        df = metrics_dfs[metric]
        columns_to_summarise = [col for col in df.columns if col != 'Case_Name']  # Exclude 'Case_Name

        peak_scores = df[columns_to_summarise].max(axis=1)  # Get the peak scores for each case. 
        
        peak_summary = pd.DataFrame({
            f'{metric}_peak_mean': [peak_scores.mean()],
            f'{metric}_peak_median': [peak_scores.median()],
            f'{metric}_peak_std': [peak_scores.std()],
            f'{metric}_peak_lq': [peak_scores.quantile(0.25)],
            f'{metric}_peak_uq': [peak_scores.quantile(0.75)]
        })
        peak_summaries[metric] = peak_summary

    #Now we need to calculate the normalised AUC metrics. 
    auc_summaries = dict()
    for metric in input_metrics:
        df = metrics_dfs[metric]
        columns_to_summarise = [col for col in df.columns if col != 'Case_Name']
        auc_scores = []
        for row in df[columns_to_summarise].iterrows():
            scores=row[1].values
            #Calculate the normalised AUC.

            #We we will use the trapezoidal rule to calculate this. Why is this reasonable? Because we don't want to
            #over-smooth or overshoot any representation of the area under the curve. There is no interpretation for
            #scores between the iterations, so piece-wise linear interpolation is a reasonable approach (as opposed to
            # using polynomial interpolation or something to that effect). 

            #We essentially want to penalise any cases where the score is not monotonically increasing.

            normalised_auc = integrate.trapezoid(scores)/(len(scores) - 1)  
            #Normalised AUC calculation
            auc_scores.append(normalised_auc)
        
        auc_scores = np.array(auc_scores)
        auc_summary = pd.DataFrame({
            f'{metric}_auc_mean': [auc_scores.mean()],
            f'{metric}_auc_median': [np.median(auc_scores)],
            f'{metric}_auc_std': [np.std(auc_scores)],
            f'{metric}_auc_lq': [np.quantile(auc_scores, 0.25)],
            f'{metric}_auc_uq': [np.quantile(auc_scores, 0.75)]
        })
        auc_summaries[metric] = auc_summary
    
    
    #Now writing the summarised metrics to a new CSV file
    # print(metric_summaries)
    # Concatenate all metric summaries vertically for iteration-wise stats
    all_iteration_summaries = pd.concat([metric_summaries[metric] for metric in input_metrics], axis=1)
    all_iteration_summaries.to_csv(os.path.join(output_folder, 'all_iteration_summaries.csv'))

    # Concatenate all peak and auc summaries
    all_peak_summaries = pd.concat([peak_summaries[metric] for metric in input_metrics], axis=1)
    all_peak_summaries.to_csv(os.path.join(output_folder, 'all_peak_summaries.csv'))

    all_auc_summaries = pd.concat([auc_summaries[metric] for metric in input_metrics], axis=1)
    all_auc_summaries.to_csv(os.path.join(output_folder, 'all_auc_summaries.csv'))



def set_parse():
    parser = argparse.ArgumentParser(description='Summarise standard metrics from a CSV file.')
    parser.add_argument('--algorithm_results_root', type=str, required=True, help='Root path to the algorithm results')
    parser.add_argument('--output_folder_root', type=str, required=True, help='Root path to the output folder for results summarisation')
    parser.add_argument('--infer_info', type=str, required=True, help='JSON string containing inference information such as init and edit interaction max')
    parser.add_argument('--filename', type=str, required=True, help='Filename of the CSV file containing the metrics.')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = set_parse()
   
    input_results_root=  args.algorithm_results_root
    output_folder = args.output_folder_root
    filename = args.filename

    infer_info = json.loads(args.infer_info)
    
    os.makedirs(output_folder, exist_ok=True) 
    input_metrics = ('Dice', 'NSD')
    interval_metric_epochs = ('Init', 1, 5, 50, 100) 
    # print(output_folder)
    summarise_standard_metrics(
        input_results_root, 
        output_folder, 
        input_metrics,
        infer_info,
        filename,
        interval_metric_epochs)
    