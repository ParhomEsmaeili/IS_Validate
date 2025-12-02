import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
from scipy.stats import norm, t, laplace, beta
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import warnings 

def plot_fitted_distributions(data, fit_params, metric, fit_type, threshold, line_name, output_folder):
    """
    Plot the histogram of the data and overlay the fitted distributions.
    fit_params: dict with keys 'gaussian', 'student', 'laplace' and values as tuples of fitted params.
    """
    x = np.linspace(min(data), max(data), 200)
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=20, density=True, alpha=0.5, label='Data')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({fit_type}) = {threshold:.2f}')
    if fit_type == 'gaussian':
        mu, sigma = fit_params['gaussian']
        plt.plot(x, norm.pdf(x, mu, sigma), label='Gaussian', lw=2)
    elif fit_type == 'student':
        df, loc, scale = fit_params['student']
        plt.plot(x, t.pdf(x, df, loc, scale), label="Student's t", lw=2)
    elif fit_type == 'laplace':
        loc, scale = fit_params['laplace']
        plt.plot(x, laplace.pdf(x, loc, scale), label='Laplace', lw=2)
    elif fit_type == 'beta':
        a, b, loc, scale = fit_params['beta']
        plt.plot(x, beta.pdf(x, a, b, loc, scale), label='Beta', lw=2)
    elif fit_type == 'quantile':
        percentile = fit_params['quantile'][0]
        # line_name = f'Q = {percentile:.2f}'
        plt.axvline(percentile, color='green', linestyle='--', label=line_name)
    else:
        raise ValueError(f"Unknown fit type: {fit_type}. Supported types are 'gaussian', 'student', 'laplace'.")
    plt.title(f"Fitted distributions for {metric}")
    plt.xlabel('Metric value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'fitted_oracle_distributions_{metric}_{fit_type}_{line_name}.png'))
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate number of interactions")
    parser.add_argument('--algo_results_path', type=str, required=True)
    parser.add_argument('--output_result_root', type=str, required=True)
    
    parser.add_argument('--metric', type=str, required=True, action='append', help='Reference metric for the threshold')
    # parser.add_argument('--nnunet_fraction', type=float, nargs='+', default=[0.9,1], help='Fraction of nnUNet performance for thresholding against')
    parser.add_argument('--reference_path', type=str, required=True)
    parser.add_argument('--nnunet_statistic', required=True, nargs='+', help='Statistical fit for selecting threshold.')
    parser.add_argument('--nnunet_bound', required=True, nargs='+', help='Bound for the nnUNet metric thresholding wrt statistic.')
    parser.add_argument('--infer_info', type=str, required=True, help='json string containing a dictionary describing the inference for determinining the reference columns')
    
    # parser.add_argument('--output_base_folder', type=str, required=True, help='Output folder for metrics')
    args = parser.parse_args()
    os.makedirs(args.output_result_root, exist_ok=True)
    # algo_input_folder = os.path.join(parent_dir, 'results', args.dataset_name, args.experiment_name, 'metrics')
    # output_folder = os.path.join(parent_dir, 'results_summary', args.dataset_name, args.experiment_name) 
    # nnunet_folder = os.path.join(parent_dir, 'results_summary', args.dataset_name, 'nnUNet_metrics')

    # edit_interaction_max = 100 

    std_bounded_statistics = ('gaussian', 'student', 'laplace', 'beta')
    quantile_statistics = ('quantile',)

    infer_info = json.loads(args.infer_info)
    init = infer_info.get('init')
    edit_interaction_max = infer_info.get('edit') 

    if init is None:
        raise ValueError("Inference info must contain 'init' key for initial metric column name.")
    if edit_interaction_max is None:
        raise ValueError("Inference info must contain 'edit' key for maximum number of edit interactions.")

    #now we read the csv files in the nnunet folder, and the reference algorithm as we want to calculate the 
    # metric for.
    algo_metrics_dfs = dict() 

    for metric in args.metric:
        #NOTE: Hardcoded the file name itself for now as we have only had the capability of running with binary semantic
        #segmentation tasks so far. 
        
        column_headers = ['Case_Name', init] + [f'Interactive Edit Iter {i + 1}' for i in range(edit_interaction_max)]
        algo_metrics_dfs[metric] = pd.read_csv(args.algo_results_path, skiprows=1, names=column_headers) 
        #NOTE: This is because the first row contained
        #a column with a header describing the case ID which was messing with the pandas read function. 

    #Now we read the nnUNet metrics csv files.
    nnunet_metrics_dfs = dict()
    for metric_id, metric in enumerate(args.metric):
        column_headers = ['Case_Name', 'Automatic Init']
        nnunet_metrics_dfs[metric] = pd.read_csv(args.reference_path, skiprows=1, names=column_headers) 
        #NOTE: This is because the first row contained
        #a column with a header describing the case ID which was messing with the pandas read function.

    noi_per_statistic = dict() #Number of interactions per fitting statistic.
    print(args.nnunet_statistic)
    for statistic_id, fit in enumerate(args.nnunet_statistic):
        if fit in std_bounded_statistics and (float(args.nnunet_bound[statistic_id]) < 0):
            raise ValueError(f"Invalid nnUNet bound used")
        if fit in quantile_statistics and (float(args.nnunet_bound[statistic_id]) < 0 or float(args.nnunet_bound[statistic_id]) > 1):
            raise ValueError(f"Invalid nnUNet bound used")

        #Now we calculate the number of interactions for each case.
        num_interactions_dict = dict()
    
        for metric in args.metric:
            algo_df = algo_metrics_dfs[metric]
            nnunet_df = nnunet_metrics_dfs[metric]

            # Ensure both DataFrames have the same cases, as a sanity check.
            if not set(algo_df['Case_Name']) == set(nnunet_df['Case_Name']):
                raise ValueError("Mismatch in case names between algorithm and nnUNet metrics.")

            #We calculate the threshold for the nnUNet metric based on the specified statistic and standard deviation bound.
            if fit == 'gaussian':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                # Fit a Gaussian distribution to the nnUNet metric values
                mu, sigma = norm.fit(nnunet_df['Automatic Init'])
                mean = mu
                std = sigma

                # mean = nnunet_df['Automatic Init'].mean()
                # std = nnunet_df['Automatic Init'].std()
                threshold = mean - float(args.nnunet_bound[statistic_id]) * std

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
                threshold = nnunet_df['Automatic Init'].quantile(float(args.nnunet_bound[statistic_id]))
                # threshold = percentile
                line_name = f'Q = {args.nnunet_bound[statistic_id]}'
            else:
                NotImplementedError(f"Unknown fit type: {fit}. Supported types are 'gaussian', 'student', 'laplace', 'beta', 'quantile'.")

            if threshold < 0:
                threshold = 0
                warnings.warn(f"Threshold for {metric} is negative after applying std bound. Setting to 0.")
            
            #Just plotting the fitted distributions for sanity check.
            fit_params = {}
            if fit == 'gaussian':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                fit_params['gaussian'] = (mean, std)
            elif fit == 'student':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                fit_params['student'] = (df, loc, scale)
            elif fit == 'laplace':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                fit_params['laplace'] = (loc, scale)
            elif fit == 'beta':
                raise NotImplementedError(f'Re-evaluating how to select a threshold with {fit} fitting.')
                fit_params['beta'] = (a, b, loc, scale)
            # Only plot once per metric (after all fits)
            elif fit == 'quantile':
                fit_params['quantile'] = (threshold,)
            else:
                raise NotImplementedError(f"Unknown fit type: {fit}. Supported types are 'gaussian', 'student', 'laplace', 'beta', 'quantile'.")

            data = nnunet_df['Automatic Init'].values
            plot_fitted_distributions(
                data,
                fit_params,
                metric,
                fit_type=fit,
                threshold=threshold,
                line_name=line_name,
                output_folder=args.output_result_root
            )




            # Calculate the number of interactions for each case
            
            num_interactions = []
            failure_cases = [] #To keep track of cases that do not manage to exceed the nnUNet metric.
            for case in algo_df['Case_Name']:
                algo_row = algo_df[algo_df['Case_Name'] == case]
                # nnunet_row = nnunet_df[nnunet_df['Case_Name'] == case]

                # Find the first interaction where the metric exceeds the nnUNet metric. 
                #We need to filter out the first column which is the case name.
                # if nnunet_row.empty or algo_row.empty:
                    # raise ValueError(f"Case {case} not found in one of the DataFrames.")
                
                bool_check = algo_row.iloc[0, 1:] > threshold 
                if np.any(bool_check):
                    #In this case it managed to exceed the nnunet metric.
                    interaction_count = np.argmax(bool_check) + 1 #bool is 1 so the max is 1, it finds us the first True.
                    num_interactions.append(interaction_count)
                    failure_cases.append(False)
                else:
                    #If there is no interaction that exceeds the nnUNet metric we set to maximum number of interactions.
                    interaction_count = edit_interaction_max + 1 # + 1 to account for the initial interaction.
                    num_interactions.append(interaction_count)
                    failure_cases.append(True)

            
            num_interactions_dict[metric] = {
                'cases': algo_df['Case_Name'].tolist(),
                'noi': num_interactions,
                'failure_cases': failure_cases,
            } 
            #NOTE: The failure cases are to keep track of cases that did not manage to exceed the nnUNet metric.

        # noi_per_statistic[fit] = num_interactions_dict
        noi_per_statistic[f'{fit}_{line_name}'] = num_interactions_dict

    # Build a DataFrame for all cases, showing NOIs and failures for each threshold and metric
    all_cases = set()
    # print(noi_per_statistic)
    for threshold_name in noi_per_statistic:
        for metric in noi_per_statistic[threshold_name]:
            all_cases.update(noi_per_statistic[threshold_name][metric]['cases'])

    all_cases = sorted(all_cases)
    rows = []

    for case in all_cases:
        row = {'Case_Name': case}
        for threshold_name in noi_per_statistic:
            for metric in noi_per_statistic[threshold_name]:
                cases = noi_per_statistic[threshold_name][metric]['cases']
                noi_list = noi_per_statistic[threshold_name][metric]['noi']
                fail_list = noi_per_statistic[threshold_name][metric]['failure_cases']
                if case in cases:
                    idx = cases.index(case)
                    # row[f'NOI_{metric}_thr_{threshold_name}_{float(args.nnunet_bound[id])}'] = noi_list[idx]
                    # row[f'Fail_{metric}_thr_{threshold_name}_{float(args.nnunet_bound[id])}'] = fail_list[idx]
                    row[f'NOI_{metric}_thr_{threshold_name}'] = noi_list[idx]
                    row[f'Fail_{metric}_thr_{threshold_name}'] = fail_list[idx]
                else:
                    # row[f'NOI_{metric}_thr_{threshold_name}_{float(args.nnunet_bound[id])}'] = None
                    # row[f'Fail_{metric}_thr_{threshold_name}_{float(args.nnunet_bound[id])}'] = None
                    row[f'NOI_{metric}_thr_{threshold_name}'] = None 
                    row[f'Fail_{metric}_thr_{threshold_name}'] = None
        rows.append(row)

    casewise_df = pd.DataFrame(rows)
    casewise_output_file = os.path.join(args.output_result_root, 'casewise_num_interactions_fitting.csv')
    casewise_df.to_csv(casewise_output_file, index=False)
    print(f"Casewise number of interactions saved to {casewise_output_file}")


    #Now for the summarisation of the number of interactions metrics. 

    #Now we will have to calculate the median and mean number of interactions, and normalise by the maximum number of 
    # interactions.

    summarised_noi = dict()
    for threshold_name in noi_per_statistic.keys():#args.nnunet_statistic):
        summarised_noi[threshold_name] = {}
        for metric in args.metric:
            noi_data = noi_per_statistic[threshold_name][metric]['noi']
            failure_cases = noi_per_statistic[threshold_name][metric]['failure_cases']

            # Calculate the median number of interactions
            median_noi = np.median(noi_data)
            #Calculate the mean number of interactions.
            mean_noi = np.mean(noi_data) 
            max_noi = edit_interaction_max + 1
            # Normalise by the maximum number of interactions
            normalised_median_noi = median_noi / max_noi
            normalised_mean_noi = mean_noi / max_noi 

            summarised_noi[threshold_name][metric] = {
                'median_noi': median_noi,
                'mean_noi': mean_noi,
                'normalised_median_noi': normalised_median_noi,
                'normalised_mean_noi': normalised_mean_noi, 
                'failure_cases': np.sum(failure_cases),  # Count of cases that failed to exceed nnUNet metric
                'failure_cases_fraction': 100 * (np.sum(failure_cases) / len(noi_data))  # Fraction of failure cases    
            }


    # Convert summarised_noi to a DataFrame
    summary_rows = []
    for threshold_name in summarised_noi:
        for metric in summarised_noi[threshold_name]:
            row = {
                'Threshold_Name': threshold_name,
                'Metric': metric,
                'Median_NOI': summarised_noi[threshold_name][metric]['median_noi'],
                'Normalised_Median_NOI': summarised_noi[threshold_name][metric]['normalised_median_noi'],
                'Normalised_Mean_NOI': summarised_noi[threshold_name][metric]['normalised_mean_noi'],
                'Failure_Cases': summarised_noi[threshold_name][metric]['failure_cases'],
                'Failure_Cases_Fraction': summarised_noi[threshold_name][metric]['failure_cases_fraction'],
            }
            summary_rows.append(row)

    summarised_df = pd.DataFrame(summary_rows)
    summarised_output_file = os.path.join(args.output_result_root, 'summarised_num_interactions_fitting.csv')
    summarised_df.to_csv(summarised_output_file, index=False)
    print(f"Summarised number of interactions saved to {summarised_output_file}")

    #Now we write the config used for this particular processing method in order to contextualise the result.
    config_output = {
        'algo_results_path': args.algo_results_path,
        'reference_path': args.reference_path,
        'metric': args.metric,
        'nnunet_statistic': args.nnunet_statistic,
        'nnunet_bound': args.nnunet_bound,
        'infer_info': infer_info,
    }
    config_output_file = os.path.join(args.output_result_root, 'num_of_interactions_fitting_config.json')
    with open(config_output_file, 'w') as f:
        json.dump(config_output, f, indent=4)
    print(f"Configuration saved to {config_output_file}")