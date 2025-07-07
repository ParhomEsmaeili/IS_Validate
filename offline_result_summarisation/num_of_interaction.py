import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.stats import norm, t, laplace, beta
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import warnings 

def plot_fitted_distributions(data, fit_params, metric, fit_type, threshold, output_folder):
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
        plt.axvline(percentile, color='green', linestyle='--', label=f'LQ = {percentile:.2f}')
    else:
        raise ValueError(f"Unknown fit type: {fit_type}. Supported types are 'gaussian', 'student', 'laplace'.")
    plt.title(f"Fitted distributions for {metric}")
    plt.xlabel('Metric value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'fitted_distributions_{metric}_{fit_type}.png'))
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate number of interactions")
    parser.add_argument('--datetime', type=str, default='20250614_025617', help='Path to the segmentation folder')
    parser.add_argument('--metrics', type=str, nargs='+', help='Reference metric for the threshold', default=['Dice', 'NSD'])
    # parser.add_argument('--nnunet_fraction', type=float, nargs='+', default=[0.9,1], help='Fraction of nnUNet performance for thresholding against')
    parser.add_argument('--nnunet_statistic', type=str, nargs='+', default=('gaussian', 'student', 'laplace', 'beta', 'quantile'), help='Statistical fit for selecting threshold.')
    parser.add_argument('--nnunet_std_bound', type=float, default=1.0, help='Standard deviation below mean for the nnUNet metric thresholding.')
    parser.add_argument('--dataset_name', type=str, default='Dataset007_Pancreas', help='Name of the dataset')
    # parser.add_argument('--output_base_folder', type=str, required=True, help='Output folder for metrics')
    args = parser.parse_args()

    algo_input_folder = os.path.join(parent_dir, 'results', args.dataset_name, args.datetime, 'metrics')
    output_folder = os.path.join(parent_dir, 'results_summary', args.dataset_name, args.datetime) 
    nnunet_folder = os.path.join(parent_dir, 'results_summary', args.dataset_name, 'nnUNet_metrics')

    edit_interaction_max = 100 
    #now we read the csv files in the nnunet folder, and the reference algorithm as we want to calculate the 
    # metric for.
    algo_metrics_dfs = dict() 

    for metric in args.metrics:
        #NOTE: Hardcoded the file name itself for now as we have only had the capability of running with binary semantic
        #segmentation tasks so far. 
        input_file = os.path.join(algo_input_folder, metric, f'cross_class_scores.csv')
        column_headers = ['Case_Name', 'Interactive Init'] + [f'Interactive Edit Iter {i + 1}' for i in range(edit_interaction_max)]
        algo_metrics_dfs[metric] = pd.read_csv(input_file, skiprows=1, names=column_headers) 
        #NOTE: This is because the first row contained
        #a column with a header describing the case ID which was messing with the pandas read function. 

    #Now we read the nnUNet metrics csv files.
    nnunet_metrics_dfs = dict()
    for metric in args.metrics:
        #NOTE: Hardcoded the file name itself for now as we have only had the capability of running with binary semantic
        #segmentation tasks so far. 
        input_file = os.path.join(nnunet_folder, metric, f'cross_class_scores.csv')
        column_headers = ['Case_Name', 'Automatic Init']
        nnunet_metrics_dfs[metric] = pd.read_csv(input_file, skiprows=1, names=column_headers) 
        #NOTE: This is because the first row contained
        #a column with a header describing the case ID which was messing with the pandas read function.

    noi_per_statistic = dict() #Number of interactions per fitting statistic.
    for fit in args.nnunet_statistic:
        if args.nnunet_std_bound < 0:
            raise ValueError(f"Invalid nnUNet std bound: {args.nnunet_std_bound}. It should be greater than 0")

        #Now we calculate the number of interactions for each case.
        num_interactions_dict = dict()
    
        for metric in args.metrics:
            algo_df = algo_metrics_dfs[metric]
            nnunet_df = nnunet_metrics_dfs[metric]

            # Ensure both DataFrames have the same cases, as a sanity check.
            if not set(algo_df['Case_Name']) == set(nnunet_df['Case_Name']):
                raise ValueError("Mismatch in case names between algorithm and nnUNet metrics.")

            #We calculate the threshold for the nnUNet metric based on the specified statistic and standard deviation bound.
            if fit == 'gaussian':
                # Fit a Gaussian distribution to the nnUNet metric values
                mu, sigma = norm.fit(nnunet_df['Automatic Init'])
                mean = mu
                std = sigma

                # mean = nnunet_df['Automatic Init'].mean()
                # std = nnunet_df['Automatic Init'].std()
                threshold = mean - args.nnunet_std_bound * std

            elif fit == 'student':
                # Fit a Student's t-distribution to the nnUNet metric values
                # mean = nnunet_df['Automatic Init'].mean()
                # std = nnunet_df['Automatic Init'].std()
                df, loc, scale = t.fit(nnunet_df['Automatic Init'])
                mean = t.mean(df, loc=loc, scale=scale)
                std = t.std(df, loc=loc, scale=scale)
                threshold = mean - args.nnunet_std_bound * std

            elif fit == 'laplace':
                # Fit a Laplace distribution to the nnUNet metric values
                loc, scale = laplace.fit(nnunet_df['Automatic Init'])
                mean = laplace.mean(loc=loc, scale=scale)
                std = laplace.std(loc=loc, scale=scale) 
                threshold = mean - args.nnunet_std_bound * std

            elif fit == 'beta':
                #Fit a beta distribution to the nnUNet metric values as it is bounded between 0 and 1
                #and flexible to asymmetric distributions. 
                
                #We clip the values to the open interval (0, 1) to avoid issues with beta fitting.
                #We use a very very small epsilon.
                eps = 1e-6
                data = np.clip(data, eps, 1 - eps)
                a, b, loc, scale = beta.fit(data, floc=0, fscale=1) 
                mean = beta.mean(a, b, loc=loc, scale=scale)
                std = beta.std(a, b, loc=loc, scale=scale)
                threshold = mean - args.nnunet_std_bound * std

            elif fit == 'quantile':
                #We use a quantile based thresholding for this... more ad-hoc.
                percentile = nnunet_df['Automatic Init'].quantile(0.25) #We go for the lower quartile.
                threshold = percentile 

            if threshold < 0:
                threshold = 0 
                warnings.warn(f"Threshold for {metric} is negative after applying std bound. Setting to 0.")
            
            #Just plotting the fitted distributions for sanity check.
            fit_params = {}
            if fit == 'gaussian':
                fit_params['gaussian'] = (mean, std)
            if fit == 'student':
                fit_params['student'] = (df, loc, scale)
            if fit == 'laplace':
                fit_params['laplace'] = (loc, scale)
            if fit == 'beta':
                fit_params['beta'] = (a, b, loc, scale)
            # Only plot once per metric (after all fits)
            if fit == 'quantile':
                fit_params['quantile'] = (percentile,)
            
            data = nnunet_df['Automatic Init'].values
            plot_fitted_distributions(
                data,
                fit_params,
                metric,
                fit_type=fit,
                threshold=threshold,
                output_folder=output_folder
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

        noi_per_statistic[fit] = num_interactions_dict 
    
    # Build a DataFrame for all cases, showing NOIs and failures for each threshold and metric
    all_cases = set()
    for fit in noi_per_statistic:
        for metric in noi_per_statistic[fit]:
            all_cases.update(noi_per_statistic[fit][metric]['cases'])

    all_cases = sorted(all_cases)
    rows = []

    for case in all_cases:
        row = {'Case_Name': case}
        for fit in noi_per_statistic:
            for metric in noi_per_statistic[fit]:
                cases = noi_per_statistic[fit][metric]['cases']
                noi_list = noi_per_statistic[fit][metric]['noi']
                fail_list = noi_per_statistic[fit][metric]['failure_cases']
                if case in cases:
                    idx = cases.index(case)
                    row[f'NOI_{metric}_thr_{fit}_{args.nnunet_std_bound}'] = noi_list[idx]
                    row[f'Fail_{metric}_thr_{fit}_{args.nnunet_std_bound}'] = fail_list[idx]
                else:
                    row[f'NOI_{metric}_thr_{fit}_{args.nnunet_std_bound}'] = None
                    row[f'Fail_{metric}_thr_{fit}_{args.nnunet_std_bound}'] = None
        rows.append(row)

    casewise_df = pd.DataFrame(rows)
    casewise_output_file = os.path.join(output_folder, 'casewise_num_interactions_fitting.csv')
    casewise_df.to_csv(casewise_output_file, index=False)
    print(f"Casewise number of interactions saved to {casewise_output_file}")



    #Now for the summarisation of the number of interactions metrics. 

    #Now we will have to calculate the median and mean number of interactions, and normalise by the maximum number of 
    # interactions.

    summarised_noi = dict()
    for fit in args.nnunet_statistic:
        summarised_noi[fit] = {}
        for metric in args.metrics:
            noi_data = noi_per_statistic[fit][metric]['noi']
            failure_cases = noi_per_statistic[fit][metric]['failure_cases']
            
            # Calculate the median number of interactions
            median_noi = np.median(noi_data)
            #Calculate the mean number of interactions.
            mean_noi = np.mean(noi_data) 
            max_noi = edit_interaction_max + 1
            # Normalise by the maximum number of interactions
            normalised_median_noi = median_noi / max_noi
            normalised_mean_noi = mean_noi / max_noi 

            summarised_noi[fit][metric] = {
                'median_noi': median_noi,
                'normalised_median_noi': normalised_median_noi,
                'normalised_mean_noi': normalised_mean_noi, 
                'failure_cases': np.sum(failure_cases),  # Count of cases that failed to exceed nnUNet metric
                'failure_cases_fraction': 100 * (np.sum(failure_cases) / len(noi_data))  # Fraction of failure cases    
            }


    # Convert summarised_noi to a DataFrame
    summary_rows = []
    for fit in summarised_noi:
        for metric in summarised_noi[fit]:
            row = {
                'Fit': fit,
                'Metric': metric,
                'Median_NOI': summarised_noi[fit][metric]['median_noi'],
                'Normalised_Median_NOI': summarised_noi[fit][metric]['normalised_median_noi'],
                'Normalised_Mean_NOI': summarised_noi[fit][metric]['normalised_mean_noi'],
                'Failure_Cases': summarised_noi[fit][metric]['failure_cases'],
                'Failure_Cases_Fraction': summarised_noi[fit][metric]['failure_cases_fraction'],
            }
            summary_rows.append(row)

    summarised_df = pd.DataFrame(summary_rows)
    summarised_output_file = os.path.join(output_folder, 'summarised_num_interactions_fitting.csv')
    summarised_df.to_csv(summarised_output_file, index=False)
    print(f"Summarised number of interactions saved to {summarised_output_file}")

    # #Now we write the summarised results to a csv file.
    
    # summarised_output_file = os.path.join(output_folder, 'summarised_num_interactions_fitting.csv')

    # summarised_df.to_csv(summarised_output_file)
    # print(f"Summarised number of interactions saved to {summarised_output_file}")
    