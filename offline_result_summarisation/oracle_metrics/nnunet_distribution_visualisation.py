import pandas as pd
import matplotlib.pyplot as plt
import os 
import argparse 

def plot_histogram(csv_path, column_to_plot, dataset_name, output_path):
    # Read the CSV file
    column_headers = [column_to_plot]
    df = pd.read_csv(csv_path, skiprows=1, names=column_headers) 
    # Plot histogram
    plt.figure(figsize=(8, 6))
    df[column_to_plot].hist(bins=30, edgecolor='black')
    plt.xlabel(column_to_plot)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_to_plot} ({dataset_name})')
    plt.grid(True)
    # plt.show()
    plt.savefig(output_path)
    print('Now lets look.')
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
#     base_path = '/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/results_summary'
# dataset_name = 'Dataset007_Pancreas'
    argparser.add_argument('--root_input_path', type=str, required=True)
    argparser.add_argument('--root_output_path', type=str, required=True)
    argparser.add_argument('--dataset_name', type=str, required=True)
    argparser.add_argument('--reference_metric', type=str, default='Dice', required=True)
    argparser.add_argument('--reference_file', type=str, default='cross_class_scores.csv',required=True)
    argparser.add_argument('--reference_column', type=str, default='Automatic Init', required=True)
    args = argparser.parse_args()
    
    input_csv_path = os.path.join(
        args.root_input_path, 
        args.dataset_name, 
        'nnUNet_metrics', 
        args.reference_metric, 
        args.reference_file, #'cross_class_scores.csv'
    )
    output_path = os.path.join(
        args.root_output_path,
        args.dataset_name,
        'nnunet_' + args.reference_metric.replace(" ", "_").lower() + '_histogram.png'
    )
    os.makedirs(os.path.join(args.root_output_path, args.dataset_name), exist_ok=True)

    plot_histogram(input_csv_path, args.reference_column, args.dataset_name, output_path=output_path)