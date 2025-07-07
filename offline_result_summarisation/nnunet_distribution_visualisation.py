import pandas as pd
import matplotlib.pyplot as plt
import os 

# Dataset name variable
base_path = '/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/results_summary'
dataset_name = 'Dataset007_Pancreas'

# Replace 'your_file.csv' with your CSV file path
csv_path = os.path.join(base_path, dataset_name, 'nnUNet_metrics', 'Dice', 'cross_class_scores.csv')

# Select a column to visualize
# Replace 'column_name' with the actual column you want to plot
column_to_plot = 'Automatic Init'

def plot_histogram(csv_path, column_to_plot, dataset_name):
    # Read the CSV file
    column_headers = ['Automatic Init']
    df = pd.read_csv(csv_path, skiprows=1, names=column_headers) 
    # Plot histogram
    plt.figure(figsize=(8, 6))
    df[column_to_plot].hist(bins=30, edgecolor='black')
    plt.xlabel(column_to_plot)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_to_plot} ({dataset_name})')
    plt.grid(True)
    plt.show()
    print('Now lets look.')
if __name__ == "__main__":
    plot_histogram(csv_path, column_to_plot, dataset_name)