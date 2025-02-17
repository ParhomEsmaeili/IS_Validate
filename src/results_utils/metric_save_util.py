import logging
import os
import time
import torch
import shutil
import numpy as np
# import pandas

def save_dice():
    pass 

def initialise_csvs(
    metric_types: tuple[str],
    include_background_metric: bool,
    per_class_scores: bool
):
    pass 

def save_csvs(
    patient_name: str,
    csv_paths: dict[str, dict],
    tracked_metrics: dict, 
    metrics_configs: dict,  
    class_configs_dict: dict 
):
    '''
    Function which saves the metric outcomes for each patient to the corresponding csv files.

    Inputs: 

    patient_name: A string denoting the corresponding filename under consideration.
    
    csv_paths: A dictionary with structure: metric_type: tuple()
    
    tracked_metrics: A thrice nested dictionary containing the metrics across the iterative refinement process separated by
    iter name, and then by metric type, and then a dictionary split into "cross_class_scores" (a torch tensor size 1) 
    and "per_class_scores" a dict separated by the class label for each per class score being saved (each is a torch tensor size 1)
    
    class_configs_dict: A dictionary mapping class labels to class integer codes.
    
    '''
    if not set(csv_paths) == set(metrics_configs):
        raise Exception('There is not a csv directory containing initialised csvs for every metric that needs to be saved')
    

# def save_csv(args, logger, patient_list,
#              loss, loss_nsd,
#              ):
#     save_predict_dir = os.path.join(args.save_base_dir, 'csv_file')
#     if not os.path.exists(save_predict_dir):
#         os.makedirs(save_predict_dir)

#     df_dict = {'patient': patient_list,
#                'dice': loss,
#                'nsd': loss_nsd,
#                }

#     df = pandas.DataFrame(df_dict)
#     df.to_csv(os.path.join(save_predict_dir, 'prompt_' + str(args.num_prompts)
#                            + '_' + str(args.save_name) + '.csv'), index=False)
#     logger.info("- CSV saved")

