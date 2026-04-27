import json
import logging
import os
from os.path import dirname as up
import sys
sys.path.append(up(up(up(os.path.abspath(__file__)))))
from src.general_utils.dict_utils import extractor, sort_infer_calls
import csv
import re

# import pandas

def init_metric_csvs(
        dir_path: str | dict[str, str], 
        metric_configs: dict, 
        config_labels_dict:dict):
    '''
    This is a function which initialises the csvs being used for a given metric type. Performs only three operations:

    1) Initialises a cross-class saver (this will always be required).
    2) Optionally initialises per-class savers. 
    3) If per-class savers/scores are disabled, then the dict for this will be a None instead.
        If not, for each class that is not required (e.g. include_background = False) it will place a NoneType for that
        given class. 

    
    input:

    dir_path: The directory in which all of the corresponding csvs will be initialised and placed.
    
    metric_configs: The metric configuration for the given metric type under consideration (the metric NAME) will 
    not be included in the csv filename.
    
    config_labels_dict: The dictionary mapping class labels to class integer codes.

    returns: 
    
    Dictionary containing two fields:

    cross_class_score: str denoting path to the full abspath to the csv.
    per_class_scores: Optional dict[str, Union[str, NoneType]] or NoneType denoting the corresponding abspaths to 
    the csv files.

    '''
    #Initialising the dictionary of paths.
    metric_paths_dict = dict()
    
    #Initialising the first row (header: contains patient name, empty space after that because each metric has their
    #own specifics wrt which iter is being presented etc.)

    header = ['Case Name']

    #Checking whether we have at least one valid class otherwise everything will break.
    valid_classes = tuple(config_labels_dict) if metric_configs['include_background_metric'] else tuple({key for key in config_labels_dict.keys() if key.title() != "Background"})
    if len(valid_classes) < 1:
        raise Exception('At least one valid class is required!')

    #Initialising the cross_class ALWAYS:

    cross_class_path = os.path.join(dir_path, 'cross_class_scores.csv')
    with open(cross_class_path,'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    metric_paths_dict['cross_class_scores'] = cross_class_path

    #Initialising the per class scores.

    if not metric_configs['include_per_class_scores']:
        #In this case, just place a NoneType in the place for the per class scores.
        metric_paths_dict['per_class_scores'] = None 
    else:
        placeholder = dict()

        for class_lb in config_labels_dict:
            #If class label is in the valid ones being saved then init and provide the path.
            if class_lb in valid_classes:
                filepath = os.path.join(dir_path, f'{class_lb}_class_scores.csv')
                
                with open(filepath,'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                placeholder[class_lb] = filepath 
            #Else: set the path to None 
            else:
                placeholder[class_lb] = None 

        metric_paths_dict['per_class_scores'] = placeholder 
    
    return metric_paths_dict

def init_all_csvs(
    metrics_save_dir: str,
    metric_configs: dict,
    config_labels_dict: dict
    ):
    '''
    Function which initialises the 1) metric type subfolders, 2) csvs for saving the metrics. 

    Returns the paths for each of the csvs in a dictionary format: metric_type : {
    
    'cross_class_scores':path,
    'per_class_scores':
        {'background_class_scores':path OR NoneType (NoneType if the specified class is not being saved/used in metrics)
        etc.
        } OR

        NoneType (if per class scores are not desired.)
    }    
    
    Inputs: 
    
    metrics_save_dir: Absolute path to the directory containing all the subdirs for each metric type.
    
    metrics_configs: A nested dictionary containing the configs for the metrics with general structure:
    metric_type [metric_specific_config]. Within each metric_specific_config is a set of metric configs.

    Of note are fields that state whether:
    1) per_class_scores are provided and
    2) and whether background is included in reporting the metrics (on a cross-class and/or per-class basis).

    config_labels_dict: A dictionary containing the class-label - class-integer code mapping.

    returns:
    A nested dictionary containing the paths to the csvs for each metric type,
    and if applicable, for each subgroup within the metric type (e.g., if there
    is a parameterisation which is not fixed/single).)
    '''
    
    complete_paths_dicts = dict() 

    for metric_type, config in metric_configs.items():
        #creating the subdir for the metric type
    
        metric_subdir_abspath = os.path.join(metrics_save_dir, metric_type)
        os.makedirs(metric_subdir_abspath, exist_ok=False) #False, folder should not exist! 
        if 'multiple_parameter_values' in config:
            #for each parameterisation we want to also create a subfolder.
            parameter_name = config['multiple_parameter_values']
            parameter_values = config[parameter_name]
            #We will add the parameter value to the metric type in the paths dict for downstream use
            metric_paths_dict = dict()
            for parameter_idx, parameter_value in parameter_values.items():
                metric_final_abspath = os.path.join(
                    metric_subdir_abspath,
                    f'{parameter_name}_idx_{parameter_idx}'
                    )
                os.makedirs(metric_final_abspath, exist_ok=False) #False, folder should not exist!
                #Save a json which contains the parameter value in question.
                with open(os.path.join(metric_final_abspath, 'parameter_values.json'), 'w') as f:
                    json.dump({parameter_name: parameter_value}, f)
                metric_paths_dict[parameter_idx] = init_metric_csvs(dir_path=metric_final_abspath, metric_configs=config, config_labels_dict=config_labels_dict)

        else:
            metric_final_abspath = metric_subdir_abspath
            metric_paths_dict =init_metric_csvs(dir_path=metric_final_abspath, metric_configs=config, config_labels_dict=config_labels_dict)

        
        complete_paths_dicts[metric_type] = metric_paths_dict

    if len(complete_paths_dicts) < 1:
        raise Exception('There must be at least one metric being saved!')
    
    return complete_paths_dicts


def try_float_or_keep(x):
    #Func which will convert a value to a float if possible. If presented with a datatype which cannot be converted, if 
    #it is a datatype which would raise a valueerror or a typeerror, then it will return the value as is. Otherwise, it will
    #raise an exception, because we do not want to silently fail.
    try:
        return float(x)
    except (ValueError, TypeError):
        #If we have a valueerror or a typeerror, then we will just return the value as is.
        return x
    except Exception as e:
        #If we have an unexpected exception, then we will raise it.
        raise Exception(f'Unexpected exception when trying to convert {x} to a float: {e}') from e

def write_row(
        case_name: str, 
        save_path: str, 
        extraction_tuples: tuple[tuple], 
        metrics_dict: dict[str, dict]):
    
    '''
    Function which writes a row of results to their corresponding csv file.

    Inputs:
    case_name: String denoting the case name under consideration
    
    save_path: The path to the csv file which the scores are being written to.
    
    extraction_tuple: The nested tuple denoting the tuple paths into the metrics dict which provides the corresponding 
    metrics necessary for writing the row. It should be separated according to the infer mode E.g.:

        ((infer_mode_1, cross_class_scores), (infer_mode_2, cross_class_scores), ...)

        or 

        ((infer_mode_1, per_class_scores, class_1), (infer_mode_2, per_class_scores, class_1), ...)
    
    metrics_dict: The dictionary from which the scores will be extracted. Expected to be in the following structure:

    {
    'infer_mode_1':
        {
        'cross_class_scores': __,
        'per_class_scores': NONE or 
            {
            class_1: __, (or NONE)
            class_2: __, (or NONE)
            }
        },
    .
    .
    .
    }
    ''' 
    #Lets first check that there does not already exist a case name, if it does then we need to delete it
    #to avoid duplicates.
    with open(save_path,'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        #We can just look at the last row because we always append at the end, and if we are restarting it is 
        #only going to re-continue from a point where the checkpointing had not established the callback on this
        #case name was done. 
        if len(rows) > 1 and rows[-1][0] == case_name:
            #In this case, we need to remove the last row.#Rewriting the file without the last row.
            rows = rows[:-1]
            with open(save_path,'w') as f_w:
                writer = csv.writer(f_w)
                writer.writerows(rows)
    
    with open(save_path,'a') as f:
        writer = csv.writer(f)
        base_row = [case_name]
        # base_row.extend([float(extractor(metrics_dict, path)) for path in extraction_tuples if extractor(metrics_dict, path)]) 
        base_row.extend([
            try_float_or_keep(val)
            for path in extraction_tuples
            for val in [extractor(metrics_dict, path)]
        ])
        #We will typically use a string to denote that the process terminated early, or that the foreground was empty.
        writer.writerow(base_row)
    
def write_to_csvs(
    case_name: str,
    csv_paths: dict[str, dict],
    tracked_metrics: dict,
    metrics_configs: dict
):
    '''
    Function which saves the metric outcomes for each patient to the corresponding csv files.

    Inputs: 

    case_name: A string denoting the corresponding case under consideration.
    
    csv_paths: A nested dictionary with structure: metric_type: {
        'cross_class_scores': 'cross_class_path',
        'per_class_scores': 
            {
            'background_class_path': __, (NOTE: If include_background_metric! Else, it is a NoneType)
            'foreground1_class_path': __,
            }
            or a NoneType if we do not want per-class-scores.
        }

    tracked_metrics: A thrice nested dictionary containing the metrics across the iterative refinement process 
    separated by metric type, then by the infer mode/name, and then a dictionary with fields:
    
    "cross_class_scores" (a torch tensor or Metatensor) and 
    "per_class_scores" an optional dict separated by the class label for each per class score being saved 
    (each is a torch tensor or metatensor). In instances where per_class_scores were not desired, this is a 
    NoneType.

    metrics_configs: A dictionary containing information regarding the metrics configurations. 
    This can inform the process of saving the metrics, for example whether we have
    multiple parameterisations for a given metric type.
    '''

    if not set(csv_paths) == set(tracked_metrics):
        raise Exception('There is not a csv directory containing initialised csvs for every metric that needs to be saved')
    
    #Here we save the row to the given csv file. We do this according to the dictionary of csv paths such that we 
    # skip past instances where a given score is not desired to be saved.
    
    for metric_type, metrics_dict in tracked_metrics.items():
        #We extract the csv paths:
        current_metric_csvs = csv_paths[metric_type]

        #Create an ordered tuple of infer mode call names from the tracked metrics dict for this given metric.
        infer_call_names = set((metrics_dict.keys()))
        sorted_infer_calls = sort_infer_calls(infer_call_names=infer_call_names)
        
        #Writing the cross class scores, must always be provided.: 
        
        if 'multiple_parameter_values' in metrics_configs[metric_type]:
            #In this case we need to convert the metrics to a format compatible with the
            #write_row function. Therefore we need to convert torch tensors of size (num_parameter values)
            #into separate tensors.
            parameter_name = metrics_configs[metric_type]['multiple_parameter_values']
            for parameter_idx in metrics_configs[metric_type][parameter_name]:
                path_tuples = tuple([(infer_mode, 'cross_class_scores', parameter_idx) for infer_mode in sorted_infer_calls])
                write_row(case_name=case_name, save_path=current_metric_csvs[parameter_idx]['cross_class_scores'], extraction_tuples=path_tuples,  metrics_dict=metrics_dict)
        
                if current_metric_csvs[parameter_idx]['per_class_scores'] is None:
                    pass #Explicit for debugging purposes
                elif isinstance(current_metric_csvs[parameter_idx]['per_class_scores'], dict):
                #In this case, we want to iterate through the non-NoneType classes.
                    for class_lb, path in current_metric_csvs[parameter_idx]['per_class_scores'].items():
                        if path is None:
                            pass #Explicit for debugging purposes
                        else:
                            path_tuples = tuple([(infer_mode, 'per_class_scores', class_lb, parameter_idx) for infer_mode in sorted_infer_calls])
                            write_row(case_name=case_name, save_path=path, extraction_tuples=path_tuples, metrics_dict=metrics_dict)
                else:
                    Exception(f'Unexpected datatype for the per-class scores csv paths in {metric_type}')
        else:
            #Generating the tuples for the paths:
            path_tuples = tuple([(infer_mode, 'cross_class_scores') for infer_mode in sorted_infer_calls])
            write_row(case_name=case_name, save_path=current_metric_csvs['cross_class_scores'], extraction_tuples=path_tuples,  metrics_dict=metrics_dict)
    
            # per_class_score handling, since this is not always provided at all, or across all classes:
            if current_metric_csvs['per_class_scores'] is None:
                pass #Explicit for debugging purposes
            elif isinstance(current_metric_csvs['per_class_scores'], dict):
                #In this case, we want to iterate through the non-NoneType classes.
                for class_lb, path in current_metric_csvs['per_class_scores'].items():
                    if path is None:
                        pass #Explicit for debugging purposes
                    else:
                        path_tuples = tuple([(infer_mode, 'per_class_scores', class_lb) for infer_mode in sorted_infer_calls])
                        write_row(case_name=case_name, save_path=path, extraction_tuples=path_tuples, metrics_dict=metrics_dict)
            else:
                Exception(f'Unexpected datatype for the per-class scores csv paths in {metric_type}')






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

