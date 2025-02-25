import argparse
import json
import logging 
import sys 
import os 
import datetime
import tempfile
codebase_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
sys.path.append(codebase_dir)
from src.front_back_interactor.pseudo_ui import FrontEndSimulator 
from src.utils.logging import experiment_args_logger

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default='Task10_colon')
    parser.add_argument('--is_seg_tmp', action='store_false')
    parser.add_argument('--metric_conf_filename', type=str, default='')
    parser.add_argument('--prompt_conf_filename', type=str, default='')
    args = parser.parse_args()
    return args


def main():

    args = set_parse() 
    args = vars(args)



    #Initialise the results folders if it doesn't exist, saves it according to the dataset name?
    results_dir = os.path.join(codebase_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    #Initialise the folder for the dataset at hand. 
    result_dataset_subdir = os.path.join(results_dir, args['dataset_name'])
    os.makedirs(result_dataset_subdir, exist_ok=True) 

    #Creating a folder for the experiment to store results.
    exp_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_results_dir = os.path.join(result_dataset_subdir, exp_datetime)

    #Creating the experiment results dir:
    os.makedirs(exp_results_dir, exist_ok=False) #Should not already exist.

    #Creating the subdirectories for the metrics
    metrics_dir = os.path.join(exp_results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=False)

    #Based on whether the segmentation is being saved permanently, create the corresponding classes. 
    if not args['is_seg_tmp']:

        #Add a function for creating the directories for savin g the metrics according to the inference run type. 
        raise NotImplementedError('Need to implement func for initialisation of the save directories.')

    if not True:
        raise NotImplementedError('Need to fix the infer app builder')
        infer_app = build_app()

        if not callable(infer_app):
            raise Exception('The inference app must be callable class.')
        else:
            #Check if it has a call attribute!
            try:
                callback = getattr(infer_app, "__call__")
            except:
                raise Exception('The inference app did not have a function __call__')
            
            #Check if it has a app_configs attribute!
            try:
                app_configs_callback = getattr(infer_app, "app_configs")
            except:
                raise Exception('The inference app did not have a function app_configs (which can be empty!), for saving the app configs to the experiment logger file.')

            if not callable(callback):
                raise Exception('The initialised inference app object had a __call__ attribute which was not a callable function.') 
            
            if not callable(app_configs_callback):
                raise Exception("The initialised inference app object had a 'app_configs' attribute which was not a callable function. ")

    
    #Save the info about the experiment args, save the info about the application config which should be spit out as a method of the callable infer class.
    
    logger_save_name = f'experiment_{exp_datetime}_logs'
    experiment_args_logger(logger_save_name=logger_save_name, root_dir=exp_results_dir, screen=True, tofile=True)
    exp_setup_logger = logging.getLogger(logger_save_name)
    exp_setup_logger.info(str(args))
    
    

    # Create a func which creates the dictionary which is passed through to the dataset constructor.
    # Add a func for iterating through using the built app. Generates a temporary directory, need to add a path to it. 

    
if __name__=='__main__':
    main()
