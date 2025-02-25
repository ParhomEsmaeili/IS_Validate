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
from src.data.utils import data_instance_reformat, iterate_dataloader_check, init_data


def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default='Task10_colon')
    parser.add_argument('--is_seg_tmp', action='store_false')
    parser.add_argument('--metric_conf_filename', type=str, default='')
    parser.add_argument('--prompt_conf_filename', type=str, default='')
    parser.add_argument('--application_name', type=str, default='')

    args = parser.parse_args()
    return args


def init_fe():
    #Function which initialises the front-end simulator.
    pass 

def run_instances(dataloader, fe_sim_obj):
    #Function is intended for iterating through the constructed dataset.
    
    #Iterating through the dataloader.
    for data_instance in dataloader:
        #Running a check on the data_instance loaded.
        iterate_dataloader_check(data_instance=data_instance)
        
        #Reformat the data instance
        data_instance_reformat(data_instance=data_instance)
        
        #Initialising the temporary directory.
        tempdir_obj = tempfile.TemporaryDirectory(dir=codebase_dir)

        try:
            #Calling the front-end simulator
            fe_sim_obj(data_instance=data_instance, tmp_dir_path=tempdir_obj.name) 
        finally:
            tempdir_obj.cleanup() 
            raise Exception('There was an error in the front-end simulator, running cleanup.')


def main():

    experiment_args = set_parse() 
    experiment_args = vars(experiment_args)



    #Initialise the results folders if it doesn't exist, saves it according to the dataset name?
    results_dir = os.path.join(codebase_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    #Initialise the folder for the dataset at hand. 
    result_dataset_subdir = os.path.join(results_dir, experiment_args['dataset_name'])
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
    if not experiment_args['is_seg_tmp']:

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
    exp_setup_logger.info(str(experiment_args))
    if not True:
        app_config_dict = infer_app.app_configs()
        if 'app_name' not in app_config_dict:
            warnings.warn('Should return the application name')
        exp_setup_logger.info(str(app_config_dict))

    #Configuring the experimental data configs for extraction.
    exp_data_configs = {
        'test_mode':experiment_args['data_mode'],
        'data_fold': experiment_args['data_fold']
    }
    config_labels_dict, dataloader = init_data(
        codebase_dir=codebase_dir,
        dataset_name=experiment_args['dataset_name'], 
        exp_data_configs=exp_data_configs)
    
    #Load the device, takes device arg, if not then reverts to cpu.

    #Build the front-end simulator: 

    fe_sim_obj = init_fe(config_labels_dict,)

    # Funcs for iterating through using the built app. Generates a temporary directory, need to add a path to it. 
    run_instances(dataloader=dataloader, fe_sim_obj=fe_sim_obj)

if __name__=='__main__':
    main()


