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
from src.results_utils.metric_save_util import init_all_csvs 
import torch 
import warnings 

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    
    #Data related args
    parser.add_argument('--dataset_name', type=str, default='BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised')
    parser.add_argument('--test_mode', type=str, default='test')
    parser.add_argument('--data_fold', type=str, default=None)
    
    #Experimental process/method related args 
    parser.add_argument('--app_name', type=str, default='Sample_TEST') #This acts as the name of the app, but also temporarily acts as the relative path name within the input_applications folder.
    parser.add_argument('--random_seed', type=int, default=23102002)
    parser.add_argument('--device_idx', type=int, default=None)
    parser.add_argument('--infer_init', type=str, default='Interactive Init')
    parser.add_argument('--infer_not_edit_bool', action='store_false', default=True)
    parser.add_argument('--infer_edit_nums', type=int, default=10)

    #Validation utilised constructors build args
    parser.add_argument('--metric_conf_filename', type=str, default='metrics_configs.txt')
    parser.add_argument('--prompt_conf_filename', type=str, default='prompts_configs.txt')
    parser.add_argument('--metric_conf_name', type=str, default='prototype')
    parser.add_argument('--init_prompt_conf_name', type=str, default='prototype')
    parser.add_argument('--edit_prompt_conf_name', type=str, default='prototype')
    #TODO: Put use_mem and other related args like that for the im etc in here. 
    parser.add_argument('--use_mem_inf_edit', action='store_true', default=False) #Whether im is used for conditioning prompt gen.
    parser.add_argument('--im_conf_remove_init', action='store_true', default=False) #Bool for whether the init state in im will be removed from memory.
    parser.add_argument('--im_conf_mem_len', type=int, default=-1)
    #For the output processor.
    parser.add_argument('--is_seg_tmp', action='store_true', default=False)
    

    args = parser.parse_args()
    return args

def gen_experiment_args(args):
    #Takes an argparse namedspace obj and constructs a dictionary required for the the run script (most of which will be inherited for the front end init).

    output_dict = dict() 

    #Setting the app name for the experiment:
    output_dict['app_name'] = args.app_name 

    #Creating paths
    
    #Creating the relative path to the base build dir within the app.
    output_dict['build_app_rel_path'] = 'src_validate'
    #Temporarily creating an abspath using this relative path:
    output_dict['build_app_abspath'] = os.path.join(codebase_dir, 'input_application', output_dict['app_name'], output_dict['build_app_rel_path'])

    #Paths for results and logging. 
    output_dict['results_dir'] = os.path.join(codebase_dir, 'results')
    output_dict['input_dataset_dir'] = os.path.join(codebase_dir, 'datasets', args.dataset_name)
    output_dict['results_dataset_subdir'] = os.path.join(output_dict['results_dir'], args.dataset_name)
    output_dict['exp_datetime'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dict['exp_results_dir'] = os.path.join(output_dict['results_dataset_subdir'], output_dict['exp_datetime'])

    ########################################################################################################################

    #Configuring the experimental data-selection configs for extraction.
    output_dict['exp_data_configs'] = {
        'test_mode':args.test_mode,
        'data_fold': args.data_fold
    }

    #Configuring the experimental configs, first the infer run configs:
    if not args.infer_not_edit_bool: #Then init only.
        output_dict['infer_run_configs'] = {
            'init':args.infer_init,
            'edit_bool':False,
            'num_iters': None
        }
    else:
        output_dict['infer_run_configs'] = {
            'init':args.infer_init,
            'edit_bool':True,
            'num_iters': args.infer_edit_nums
        }

    #Extracting the configs dicts for the metrics and the prompt configs.
    exp_conf_dir = os.path.join(codebase_dir, 'exp_configs') 

    output_dict['metrics_configs'] = extract_config(os.path.join(exp_conf_dir, args.metric_conf_filename), args.metric_conf_name)
    
    output_dict['inf_init_prompt_config'] =  extract_config(os.path.join(exp_conf_dir, args.prompt_conf_filename), args.init_prompt_conf_name)

    output_dict['inf_edit_prompt_config'] = extract_config(os.path.join(exp_conf_dir, args.prompt_conf_filename), args.edit_prompt_conf_name)
    
    # output_dict['metrics_prompts_configs']
    
    #Extracting the random seed:

    output_dict['random_seed'] = args.random_seed

    #####################################################################################

    #Extracting the writer info. 
    output_dict['is_seg_tmp'] = args.is_seg_tmp

    #Extracting the inference and prompt gen device info. 
    if args.device_idx is None:
        output_dict['device'] = torch.device('cpu')
    elif isinstance(args.device_idx, int):
        try:
            #Check how many cuda devices there are, if none then cpu. If there is, then pick the one selected by the argparse arg.
            device_count = torch.cuda.device_count()
            if device_count:
                try:
                    output_dict['device'] = torch.device(args.device_idx)
                    #Try to use the device they provided.
                    # Get the corresponding device names
                    device_name = torch.cuda.get_device_name(args.device_idx)
                    print(f'Valid device idx selected, using device idx:{args.device_idx} - {device_name}')


                except:
                    #Revert to rank=0 valid device. 
                    output_dict['device'] = torch.device(0)

                    # Get the corresponding device names
                    device_name = torch.cuda.get_device_name(0)
                    print(f'Invalid device idx selected, using device idx:0 - {device_name}')
                


            else:
                output_dict['device'] = torch.device('cpu')
                print('No cuda device visible, using cpu.')
        except:
            #If there is any failure (e.g. if cuda package not installed.)
            output_dict['device'] = torch.device('cpu')
            warnings.warn('Check if CUDA is installed.')
            print('Cuda related error occured, using cpu')
    else:
        raise TypeError('Device idx must be a None (i.e. cpu) or an int.')
    
    #Extracting the info about the interaction memory usage:

    #The use of inf im for conditioning the prompt generation.
    output_dict['use_mem_inf_edit'] = args.use_mem_inf_edit
    #Handling the im configs in the front-end-simulator (e.g. memory len, keeping init)
    output_dict['im_config'] = {
        'keep_init':args.im_conf_keep_init,
        'im_len':args.im_conf_mem_len 
    }

    return output_dict 


# def extract_fe_dict(experiment_args_dict):
#     #Takes the experiment args dict and returns the sub-dict required for the front-end-simulator initialisation. 
#     pass 


def create_seg_dirs(exp_results_dir, infer_run_conf):
    #Function which creates the directories for saving the segmentations (if desired), according to the inference run configuration.
    
    #Create the segmentation directory within the experiment results directory.
    seg_base_dir = os.path.join(exp_results_dir, 'segmentations')  
    os.makedirs(seg_base_dir, exist_ok=False)
    #Create the subdirs for each of the iterations according to the infer_run_config dictionary.

    #Create the init (ALWAYS PROVIDED!) dir
    init_dir = os.path.join(seg_base_dir, infer_run_conf['init'].title())
    os.makedirs(init_dir, exist_ok=False)

    #Create the edits (optional) dirs 
    if not isinstance(infer_run_conf['edit_bool'], bool):
        raise TypeError('The edit bool in the infer run configs must be a bool')
    else:
        if infer_run_conf['edit_bool']:

            if isinstance(infer_run_conf['num_iters'], int):
                for iter in range(1, infer_run_conf['num_iters'] + 1):
                    os.makedirs(os.path.join(seg_base_dir, f'Interactive Edit Iter {iter}'), exist_ok=False)
            else:
                raise TypeError('If running editing, needs to be an int type for the number of iterations performed.')

def init_metrics_saves(exp_results_dir, metrics_configs, configs_labels_dict):
    #Function which creates metrics dirs and initialises the csvs for the metrics saver, takes the results base dir and the metrics configs dict.
    metrics_dir = os.path.join(exp_results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=False)

    init_all_csvs(metrics_dir, metrics_configs, configs_labels_dict)


def extract_config(path, name):
    #Function which extracts configs dicts from json or txt files. Takes the path to the file, and the name of the specific config desired. Returns a dict.

    if not os.path.exists(path):
        raise Exception('The path was not a valid one. Please check.')    

    #Loading the file:
    with open(path) as f:
        configs_registry = json.load(f)
        config_dict = configs_registry[name]

    return config_dict 
    
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



    ################################# Extraction and construction of any experiment related information #######################################################

    #Extracting parser args.
    args = set_parse() 

    #Reformulate the args into the format required for the experiment:
    experiment_args = gen_experiment_args(args)


    ################################## Configuration and extraction of data-related info.  ######################################################################
    
    #Extraction of the config labels dictionary, and the initialisation of the dataloader
    config_labels_dict, dataloader = init_data(
        dataset_dir=experiment_args['input_dataset_dir'], 
        exp_data_configs=experiment_args['exp_data_configs'],
        file_ext='.nii.gz')
    

    ############################## Initialisation of any required directories for this experiment ####################################################################

    #Initialise the base results folders if it doesn't exist.
    os.makedirs(experiment_args['results_dir'], exist_ok=True)
    
    #Initialise the base results folder for the dataset at hand. 
    os.makedirs(experiment_args['results_dataset_subdir'], exist_ok=True) 

    #Creating a base dir for the experiment to store results.
    os.makedirs(experiment_args['exp_results_dir'], exist_ok=False) #Should not already exist.

    #Creating the subdirectories and the csv files for the metrics
    init_metrics_saves(exp_results_dir=experiment_args['exp_results_dir'], metrics_configs=experiment_args['metrics_configs'], configs_labels_dict=config_labels_dict)

    #Based on whether the segmentation is being saved permanently, create the corresponding subdirectories or not. 
    if not experiment_args['is_seg_tmp']:
        create_seg_dirs(exp_results_dir=experiment_args['exp_results_dir'], infer_run_conf=experiment_args['infer_run_configs'])
    

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
    
    logger_save_name = f'experiment_{experiment_args["exp_datetime"]}_logs'
    experiment_args_logger(logger_save_name=logger_save_name, root_dir=experiment_args['exp_results_dir'], screen=True, tofile=True)
    exp_setup_logger = logging.getLogger(logger_save_name)
    exp_setup_logger.info(f'Experiment arguments: \n {str(experiment_args)}')


    if not True:
        app_config_dict = infer_app.app_configs()
        if 'app_name' not in app_config_dict:
            warnings.warn('Should return the application name')
        exp_setup_logger.info(f'Application configuration arguments \n {str(app_config_dict)}')



    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #Build the front-end simulator: 

    # fe_sim_obj = init_fe(config_labels_dict,)




    # Iterating through the dataset:

    # run_instances(dataloader=dataloader, fe_sim_obj=fe_sim_obj)
if __name__=='__main__':
    main()


