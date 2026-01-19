import argparse
import json
import logging 
import sys 
import os 
import datetime
import tempfile 
import importlib
import torch
import pickle 
import warnings
import base64
from typing import Optional
codebase_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
sys.path.append(codebase_dir)
from src.front_back_interactor.pseudo_ui import FrontEndSimulator 
from src.general_utils.logging import experiment_args_logger
from src.general_utils.dict_utils import extractor
from src.data.utils import data_instance_reformat, iterate_dataloader_check, init_task_cases
from src.results_utils.metric_save_util import init_all_csvs
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    
    #Experimental name/job-continuation related args
    parser.add_argument('--experiment_name', type=str, required=False, default=None)#'debugging_continual_adapt')#None
    parser.add_argument('--continue_execution', action='store_true', default=False)#True) #False)
    parser.add_argument('--continue_exec_root', type=str, default='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/continue_execution_files') #None
    #Data and application root related args
    
    parser.add_argument('--data_root', type=str, default=codebase_dir)
    parser.add_argument('--dataset_name', type=str, default='Dataset001_BrainTumour')#'Dataset005_Prostate')
    parser.add_argument('--app_root', type=str, default= 
                        '/home/parhomesmaeili/IS_Codebase_Forks/nnInteractive_Fork')
                        #'/home/parhomesmaeili/MY METHOD') #NOTE:Just set for debugging purposes.

    #This acts as the name of the app, but also temporarily acts as the relative path name within the input_applications folder in the app root folder.
    parser.add_argument('--app_name', type=str, default='nnInteractive_App') #'AdaptiveIS')
    parser.add_argument('--metrics_root', type=str, default=os.path.join(codebase_dir, 'results'))
    parser.add_argument('--seg_root', type=str, default=os.path.join(codebase_dir, 'results'))

    #Experimental process/method related args
    parser.add_argument('--adaptation_config_name', type=str, default=None)#'adapt_prototype_1') #None
    parser.add_argument('--enable_adaptation', action='store_true', default=False)# True)#False)
    parser.add_argument('--provide_gold_standard_after_inference', action='store_true', default=False)#True)#False)
    #Adaptation requires the capability to enable the algorithm to adapt, and so requires some knowledge of the 
    #annotation. This boolean controls whether adaptation is enabled or not. This also requires some extra handling for
    #continuing execution, as checkpoints will need to be restored. In addition to other information, e.g. position in the
    #data stream etc. 
    parser.add_argument('--shuffle_cases', action='store_true', default=False) #True)
    parser.add_argument('--random_seed', type=int, default=341103)
    parser.add_argument('--cuda_deterministic_disable', action='store_true', default=False)
    parser.add_argument('--torch_deterministic_disable', action='store_true', default=False)
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--infer_init', type=str, default='Interactive Init')
    parser.add_argument('--infer_not_edit_bool', action='store_false', default=True)
    parser.add_argument('--infer_edit_nums', type=int, default=100)
    parser.add_argument('--dice_termination_thresh', type=float, default=1.0)

    #Validation utilised constructors build args
    parser.add_argument('--metric_conf_filename', type=str, default='metrics_configs.txt')
    parser.add_argument('--prompt_conf_filename', type=str, default='prompts_configs.txt')
    parser.add_argument('--task_conf_filename', type=str, default='task_configs.txt')
    parser.add_argument('--metric_conf_name', type=str, default='prototype')
    parser.add_argument('--task_conf_name', type=str, default='task_id_2')
    parser.add_argument('--init_prompt_conf_name', type=str, default='prototype')
    parser.add_argument('--edit_prompt_conf_name', type=str, default='prototype')
    parser.add_argument('--metric_prompt_procedure_type', type=str, default='heuristic')
    parser.add_argument('--inf_prompt_procedure_type', type=str, default='heuristic')
    parser.add_argument('--sim_empty_fg_automatic', action='store_true', default=False)
    #TODO: Put use_mem and other related args like that for the im etc in here. 
    parser.add_argument('--use_mem_inf_edit', action='store_true', default=False) #Whether im is used for conditioning prompt gen.
    parser.add_argument('--im_conf_remove_init', action='store_true', default=True) 
    #Bool for whether the init state in im will be removed from memory.
    parser.add_argument('--im_conf_mem_len', type=int, default=1)#-1)
    #Int which determines the memory length used at cleanup after the interaction memory is updated with the current edit iteration's interaction state (inclusive of current state). 
    # This functionally has the same thing as using a memory length of N (where N is our variable here) for conditioning
    #the prompt generation of the next iteration (if memory is being used for conditioning.) N is strictly > 0 or N = - 1, where N=-1 indicates full memory length paradoxically. 
    # as N = 0 would remove the current iteration's interaction for inference, and also would be the same as ignoring the memory for prompt generation (for which we have a separate variable.)
    #For now we will set both these parameters to being true (i.e. to delete) because we are having lots of memory issues.


    #For the output processor/writing args which are optional.
    parser.add_argument('--write_segmentation', action='store_true', default=False)
    # Whether to write the segmentations at all, or not (default is false for now)
    parser.add_argument('--is_seg_tmp', action='store_true', default=False)
    # Whether the segmentations are written as temp files, if they are written. Or whether they are permanent files.
    parser.add_argument('--save_prompts', action='store_true', default=False)
    

    args = parser.parse_args()
    return args

def gen_experiment_args(args):
    #Takes an argparse namedspace obj and constructs a dictionary required for the the run script (most of which will be inherited for the front end init).

    output_dict = dict() 
    #Setting the app name for the experiment, also available for the build script. 
    output_dict['app_name'] = args.app_name 

    #Creating paths
    
    #Creating the relative path to the base build dir within the app.
    output_dict['build_app_rel_path'] = 'src_validate'
    #Temporarily creating an abspath using this relative path:
    output_dict['build_app_abspath'] = os.path.join(args.app_root, output_dict['app_name'], output_dict['build_app_rel_path'])
    # output_dict['build_app_abspath'] = os.path.join(codebase_dir, 'input_application', output_dict['app_name'], output_dict['build_app_rel_path'])

    #Paths for results and logging etc. 
    output_dict['metrics_root'] = args.metrics_root #For the metrics themselves.
    output_dict['seg_root'] = args.seg_root #For the base directory for the segmentations. This will typically be the same as results, but in case of separate 
    #mounts we want to be able to store these large files externally.
    
    if args.metrics_root == args.seg_root:
        #In this case we are storing the metrics and the segmentations in the same root directory, and so will require subdirectories for each of these.
        output_dict['root_match'] = True #Being EXPLICIT, we could have just set the variable to the bool directly.
    else:
        output_dict['root_match'] = False
    

    output_dict['input_dataset_dir'] = os.path.join(args.data_root, 'datasets', args.dataset_name) #os.path.join(codebase_dir, 'datasets', args.dataset_name)

    output_dict['results_dataset_subdir'] = os.path.join(output_dict['metrics_root'], args.dataset_name) #Subdir for the dataset which is being used in the task
    #of the experiment at hand
    output_dict['seg_dataset_subdir'] = os.path.join(output_dict['seg_root'], args.dataset_name) #Subdir for the dataset which is being used in the task, this will
    #typically be the same as the results dataset subdir, but in case of separate mounts we want to be able to store these large files externally.

    #Experiment logging related arguments (i.e. for a specific run of the evaluation script!)
    if args.continue_execution:
        if args.experiment_name == None:
            raise ValueError('If configured for continuing an experiment execution, must provide the experiment name to \n ' \
            'continue from otherwise we cannot proceed.')
        else:
            output_dict['experiment_name'] = args.experiment_name
        
        if args.continue_exec_root == None:
            raise ValueError('If continuing an experiment execution, must provide the root directory of the \n'
            'files to write/read where to continue from.')
        else:
            output_dict['continue_exec_path'] = os.path.join(args.continue_exec_root, args.experiment_name + '.pkl')
    else:
        if args.experiment_name == None:
            output_dict['experiment_name'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            raise Exception('If not continuing an experiment execution, cannot provide an experiment name, \n'
            'this is redundant.')
    if args.enable_adaptation:
        if not args.continue_execution:
            raise ValueError('Adaptation almost certainly requires the capability to continue execution, please enable \n'
            'the continue execution boolean flag to proceed for reassurance.')
        else:
            output_dict['adaptation_config_name'] = args.adaptation_config_name
            output_dict['enable_adaptation'] = args.enable_adaptation
            output_dict['provide_gold_standard_after_inference'] = args.provide_gold_standard_after_inference
    else:
        output_dict['enable_adaptation'] = args.enable_adaptation
        assert args.adaptation_config_name == None, 'If adaptation is disabled, cannot provide adaptation config name, please check your input arguments.'
        output_dict['adaptation_config_name'] = None #Irrelevant if adaptation is disabled.
        assert args.provide_gold_standard_after_inference == False, 'If adaptation is disabled, cannot provide gold standard after inference, please set this flag or check your input arguments.'
        output_dict['provide_gold_standard_after_inference'] = False #Irrelevant if adaptation is disabled.

    output_dict['continue_execution'] = args.continue_execution #Storing this boolean.
    output_dict['exp_results_dir'] = os.path.join(output_dict['results_dataset_subdir'], output_dict['experiment_name'])
    output_dict['exp_seg_dir'] = os.path.join(output_dict['seg_dataset_subdir'], output_dict['experiment_name'])

    ########################################################################################################################

    #Configuring more generic experiment related configs.


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

    #Configuring more experiment specific configs:

    #Extracting the configs dicts for the dataloading, metrics and the prompt configs.
    exp_conf_dir = os.path.join(codebase_dir, 'exp_configs', args.dataset_name) 
    # output_dict['dataloading_type'] = args.dataloading_type
    output_dict['task_id'] =args.task_conf_name
    output_dict['task_configs'] = extract_config(os.path.join(exp_conf_dir, args.task_conf_filename), args.task_conf_name)
    output_dict['seg_problem'] = output_dict['task_configs']['seg_problem']
    
    #Loading in the relevant information from the dataset

    output_dict['dataset_info'] = {
    'dataset_name': args.dataset_name,
    'dataset_image_channels': extract_config(os.path.join(args.data_root, 'datasets', args.dataset_name, 'dataset.json'), 'channel_names'),
    'task_channels': extractor(output_dict['task_configs'], ('data_sampling', 'image_conf', 'image_channel'))
    }


    output_dict['metrics_configs'] = extract_config(os.path.join(exp_conf_dir, args.metric_conf_filename), args.metric_conf_name)
    
    #Extracting the config dict for handling the empty fg....? 
    output_dict['sim_empty_fg_automatic'] = args.sim_empty_fg_automatic  

    # output_dict['metric_prompt_procedure_type']
    output_dict['inf_prompt_procedure_type'] = args.inf_prompt_procedure_type 

    #Extracting from inf prompt registry according to procedural type: 
    inf_prompt_registry = extract_config(os.path.join(exp_conf_dir, args.prompt_conf_filename), args.inf_prompt_procedure_type)

    output_dict['inf_init_prompt_config'] =  inf_prompt_registry[args.init_prompt_conf_name]

    output_dict['inf_edit_prompt_config'] = inf_prompt_registry[args.edit_prompt_conf_name]
    
    # output_dict['metrics_prompts_configs']
    
    #Extracting the random seed/randomness related info:
    output_dict['shuffle_cases'] = args.shuffle_cases
    output_dict['random_seed'] = args.random_seed
    output_dict['cuda_deterministic'] = not args.cuda_deterministic_disable
    output_dict['torch_deterministic'] = not args.torch_deterministic_disable

    #Now we extract the termination condition threshold:
    output_dict['dice_termination_thresh'] = args.dice_termination_thresh 
    #####################################################################################

    #Extracting the writer info.
    output_dict['write_segmentation'] = args.write_segmentation
    output_dict['is_seg_tmp'] = args.is_seg_tmp
    output_dict['save_prompts'] = args.save_prompts
    ###########################################################################################
    
    #Extracting the inference and prompt gen device info. 
    if args.device_idx == None:
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
    
    ###########################################################################################
    
    #Extracting the info about the interaction memory usage:

    #The use of inf im for conditioning the prompt generation.
    output_dict['use_mem_inf_edit'] = args.use_mem_inf_edit
    #Handling the im configs in the front-end-simulator (e.g. memory len, keeping init)
    output_dict['im_config'] = {
        'keep_init':not args.im_conf_remove_init,
        'im_len':args.im_conf_mem_len 
    }

    return output_dict 


def create_seg_dirs(exp_seg_dir, infer_run_conf, exist_ok=False):
    '''
    Function which creates the directories for saving the segmentations (if desired), according to the inference run configuration.
    Args: 
        exp_seg_dir: str, the base dir for the experiment segmentations to be stored.
        root_match: bool, whether the root of the segmentations is the same as the results root or not
        infer_run_conf: dict, the inference run configuration dictionary.
    '''
    #Function which creates the directories for saving the segmentations (if desired), according to the inference run configuration.
    exp_seg_dir = os.path.join(exp_seg_dir, 'segmentations')
    #Create the subdirs for each of the iterations according to the infer_run_config dictionary.

    #Create the init (ALWAYS PROVIDED!) dir
    init_dir = os.path.join(exp_seg_dir, infer_run_conf['init'].title())
    os.makedirs(init_dir, exist_ok=exist_ok)

    #Create the edits (optional) dirs 
    if not isinstance(infer_run_conf['edit_bool'], bool):
        raise TypeError('The edit bool in the infer run configs must be a bool')
    else:
        if infer_run_conf['edit_bool']:

            if isinstance(infer_run_conf['num_iters'], int):
                for iter in range(1, infer_run_conf['num_iters'] + 1):
                    os.makedirs(os.path.join(exp_seg_dir, f'Interactive Edit Iter {iter}'), exist_ok=exist_ok)
            else:
                raise TypeError('If running editing, needs to be an int type for the number of iterations performed.')

def init_metrics_saves(exp_results_dir, metrics_configs, configs_labels_dict):
    #Function which creates metrics dirs and initialises the csvs for the metrics saver, takes the results base dir and the metrics configs dict.
    metrics_dir = os.path.join(exp_results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=False)

    return init_all_csvs(metrics_dir, metrics_configs, configs_labels_dict)


def extract_config(path, name):
    #Function which extracts configs dicts from json or txt files. Takes the path to the file, and the name of the specific config desired.

    if not os.path.exists(path):
        raise Exception(f'The path {path} was not a valid one. Please check.')    

    #Loading the file:
    with open(path) as f:
        configs_registry = json.load(f)
        config = configs_registry[name]

    return config 
    

def log_config_writer(args_name, args_dict, logger):
    logger.info(f'Printing {args_name}: {os.linesep}')
    #Easier to use this method as not everything is json serialisable.
    for key, value in args_dict.items():
        try:
            logger.info(f"{key}: {json.dumps(value, indent=4)}") #Where possible try to serialise in a manner where its readable. 
        except:
            logger.info(f"{key}: {value}")


def generate_base64_filename(length=12):
    random_bytes = os.urandom(length)
    encoded = base64.urlsafe_b64encode(random_bytes).decode('utf-8')
    return encoded.rstrip('=')  # Remove padding


def build_infer_app(build_app_path, device, adaptation_config_name, algorithm_state: dict, enable_adaptation: bool, algo_cache_name:str):

    build_app_dir = os.path.join(build_app_path, 'build_app')
    # print(build_app_dir)
    if not os.path.exists(build_app_dir):
        raise ValueError('The provided path does not exist')
    
    try: #Try to use __init__.
    
        MODULE_PATH = os.path.join(build_app_dir, "__init__.py")
        MODULE_NAME = "InferApp"
        
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module 
        spec.loader.exec_module(module)
        
        from BuildInferApp import InferApp

    except: #If only providing a .py with no init. 
        spec = importlib.util.spec_from_file_location("BuildInferApp", os.path.join(build_app_dir, 'infer_app.py'))
        foo = importlib.util.module_from_spec(spec)
        sys.modules["BuildInferApp"] = foo
        spec.loader.exec_module(foo)
        InferApp = foo.InferApp

    return InferApp(device, adaptation_config_name, algorithm_state, enable_adaptation, algo_cache_name)

def init_infer_app(experiment_args:dict, loaded_experiment_checkpoint: Optional[dict] = None): 

    #Function which finds and initialises the inference app using the build script, then checks it has the necessary methods. 
    
    #We will DEMAND that there is some algorithm state info, even if empty for a non adaptive algorithm. 
    #Even if its an empty dict for a non-adapting model (this is just a hack for consistency).
    
    
    #Algo state either = empty dict in checkpoint if auto continue enabled but not adapting,
    # empty dict if not auto continue. 
    # empty dict if adapting but not yet saved the checkpoint before starting evaluation.
    enable_adaptation = experiment_args.get('enable_adaptation', False)
    #Adding another safety check:
    if not enable_adaptation:

        assert experiment_args.get('adaptation_config_name') == None, 'If adaptation is disabled, cannot provide adaptation config name, please check your input arguments.'
        #if not adapting then algorithm state is non-existent.
        algorithm_state = {}
        # ('If adaptation is not enabled, then the algorithm state in the loaded experiment checkpoint \n'    
        # 'must be an empty dict (if auto-continue enabled). Please check your input arguments.')
        algo_cache_name = None #cache name irrelevant if not adapting.
    else:
        #Another safety check
        if not experiment_args['continue_execution']:
            raise ValueError('If adaptation is enabled, must also enable continue execution to proceed safely.')
        if experiment_args.get('algo_cache_name', None) == None:
            raise ValueError('If adaptation is enabled, must provide a valid algo_cache_name in the experiment args \n'
                             'to proceed.')
        else:
            algo_cache_name = experiment_args['algo_cache_name']

        #If we have not yet stored any checkpoint then loaded checkpoint would be none, so lets set the algorithm state
        #to an empty dict for now - i.e., we don't have any state to load at all!
        if loaded_experiment_checkpoint == None:
            algorithm_state = {}
        elif loaded_experiment_checkpoint != None:
            #Some checkpoint has been saved.
            # For adaptation, even if the model has not adapted, we will need this info to restore the state of the algorithm 
            # for auto-continuation.          
            algorithm_state = loaded_experiment_checkpoint.get('algorithm_state')
            if algorithm_state == None:
                raise ValueError('If adaptation is enabled and a checkpoint is loaded, the algorithm state must be present in the checkpoint, even if empty')
            
            if loaded_experiment_checkpoint['eval_state']['last_completed_idx'] > -1 and loaded_experiment_checkpoint['algorithm_state'] == {}:
                #The presumption here is that if a sample has been completed, then the algorithm state cannot be
                #empty because it should have stored something about the meta-state of the algorithm.
                raise ValueError('If adaptation is enabled and a sample is completed, the algorithm state in the loaded experiment checkpoint \n'
                                'cannot be empty. Please provide a valid algorithm state.')

        

    infer_app = build_infer_app(
        build_app_path=experiment_args['build_app_abspath'], 
        device=experiment_args['device'], 
        adaptation_config_name=experiment_args.get('adaptation_config_name', None),
        algorithm_state=algorithm_state, 
        enable_adaptation=enable_adaptation,
        algo_cache_name=algo_cache_name)
    
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

        if enable_adaptation:
            #If adaptation is enabled, check if it has an accept_new_sample method.
            try:
                accept_new_sample_callback = getattr(infer_app, "accept_new_sample")
            except:
                raise Exception('The inference app did not have a function accept_new_sample required for adaptation.')

            if not callable(accept_new_sample_callback):
                raise Exception('The initialised inference app object had an accept_new_sample attribute which was not a callable function.')
            
            try:
                trigger_adaptation_callback = getattr(infer_app, "trigger_adaptation")
            except:
                raise Exception('The inference app did not have a function trigger_adaptation required for adaptation.')
            
            if not callable(trigger_adaptation_callback):
                raise Exception('The initialised inference app object had a trigger_adaptation attribute which was not a callable function.')


        if not callable(callback):
            raise Exception('The initialised inference app object had a __call__ attribute which was not a callable function.') 
        
        if not callable(app_configs_callback):
            raise Exception("The initialised inference app object had a 'app_configs' attribute which was not a callable function. ")
        
    return infer_app

def write_to_checkpoint(
        checkpoint_path:str,
        event: str,
        checkpoint_info: dict, 
        updated_fields_eval: dict,
        updated_fields_algo: dict
    ):
    if checkpoint_path == None:
        raise ValueError('Checkpoint path cannot be None when writing a checkpoint.')
    else:
        assert type(updated_fields_algo) == dict, 'The updated fields for the algorithm state must be provided as a dict (empty dict if no updates).'
        assert type(updated_fields_eval) == dict, 'The updated fields for the evaluation state must be provided as a dict.'
        if event == 'pre_loop':
            #In this case we are writing the initial info before the loop starts, in case the subprocess fails before any
            # sample completed
            pass #NOTE: No need to update fields here. 

        elif event == 'pre_subprocess':
            if updated_fields_eval == {}: #Empty dict will have a falsey, but for CLARITY we will be explicit.
                raise ValueError('When writing a checkpoint pre_subprocess, must provide the updated eval fields dict.')
            #In this case, we are writing some initial info before a sample is passed through for evaluation.
            
            #NOTE: We will be measured about what variables can be updated here. We do not want to be too hasty.
            
            required_eval_fields = ('current_temp_dir',)
            if not all([key in updated_fields_eval.keys() for key in required_eval_fields]):
                raise ValueError(f'When writing a checkpoint pre_subprocess, must only update the following fields: {required_eval_fields}')

            checkpoint_info["eval_state"].update(updated_fields_eval)
            if updated_fields_algo != {}: #Empty dict will have a falsey, but for CLARITY we will be explicit.
                checkpoint_info['algorithm_state'].update(updated_fields_algo) #We will not check any specifics
                #because this is algorithm dependent. 

        elif event == 'post_subprocess':
            if updated_fields_eval == {}: #Empty dict will have a falsey, but for CLARITY we will be explicit.
                raise ValueError('When writing a checkpoint post_subprocess, must provide the updated fields dict.')
            
            required_eval_fields = ('last_completed_case', 'last_completed_idx',)
            if not all([key in updated_fields_eval.keys() for key in required_eval_fields]):
                raise ValueError(f'When writing a checkpoint post_subprocess, must only update the following fields: {required_eval_fields}')
            checkpoint_info["eval_state"].update(updated_fields_eval)

            if updated_fields_algo != {}: #Empty dict will have a falsey, but for CLARITY we will be explicit.
                checkpoint_info['algorithm_state'].update(updated_fields_algo) #We will not check any specifics
                #because this is algorithm dependent.

        elif event == 'post_cleanup':
            #Only intended for when the temporary dir path is to be removed after cleanup.
            if updated_fields_eval == {}: #Empty dict will have a falsey, but for CLARITY we will be explicit.
                raise ValueError('When writing a checkpoint post_cleanup, must provide the updated fields dict.')
            required_eval_fields = ('current_temp_dir',)
            if not all([key in updated_fields_eval.keys() for key in required_eval_fields]):
                raise ValueError(f'When writing a checkpoint post_cleanup, must only update the following fields: {required_eval_fields}')
            checkpoint_info["eval_state"].update(updated_fields_eval)

            if updated_fields_algo != {}:
                raise ValueError('When writing a checkpoint post_cleanup, cannot update any algorithm state info, as no algorithm changes occur at this point \n' \
                'this is only intended for cleaning up temporary files and directories on the evaluation side.')
        else:
            raise ValueError(f'Event type {event} not recognised for writing to checkpoint.')

        #Finally writing the checkpoint info to the checkpoint path.
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_info, f) 
 
def run_instances(
        dataloader, 
        fe_sim_obj, 
        logger, 
        loaded_experiment_checkpoint:dict = None,
        experiment_checkpoint_path:str = None,
        enable_adaptation: bool = False,
        resume_bool: bool = False
    ):
    
    if resume_bool:
        logger.info(f'Resuming after {loaded_experiment_checkpoint["eval_state"]["last_completed_case"]} from {dataloader.data[0]["case_name"]}')
        resumed_idx = loaded_experiment_checkpoint["eval_state"]['last_completed_idx'] + 1

    if experiment_checkpoint_path != None:
        write_to_checkpoint(
            checkpoint_path=experiment_checkpoint_path,
            event='pre_loop',
            checkpoint_info=loaded_experiment_checkpoint,
            updated_fields_eval={},
            updated_fields_algo={} #Both are always empty dicts here. We are just writing the initial checkpoint to disk.
    )

    #Iterating through the dataloader.
    for case_idx, data_instance in enumerate(dataloader):
        #Running a check on the data_instance loaded.
        iterate_dataloader_check(data_instance=data_instance)
        

        #Reformat the data instance to pass into the simulation.
        data_instance, case_name = data_instance_reformat(data_instance=data_instance)
        
        #Logging the progress.
        if resume_bool:
            true_idx = case_idx + resumed_idx
            num_samples = len(dataloader) + resumed_idx
            logger.info(f'Sample  {true_idx + 1} of {num_samples}, case_name: {case_name}')
        else:
            num_samples = len(dataloader)
            logger.info(f'Sample {case_idx + 1} of {num_samples}, case_name: {case_name}')


        #Initialising the temporary directory for this data instance.
        tempdir_obj = tempfile.TemporaryDirectory(dir=os.path.dirname(fe_sim_obj.args['seg_root'])) 
        #By default it is the codebase_dir but if externally mounted drives for the results are
        #required then this will be changed to the root of the directory which contains the segmentations directory)
        if experiment_checkpoint_path != None:
            write_to_checkpoint(
                checkpoint_path=experiment_checkpoint_path,
                event='pre_subprocess',
                checkpoint_info=loaded_experiment_checkpoint,
                updated_fields_eval={'current_temp_dir': tempdir_obj.name},
                updated_fields_algo={} #No algo fields to update yet.

        )
        saved_exc = saved_tb = None
        try:
            #Calling the front-end simulator
            algorithm_state = fe_sim_obj(data_instance=data_instance, case_name=case_name, tmp_dir_path=tempdir_obj.name) 
            
            if experiment_checkpoint_path != None:
                #In this case, we are writing the experiment checkpoints - i.e. auto-rerun is configured. 
                if enable_adaptation and algorithm_state == {}:
                    raise Exception('If auto-re run is being configured, then the algorithm state \n' \
                    'cannot be a NoneType when adaptation is enabled. If no adaptation is being performed then' \
                    'adaptation should not have been enabled in the first place.')

                elif enable_adaptation and algorithm_state != {} and list(algorithm_state['meta_algorithm_state'].keys()) != ['algo_cache_name']:
                    #ANOTHER checking point, we placed this condition on the ui-simulator, but being very careful! 
                    write_to_checkpoint(
                        checkpoint_path=experiment_checkpoint_path,
                        event='post_subprocess',
                        checkpoint_info=loaded_experiment_checkpoint,
                        updated_fields_eval={
                            'last_completed_case': case_name, 
                            'last_completed_idx': true_idx if resume_bool else case_idx
                            },
                        updated_fields_algo=algorithm_state
                    )
                    if algorithm_state['meta_algorithm_state'].get('write_state', False):
                        log_config_writer('Current meta-algo state after adaptation triggered', algorithm_state['meta_algorithm_state'], logger)
                        log_config_writer('Number of samples in memory buffer', {'memory buffer num_samples': len(algorithm_state['meta_algorithm_state']['memory_buffer_disk'])}, logger)
                elif not enable_adaptation:
                    write_to_checkpoint(
                            checkpoint_path=experiment_checkpoint_path,
                            event='post_subprocess',
                            checkpoint_info=loaded_experiment_checkpoint,
                            updated_fields_eval={
                                'last_completed_case': case_name, 
                                'last_completed_idx': true_idx if resume_bool else case_idx
                                },
                            updated_fields_algo={} #No algo state fields to update if not adapting
                        )
                else:
                    raise Exception('Unknown subroutine, something went wrong!')
        
        except Exception as e:
            logger.info(f'Exception occurred in simulation for case {case_name}, running cleanup. Exception details: {e}')
            saved_exc = e
            saved_tb = sys.exc_info()[2]
        finally:
            tempdir_obj.cleanup()

            if experiment_checkpoint_path != None:
                write_to_checkpoint(
                    checkpoint_path=experiment_checkpoint_path,
                    event='post_cleanup',
                    checkpoint_info=loaded_experiment_checkpoint,
                    updated_fields_eval={
                        'current_temp_dir': None
                    },
                    updated_fields_algo = {} #No algo state fields to update at cleanup currently.
                )
            if saved_exc != None:
                raise saved_exc.with_traceback(saved_tb)
            
    logger.info('Successfully completed!')

def init_fe(infer_app, experiment_args):
    #Function which initialises the front-end simulator.
    keep_key_list = [
        ##Variables related to the experimental configuration
        'configs_labels_dict', #NOTE: Currently just assuming semantic segmentation support. 
        #Variable related to handling empty foreground cases for automatic segmentation configurations
        'sim_empty_fg_automatic',
        'infer_run_configs',
        'metrics_configs',
        'inf_prompt_procedure_type',
        'inf_init_prompt_config',
        'inf_edit_prompt_config',
        #Variables related to reproducibility and device.
        'random_seed',
        'cuda_deterministic',
        'torch_deterministic',
        'device',
        #Variables related to interaction memory
        'use_mem_inf_edit',
        'im_config', 
        #Variables related to metrics and saving metrics/
        'dice_termination_thresh',
        'metrics_savepaths',
        'exp_results_dir',
        #Variables related to saving segmentations and prompt info
        'seg_root',
        'exp_seg_dir',
        'save_prompts',
        'is_seg_tmp',
        'write_segmentation',
        #Variables related to api-structure
        'dataset_info',
        'enable_adaptation',
        'provide_gold_standard_after_inference'
    ]
    args = {key:val for key,val in experiment_args.items() if key in keep_key_list}

    return FrontEndSimulator(infer_app=infer_app, args=args)


def main():



    ################################# Extraction and construction of any experiment related information #######################################################

    #Extracting parser args.
    args = set_parse() 

    #Reformulate the args into the format required for the experiment:
    experiment_args = gen_experiment_args(args)


    ############################## Initialisation of any required directories for this experiment ####################################################################

    #Initialise the base results folders if it doesn't exist.
    os.makedirs(experiment_args['metrics_root'], exist_ok=True)
    
    #Initialise the base results and seg folders for the dataset at hand. 
    os.makedirs(experiment_args['results_dataset_subdir'], exist_ok=True) 
    os.makedirs(experiment_args['seg_dataset_subdir'], exist_ok=True)

    ############################## Loading a pickle file if continuing an experiment execution ##############################################################
    
    loaded_experiment_checkpoint = None 
    if experiment_args['continue_execution']:
        if os.path.exists(experiment_args['continue_exec_path']):
            with open(experiment_args['continue_exec_path'], 'rb') as f:
                loaded_experiment_checkpoint = pickle.load(f)
        #Only update the checkpoint variables if the file existed and was loaded.

    #Checking that the random seed is the same if continuing an experiment execution.
    if loaded_experiment_checkpoint != None:
        if loaded_experiment_checkpoint["eval_state"]['random_seed'] != experiment_args['random_seed']:
            raise ValueError(f'Random seed has changed from {loaded_experiment_checkpoint["eval_state"]["random_seed"]} to {experiment_args["random_seed"]}.')

        if loaded_experiment_checkpoint["eval_state"]['shuffle_cases'] != experiment_args['shuffle_cases']:
            raise ValueError(f'Shuffle cases boolean has changed from {loaded_experiment_checkpoint["eval_state"]["shuffle_cases"]} to {experiment_args["shuffle_cases"]}.')
        
        #################################### Clearing any untidied temp files ############################
        if loaded_experiment_checkpoint["eval_state"]['current_temp_dir'] != None:
            if os.path.exists(loaded_experiment_checkpoint["eval_state"]['current_temp_dir']):
                #If the temp dir still exists, we remove it.
                try:
                    import shutil
                    shutil.rmtree(loaded_experiment_checkpoint["eval_state"]['current_temp_dir'])
                except Exception as e:
                    raise Exception(f'Could not remove the temporary directory at {loaded_experiment_checkpoint["eval_state"]["current_temp_dir"]} during cleanup before resuming experiment execution. Exception details: {e}')
        experiment_args['resume_bool'] = True

        if experiment_args['enable_adaptation']:
            experiment_args['algo_cache_name'] = loaded_experiment_checkpoint['algorithm_state']['meta_algorithm_state'].get('algo_cache_name')
            if experiment_args['algo_cache_name'] == None:
                raise ValueError('If adaptation is enabled, the algorithm cache name must be provided in the loaded experiment checkpoint algorithm state to proceed.')
    else:
        experiment_args['resume_bool'] = False

        #We generate a new algo cache name if adaptation is enabled, and we do not have a loaded checkpoint.
        if experiment_args['enable_adaptation']:
            experiment_args['algo_cache_name'] = generate_base64_filename(length=12)

    #################################### END OF LOADING CONTINUATION FILE  ###############################################################


    ################################## Configuration and extraction of data-related info.  ######################################################################
    
    #Extraction of the config labels dictionary, and the initialisation of the dataloader
    configs_labels_dict, dataloader = init_task_cases(
        dataset_dir=experiment_args['input_dataset_dir'],
        exp_task_configs=experiment_args['task_configs'],
        shuffle_bool=experiment_args['shuffle_cases'],
        random_seed=experiment_args['random_seed'],
        last_completed_case=loaded_experiment_checkpoint["eval_state"]['last_completed_case'] if loaded_experiment_checkpoint != None else None,
        last_completed_idx=loaded_experiment_checkpoint["eval_state"]['last_completed_idx'] if loaded_experiment_checkpoint != None else None
        )
    #Appending some relevant dataset information to the experiment args for passing through to the front-end simulator.
    num_samples = loaded_experiment_checkpoint["eval_state"]['last_completed_idx'] + 1 + len(dataloader) if loaded_experiment_checkpoint != None else len(dataloader)
    experiment_args['dataset_info']['num_samples'] = num_samples


    #We append the config labels dict to the experiment args. 
    experiment_args['configs_labels_dict'] = configs_labels_dict

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    #Creating a base dir for the experiment to store metrics.
    if not experiment_args['continue_execution'] or loaded_experiment_checkpoint == None:
        os.makedirs(experiment_args['exp_results_dir'], exist_ok=False) 
        #Should not already exist, if continuing is not configured.
        if not experiment_args['root_match']:
            #If we store the metrics and segmentations in the same root, then we do not need to create the base directory again.

            #Creating a base dir for the experiment segmentations to be stored.
            #Should not already exist for the current experimental run if not configured to permit continuation.
            os.makedirs(experiment_args['exp_seg_dir'], exist_ok=False)
    else:
        pass #If continuing an experiment execution and the checkpoint was loaded, then these directories should already exist.

    #Creating the subdirectories and the csv files for the metrics, then returning a dict of the savepaths according to each and every one of these.
    if loaded_experiment_checkpoint == None:
        experiment_args['metrics_savepaths'] = init_metrics_saves(exp_results_dir=experiment_args['exp_results_dir'], metrics_configs=experiment_args['metrics_configs'], configs_labels_dict=configs_labels_dict)
    else:
        #We will load the path from the experiment's configuration pickle file.
        experiment_args['metrics_savepaths'] = loaded_experiment_checkpoint["eval_state"]['metrics_savepaths']
    

    #Based on whether the segmentation is being saved permanently, create the corresponding subdirectories or not. 
    if not experiment_args['is_seg_tmp']:
        create_seg_dirs(
            exp_seg_dir=experiment_args['exp_seg_dir'], 
            infer_run_conf=experiment_args['infer_run_configs'], 
            exist_ok=experiment_args['continue_execution'] #If continuation capability has been configured, then
            # we permit existing dirs.
        )

    #Save the info about the experiment args, save the info about the application config which should be spit out as a method of the callable infer class.

    logger_save_name = f'experiment_{experiment_args["experiment_name"]}_logs'
    exp_setup_logger = experiment_args_logger(logger_save_name=logger_save_name, root_dir=experiment_args['exp_results_dir'], screen=True, tofile=True)
    # exp_setup_logger = logging.getLogger(logger_save_name)
    
    
    #Here we will load the existing information from the interrupted run, if continuing an experiment execution is configured.
    #NOTE: The file might not yet exist if this is the first time we are starting an experiment! We will only be creating
    # this file on the first instance where a case is completely finished! 

    if loaded_experiment_checkpoint == None:
        #In this case, nothing to continue. So we start from scratch.
        exp_setup_logger.info(f'Starting up! {os.linesep}')
        log_config_writer('Experiment Args', experiment_args, exp_setup_logger)

        exp_setup_logger.info(f'Moving onto build now!: {os.linesep} {os.linesep} {os.linesep}')

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Now we move onto building the app
    if loaded_experiment_checkpoint != None:
        #In this case we have a prior experiment checkpoint. 
        infer_app = init_infer_app(experiment_args, loaded_experiment_checkpoint=loaded_experiment_checkpoint)
    else:
        infer_app = init_infer_app(experiment_args, loaded_experiment_checkpoint=None)
    #Extract the app configs using the required method. 
    app_config_dict = infer_app.app_configs()

    if not app_config_dict:
        raise Exception('Should at least return the application name in the app_configs method.')

    #Writing app configs, only if we are not continuing an experiment.
    if loaded_experiment_checkpoint == None:
        log_config_writer('Application Args', app_config_dict, exp_setup_logger)
    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #Build the front-end simulator: 

    fe_sim_obj = init_fe(infer_app=infer_app, experiment_args=experiment_args)


    # Iterating through the dataset:

    if loaded_experiment_checkpoint == None and experiment_args['continue_execution']:
        #Here we will create the first dictionary for saving the experiment checkpoint info.
        if experiment_args['enable_adaptation']:
            loaded_experiment_checkpoint = {
                "eval_state": {
                    'metrics_savepaths': experiment_args['metrics_savepaths'],
                    'random_seed': experiment_args['random_seed'],
                    'shuffle_cases': experiment_args['shuffle_cases'],
                    'last_completed_case': None,
                    'last_completed_idx': -1,
                    'current_temp_dir': None
                    }
                ,
                'algorithm_state': 
                {
                    'meta_algorithm_state': {'algo_cache_name': experiment_args['algo_cache_name']},
                }
            }
        else:
            loaded_experiment_checkpoint = {
                "eval_state": {
                    'metrics_savepaths': experiment_args['metrics_savepaths'],
                    'random_seed': experiment_args['random_seed'],
                    'shuffle_cases': experiment_args['shuffle_cases'],
                    'last_completed_case': None,
                    'last_completed_idx': -1,
                    'current_temp_dir': None
                    }
                ,
                "algorithm_state": { }
            }


    run_instances(
        dataloader=dataloader, 
        fe_sim_obj=fe_sim_obj, 
        logger=exp_setup_logger, 
        loaded_experiment_checkpoint=loaded_experiment_checkpoint,
        experiment_checkpoint_path=experiment_args['continue_exec_path'] if experiment_args['continue_execution'] else None,
        enable_adaptation=experiment_args['enable_adaptation'],
        resume_bool=experiment_args['resume_bool']
        )

if __name__=='__main__':
    main()


