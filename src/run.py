import argparse
import copy
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
import re 
from typing import Optional
codebase_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
sys.path.append(codebase_dir)
from front_back_interactor.simulation_orchestrator import FrontEndSimulator 
from src.general_utils.logging import experiment_args_logger
from src.general_utils.dict_utils import (
    extractor, 
    extract_config, 
    dict_deep_equals,
    has_path
)
from src.data.utils import data_instance_reformat, iterate_dataloader_check, init_task_cases
from src.results_utils.metric_save_util import init_all_csvs
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
run_vs_seed = {
    'run1': 341103,
    'run2': 55432,
    'run3': 754537
}

def str2bool(v):
    assert v.lower() in ('true', 'false'), 'Boolean value expected, please check your input arguments, received {}.'.format(v)
    return v.lower() in ('true')

def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    
    #Pathing arguments
    parser.add_argument('--data_root', type=str, default=codebase_dir)
    parser.add_argument('--dataset_name', type=str, default='Dataset005_Prostate')
    parser.add_argument('--app_root', type=str, 
                        # default='/home/parhomesmaeili/IS_Codebase_Forks/nnInteractive_Fork'
                        default='/home/parhomesmaeili/MY METHOD/'
    ) 
    #NOTE:Just set for debugging purposes.
    #This acts as the name of the app, but also temporarily acts as the relative path name within the input_applications folder in the app root folder.
    parser.add_argument('--app_name', type=str, default='CLoPA') #'nnInteractive_App')
    parser.add_argument('--metrics_root', type=str, default=os.path.join(codebase_dir, 'results'))
    parser.add_argument('--seg_root', type=str, default=os.path.join(codebase_dir, 'results'))
    parser.add_argument('--continue_exec_root', type=str, default='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/continue_execution_files') #None
    
    #Experiment config paths:
    parser.add_argument('--configs_root', type=str, default=os.path.join(codebase_dir, 'exp_configs'))
    #The following arguments are relative to the expermiments configs directory 
    parser.add_argument('--experiment_manifest_filename', type=str, default='experiment_manifest.json')
    parser.add_argument('--metric_conf_filename', type=str, default='metrics_configs.txt')
    #prompter pathing. prompter manifest = the set of prompters that will be used in a given experiment,
    #it pairs init and edit prompt configs together. prompt_conf_filename contains configurations which
    #are referenced by the manifest and actually contain a set of configurations for potential prompters.
    parser.add_argument('--prompter_manifest_filename', type=str, default='prompter_manifest.json')
    parser.add_argument('--prompt_conf_filename', type=str, default='prompts_configs.txt')
    parser.add_argument('--task_conf_filename', type=str, default='task_configs.txt')

    
    #########################
    
    #Name of the experiment, used to store results, checkpoints for continuation/auto
    parser.add_argument('--experiment_basename', type=str, required=False, default='clopatrain_debug_experiment6_run1') #None)
    #Runtime environment arguments/system control.
    parser.add_argument('--continue_execution', type=str2bool, default=False)
    #seeding
    parser.add_argument('--shuffle_cases', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=341103)
    parser.add_argument('--run_num', type=str, required=False, default='run1')
    #cuda and determinism arguments
    parser.add_argument('--cuda_deterministic_disable', type=str2bool, default=False)
    parser.add_argument('--torch_deterministic_disable', type=str2bool, default=False)
    parser.add_argument('--device_idx', type=int, default=0)

    #Auxiliary output arguments/controls whether to write additional outputs in addition to the
    #metrics.
    parser.add_argument('--write_segmentation', type=str2bool, default=False)
    # Whether to write the segmentations at all, or not (default is false for now)
    parser.add_argument('--is_seg_tmp', type=str2bool, default=True)
    # Whether the segmentations are written as temp files, if they are written. Or whether they are permanent files.
    parser.add_argument('--save_prompts', type=str2bool, default=False)
    
    #Adaptation training related arguments.

    #Whether to skip the metric and prompt generation steps.
    #May be used to execute adaptation with gold-standard annotations.
    parser.add_argument('--skip_metric', type=str2bool, default=False) 
    parser.add_argument('--skip_prompt', type=str2bool, default=False) 
    
    #This is a bool which controls the adaptation config name, to steer the registry to pull the relevant
    #config for adaptation. Realistically this is just a byproduct of how we have intertwined the
    #apps and this framework... probably not ideal. 
    parser.add_argument('--adaptation_config_name', type=str, default=None)
    #Self explanatory, whether adaptation is enabled or not.
    parser.add_argument('--enable_adaptation', type=str2bool, default=False)
    
    #Adapted method hold-out inference related arguments:
 
    #Whether we are going to execute on a pre-adapted method or not -> this informs whether
    #we initialise an app which contained adaptation mechanisms with an adapted checkpoint.
    #(We perform holdout inference on checkpoints even with adaptive methods as this represents expected
    #performance as a function of training data).
    parser.add_argument('--execute_on_adapted', type=str2bool, default=False)
    #Training episode number to pull the checkpoint from for hold out inference.
    #ONLY INTENDED when execute_on_adapted is enabled.
    parser.add_argument('--adaptation_episode', type=int, default=None) 
    #What the cache name is within the algo cache directory (once again a byproduct of how we have
    #intertwined the apps and framework to handle job submissions which can be interrupted...)
    parser.add_argument('--algo_cache_name', type=str, default=None)
    #This is a string which denotes the reference name/run from which we have stored an experiment
    # execution checkpoint, from which we will be pulling episodes from.
    parser.add_argument('--reference_experiment_checkpoint', type=str, default=None)
    #This bool controls whether we provide the gold standard annotation from the reference annotation,
    #or if we pass through an inferred segmentation (i.e., through what the app has generated at
    #termination of the editing process) for the adaptation step.
    parser.add_argument('--provide_gold_standard_after_inference', type=str2bool, default=False)

    #Experiment configuration arg
    parser.add_argument('--experiment_conf_id', type=int, default=6)
    args = parser.parse_args()
    return args

def gen_experiment_args(args):
    #Takes an argparse namedspace obj and constructs a dictionary required for the the run script (most of which will be inherited for the front end init).

    output_dict = dict() 


    ################################ Pathing related arguments #############################################

    #Setting the app name for the experiment, also available for the build script. 
    output_dict['app_name'] = args.app_name 
    #Creating the relative path to the base build dir within the app.
    output_dict['build_app_rel_path'] = 'src_validate'
    #Temporarily creating an abspath using this relative path:
    output_dict['build_app_abspath'] = os.path.join(args.app_root, output_dict['app_name'], output_dict['build_app_rel_path'])
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

    #Config pathing directory arguments:
    exp_conf_dir = os.path.join(args.configs_root, args.dataset_name) 
    exp_manifest_path = os.path.join(exp_conf_dir, args.experiment_manifest_filename)
    prompter_manifest_path = os.path.join(exp_conf_dir, args.prompter_manifest_filename)
    prompt_conf_path = os.path.join(exp_conf_dir, args.prompt_conf_filename) 
    task_conf_path = os.path.join(exp_conf_dir, args.task_conf_filename)
    metric_conf_path = os.path.join(exp_conf_dir, args.metric_conf_filename)
    

    ############################ Runtime environment/system control related arguments ########################################

    #Auto-continuation related arguments, requires a fixed experiment name such that it need not be
    #configured manually for continuation. 
    if args.continue_execution:
        if args.experiment_basename == None:
            raise ValueError('If configured for continuing an experiment execution, must provide the experiment basename to \n ' \
            'continue from otherwise we cannot proceed.')
        else:
            output_dict['experiment_name'] = f'{args.experiment_basename}_{args.run_num}' #_{args.split_name}'
            # named_experiment = True 
            assert f'experiment{args.experiment_conf_id}' in args.experiment_basename, 'If using named experiments, the experiment manifest filename must contain the experiment name to ensure consistency and clarity, please check your input arguments and the experiment manifest filename to ensure this is the case.'
        if args.continue_exec_root == None:
            raise ValueError('If continuing an experiment execution, must provide the root directory of the \n'
            'files to write/read where to continue from.')
        else:
            output_dict['continue_exec_path'] = os.path.join(args.continue_exec_root, output_dict['experiment_name'] + '.pkl')
    else:
        if args.experiment_basename == None:
            output_dict['experiment_name'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # named_experiment = False
        else:
            raise Exception('If not continuing an experiment execution, cannot provide an experiment basename, \n'
            'this is redundant.')
    
    #Storing the directory for the results, segs, with the experiment name.
    output_dict['exp_results_dir'] = os.path.join(output_dict['results_dataset_subdir'], output_dict['experiment_name'])
    output_dict['exp_seg_dir'] = os.path.join(output_dict['seg_dataset_subdir'], output_dict['experiment_name'])

    #Storing the continue execution bool, this informs downstream checkpointing.
    output_dict['continue_execution'] = args.continue_execution

    #Extracting the random seed/randomness related info, cross-checking it with the experiment 
    #number (for repeats).
    output_dict['shuffle_cases'] = args.shuffle_cases
    output_dict['random_seed'] = args.random_seed
    if args.random_seed != None:
        #Lets extract the run num. 
        run_num = args.run_num #re.search(r'run(\d+)', output_dict['experiment_name'])
        if run_num is None:
            raise ValueError('No run num was found at all with a pre-determined random seed!')
        if run_num not in run_vs_seed:
            raise ValueError(f'If a random seed is provided, the experiment name must contain run-num substring \n'
            f'where {run_num} is in {run_vs_seed.keys()} to match the available seeds in the run_vs_seed dictionary.')
        
        assert run_vs_seed.get(run_num) == args.random_seed, f'The provided random seed {args.random_seed} does not match the expected seed {run_vs_seed.get(run_num)} for run num {run_num}. Please check your input arguments.'
    output_dict['cuda_deterministic'] = not args.cuda_deterministic_disable
    output_dict['torch_deterministic'] = not args.torch_deterministic_disable

    #Extracting the device info. 
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
    
    ########################### Auxiliary output related arguments ####################################
    output_dict['write_segmentation'] = args.write_segmentation
    output_dict['is_seg_tmp'] = args.is_seg_tmp
    output_dict['save_prompts'] = args.save_prompts
    if output_dict['is_seg_tmp'] == output_dict['write_segmentation']:
        raise ValueError('The is_seg_tmp and write_segmentation flags cannot be the same, please check your input arguments. If write_segmentation is false, then it does not make sense for is_seg_tmp to be true, as we are not writing segmentations at all. If write_segmentation is true, then it does not make sense for is_seg_tmp to be false, as we want to write the segmentations as temp files.')

#################################### Adaptation Related Arguments ###########################################

    # First we check whether adaptation is enabled or not.
    #If it is enabled, we assert that we cannot be running "execute on adapted", which is the 
    #process of performing hold-out inference on an adapted model's checkpoints.
    if args.enable_adaptation:
        assert not args.execute_on_adapted, 'If adaptation is enabled, then executing on pre-trained adapted should not be enabled, please check your input arguments.'
        if not args.continue_execution:
            raise ValueError('Adaptation almost certainly requires the capability to continue execution, please enable \n'
            'the continue execution boolean flag to proceed for reassurance.')
        else:
            assert args.algo_cache_name is None, 'If adaptation is enabled, must not provide the algorithm cache name as an input argument, please check your input arguments. The algorithm cache name needs to be extracted from the adaptation experiment_checkpoint.'
            assert args.adaptation_config_name is not None, 'If adaptation is enabled, must provide the adaptation config name as an input argument, please check your input arguments.'
            output_dict['adaptation_config_name'] = args.adaptation_config_name
            output_dict['enable_adaptation'] = args.enable_adaptation
            output_dict['provide_gold_standard_after_inference'] = args.provide_gold_standard_after_inference
    else:
        output_dict['enable_adaptation'] = args.enable_adaptation
        assert args.adaptation_config_name == None, 'If adaptation is disabled, cannot provide adaptation config name, please check your input arguments.'
        output_dict['adaptation_config_name'] = None #Irrelevant if adaptation is disabled.
        #We cannot provide gold standard after inference because there should not be adaptation in the first place!
        assert args.provide_gold_standard_after_inference == False, 'If adaptation is disabled, cannot provide gold standard after inference, please set this flag or check your input arguments.'
        output_dict['provide_gold_standard_after_inference'] = False #Irrelevant if adaptation is disabled.

    #Now we handle some arguments for the config where we want to skip metric computation and prompting
    #this may be the case if we just want to run the adaptation process with gold standard annotations.

    output_dict['skip_metric'] = args.skip_metric
    output_dict['skip_prompt'] = args.skip_prompt
        
    if args.skip_metric or args.skip_prompt:
        assert args.enable_adaptation, 'If skipping metric or prompt generation, then adaptation must be enabled, as we need the adaptation to be able to run without the metrics and prompting generation steps, please check your input arguments.'
        if 'enable_adaptation' in output_dict:
            assert output_dict['enable_adaptation'] == args.enable_adaptation, 'The enable adaptation value in the output dict does not match the input argument, please check your input arguments and the logic for setting the enable adaptation value in the output dict to ensure consistency and clarity.'
        else:
            raise Exception('We should have enabled adaptation by this point, please check the logic, safety precautions implemented.')
        assert args.continue_execution, 'Continue execution must be enabled as skip metric/prompt can only be used'
        'for adaptation.'
        if 'continue_execution' in output_dict:
            assert output_dict['continue_execution'] == args.continue_execution, 'The continue execution value in the output dict does not match the input argument, please check your input arguments and the logic for setting the continue execution value in the output dict to ensure consistency and clarity.'
        else:
            raise Exception('We should have set the continue execution value in the output dict by this point, please check the logic, safety precautions implemented.')
    if output_dict['skip_prompt']:
        #If we skip prompting during adaptation, then there is no mechanism to generate a mask for training, other than
        #to pass through a gold standard annotation.
        assert args.provide_gold_standard_after_inference, 'If skipping prompting generation, then '
        'providing gold standard after inference must be enabled. We need to be able to provide an annotation for'
        'the adaptation step, please check your input arguments.'
        assert args.skip_metric, 'If skipping prompting generation, then skipping metric generation mus be enabled. This is because if we are skipping prompt generation, then we are likely skipping metric generation as well since the metrics are dependent on the prompts for generating predictions, please check your input arguments.'     
    #Somewhat amusingly, I think we have implemented assertion checks which invert the logic of
    #the checks implemented when adaptation is enabled. Let us leave this be. 
    output_dict['execute_on_adapted'] = args.execute_on_adapted
    if output_dict['execute_on_adapted']:
        if args.enable_adaptation:
            raise ValueError('If executing on adapted, then adaptation should not be enabled, as we are executing on a pre-adapted method, please check your input arguments.')
        else:
            if 'enable_adaptation' in output_dict:
                assert output_dict['enable_adaptation'] == args.enable_adaptation, 'The enable adaptation value in the output dict does not match the input argument, please check your input arguments and the logic for setting the enable adaptation value in the output dict to ensure consistency and clarity.'
            else:
                raise Exception('We should have set the enable adaptation value in the output dict by this point, please check the logic, safety precautions implemented.')
            assert args.adaptation_episode is not None, 'If executing on adapted, must provide an adaptation episode number to pull the checkpoint from, please check your input arguments.'
            output_dict['adaptation_episode'] = args.adaptation_episode
            assert args.algo_cache_name is not None, 'If executing on adapted, must provide the algorithm cache name to pull the checkpoint from, please check your input arguments.'
            output_dict['algo_cache_name'] = args.algo_cache_name
            assert args.reference_experiment_checkpoint is not None, 'If executing on adapted, must provide the reference experiment checkpoint name to pull info from, please check your input arguments.'
            output_dict['reference_experiment_checkpoint'] = os.path.join(args.continue_exec_root, args.reference_experiment_checkpoint)
    else:
        assert args.adaptation_episode is None, 'If not executing on adapted, cannot provide an adaptation episode number, please check your input arguments.'
        output_dict['adaptation_episode'] = None
        assert args.reference_experiment_checkpoint is None, 'If not executing on adapted, cannot provide a reference experiment checkpoint, please check your input arguments.'
        output_dict['reference_experiment_checkpoint'] = None

    ############################# Experiment configuration arguments #########################################
    exp_config_name = f'exp_config_{args.experiment_conf_id}'
    #First we pull the experiment config from the manifest.
    output_dict['experiment_config'] = extract_config(
        exp_manifest_path, 
        exp_config_name
    )

    #Now we will pull the task, metric configs, prompt configs and prompt manifest for cross-checking.
    orig_task_configs = extract_config(
        task_conf_path,
        None
    )
    orig_prompter_manifest = extract_config(
        prompter_manifest_path,
        None
    )
    orig_prompt_configs = extract_config(
        prompt_conf_path,
        None
    )
    orig_metric_configs = extract_config(
        metric_conf_path,
        None
    )

    #Now we will cross-check the configs provided with the experiment manifest against the
    #original task, prompt, metric configs to ensure that the config provided in the manifest is
    # consistent with the referenced configs.
    task_id = output_dict['experiment_config']['task']['task_id']
    metric_id = output_dict['experiment_config']['metrics']['metrics_config_id']
    prompter_id = output_dict['experiment_config']['prompter']['prompter_id']

    orig_task_pulled = extractor(orig_task_configs, (task_id,))
    orig_metric_pulled = extractor(orig_metric_configs, (metric_id,))
    orig_prompter_pulled = extractor(orig_prompter_manifest, (prompter_id,))
    #We do a dict deep equals
    assert dict_deep_equals(orig_task_pulled, output_dict['experiment_config']['task']['config']), 'The task config pulled based on the experiment manifest does not match the original task config with the corresponding id, please check your experiment manifest and task configs for consistency and clarity.'
    assert dict_deep_equals(orig_metric_pulled, output_dict['experiment_config']['metrics']['config']), 'The metric config pulled based on the experiment manifest does not match the original metric config with the corresponding id, please check your experiment manifest and metric configs for consistency and clarity.'
    assert dict_deep_equals(orig_prompter_pulled, output_dict['experiment_config']['prompter']['config']), 'The prompt config pulled based on the experiment manifest does not match the original prompt config with the corresponding id, please check your experiment manifest and prompt configs for consistency and clarity.'


    #If all ok, lets assign the relevant configs
    output_dict['task_configs'] = output_dict['experiment_config']['task']['config']
    output_dict['metrics_configs'] = output_dict['experiment_config']['metrics']['config']
    output_dict['prompter_configs'] = output_dict['experiment_config']['prompter']['config']

    #Lets make some assertions on what keypaths need to be provided.
    #tasks first
    assert has_path(output_dict['task_configs'], ('data_sampling', 'sample_group_category'))
    assert has_path(output_dict['task_configs'], ('data_sampling', 'image_conf'))
    assert has_path(output_dict['task_configs'], ('infer_info',))
    assert has_path(output_dict['task_configs'], ('seg_problem',))
    assert has_path(output_dict['task_configs'], ('data_transforms', 'semantic_class_mapping'))
    #prompters
    assert has_path(output_dict['prompter_configs'], ('init_prompt_conf',))
    assert has_path(output_dict['prompter_configs'], ('edit_prompt_conf',))
    assert has_path(output_dict['prompter_configs'], ('infer_edit_nums',))
    assert has_path(output_dict['prompter_configs'], ('use_mem_inf_edit',))
    assert has_path(output_dict['prompter_configs'], ('im_conf_remove_init',))
    assert has_path(output_dict['prompter_configs'], ('im_conf_mem_len',))
    assert has_path(output_dict['prompter_configs'], ('annotation_conf',))

    #metrics
    assert has_path(output_dict['metrics_configs'], ('metrics',))
    assert has_path(output_dict['metrics_configs'], ('early_termination_criterion','metric'))
    assert has_path(output_dict['metrics_configs'], ('early_termination_criterion','threshold'))
    assert has_path(output_dict['metrics_configs'], ('data_sampling', 'annotation_conf'))

    ######################   Task definition arguments ######################################
    output_dict['seg_problem'] = output_dict['task_configs']['seg_problem']
    #Configuring the experimental configs, first the infer run configs:
    infer_info = output_dict['task_configs']['infer_info']
    assert 'infer_init' in infer_info
    assert 'infer_edit_bool' in infer_info
    assert 'sim_empty_fg_automatic' in infer_info
    num_edits = output_dict['prompter_configs']['infer_edit_nums']

    #Pulling the number of iters for editing from the prompter config, we will cross-check this against
    #the infer info. Why are they separate? Because prompter behaviour -> number of edits, meanwhile
    #init mode, or editing use are mostly informing what the expected api should be? Edits (aside from interaction
    #memory are fairly homogenous in the request structure otherwise/or what is expected), and different
    #to the init sometimes. E.g., prompts which can only be used for init, but not edits.

    #The infer_modes inform part of what the fingerprint of the segmentation task is expected, 
    # meanwhile the prompter defines what will be used to generate prompts.
    
    #However this will be wrapped together in the higher-level of "init, edit format" and "number of
    # "edits" as on the prompting side we need to loop iteratively.

    if infer_info['infer_edit_bool']:
        assert type(num_edits) == int, 'If infer_edit_bool is true, then the number of edits must be an integer, please check your experiment manifest and prompter configs for consistency and clarity.'
        assert num_edits != None and num_edits > 0, 'If infer_edit_bool is true, then the number of edits must be a positive integer, please check your experiment manifest and prompter configs for consistency and clarity.'
        output_dict['infer_run_configs'] = {
            'init':infer_info['infer_init'],
            'edit_bool':True,
            'num_edits': num_edits
        }
    else:
        assert num_edits == 0 or num_edits == None, 'If infer_edit_bool is false, then the number of edits must be 0 or None, please check your experiment manifest and prompter configs for consistency and clarity.'
        output_dict['infer_run_configs'] = {
            'init':infer_info['infer_init'],
            'edit_bool':False,
            'num_edits': None
        }
    #Extracting the config dict for handling the empty fg....? 
    output_dict['sim_empty_fg_automatic'] = infer_info['sim_empty_fg_automatic']

    #Loading in the relevant information from the dataset.

    #We pull the spacing config info here as it may be needed. If it does not exist we raise an error
    #if the metrics configs require it (e.g. for NSD with tolerance sf instead of mms), but we don't want to pull it if it isn't needed to avoid unnecessary dependencies on the dataset configs.
    try:
        spacing_config = extract_config(
            os.path.join(exp_conf_dir, 'spacing_config.json'), 
            None
            )
        reference_spacing = spacing_config['reference_spacing']
    except:
        spacing_config = None 
        reference_spacing = None 

    #TODO: Update this for the new dataset-level contract.
    output_dict['dataset_level_data_schema'] = {
    'dataset_name': args.dataset_name,
    'dataset_image_channels': extract_config(os.path.join(args.data_root, 'datasets', args.dataset_name, 'dataset.json'), 'channel_names'),
    'task_channels': extractor(output_dict['task_configs'], ('data_sampling', 'image_conf', 'image_channel')),
    'spacing_info': spacing_config
    }

    ########################### Experiment manifest prompter arguments ##############################

    #Extracting the prompting class initialisation configs...
    #first lets cross-check that the config subdict is matching with what is in the prompt configs..
    if output_dict['prompter_configs']['init_prompt_conf']['name'] is not None:
        assert dict_deep_equals(
            orig_prompt_configs[output_dict['prompter_configs']['init_prompt_conf']['name']],
            output_dict['prompter_configs']['init_prompt_conf']['config']
        )
    if output_dict['prompter_configs']['edit_prompt_conf']['name'] is not None:
        assert dict_deep_equals(
            orig_prompt_configs[output_dict['prompter_configs']['edit_prompt_conf']['name']],
            output_dict['prompter_configs']['edit_prompt_conf']['config']
        )
    #If we passed it, then we can now assign the config dict.
    output_dict['inf_init_prompt_config'] =  output_dict['prompter_configs']['init_prompt_conf']['config'] 
    output_dict['inf_edit_prompt_config'] = output_dict['prompter_configs']['edit_prompt_conf']['config']

    if infer_info['infer_init'] == 'Interactive Init':
        assert output_dict['inf_init_prompt_config'] is not None, 'If infer init is interactive init, then the init prompt config must not be None, please check your experiment manifest and prompter configs for consistency and clarity.'
    if infer_info['infer_edit_bool']:
        assert output_dict['inf_edit_prompt_config'] is not None, 'If infer edit bool is true, then the edit prompt config must not be None, please check your experiment manifest and prompter configs for consistency and clarity.'

    #Extracting configurations for interaction memory usage -> informs use of interaction memory in
    #prompting and the memory length for conditioning the prompt generation.

    #The use of inf im for conditioning the prompt generation.
    output_dict['use_mem_inf_edit'] = output_dict['prompter_configs']['use_mem_inf_edit'] 
    #Memory maintenance config in the front-end-simulator (e.g. memory len, keeping init)
    output_dict['im_config'] = {
        'keep_init': not output_dict['prompter_configs']['im_conf_remove_init'],
        'im_len':output_dict['prompter_configs']['im_conf_mem_len'], 
    }

    ####################### Experiment manifest metric arguments ################################

    output_dict['metrics_configs'] = process_metric_config(
        output_dict['metrics_configs'],
        output_dict['skip_metric'],
        reference_spacing
    )
    #Now we extract the termination condition threshold:
    output_dict['early_termination_criterion'] = output_dict['metrics_configs']['early_termination_criterion'] 
    #deprecated -> ['dice_termination_thresh'] = args.dice_termination_thresh 
    
    return output_dict 

def process_metric_config(
        metric_config: dict, 
        skip_metric: bool,
        reference_spacing: list[float]
    ) -> dict:
    #Function which processes the metric configs for cases where we do not have a trivial config.

    #First we check whether we are calculating metrics to be stored, or if we are skipping calculation. If we are 
    #skipping calculation (we will still require that a termination criterion is provided )
    if skip_metric:
        #If we are skipping metrics for storage then the only metric configs we need to retain are those which are
        #relevant for early termination criteria. 
        assert 'early_termination_criterion' in metric_config, 'If skipping metric calculation, then the early termination criterion must be provided in the metric configs, please check your metric configs and input arguments for consistency and clarity.'
        #This should have already been checked but just being safe here.
        
        #we pull the metric name.
        metric_name = metric_config['early_termination_criterion'].get('metric', None)
        if metric_name is None:
            raise ValueError('Even when skipping metric calculation, then the early termination criterion must still specify a metric name, please check your metric configs and input arguments for consistency and clarity.')
        #We check that the metric name is in the metric configs provided, as we will need to have the config available
        #for the termination criterion metric to be configured!
        metrics_to_remove = set()
        for metric_name, conf in metric_config['metrics'].items():
            if metric_name != metric_config['early_termination_criterion']['metric']:
                #we delete the config for this metric as we will not be calculating it at all!
                metrics_to_remove.add(metric_name)
        #So unglamourous.
        for metric_name in metrics_to_remove:
            metric_config['metrics'].pop(metric_name)

    #E.g., for NSD where we may want to calculate across multiple tolerance values.
    for metric_name, conf in metric_config['metrics'].items():
        if metric_name == 'NSD':
            if 'tolerance_mm' not in conf:
                assert 'tolerance_sf' in conf, 'If no tolerance_mm provided for NSD metric config, then must provide a tolerance_sf value to calculate the tolerance based on the dataset voxel spacing, please check your metric configs.'
                #If we have a tolerance sf, then we must calculate tolerance mms from the tolerance sf. 
                if reference_spacing:
                    tolerance_mms = [float(i) * float(reference_spacing) for i in conf['tolerance_sf']]
                    assert isinstance(tolerance_mms, list), 'The calculated tolerance mms values must be a list, even if there is only one value, to be consistent with the case where the tolerance mms values are provided directly in the metric configs, please check your metric configs and dataset spacing config to ensure this is the case.'
                    conf['tolerance_mms'] = {index:tolerance_mm for index,tolerance_mm in enumerate(tolerance_mms)}

                    #We add the tolerance mms values to the config dict for the metric, we will use these for the metric calculations. We deepcopy just to be safe and avoid any weird pointer issues.
                    del conf['tolerance_sf'] #We remove the tolerance sf value as it is no longer needed and to avoid confusion.
                else:
                    raise ValueError('If using a tolerance sf value for NSD metric config, then the dataset spacing config must be provided to calculate the tolerance_mm values, please check your metric configs and dataset configs to ensure this is the case.')
            else:
                tolerance_mms = copy.deepcopy(conf['tolerance_mm'])
                del conf['tolerance_mm'] #We remove the tolerance mm value as it is no longer needed and to avoid confusion, we will just use the tolerance mms values for the metric calculations. We deepcopy just to be safe and avoid any weird pointer issues.
                conf['tolerance_mms'] = {0: tolerance_mms} #We just rename the key to be consistent with the case where we calculate the tolerance mms from the tolerance sf, this is just
            if len(conf['tolerance_mms']) > 1:
                #We add a new key which says that we have multiple nsds so downstream it knows to treat
                #this differently!. It points to the key where that corresponding list of parameterisations
                #are.
                conf['multiple_parameter_values'] = 'tolerance_mms'
    return metric_config


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

            if isinstance(infer_run_conf['num_edits'], int):
                for iter in range(1, infer_run_conf['num_edits'] + 1):
                    os.makedirs(os.path.join(exp_seg_dir, f'Interactive Edit Iter {iter}'), exist_ok=exist_ok)
            else:
                raise TypeError('If running editing, needs to be an int type for the number of iterations performed.')

def init_metrics_saves(exp_results_dir, metrics_configs, semantic_id_dict):
    #Function which creates metrics dirs and initialises the csvs for the metrics saver, takes the results base dir and the metrics configs dict.
    metrics_dir = os.path.join(exp_results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=False)

    return init_all_csvs(metrics_dir, metrics_configs, semantic_id_dict)

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

def generate_dataset_level_schema(
    experiment_args: dict
    ):
    assert experiment_args != None, 'The experiment args must be provided to generate the dataset level schema, please check your input arguments.'
    assert 'dataset_level_data_schema' in experiment_args, 'The dataset level data schema must be provided in the experiment args to generate the dataset level schema'
    
    dataset_level_schema = dict()
    #Appending the data schema
    dataset_level_schema['data_schema'] = experiment_args['dataset_level_data_schema']
    #Appending some segmentation task schema.
    dataset_level_schema['segmentation_task_schema'] = {
    'semantic_id_dict': experiment_args['semantic_id_dict']
    }
    #Now appending the full image cache.
    dataset_level_schema['full_image_cache'] = experiment_args['full_image_cache']
    # dataset_level_schema['data_root'] = os.path.join(experiment_args['input_dataset_dir'])
    dataset_level_schema['full_image_cache'] = {
        case_id: {k_1: {
            k_2: os.path.join(experiment_args['input_dataset_dir'], v_2) for k_2,v_2 in v_1.items()
            } if isinstance(v_1, dict) else v_1 for k_1,v_1 in case_cache.items()}
        for case_id, case_cache in experiment_args['full_image_cache'].items()
    }
    #Assigning the dataset level schema. 
    experiment_args['dataset_level_schema'] = dataset_level_schema

    return experiment_args

def build_app(
    build_app_path, 
    device, 
    dataset_level_schema: dict,
    adaptation_config_name: str,
    algorithm_state: dict, 
    enable_adaptation: bool,
    execute_on_adapted: bool,
    adaptation_episode: int | None, 
    algo_cache_name:str):

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

    assert device != None, 'Device must be provided to build the inference app, please check your input arguments and the build app script to ensure this is the case.'
    assert dataset_level_schema != None, 'The dataset level config must be provided to build the inference app, please check your input arguments and the build app script to ensure this is the case.'

    return InferApp(
        device, 
        dataset_level_schema,
        adaptation_config_name, 
        algorithm_state, 
        enable_adaptation, 
        execute_on_adapted, 
        adaptation_episode, 
        algo_cache_name
        )

def init_app(
    experiment_args:dict, 
    loaded_experiment_checkpoint: Optional[dict] = None, 
    reference_experiment_checkpoint: Optional[dict] = None
    ): 

    #Function which finds and initialises the app using the build script, then checks it has the necessary methods. 
    
    #We will DEMAND that there is some algorithm state info, even if empty for a non adaptive algorithm. 
    #Even if its an empty dict for a non-adapting model (this is just a hack for consistency).
    
    
    #Algo state either = empty dict in checkpoint if auto continue enabled but not adapting,
    # empty dict if not auto continue. 
    # empty dict if adapting but not yet saved the checkpoint before starting evaluation.
    enable_adaptation = experiment_args.get('enable_adaptation', False)
    #Adding another safety check:
    if not enable_adaptation:

        assert experiment_args.get('adaptation_config_name') == None, 'If adaptation is disabled, cannot provide adaptation config name, please check your input arguments.'
        if experiment_args.get('execute_on_adapted') == None:
            raise Exception('The execute_on_adapted argument must be provided as a boolean in the experiment args, please check your input arguments.')
        
        if experiment_args.get('execute_on_adapted'):
            if reference_experiment_checkpoint == None:
                raise ValueError('We need a loaded experiment_checkpoint for loading adapted info')
            else:
                algorithm_state = reference_experiment_checkpoint['algorithm_state']

            if experiment_args.get('algo_cache_name', None) == None:
                raise ValueError('If exec on adapt is enabled, must provide a valid '
                    'algo_cache_name in the experiment args \n'
                    'to proceed.')
            else:
                algo_cache_name = experiment_args['algo_cache_name']
            #sanity check!
            assert algo_cache_name == algorithm_state['meta_algorithm_state']['algo_cache_name']
            #

        else:
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
    
    app = build_app(
        build_app_path=experiment_args['build_app_abspath'], 
        device=experiment_args['device'], 
        dataset_level_schema=experiment_args['dataset_level_schema'],
        adaptation_config_name=experiment_args.get('adaptation_config_name', None),
        algorithm_state=algorithm_state, 
        enable_adaptation=enable_adaptation,
        execute_on_adapted=experiment_args.get('execute_on_adapted', False),
        adaptation_episode=experiment_args.get('adaptation_episode', None),
        algo_cache_name=algo_cache_name)
    
    if not callable(app):
        raise Exception('The inference app must be callable class.')
    else:
        #Check if it has a call attribute!
        try:
            callback = getattr(app, "__call__")
        except:
            raise Exception('The inference app did not have a function __call__')
        
        #Check if it has a app_configs attribute!
        try:
            app_configs_callback = getattr(app, "app_configs")
        except:
            raise Exception('The inference app did not have a function app_configs (which can be empty!), for saving the app configs to the experiment logger file.')

        if enable_adaptation:
            #If adaptation is enabled, check if it has an accept_new_sample method.
            try:
                accept_new_sample_callback = getattr(app, "accept_new_sample")
            except:
                raise Exception('The inference app did not have a function accept_new_sample required for adaptation.')

            if not callable(accept_new_sample_callback):
                raise Exception('The initialised inference app object had an accept_new_sample attribute which was not a callable function.')
            
            try:
                trigger_adaptation_callback = getattr(app, "trigger_adaptation")
            except:
                raise Exception('The inference app did not have a function trigger_adaptation required for adaptation.')
            
            if not callable(trigger_adaptation_callback):
                raise Exception('The initialised inference app object had a trigger_adaptation attribute which was not a callable function.')


        if not callable(callback):
            raise Exception('The initialised inference app object had a __call__ attribute which was not a callable function.') 
        
        if not callable(app_configs_callback):
            raise Exception("The initialised inference app object had a 'app_configs' attribute which was not a callable function. ")
        
    return app

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
    else:
        log_config_writer('Case list for this experiment', {'Case list':[data_instance['case_name'] for data_instance in dataloader]}, logger)
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

def init_fe(app, experiment_args):
    #Function which initialises the front-end simulator.
    keep_key_list = [
        ##Variables related to the experimental configuration
        'semantic_id_dict', #NOTE: Currently just assuming semantic segmentation support. 
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
        'early_termination_criterion', # 'dice_termination_thresh',
        'metrics_savepaths',
        'exp_results_dir',
        #Variables related to saving segmentations and prompt info
        'seg_root',
        'exp_seg_dir',
        'save_prompts',
        'is_seg_tmp',
        'write_segmentation',
        #schema variables, which will provide dataset level information, which could be useful for sample-level
        #schema formatting.
        'dataset_level_schema',
        #Adaptation configs
        'enable_adaptation',
        'provide_gold_standard_after_inference',
        'skip_metric',
        'skip_prompt'
    ]
    args = {key:val for key,val in experiment_args.items() if key in keep_key_list}

    return FrontEndSimulator(app=app, args=args)


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
    #Here we load the reference checkpoint for executing inference on an adapted model.
    reference_experiment_checkpoint = None
    if experiment_args['execute_on_adapted']:
        if not os.path.exists(experiment_args['reference_experiment_checkpoint']): 
            raise ValueError(f'The provided reference experiment checkpoint path {experiment_args["reference_experiment_checkpoint"]} does not exist, please check your input arguments.')
        else:
            with open(experiment_args['reference_experiment_checkpoint'], 'rb') as f:
                reference_experiment_checkpoint = pickle.load(f) #Here we load the checkpoint for continuing an experiment execution, 
            
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
    
    #Extraction of the semantic id dictionary, and the initialisation of the dataloader
    semantic_id_dict, full_image_cache, dataloader = init_task_cases(
        dataset_dir=experiment_args['input_dataset_dir'],
        exp_task_configs=experiment_args['task_configs'],
        metric_configs=experiment_args['metrics_configs'],
        prompter_configs=experiment_args['prompter_configs'],
        shuffle_bool=experiment_args['shuffle_cases'],
        random_seed=experiment_args['random_seed'],
        last_completed_case=loaded_experiment_checkpoint["eval_state"]['last_completed_case'] if loaded_experiment_checkpoint != None else None,
        last_completed_idx=loaded_experiment_checkpoint["eval_state"]['last_completed_idx'] if loaded_experiment_checkpoint != None else None
        )
    #Appending some relevant dataset information to the experiment args for passing through to the front-end simulator.
    num_samples = loaded_experiment_checkpoint["eval_state"]['last_completed_idx'] + 1 + len(dataloader) if loaded_experiment_checkpoint != None else len(dataloader)
    experiment_args['dataset_level_data_schema']['num_samples'] = num_samples
    experiment_args['full_image_cache'] = full_image_cache #We store it in the experiment args, however we will
    #NOT carry this forward into the evaluation side, it is just tidier to have it in the dictionary for
    #generating the dataset level schema.

    #We append the semantic id dict dict to the experiment args. 
    experiment_args['semantic_id_dict'] = semantic_id_dict

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
        if experiment_args['skip_metric']: 
            experiment_args['metrics_savepaths'] = None #If skipping metric saving, then we set the metrics savepaths to None for the metrics handler, which will cause it to skip over the saving procedure in the metrics handler.
        else:
            experiment_args['metrics_savepaths'] = init_metrics_saves(exp_results_dir=experiment_args['exp_results_dir'], metrics_configs=experiment_args['metrics_configs'], semantic_id_dict=semantic_id_dict)
    else:
        if experiment_args['skip_metric']:
            experiment_args['metrics_savepaths'] = None #If skipping metric saving, then we set the metrics savepaths to None for the metrics handler, which will cause it to skip over the saving procedure in the metrics handler.
            assert loaded_experiment_checkpoint["eval_state"]['metrics_savepaths'] == None, 'If skipping metric saving, the loaded experiment checkpoint metrics savepaths must also be None. Please check your input arguments and loaded checkpoint.'
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

    

    ############################### Formulating config schemas ########################################################
    
    #Here we generate the schema which will be passed to the app at initialisation.
    experiment_args = generate_dataset_level_schema(
        experiment_args
    )
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Now we move onto building the app
    if loaded_experiment_checkpoint != None:
        #In this case we have a prior experiment checkpoint. 
        app = init_app(experiment_args, loaded_experiment_checkpoint=loaded_experiment_checkpoint, reference_experiment_checkpoint=reference_experiment_checkpoint)
    else:
        app = init_app(experiment_args, loaded_experiment_checkpoint=None, reference_experiment_checkpoint=reference_experiment_checkpoint)
    #Extract the app configs using the required method. 
    app_config_dict = app.app_configs()

    if not app_config_dict:
        raise Exception('Should at least return the application name in the app_configs method.')

    #Writing app configs, only if we are not continuing an experiment.
    if loaded_experiment_checkpoint == None:
        log_config_writer('Application Args', app_config_dict, exp_setup_logger)
    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #Build the front-end simulator: 

    fe_sim_obj = init_fe(app=app, experiment_args=experiment_args)


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

    if len(dataloader) == 0:
        exp_setup_logger.info('No samples to iterate through in the dataloader, finishing execution. Check if intended.')
    else:
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


