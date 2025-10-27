'''
This script is intended for simulating inference from the pseudo-front end, as part of the front-end to back-end setup of an end-to-end interactive seg. application. 
'''
from email.mime import image
from typing import Callable, Union
import logging
import time
import copy
import warnings
import os
import sys
import re 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import random
import numpy as np

from src.data.interaction_state_construct import HeuristicInteractionState 
from src.output_processing.processor import OutputProcessor
from src.output_processing.scoring import MetricsHandler 
from src.data.memory_cleanup import memory_cleanup
from src.utils_checks.pseudo_ui_check import (
    check_empty,
    check_config_labels

)
from src.general_utils.dict_utils import dict_path_create, sort_infer_calls
# logger = logging.getLogger(__name__)

class FrontEndSimulator:
    '''
    This class serves as an "interface" for the pseudo-ui with operations such as: 
    
    Generating prompts in the "UI" domain, 
    Storing the interaction states through the iterative segmentation process with configured interaction memory params,
    Saving the segmentation states throughout the iterative segmentation process
    Computing and storing the metrics throughout the iterative segmentation process


    etc.

    
    Input request dictionary for application contains the following input fields:

    NOTE: All input arrays, tensors etc, will be on CPU. NOT GPU. 

    NOTE: Orientation convention is always assumed to be RAS+ (Nibabel)! 

        image: A dictionary containing a pre-loaded (UI) metatensor objects 
        {
        'metatensor':monai metatensor object containing image, torch.float datatype.
        'meta_dict': image_meta_dictionary, contains the original affine from loading and current affine array in the ui-domain.}

        infer_mode: A string denoting the inference "mode" being simulated/requested, has three options: 
                1) Automatic Segmentation, denoted: 'IS_autoseg' 
                2) Interactive Initialisation: 'IS_interactive_init'
                3) Interactive Editing: 'IS_interactive_edit'
        
        config_labels_dict: A dictionary containing the semantic class label - class integer code mapping relationship being used. 
        note that the codes are >= 0 with 0 = background always, and that the labels are pre-normalised. E.g., 0,1,2,3... and never 0,2,3,5.

        i_state: An interaction dictionary containing the current input interaction states:
              
            Within the interaction state we also have prompt information stored under the following keys:

                interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor] format 
                {'interactions':dict[prompt_type_str[list[torch.tensor/metatensor] OR NONE ]], 
                'interactions_labels':dict[prompt_type_str_labels [list[torch.tensor/metatensor] OR NONE]],
                }
                interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
                    (where each prompt spatial coord is represented as a sublist).  
                    dict[prompt_type_str[class[list[list]] OR NONE]]


            -------------------------------------------------------------------------------------------------------

    Inference app must generate the output in a dict format with the following fields:

    NOTE: Checks will be put in place to ensure that voxel count, spacing, orientation will be matching & otherwise 
    the code will be non-functional.

        'probs': Dict which contains the following fields:

            'metatensor': MetaTensor or Torch tensor object, ((torch.float dtype)), multi-channel probs map (CHWD), where C = Number of Classes (channel first format)
        
            'meta_dict: Meta information in dict format,  ('affine must match the input-images' affine info).
        
        'pred': Dict which contains the following fields:
            metatensor: MetaTensor or Torch tensor object ((torch.int dtype)) containing the discretised prediction (shape 1HWD)
            meta_dict: Meta information in dict format, which corresponds to the header of the prediction (affine array must match the input image's meta-info)

        NOTE: The meta dictionaries will be expected to contain a key:item pair denoted as "affine", containing the 
        affine array. NOTE: The affine must be a torch tensor or numpy array.
        
        NOTE: MetaTensor objects are permitted as they are lightweight wrappers around torch tensors with meta-info, provided
        as part of MONAI. However, they need not be used if the user prefers to use torch tensors and meta-dicts directly. 

    NOTE: These outputs must be stored/provided on cpu at the inferface level.


    --------------------------------------------------------------------------------------------------------------------------

    The simulation process stores the memory of interaction states: this may optionally be used in future versions for guiding the
    prompt generation process in a manner which is dependent on prior prompts and algorithm outputs.

    im: An dictionary containing the history input interaction states

        Keys = Infer mode + optionally (for Edit) the inference iter num (1, ...).       

        Within the interaction state:    
         
        prev_probs: A dictionary containing: {
                'metatensor': Non-modified (CHWD) metatensor/torch tensor that is forward-propagated from the prior output (CHWD).
                'meta_dict': Non-modified meta dictionary that is forward propagated.
                }
        prev_pred: A dictionary containing: {
                'metatensor': Non-modified metatensor/torch tensor that is forward-propagated from the prior output (1HWD).
                'meta_dict': Non-modified meta dictionary that is forward propagated.
                }

        prompt information

        interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor] format 
            {'interactions':dict[prompt_type_str[list[torch.tensor/metatensor] OR NONE ]], 
            'interactions_labels':dict[prompt_type_str_labels [list[torch.tensor/metatensor] OR NONE]],
            }
        interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
            (where each prompt spatial coord is represented as a sublist).  
            dict[prompt_type_str[class[list[list]] OR NONE]]

    The memory length (in discrete interaction states) is configurable at runtime via the im_config field in the input
    arguments for the experimental run. 
    '''

    def __init__(self, 
                infer_app: Callable, 
                args: dict):
        '''
        Inputs: 
    
        infer_app: Initialised inference application which can be called to process an input request (structure noted above).
        
        args: Dictionary containing the information required for performing the experiment, e.g.: 
        
        Variables related to the experimental configuration
        config_labels_dict: Dictionary mapping class labels and integer codes #NOTE: Currently just assuming semantic segmentation support. 
            
        #Variable related to handling empty foreground cases for automatic segmentation configurations
        
        sim_empty_fg_automatic: Boolean denoting whether to simulate/attempt automatic segmentation in cases where the foreground are empty (non-background). 
        infer_run_configs: Dict containing inference run configs: (e.g., modes, number of refinement iterations)
    
        metrics_configs: metrics being computed, prompt generation configs for parameter-dependent metrics (NOTE: latter not yet supported), etc.
            
        
        inf_prompt_procedure_type: inf_prompt_procedure_type: String denoting the inference prompt generator type: Heuristic is the only 
        one with support currently.

        inf_init_prompt_config
        inf_edit_prompt_config
        {"use mode" specific prompt generation config dictionaries for inference prompt generation}

        metric_init_prompt_config (and metric_edit_prompt_config): Same thing but for the metrics NOTE: currently not supported. 
            
            

        #Variables related to reproducibility and device.
        random_seed: The optional int denoting the seed being used for this instance of validation. Otherwise, it is
        None and there is no determinism. 

        cuda_deterministic - Boolean denoting whether to use deterministic algorithms for CUDA operations.
        torch_deterministic - Boolean denoting whether to use deterministic algorithms for PyTorch operations.
        device - torch device

        #Variables related to interaction memory
        use_mem_inf_edit: 
        im_config : configs for how the interaction states will be stored for prompting, 
        Latter contains fields 'im_conf_memory_len' (denotes the state memory, inclusive of the initialisation.)
        E.g., im_config = 
        {
            'im_conf_memory_len': int (this denotes the retained memory backwards, -1 denotes full memory, otherwise it 
            denotes the memory retained relative to the "current" iter)
        }  
        
        #Variables related to metrics and saving metrics/
        dice_termination_thresh - threshold for early termination based on dice score.
        metrics_savepaths - dictionary mapping metric names to their save paths.
        exp_results_dir - base directory for saving experimental results.

        #Variables related to saving segmentations and prompt info
        
        seg_root - Root directory for saving segmentations.
        exp_seg_dir - Directory for saving experiment segmentations.
        save_prompts - Boolean denoting whether to save the prompts used during inference.
        is_seg_tmp - Boolean denoting whether to use a temporary directory for saving segmentations.
        write_segmentation - Boolean denoting whether to write the segmentation results to disk.

        #Variables related to api-structure
        dataset_info - Dictionary containing dataset information required for the application API structure (e.g., name, 
        modality, etc.)
        
        '''
        
        if not callable(infer_app):
            raise Exception('The inference app must be callable class.')
        else:
            #Check if it has a call attribute!
            try:
                callback = getattr(infer_app, "__call__")
            except:
                raise Exception('The inference app did not have a function __call__')
            
            if not callable(callback):
                raise Exception('The initialised inference app object had a __call__ attribute which was not a callable function') 
            
        self.infer_app = infer_app
        self.args = args

        self.check_args()
        self.init_classes()
    
    def check_args(self):
        '''
        Function is intended for all the checks required for the config arguments in the input args at initialisation to 
        catch early-breaking conditions in code.
        '''
        
        #Running checks on input args..

        #Running a check on the class configs dictionary.
        check_config_labels(self.args['configs_labels_dict'])

        #Running a check on random seed is not required, any non-NoneType TypeErrors will be caught by init_seeds.

        #TODO: Add more to this. 
    
    def init_classes(self):
        '''
        Function is for initialisation of all classes used in other methods (except for the inference app). 
        '''
        #TODO: Expand this with any further modifications implemented.

        #Initialising the prompt generation classes.
        self.inf_prompt_gen_init()
        self.metric_prompt_gen_init()
        
        #Initialising the output processors and writers.
        self.post_handlers_init()

    def init_vars(self):
        #initialising any remaining variables which are not class objects, i.e., variables which are derived from some input args.
        pass 

    def set_seeds(
        self, 
        seed: Union[int, None], 
        cuda_deterministic:bool,
        torch_deterministic: bool):
        if seed is None:
            print('We do not have a fixed seed!')
        
        else:
            warnings.warn('You have set a deterministic seed which will be re-initialised for each data instance, please check whether this is intended. \n')
            if isinstance(seed, int):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            else:
                raise TypeError('Seed must be an int if it is set for determinism.')
                    
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic=False 
            torch.backends.cudnn.benchmark=True
        if torch_deterministic:
            torch.use_deterministic_algorithms(True)
        else:
            torch.use_deterministic_algorithms(False)

    def post_handlers_init(self):
        '''
        This function initialises the class objects which can be used for processing the outputs of calls to the 
        inference app, generating metrics, saving metrics etc.
        '''
        self.metrics_handler = MetricsHandler(
            calc_device=self.args['device'], 
            dice_termination_threshold=self.args['dice_termination_thresh'],
            metrics_configs=self.args['metrics_configs'],
            metrics_savepaths=self.args['metrics_savepaths'],
            config_labels_dict=self.args['configs_labels_dict']
        )

        self.output_processor = OutputProcessor(
            base_save_dir=self.args['exp_seg_dir'],
            config_labels_dict=self.args['configs_labels_dict'],
            is_seg_tmp=self.args['is_seg_tmp'],
            save_prompts=self.args['save_prompts'],
            write_segmentation=self.args['write_segmentation']
        )
    def inf_prompt_gen_init(self):
        '''
        This function initialises the class objects which can generate the interaction states for use in inference.
        '''
        if self.args['inf_prompt_procedure_type'].title() == 'Heuristic':
            # self.autoseg_state_generator = None  
            # The Autosegmentation state does not require any interaction state generators since they contain no 
            # interaction. All information for the autosegmentation state will be provided manually.

            self.inf_init_generator = HeuristicInteractionState(
                sim_device=self.args['device'],
                use_mem=False,
                prompt_configs=self.args['inf_init_prompt_config'],
                config_labels_dict=self.args['configs_labels_dict']
            )
            self.inf_edit_generator = HeuristicInteractionState(
                sim_device=self.args['device'],
                use_mem=self.args['use_mem_inf_edit'],
                prompt_configs=self.args['inf_edit_prompt_config'],
                config_labels_dict=self.args['configs_labels_dict']
            )
        else:
            raise ValueError('The selected prompt generation algorithm-type is not supported')
    
    def metric_prompt_gen_init(self):
        '''
        Function intended for initialising the prompt generators for the metric computation.
        '''
        pass 

    def metric_im_handler(self, inf_im:dict, metric_im:Union[dict, None]):
        ''' 
        NOTE: This function may be deprecated if memory proves to be obsolete (i.e., no metrics require interaction information,
        which may be likely as it is extremely challenging to parameterise such metrics in a generalisable manner!)

        Function intended for interaction memory handling, but for instances where the generated data is used for metric
        computation. Uses the inference interaction memory to access new interaction info from the input prompts.
        Uses this to generate metric interaction memory/update metric interaction memory.

        inf_im must be a dict type, cannot be a NoneType.
        metric_im can be both as at initialisation it may not be initialised.
        '''
        if not isinstance(inf_im, dict) or not inf_im:
            raise Exception('The inference im must be a non-empty dict')
        if not isinstance(metric_im, dict) and metric_im is not None:
            raise TypeError('The metric im must either be a dictionary or a NoneType')
        if isinstance(metric_im, dict) and not metric_im:
            raise ValueError('If the metric im is a dict, it cannot be empty!')

        #Does absolutely nothing for now. 

        return metric_im
    def prompting_im_handler(self,
                infer_config: dict,
                im:Union[dict, None],
                prev_output_data: Union[dict, None],
                ):
        '''
        Function which handles the interaction memory dict for the input information. Takes the following args:

        infer_config: Dict - A dictionary containing two subfields:
            'mode': str - The mode that the inference call will be made for (Automatic Init, Interactive Init, Interactive Edit)
            'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
        
        im: Union[Dict, None] - An optional dictionary containing the existing prompt-side interaction memory (None for inits)
        prev_output_data: Union[Dict, None] - An optional dictionary containing the post-processed output data from prior
        iterations' inference calls.

        Returns:
        The updated interaction memory, with any post-processing implemented for cleanup (if activated)
        '''
        if not isinstance(self.data_instance, dict) or not self.data_instance:
            raise TypeError('The data instance must be a non-empty dictionary, should have been flagged earlier.')
        
        if not isinstance(infer_config, dict) or not infer_config:
            raise TypeError('The inference call config must be a dictionary which is non-empty.')
        
        if not isinstance(im, dict) and im is not None:
            raise TypeError('The inference interaction memory should be a dict or a NoneType')
        
        if isinstance(im, dict) and not im:
            raise Exception('If the inference interaction memory exists, then it must not be empty')

        if not isinstance(prev_output_data, dict) and prev_output_data is not None:
            raise TypeError('The prev output data should be a dict or a NoneType')
        
        if isinstance(prev_output_data, dict) and not prev_output_data:
            raise Exception('If the prev output data is a dict, then it must not be empty')

    
        if infer_config['mode'].title() == 'Automatic Init':
            im = {'Automatic Init': None}

        elif infer_config['mode'].title() == 'Interactive Init':
            im = {'Interactive Init': self.inf_init_generator(
                image=self.data_instance['image']['metatensor'], 
                mode=infer_config['mode'], 
                gt=self.data_instance['label']['metatensor'], 
                prev_output_data=prev_output_data, 
                im=None)} 
            

        elif infer_config['mode'].title() == 'Interactive Edit':
            im[f'Interactive Edit Iter {infer_config["edit_num"]}'] = self.inf_edit_generator(
                image=self.data_instance['image']['metatensor'],
                mode=infer_config['mode'],
                gt=self.data_instance['label']['metatensor'],
                prev_output_data=prev_output_data,
                im=im
            )
            #Here we implement the optional use of memory clipping in instances where memory concerns may exist.

            #We only implement this in the interactive edit mode since it necessitates that there be memory in the first place!

            im, self.tracked_paths = memory_cleanup(self.args['is_seg_tmp'], self.tmp_dir_path, self.args['im_config'], im, self.tracked_paths, infer_config)
        
        return im 
    
    def im_to_is(self, im:dict):
        '''
        This is a function which converts a prompting interaction memory dictionary into an interaction state dictionary which is
        to be passed through into the inference app call.

        inputs:
        im: A dictionary containing the interaction memory object (a dictionary)

        outputs: 
        i_state: A dictionary containing the interaction state object (a dictionary), as described in the docstring of the class.
        It only contains the prompting information! Everything else is expected to be stored by the application, as per their
        own requirements.
        '''
        if not isinstance(im, dict) or not im:
            raise Exception('The interaction memory must be a non-empty dictionary, should have been flagged earlier.')
        

        #We extract the current interaction state's name (last entry in the dictionary)
        if len(im.keys()) == 1:
            last_is_key = list(im.keys())[0]
    
            i_state = {
                'interaction_torch_format': im[last_is_key]['interaction_torch_format'],
                'interaction_dict_format': im[last_is_key]['interaction_dict_format'],
            } 
        else:
            #In this case we are going to have to search through the keys to find the last one 
            iteration_names = set(im.keys())     
            sorted_iterations = sort_infer_calls(iteration_names)
            last_is_key = sorted_iterations[-1]
            i_state = {
                    'interaction_torch_format': im[last_is_key]['interaction_torch_format'],
                    'interaction_dict_format': im[last_is_key]['interaction_dict_format'],
            }

        if not i_state:
            raise Exception('The interaction state dictionary could not be constructed from the interaction memory, \n ' \
            'or was empty.')
        return i_state
    
    def update_tracked_paths(self, output_paths:dict, inf_call_config:dict):
        
        '''
        This function is intended for filling out the fields containing the strings for the segmentation paths in 
        the output data for forward propagation. NOTE: This may become deprecated in future versions, depending on how
        memory handling is adjusted on the validation-side. 
        
        inputs:
        
        output_paths: A dict containing the following fields
            
            pred: A string denoting the path to the discretised prediction of the prior inference call.
        
            probs: A list of strings denoting the paths to the channel-unrolled probs maps from the prior inference call.
            (in the order corresponding to the config labels dictionary.)

        inf_call_config: A dict denoting the current inference call configuration which we will use to store the output paths which have been given to us from the 
        prior iter.

        returns: 
        
        output_paths with pred_path and probs_paths of prior iteration output inserted in the key for the current iteration, done with the assumption that each interaction 
        state is always dependent on the output of the prior state at minimum. For initialisations this is just None... it is a dummy variable due to my own dislike 
        for seeing disordered sets but also to simplify the process of future editing iterations to harmonise the extraction of prior preds and probs
        from both the prev_output data dict, and the tracked paths dict.
        
        '''
        
        infer_config_dir = f'{inf_call_config["mode"]} Iter {inf_call_config["edit_num"]}' if inf_call_config['mode'].title() == 'Interactive Edit' else inf_call_config['mode'].title() 
            
            
        #Dict of info regarding the dict-paths for each filepath being placed after the segmentations have been saved.
        reformat_dict_info = {
            'probs': (infer_config_dir, 'prev_probs','paths'), 
            'pred': (infer_config_dir, 'prev_pred','path'),
        }     

        for key, val in reformat_dict_info.items():
            if key.title() == 'Probs':
                try: 
                    self.tracked_paths = dict_path_create(self.tracked_paths, val, output_paths['probs'])
                except:
                    self.tracked_paths = dict_path_create(self.tracked_paths, val, None)
            elif key.title() == 'Pred':
                try:
                    self.tracked_paths = dict_path_create(self.tracked_paths, val, output_paths['pred'])
                except:
                    self.tracked_paths = dict_path_create(self.tracked_paths, val, None)
            else:
                raise KeyError('Reformatter info dictionary contained an unsupported key')
    
    def app_output_processor(self,
                            # data_instance: dict,
                            # inf_req: dict,
                            output_data:dict,
                            inf_im:dict,
                            metric_im: Union[dict, None],
                            infer_call_config:dict,): 
                            # tracked_metrics:dict):
        '''
        Makes use of the output processor class. This will tie together several functionalities such as 
        reformatting the output data dictionary, writing the segmentations, computing the metrics etc.
        
        Uses: 

        self.data_instance: Dict - A dictionary containing info related to the image, gt for the current data instance we 
        are evaluating on.

        output_data: Dict - A dictionary containing the pre-processed output dictionary from the inference app call.
        
        inf_im: Dict - A dictionary containing the interaction memory for the input prompts. Cannot be a NoneType as even
        automatic initialisation has an IM.

        metric_im: Dict - A dictionary containing the interaction memory info the metrics. (e.g. parametrisations)
        (or a NoneType if initialisation!)
        NOTE: Both are provided in the circumstance where any metrics would require memory which could not be
        provided otherwise.
        
        infer_config: Dict - A dictionary containing two subfields:
            'mode': str - The mode that the inference call was be made for (Automatic Init, Interactive Init, Interactive Edit)
            'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
        
        
        Returns: 

        output_paths : Dict - A dict containing the paths for the saved outputs of the inference call.

        updated_im_metric: Dict or None - An optional dict containing the tracked im for the metrics.

        terminate_early: Bool - A bool which is returned during the updating of the tracked metrics, which determines whether
        or not to terminate the refinement process early with respect to a termination criterion (currently: Dice = 1.0).
        '''
        
        if not isinstance(self.data_instance, dict) or not self.data_instance:
            raise Exception('The data instance must be a non-empty dictionary, should have been flagged earlier.')

        # if not isinstance(inf_req, dict) or not inf_req:
        #     raise Exception('The inference request must be a dictionary which is non-empty.')
        
        if not isinstance(inf_im, dict) or not inf_im:
            raise Exception('The inference interaction memory must be a dictionary which is non-empty.')
        
        if not isinstance(metric_im, dict) and metric_im is not None:
            raise TypeError('The metric interaction memory should be a dict or a NoneType')
        
        if isinstance(metric_im, dict) and not metric_im:
            raise Exception('If the metric interaction memory exists, then it must not be empty')

        if not isinstance(infer_call_config, dict) or not infer_call_config:
            raise Exception('The inference call config must be a dictionary which is non-empty.')
        
        if not isinstance(self.tracked_metrics, dict):
            raise Exception('The tracked metrics must be a dictionary, even if empty for initialisation.')
        
        #First we call on the metric_im handler for generating any parameters required for metrics.
        updated_metric_im = self.metric_im_handler(inf_im=inf_im, metric_im=metric_im)
        
        #Then we call on the output processor, which checks that the app call provided a valid output (within a prescribed set of rules)
        # writes any desired predictions and probs, and returns the paths to the corresponding files. 
        output_paths = self.output_processor(data_instance=self.data_instance, case_name=self.case_name, output_dict=output_data, infer_call_config=infer_call_config, tmp_dir=self.tmp_dir_path)
        #Tuple.

        #Then we update the tracked metrics.
        self.tracked_metrics, terminate_early = self.metrics_handler.update_metrics(
            output_data=output_data,
            data_instance=self.data_instance,
            tracked_metrics=self.tracked_metrics,
            im_inf=inf_im,
            im_metric=updated_metric_im,
            infer_call_info=infer_call_config
        )
        return output_paths, updated_metric_im, terminate_early

    def infer_app_request_generator(self,
                            # data_instance: dict,
                            infer_call_config: dict,
                            im: Union[dict, None], 
                            prev_output_data: Union[dict, None]):
        '''
        This function generates the app request (i.e. the input dictionary to the application) which is intended 
        to be called in the iterator.

        NOTE: Each request comes with field containing the infer_mode being requested also: I.e., Automatic Segmentation, Interactive Init, 
        Interactive Edit. Users can provide three separate apps, or just repeat the same underlying algorithm but it should be packaged 
        in a manner such that the input request will be channeled appropriately for their requirements.
        
        We use the following convention, for Automatic Init: 'IS_autoseg', Interactive Init: 'IS_interactive_init', Interactive Edit: 'IS_interactive_edit'.

        This value will be stored under the "infer_mode" key in the input request. 


        Inputs:

        infer_call_config: A dict providing info about the current infer call, contains
         
            mode - The mode in which the application is being used, therefore queried in the request, and the
        
            edit_num - The editing iteration number (1, ...) or NONE (for initialisation)
        
        im - (Optional) The currently existing interaction memory (for edit) or NoneType (for initialisations) 
        
        prev_output_data - (Optional) The output dictionary from the prior iteration of inference (for editing modes). 
        or NoneType. 

        Returns:

        request - The input request dictionary for input to the app inference call.
        im - The updated prompt interaction memory dict for tracking.
        '''

        if not isinstance(self.data_instance, dict) or not self.data_instance:
            raise Exception('Data_instance should be a non-empty dictionary.')
        if not isinstance(infer_call_config, dict) or not infer_call_config:
            raise Exception('The infer call config should be a non-empty dictionary.')
        if not isinstance(im, dict) and im is not None:
            raise TypeError('The im should either exist as a dictionary for edit request generation, or be a NoneType for init.')
        if isinstance(im, dict) and not im:
            raise ValueError('The im, if a dict, must be non-empty.')
        if not isinstance(prev_output_data, dict) and prev_output_data is not None:
            raise TypeError('The prev_output_data should either exist as a dictionary for edit request generation, or be a NoneType for init.')
        if isinstance(prev_output_data, dict) and not prev_output_data:
            raise ValueError('The prev_output_data, if a dict, should be non-empty.')



        if infer_call_config['mode'].title() == 'Automatic Init':
            if prev_output_data is not None: #We choose an explicit check of Nonetype for the if statement
                raise TypeError('The previous output should not exist for initialisation')
            
            if infer_call_config['edit_num'] is not None:
                raise TypeError('The edit num in the infer call config dict should not exist for initialisation!')
            
            im = self.prompting_im_handler(
                # data_instance=selfdata_instance 
                infer_config=infer_call_config, 
                im=None, 
                prev_output_data=None)

            request = {   
                'infer_mode':'IS_autoseg', 
                }            
            # return request, im

        elif infer_call_config['mode'].title() == 'Interactive Init':
            if prev_output_data is not None: #We choose an explicit check of Nonetype for the if statement
                raise TypeError('The previous output should not exist for initialisation')
            if infer_call_config['edit_num'] is not None:
                raise TypeError('The edit num in the infer call config dict should not exist for initialisation!')
            
            im = self.prompting_im_handler(
                # data_instance=data_instance, 
                infer_config=infer_call_config, 
                im=None, 
                prev_output_data=None) 


            request = {
                'infer_mode': 'IS_interactive_init', 
                }
            
        elif infer_call_config['mode'].title() == 'Interactive Edit':
            if prev_output_data is None:
                raise ValueError('There must be a dictionary containing the outputs of the prior inference call!')
            if infer_call_config['edit_num'] is None and not isinstance(infer_call_config['edit_num'], int):
                raise TypeError('The edit num in the infer call config dict should be an int!')
            
            im = self.prompting_im_handler(
                infer_config=infer_call_config,
                im=im,
                prev_output_data=prev_output_data
            )

            request = {
                'infer_mode': 'IS_interactive_edit', 
                }
        else:
            raise ValueError('The inference mode is invalid for app request generation!')
        
        #Now we will perform some processing to ensure that the interface for the api-request is only dependent on torch
        #objects, and basic pythonic datatypes.
        request['config_labels_dict'] = self.args['configs_labels_dict']
        request['dataset_info'] = self.args['dataset_info'] 
        request['image'] = copy.deepcopy(self.data_instance['image'])
        request['image']['metatensor'] = torch.from_numpy(request['image']['metatensor'].clone().detach().numpy()) 
        i_state = self.im_to_is(im=im)        
        #Now we update the request with the interaction state and the dataset information.
        request['i_state'] = i_state
        

        return request, im
    def iterative_loop(self, empty_foreground:bool=False):
        '''
        ''' 
        
        if not isinstance(self.data_instance, dict):
            raise TypeError('Expected data instance to be provided as a dictionary.')
        if not isinstance(self.tracked_metrics, dict):
            raise TypeError('The tracked metrics must be initialised as a dictionary')
        
        if self.tracked_metrics or not self.data_instance:
            raise Exception('Either the tracked metrics were non-empty, or the data_instance was empty.')

        infer_run_configs = self.args['infer_run_configs']
        
        if not isinstance(infer_run_configs, dict):
            raise TypeError('Inference run type configs must be presented in dict, with keys of "edit_bool" and "num_iters".')
        
        if not isinstance(infer_run_configs['edit_bool'],  bool):
            raise TypeError('edit_bool value must be a boolean in the inference run configs dict.')
        
        if empty_foreground:
            if infer_run_configs['init'].title() == 'Interactive Init':
                #If the foreground is empty, we cannot perform an interactive initialisation with current prompting mechanisms, so if this is the config then we raise an error.
                raise ValueError('Cannot currently perform interactive initialisation with empty foreground with our prompting mechanisms, this should have been flagged earlier!')    
            
            if self.args['sim_empty_fg_automatic']:
                #We can perform the initialisation, but only the initialisation. We currently do not support any mechanisms for simulating prompts for empty foregrounds.
                warnings.warn('We have an empty foreground, with current prompting strategy we will only perform an automatic initialisation, if at all.')
                #We will modify the infer_run_configs to only perform an automatic initialisation.
                infer_run_configs['init'] == 'Automatic Init'
                infer_run_configs['edit_bool'] = False
                infer_run_configs['num_iters'] = 0
                #This is only a temporary fix, we will implement a more robust solution in the future, but also it is only implemented for the current 
                # data instance, the class attribute is not modified for other data instances. 
            elif not self.args['sim_empty_fg_automatic']:
                raise Exception('There was an empty foreground and the sim_empty_fg_automatic flag was not set to True, this should have been flagged earlier!')
            else:
                raise Exception('Unknown use-case.')
        
        #We use the initialisation mode provided in the inference run config to initialise the model.
        if len({infer_run_configs['init'].title()} & {'Automatic Init', 'Interactive Init'}) == 1:
            
            self.update_tracked_paths(output_paths=None, inf_call_config={'mode':infer_run_configs['init'].title(), 'edit_num': None}) 

            #Generate the inference request and initialises the inference interaction memory:
            request, inf_im = self.infer_app_request_generator(
                # data_instance=data_instance, 
                infer_call_config={'mode': infer_run_configs['init'].title(), 'edit_num': None},
                im=None,
                prev_output_data=None
            )
            #Take the generated request dict, and pass it through the callable application.
            prev_output_data = self.infer_app(request)
            #This generates the output data.
        
            # We call on the output processor for processing of the prev_output_data dictionary for 
            # writing of segs, and generation of metrics for tracking

            output_paths, metric_im, terminated_early = self.app_output_processor(
                # data_instance=data_instance,
                # inf_req=request, 
                output_data=prev_output_data,
                inf_im=inf_im,
                metric_im=None,
                infer_call_config={'mode': infer_run_configs['init'].title(), 'edit_num': None},
                )

            #We put a placeholder for handling the termination condition.
            if terminated_early:
                # raise Exception('We do not yet have any handling for early convergence')
                print(f'Reached convergence already, terminating at {infer_run_configs["init"].title()}!')
                return 0, terminated_early #0 is a placeholder for the number of editing iterations performed, which is zero in this case.
        else:
            raise KeyError('A supported initialisation mode was not selected')  
            

        #Now, we run the editing iterations, if applicable! 

        if infer_run_configs['edit_bool']:
            
            if not isinstance(infer_run_configs['num_iters'], int):
                raise TypeError('The num_iters value must be an int in the inference run configs if editing!')
            
            print('We are now performing iterative edits')

            for iter_num in range(1, infer_run_configs['num_iters'] + 1):
                
                #First we store the prior output paths here:
                self.update_tracked_paths(output_paths=output_paths, inf_call_config={'mode': 'Interactive Edit', 'edit_num': iter_num})

                if iter_num == 99:
                    print('pause')

                #Generate the inference request and initialises the inference interaction memory:
                request, inf_im = self.infer_app_request_generator(
                    # data_instance=data_instance, 
                    infer_call_config={'mode': 'Interactive Edit', 'edit_num': iter_num},
                    im=inf_im,
                    prev_output_data=prev_output_data
                )
                
                #Take the generated request dict, and pass it through the callable application.
                prev_output_data = self.infer_app(request)
                #This generates the non-processed output data.
            
                # We call on the output processor for reformatting/processing of the prev_output_data dictionary for 
                # future interaction state construction, writing of segs, and generation of metrics for tracking
    
                output_paths, metric_im, terminated_early = self.app_output_processor(
                    # data_instance=data_instance,
                    # inf_req=request,
                    output_data=prev_output_data,
                    inf_im=inf_im,
                    metric_im=metric_im, 
                    infer_call_config={'mode': 'Interactive Edit', 'edit_num': iter_num},
                    )
                #We put a placeholder for handling the termination condition.
                if terminated_early:
                    # raise Exception('We do not yet have any handling for early convergence')
                    print(f'Reached convergence already, terminating at Interactive Edit Iter {iter_num}!')
                    return iter_num, terminated_early # iter_num is the number of editing iterations performed, which is iter_num in this case.
        
        #We delete the inference and metric interaction memory, prev output data, output paths etc just to be safe so it has zero chance of leaking over into the next
        # data instance
        del inf_im, metric_im, prev_output_data, output_paths, self.tracked_paths

        return None, terminated_early 
    
    def __call__(self, 
                data_instance:dict,
                case_name: str, 
                tmp_dir_path:str):
        '''
        data_instance - A dictionary containing the set of information with respect to the image, and ground truth, 
        required for the request generation + interaction state generation:

            Contains the following fields:

                'image': dict - A dictionary containing the following subfields
                    'metatensor': Loaded MetaTensor in RAS orientation (pseudo-UI native domain) channelfirst 1HWD.
                    'meta_dict': MetaTensor's meta_dict, contains the original affine array, and the pseudo-ui affine array
                
                'label': dict - A dictionary containing the same subfields as the image! Not one-hot encoded for the MetaTensors!
        
        case_name - The string denoting the filename for the image and ground truth. 

        NOTE: KEY ASSUMPTION 1: The filename for both the image and ground truth will be the same. 

        tmp_dir_path - The path name for the current temporary directory initialised for the current data instance.
        '''

        if not isinstance(data_instance, dict) or not data_instance:
            raise Exception('The data instance must be a non-empty dictionary.')
        
        if not isinstance(case_name, str) or not case_name:
            raise Exception('The name for the image must be a string and be non-empty.')
        
        if not isinstance(tmp_dir_path, str) or not tmp_dir_path:
            raise Exception('The tmp_dir path must be a string and non-empty.')

        #Checking if the foreground for this instance is even non-empty... we will currently not be supporting this at all for interaction simulation. In most scenarios
        # it would not make sense to simulate prompts for an empty foreground, as ultimately there usually would be some salient target to have prompted the user to interact
        # with the image in the first place.
        if check_empty(data_instance['label']['metatensor'], self.args['configs_labels_dict']['background']):
            if self.args['infer_run_configs']['init'].title() == 'Automatic Init':
                if not self.args['sim_empty_fg_automatic']:
                    warnings.warn(f'The gold standard segmentation for the task foreground is completely empty and sim_empty_fg_automatic flag is false, skipping... \n'
                        f'If this is not intended, please check the data instance provided. \n'
                        f'Image path: {data_instance["image"]["path"]} \n'
                        f'Ground truth path: {data_instance["label"]["path"]} \n'
                        f'Case name: {case_name} \n')
                    return    
                else:
                    #If automatic initialisation is being used, in this case we want the algorithm to be able to appropriately handle the empty foreground case.
                    warnings.warn(f'The gold standard seg. for the foreground is empty, but the sim_empty_fg_automatic flag is set to True, so we will only perform an automatic initialisation. \n'
                        f'Image path: {data_instance["image"]["path"]} \n'
                        f'Ground truth path: {data_instance["label"]["path"]} \n'
                        f'Case name: {case_name} \n')
                    empty_foreground = True
            else:
                warnings.warn(f'The gold standard segmentation for the task foreground is completely empty, skipping... \n'
                    f'If this is not intended, please check the data instance provided. \n'
                    f'Image path: {data_instance["image"]["path"]} \n'
                    f'Ground truth path: {data_instance["label"]["path"]} \n'
                    f'Case name: {case_name} \n')
                return # We return here, since we do not want to raise an exception for this case, but rather skip the instance.
        

        #Calling on the set_seeds function to re-initialise the seeds for each data instance (this ensures early
        #termination would not cause deterministic runs to vary across different models.)
        self.set_seeds(seed=self.args['random_seed'], cuda_deterministic=self.args['cuda_deterministic'], torch_deterministic=self.args['torch_deterministic'])
        
        #Re-assigning the tmp_dir_path attribute for each data instance. 
        self.tmp_dir_path = tmp_dir_path

        #Re-assigning the Case name
        self.case_name = case_name 

        #Re-assigning the data instance
        self.data_instance = data_instance 

        #Initialising a dictionary for tracking the metrics generated, not required for the im dictionaries as this
        #will be performed within the iterative loop.
        self.tracked_metrics = {}

        #First we initialise the dictionary for storing the set of output paths. 
        self.tracked_paths = {}

        #Executing the iterative loop.
        try:       
            iter_num, terminated_early = self.iterative_loop(empty_foreground=empty_foreground)
        except:
            iter_num, terminated_early = self.iterative_loop()
            empty_foreground = False       
        #Saving the final set of tracked metrics....

        self.metrics_handler.save_metrics(
            case_name=case_name,
            empty_foreground=empty_foreground,
            terminated_early=terminated_early,
            temporary_iter_lims=(iter_num, self.args['infer_run_configs']['num_iters']), #This is a tuple containing the lower and upper iteration limits for padding the tracked metrics, 
            # if early convergence occured.....
            tracked_metrics=self.tracked_metrics
        )

        #Nothing is returned here, everything except final tmp_dir cleanup needs to be handled during the loop!
        #Final tmp_dir cleanup will occur in the script which calls this function.



if __name__ == '__main__':
    print('stop')