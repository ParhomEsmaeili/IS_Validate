'''
This script is intended for simulating inference from the pseudo-front end, as part of the front-end to back-end setup of an end-to-end interactive seg. application. 
'''
from cProfile import Profile
import profile
from pstats import SortKey, Stats 
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
    check_semantic_id_dict

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

    NOTE: Although the string referring to the image is designated as  MetaTensor, this is deprecated. It has been adjusted to
    torch tensors ONLY for the API-interface. The string has been untouched for the sake of minimising checking breakages.

    NOTE: MetaTensor objects are still used internally within the validation framework for handling meta-information which might
    be required for processing, but are not used at the interface level!

    NOTE: Orientation convention is always assumed to be RAS+ (Nibabel)! 

        image: A dictionary containing a pre-loaded (UI) metatensor objects 
        {
        'metatensor':torch tensor object containing image, torch.float datatype.
        'meta_dict': image_meta_dictionary, contains the original affine from loading and current affine array in the ui-domain.}

        infer_mode: A string denoting the inference "mode" being simulated/requested, has three options: 
                1) Automatic Segmentation, denoted: 'IS_autoseg' 
                2) Interactive Initialisation: 'IS_interactive_init'
                3) Interactive Editing: 'IS_interactive_edit'
        
        semantic_id_dict: A dictionary containing the semantic class label - class integer code mapping relationship being used. 
        note that the codes are >= 0 with 0 = background always, and that the labels are pre-normalised. E.g., 0,1,2,3... and never 0,2,3,5.
        This is currently necessary such that there forms a cross-correspondence between the channel order
        of the probability maps and the semantic labels.
        #TODO: Future versions will require an explicit contract for the relationship between the semantic
        class and the channel order that goes beyond the contiguous labelling assumption.

        i_state: An interaction dictionary containing the current input interaction states OR a NoneType for automatic segmentation mode.:
              
            Within the interaction state we also have prompt information stored under the following keys:

                interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor] format 
                {'interactions':dict[prompt_type_str[list[torch.tensor] OR NONE ]], 
                'interactions_labels':dict[prompt_type_str_labels [list[torch.tensor] OR NONE]],
                }
                interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
                    (where each prompt spatial coord is represented as a sublist).  
                    dict[prompt_type_str[class[list[list]] OR NONE]]

        sample_level_schema: A dictionary containing the sample-level schema information, 
        which is generated from the sample data instance and potentially the dataset-level schema. 
        This is intended to provide the app with the necessary context, and conventions for addressing 
        the output arrays. Contains the following fields currently:
            data_schema: A dictionary containing the information regarding the data schema for the current sample, which is generated from the sample data instance and potentially the dataset-level schema. Contains the following fields:
                task_channels: A dict denoting the channel-modality correspondence in the input image, which is 
                currently in CHWD format. E.g., {'T1': 0, 'T2': 1, 'FLAIR': 2}. This need not necessarily
                be contiguous, for now we assume any non-continguous indexing is handled by attaching an 
                empty channel of zeros in the corresponding position in the input image.
                 
                Implicit assumption: The task channel in the dataset-schema is protected. Future versions 
                may however override this protection. TBD.
            semantic_id_dict: A dictionary containing the addressing convention for the arrays to correspond
            to the semantic labels. Presently, this is assumed to align exactly with the dictionary provided at
            dataset level (i.e., the semantic code is locked to a specific class always).
            

                -------------------------------------------------------------------------------------------------------

    Inference app must generate the output in a dict format with the following fields:

    NOTE: Checks will be put in place to ensure that voxel count, spacing, orientation will be matching & otherwise 
    the code will be non-functional.

        'probs': Dict which contains the following fields:

            'metatensor': Torch tensor object, ((torch.float dtype)), multi-channel probs map (CHWD), where C = Number of Classes (channel first format)
        
            'meta_dict: Meta information in dict format,  ('affine must match the input-images' affine info).
        
        'pred': Dict which contains the following fields:
            metatensor: Torch tensor object ((torch.int dtype)) containing the discretised prediction (shape 1HWD)
            meta_dict: Meta information in dict format, which corresponds to the header of the prediction (affine array must match the input image's meta-info)

        NOTE: The meta dictionaries will be expected to contain a key:item pair denoted as "affine", containing the 
        affine array. NOTE: The affine must be a torch tensor or numpy array.
        
        NOTE: MetaTensor objects are NO LONGER supported and add additional headache with minimising code-breakage and leakage of
        metainformation into the algorithm. Any relevant meta-information will be separately provided in the meta_dict field of the
        input request. 

    NOTE: These outputs must be stored/provided on cpu at the inferface level.


    --------------------------------------------------------------------------------------------------------------------------

    The simulation process stores the memory of interaction states: this may optionally be used in future versions for guiding the
    prompt generation process in a manner which is dependent on prior prompts and algorithm outputs.

    im: An dictionary containing the history input interaction states

        Keys = Infer mode + optionally (for Edit) the inference iter num (1, ...).       

        Within the interaction state:    
         
        prev_probs: A dictionary containing: {
                'metatensor': Non-modified (CHWD) torch tensor that is forward-propagated from the prior output (CHWD).
                'meta_dict': Non-modified meta dictionary that is forward propagated.
                }
        prev_pred: A dictionary containing: {
                'metatensor': Non-modified torch tensor that is forward-propagated from the prior output (1HWD).
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
                app: Callable, 
                args: dict):
        '''
        Inputs: 
    
        app: Initialised application which can be called to process an input request (structure noted above).
        
        args: Dictionary containing the information required for performing the experiment, e.g.: 
        
        Variables related to the experimental configuration
        semantic_id_dict: Dictionary mapping class labels and integer codes #NOTE: Currently just assuming semantic segmentation support. 
            
        #Variable related to handling empty foreground cases for automatic segmentation configurations
        
        sim_empty_fg_automatic: Boolean denoting whether to simulate/attempt automatic segmentation 
        in cases where the foreground are empty (non-background). It is only acceptable/relevant in 
        cases where the inference config was configured to perform automatic initialisation. It would be
        too messy to switch initialisation modes mid-experiment as this would change the parameters for
        the experiment. 

        infer_run_configs: Dict containing inference run configs: (e.g., modes, number of refinement iterations)
    
        metrics_configs: configs for metrics being computed, eval annotation samples, prompt generation configs for parameter-dependent metrics (NOTE: latter not yet supported), etc.
            
    
        inf_init_prompt_config
        inf_edit_prompt_config
        {"use mode" specific prompt generation config dictionaries for inference prompt generation}
            inf_prompt_procedure_type: String denoting the inference prompt generator type: Heuristic is the only 
        one with support currently.
            

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
        early_termination_criterion - A dictionary containing the information regarding the early 
        termination criterion, which is based on a metric score. For now contains the following fields:
        - 'metric': The metric name to use for early termination.
        - 'threshold': The threshold for the metric score.
        NOTE: Future work may expand this into a more complex criterion.

        metrics_savepaths - dictionary mapping metric names to their save paths.
        exp_results_dir - base directory for saving experimental results.

        #Variables related to saving segmentations and prompt info
        
        seg_root - Root directory for saving segmentations.
        exp_seg_dir - Directory for saving experiment segmentations.
        save_prompts - Boolean denoting whether to save the prompts used during inference.
        is_seg_tmp - Boolean denoting whether to use a temporary directory for saving segmentations.
        write_segmentation - Boolean denoting whether to write the segmentation results to disk.
        
        '''
        
        if not callable(app):
            raise Exception('The inference app must be callable class.')
        else:
            #Check if it has a call attribute!
            try:
                callback = getattr(app, "__call__")
            except:
                raise Exception('The inference app did not have a function __call__')
            
            if not callable(callback):
                raise Exception('The initialised inference app object had a __call__ attribute which was not a callable function') 
            
        self.app = app
        self.args = args

        self.check_args()
        self.init_classes()
    
    def check_args(self):
        '''
        Function is intended for all the checks required for the config arguments in the input args at initialisation to 
        catch early-breaking conditions in code.
        '''
        
        #Running checks on input args..

        #Running a check on the semantic id dictionary.
        check_semantic_id_dict(self.args['semantic_id_dict'])

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
            early_termination_criterion=self.args['early_termination_criterion'], #self.args['dice_termination_thresh'],
            metrics_configs=self.args['metrics_configs']['metrics'],
            metrics_savepaths=self.args['metrics_savepaths'],
            semantic_id_dict=self.args['semantic_id_dict']
        )

        self.output_processor = OutputProcessor(
            base_save_dir=self.args['exp_seg_dir'],
            semantic_id_dict=self.args['semantic_id_dict'],
            is_seg_tmp=self.args['is_seg_tmp'],
            save_prompts=self.args['save_prompts'],
            write_segmentation=self.args['write_segmentation']
        )
    def inf_prompt_gen_init(self):
        '''
        This function initialises the class objects which can generate the interaction states for use in inference.
        '''
        #First we check whether initialisation or editing config is a nulltype. 
        #If init is a nulltype then no prompts are generated for init -> automatic init.
        #If edit is a nulltype then no prompts are generated for editing -> initialisation only.
        
        #First we initialise the init generator:
        if self.args['inf_init_prompt_config'] is None:
            self.inf_init_generator = None
        else:
            assert 'procedure_type' in self.args['inf_init_prompt_config'], 'The inference initialisation prompt config dictionary must contain the key "procedure_type"'
            
            if self.args['inf_init_prompt_config']['procedure_type'].title() == 'Heuristic':
                self.inf_init_generator = HeuristicInteractionState(
                    sim_device=self.args['device'],
                    use_mem=False,
                    prompt_configs=self.args['inf_init_prompt_config']['config'],
                    semantic_id_dict=self.args['semantic_id_dict']
                )
            else:
                raise ValueError('The selected prompt generation algorithm-type is not supported')
        
        if self.args['inf_edit_prompt_config'] is None:
            self.inf_edit_generator = None
        else:
            assert 'procedure_type' in self.args['inf_edit_prompt_config'], 'The inference editing prompt config dictionary must contain the key "procedure_type"'
            #Then we initialise the edit generator:
            if self.args['inf_edit_prompt_config']['procedure_type'].title() == 'Heuristic':
                self.inf_edit_generator = HeuristicInteractionState(
                    sim_device=self.args['device'],
                    use_mem=self.args['use_mem_inf_edit'],
                    prompt_configs=self.args['inf_edit_prompt_config']['config'],
                    semantic_id_dict=self.args['semantic_id_dict']
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
        
        if not isinstance(im, dict) and im != None:
            raise TypeError('The inference interaction memory should be a dict or a NoneType')
        
        if isinstance(im, dict) and not im:
            raise Exception('If the inference interaction memory exists, then it must not be empty')

        if not isinstance(prev_output_data, dict) and prev_output_data != None:
            raise TypeError('The prev output data should be a dict or a NoneType')
        
        if isinstance(prev_output_data, dict) and not prev_output_data:
            raise Exception('If the prev output data is a dict, then it must not be empty')

    
        if infer_config['mode'].title() == 'Automatic Init':
            im = {'Automatic Init': None}

        elif infer_config['mode'].title() == 'Interactive Init':
            if self.inf_init_generator is None:
                raise Exception('The prompt generator for interactive initialisation was not initialised, but the inference config requires it.')
            
            im = {'Interactive Init': self.inf_init_generator(
                image=self.data_instance['image']['metatensor'], 
                mode=infer_config['mode'], 
                gt=self.data_instance['reference_label']['metatensor'], 
                prev_output_data=prev_output_data, 
                im=None)} 
            

        elif infer_config['mode'].title() == 'Interactive Edit':
            if self.inf_edit_generator is None:
                raise Exception('The prompt generator for interactive editing was not initialised, but the inference config requires it.') 
            
            im[f'Interactive Edit Iter {infer_config["edit_num"]}'] = self.inf_edit_generator(
                image=self.data_instance['image']['metatensor'],
                mode=infer_config['mode'],
                gt=self.data_instance['reference_label']['metatensor'],
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

            if im[last_is_key] == None:
                if not last_is_key.title() == 'Automatic Init':
                    raise Exception('The interaction memory for the last interaction state was None, but the last interaction state was not Automatic Init!')
                else:
                    i_state = None 
            else:
                i_state = {
                    'interaction_torch_format': im[last_is_key]['interaction_torch_format'],
                    'interaction_dict_format': im[last_is_key]['interaction_dict_format'],
                } 
        else:
            #In this case we are going to have to search through the keys to find the last one 
            iteration_names = set(im.keys())     
            sorted_iterations = sort_infer_calls(iteration_names)
            last_is_key = sorted_iterations[-1]
            if im[last_is_key] == None:
                if not last_is_key.title() == 'Automatic Init':
                    raise Exception('The interaction memory for the last interaction state was None, but the last interaction state was not Automatic Init!')
                i_state = None 
            else:
                i_state = {
                    'interaction_torch_format': im[last_is_key]['interaction_torch_format'],
                    'interaction_dict_format': im[last_is_key]['interaction_dict_format'],
            }

        if last_is_key.title() == 'Automatic Init':
            if i_state is not None:
                raise Exception('The interaction state for automatic init must be a NoneType!')
        else:
            if not isinstance(i_state, dict) or not i_state: #Second is just a bool check to ensure it is not empty.
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

    def generate_sample_level_schema(self, request: dict):
        '''
        This function generates the sample-level schema which is to be provided in the input request to the app. 
        This is intended to provide the app with the necessary context, and conventions for addressing the output arrays
        '''
        assert 'image' in request, 'The input request dictionary must contain an "image" key for generating the sample-level data schema.'
        
        #We put a temporary assertion until we find a way to pass through the sample-level data schema in the dataloader,
        #which is that the number of image channels must be FIXED and match the dataset level schema!
        assert request['image']['metatensor'].ndim == 4, 'The input image must be in CHWD format for the sample-level data schema generation to work, as this is currently dependent on the dataset-level schema which is in CHWD format.'
        assert request['image']['metatensor'].shape[0] == len(self.args['dataset_level_schema']['data_schema']['task_channels']), 'The number of channels in the input image must match the number of channels in the dataset-level schema for the sample-level data schema generation to work, as this is currently dependent on the dataset-level schema which is in CHWD format.'
        return {
            'data_schema': {
                'task_channels': self.args['dataset_level_schema']['data_schema']['task_channels']
            },
            'segmentation_task_schema': {
                'semantic_id_dict': self.args['semantic_id_dict']
            }
        }
    def app_request_generator(self,
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
        if not isinstance(im, dict) and im != None:
            raise TypeError('The im should either exist as a dictionary for edit request generation, or be a NoneType for init.')
        if isinstance(im, dict) and not im:
            raise ValueError('The im, if a dict, must be non-empty.')
        if not isinstance(prev_output_data, dict) and prev_output_data != None:
            raise TypeError('The prev_output_data should either exist as a dictionary for edit request generation, or be a NoneType for init.')
        if isinstance(prev_output_data, dict) and not prev_output_data:
            raise ValueError('The prev_output_data, if a dict, should be non-empty.')



        if infer_call_config['mode'].title() == 'Automatic Init':
            if prev_output_data != None: #We choose an explicit check of Nonetype for the if statement
                raise TypeError('The previous output should not exist for initialisation')
            
            if infer_call_config['edit_num'] != None:
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
            if prev_output_data == None:
                raise ValueError('There must be a dictionary containing the outputs of the prior inference call!')
            if infer_call_config['edit_num'] == None and not isinstance(infer_call_config['edit_num'], int):
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
        # request['semantic_id_dict'] = self.args['semantic_id_dict']
        request['image'] = copy.deepcopy(self.data_instance['image'])
        request['image']['metatensor'] = torch.from_numpy(request['image']['metatensor'].clone().detach().numpy()) 
        request['sample_level_schema'] = self.generate_sample_level_schema(request)
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
            raise TypeError('Inference run type configs must be presented in dict, with keys of "edit_bool" and "num_edits".')
        
        if not isinstance(infer_run_configs['edit_bool'],  bool):
            raise TypeError('edit_bool value must be a boolean in the inference run configs dict.')
        
        if empty_foreground:
            if infer_run_configs['init'].title() == 'Interactive Init':
                #If the foreground is empty, we cannot perform an interactive initialisation with current prompting 
                # mechanisms, so if this is the config then we raise an error. We should NOT have reached this point.
                raise ValueError('Cannot currently perform interactive initialisation with empty foreground with our prompting mechanisms, \n '
                                'this should have been flagged earlier!')

            if self.args['sim_empty_fg_automatic']:
                #We can perform the initialisation, but only the initialisation. We currently do not support any mechanisms 
                # for simulating prompts for empty foregrounds.
                warnings.warn('We have an empty foreground, with current prompting strategy we will only perform an automatic initialisation, if at all.')
                #We will modify the infer_run_configs to only perform an automatic initialisation.
                infer_run_configs['init'] = 'Automatic Init'
                infer_run_configs['edit_bool'] = False
                infer_run_configs['num_edits'] = 0
                #This is only a temporary fix, we will implement a more robust solution in the future, but also it is only implemented for the current 
                # data instance, the class attribute is not modified for other data instances. 
            elif not self.args['sim_empty_fg_automatic']:
                raise Exception('There was an empty foreground and the sim_empty_fg_automatic flag was not set to True, this should have been flagged earlier!')
            else:
                raise Exception('Unknown use-case for handling instance where foreground is empty.')
        
        #We use the initialisation mode provided in the inference run config to initialise the model.
        if len({infer_run_configs['init'].title()} & {'Automatic Init', 'Interactive Init'}) == 1:
            
            self.update_tracked_paths(output_paths=None, inf_call_config={'mode':infer_run_configs['init'].title(), 'edit_num': None}) 

            #Generate the inference request and initialises the inference interaction memory:
            request, inf_im = self.app_request_generator(
                # data_instance=data_instance, 
                infer_call_config={'mode': infer_run_configs['init'].title(), 'edit_num': None},
                im=None,
                prev_output_data=None
            )
            #Take the generated request dict, and pass it through the callable application.
            prev_output_data = self.app(request)
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
            
            if not isinstance(infer_run_configs['num_edits'], int):
                raise TypeError('The num_edits value must be an int in the inference run configs if editing!')
            
            print('We are now performing iterative edits')

            for iter_num in range(1, infer_run_configs['num_edits'] + 1):
                
                #First we store the prior output paths here:
                self.update_tracked_paths(output_paths=output_paths, inf_call_config={'mode': 'Interactive Edit', 'edit_num': iter_num})

                #Generate the inference request and initialises the inference interaction memory:
                
                request, inf_im = self.app_request_generator(
                    # data_instance=data_instance, 
                    infer_call_config={'mode': 'Interactive Edit', 'edit_num': iter_num},
                    im=inf_im,
                    prev_output_data=prev_output_data
                )
            
                #Take the generated request dict, and pass it through the callable application.
                prev_output_data = self.app(request)
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
        data_instance - A dictionary containing the set of information with respect to the image,
        and annotations for the current data instance. This is required for the request generation 
        + interaction state generation, and even for passing through image-label pairs for adaptation:

            Contains the following fields:

                'image': dict - A dictionary containing the following subfields
                    'metatensor': Loaded MetaTensor in nibabel RAS+ orientation (pseudo-UI native domain) channelfirst 1HWD.
                    'meta_dict': meta_dict, contains the original affine array, and the pseudo-ui domainaffine array
                
                'eval_label': dict - A dictionary containing the same subfields as the image! 
                Not one-hot encoded for the MetaTensors!
                'reference_label': dict - A dictionary containing the same subfields as the image! 
                Not one-hot encoded for the MetaTensors!

        case_name - The string denoting the filename for the image and annotations. 

        NOTE: KEY ASSUMPTION 1: The case name for both the image and annotations will be the same. 

        tmp_dir_path - The path name for the current temporary directory initialised for the current data instance.
        '''

        if not isinstance(data_instance, dict) or not data_instance:
            raise Exception('The data instance must be a non-empty dictionary.')
        
        if not isinstance(case_name, str) or not case_name:
            raise Exception('The name for the image must be a string and be non-empty.')
        
        if not isinstance(tmp_dir_path, str) or not tmp_dir_path:
            raise Exception('The tmp_dir path must be a string and non-empty.')

        #Checking if the foreground for this instance is even non-empty. We will currently not be supporting this at all for 
        # interaction simulation. 

        #NOTE:
        # We check the reference label for the empty fg case as it is the one that could be used for
        #simulating prompts. Therefore, it would be the primary cause for any code-breaking if empty. 
        # 
        # The eval annotation being empty is not relevant here, as misalignment with the eval 
        # annotation is what we are testing against. Empty annotations for the eval label are still a 
        # valid use case.

        #We initialise with a nonetype, and then update it if we have an empty foreground 
        # case, which is still a valid use case. NOTE: The validity of the case depends on the
        # experiment config. If it is a fully-interactive experiment then it would never be
        # valid. For an experiment with automatic initialisation, it can be valid up until 
        # the automatic initialisation stage, if we have configured it as such
        #  (i.e., sim automated empty fg)

        empty_foreground = None 

        if check_empty(data_instance['reference_label']['metatensor'], self.args['semantic_id_dict']['background']):
            if self.args['infer_run_configs']['init'].title() == 'Automatic Init':
                if not self.args['sim_empty_fg_automatic']:
                    warnings.warn(f'The gold standard segmentation used as the reference for the task foreground is completely empty and sim_empty_fg_automatic flag is false, skipping... \n'
                        f'If this is not intended, please check the data instance provided. \n'
                        f'Case name: {case_name} \n')
                    return  #We return and just continue onto the next data instance.
                else:
                    #If automatic initialisation is being used, in this case we want the algorithm to be able to appropriately handle the empty foreground case.
                    warnings.warn(f'The gold standard seg. for the foreground is empty, but the sim_empty_fg_automatic flag is set to True, so we will only perform an automatic initialisation. \n'
                        f'Case name: {case_name} \n')
                    empty_foreground = True
                    #We set empty foreground flag to true, so that the iterative loop can handle this case appropriately.
            else:
                warnings.warn(f'The gold standard segmentation for the task foreground is completely empty, skipping... \n'
                    f'If this is not intended, please check the data instance provided. \n'
                    f'Case name: {case_name} \n')
                return # We return here, since we do not want to raise an exception for this case, but rather skip the instance.
        
        #We will also examine the eval label for the empty foreground case, this will NOT affect the 
        #validity of the case, but is still relevant to flag for interpretation of results.
        if check_empty(data_instance['eval_label']['metatensor'], self.args['semantic_id_dict']['background']):
            for metric in self.args['metrics_configs']['metrics']:
                # Process each metric
                if metric['ignore_empty']:
                    warnings.warn(f'The eval label for the fg is empty but the ignore empty flag is set to True for {metric["name"]}, so we will ignore this case for this metric. \n'
                                  'Please check whether this is intended') 

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
        
        #NOTE: We only reach here after checking for empty foregrounds, where we have a flag
        #which permits continuation of evaluation if automatic initialisation is permitted to be used in these edge cases. 
        
        #If empty_foreground is not True, then it was FALSE, and so we proceed as normal.
        if empty_foreground == None:
            empty_foreground = False
        else:
            assert empty_foreground == True, 'The empty_foreground flag should either be True or False at this point.'

        if self.args['skip_prompt']:
            assert self.args['skip_metric'], 'If prompt skipping is enabled, then metric skipping must also be enabled, as the metrics are dependent on the prompts for their generation, and therefore if no prompts are being generated then the metrics cannot be generated either. Please check the skip_prompt and skip_metric flags in the experiment config.'
            logging.info('Skipping prompting and metric generation as per the skip_prompt flag -> this is effectively just using the framework to run the app in a loop and feed it an incremental data stream..')
        else:
            iter_num, terminated_early = self.iterative_loop(empty_foreground=empty_foreground)
    
        #Saving the final set of tracked metrics....
        if self.args['skip_metric']:
            logging.info('Skipping saving of metrics as per the skip_metric flag, moving on to next data instance...')
        else:
            self.metrics_handler.save_metrics(
                case_name=case_name,
                empty_foreground=empty_foreground,
                terminated_early=terminated_early,
                temporary_iter_lims=(iter_num, self.args['infer_run_configs']['num_edits']), #This is a tuple containing the lower and upper iteration limits for padding the tracked metrics, 
                # if early convergence occured.....
                tracked_metrics=self.tracked_metrics
            )

        if self.args['enable_adaptation']:
            if self.args['skip_prompt']: 
                assert self.args['provide_gold_standard_after_inference'], 'If adaptation is enabled and prompt skipping is enabled, then the '
                'provide_gold_standard_after_inference flag must be set to True, as the adaptation procedure requires some reference annotation'
                'Since no prompting information would be provided for the sample due to the skip_prompt flag being set '
                'to True it would have not generated any intermediate predictions to use for adaptation.'

            if self.args['provide_gold_standard_after_inference']:
                self.app.accept_new_sample(
                    {
                    'image': {
                        'metatensor':torch.from_numpy(self.data_instance['image']['metatensor'].clone().detach().numpy()),
                        'meta_dict':copy.deepcopy(self.data_instance['image']['meta_dict'])
                        },
                    'label': {
                        'metatensor': torch.from_numpy(self.data_instance['reference_label']['metatensor'].clone().detach().numpy()),
                        'meta_dict': copy.deepcopy(self.data_instance['reference_label']['meta_dict'])
                        }
                    }
                )
            else:
                self.app.accept_new_sample(
                    {
                    'image': {
                        'metatensor':torch.from_numpy(self.data_instance['image']['metatensor'].clone().detach().numpy()),
                        'meta_dict':copy.deepcopy(self.data_instance['image']['meta_dict'])
                        },
                    'label': None #Assumed that the label is to be stored internally for this sample.
                    }
                )

            #Now we will make a callback to trigger a function which will handle the adaptation procedure (need not necessarily
            #make a change at every data instance, depends on the algorithm). This is solely for isolating the process of returning
            #updated meta-algorithm state for automatic continuation of experiment.
            algorithm_state = self.app.trigger_adaptation()       

            if algorithm_state == None or list(algorithm_state['meta_algorithm_state'].keys()) == ['algo_cache_name']: 
                #We need something more than the cache name! Nothing else was changed/saved!
                raise ValueError('If adaptation is enabled, then the algorithm state cannot be a NoneType! This does not \n' \
                'make sense as the adaptation procedure must have made some change to the algorithm state!')
            elif type(algorithm_state) != dict:
                raise TypeError('The algorithm state returned from adaptation must be a dictionary!')
            else:
                return algorithm_state 
        else:
            return {} #Algorithm state is empty dict if adaptation is not being used, could use a nonetype but for
            # consistency on re-run we use None, even though functionally the state stored will not change!
        
            #Nothing else is returned here, everything except final tmp_dir cleanup needs to be handled during the loop!
            #Final tmp_dir cleanup will occur in the script which calls this function.


