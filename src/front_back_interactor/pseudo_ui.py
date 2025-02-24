'''
This script is intended for simulating inference from the pseudo-front end, as part of the front-end to back-end setup of an end-to-end interactive seg. application. 
'''
from typing import Callable, Union
import logging
import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import random
import numpy as np

from src.data.interaction_state_construct import HeuristicInteractionState 
from src.output_processing.processor import OutputProcessor
from src.output_processing.scoring import MetricsHandler 
from src.data.interaction_memory_cleanup import im_cleanup
from src.utils_checks.pseudo_ui_check import (
    check_empty,
    check_config_labels

)

# logger = logging.getLogger(__name__)

class front_end_simulator:
    '''
    This class serves as an "interface" for the pseudo-ui with operations such as: 
    
    Generating prompts in the "UI" domain, 
    Storing the interaction states through the iterative segmentation process with configured interaction memory params,
    Saving the segmentation states throughout the iterative segmentation process
    Computing and storing the metrics throughout the iterative segmentation process


    etc.

    
    Input request dictionary for application contains the following input fields:

    NOTE: All input arrays, tensors etc, will be on CPU. NOT GPU. 

    NOTE: Orientation convention is always assumed to be RAS! 

        image: A dictionary containing a path & a pre-loaded (UI) metatensor objects 
        {'path':image_path, 
        'metatensor':monai metatensor object containing image, torch.float datatype.
        'meta_dict': image_meta_dictionary, contains the original affine from loading and current affine array in the ui-domain.}

        model: A string denoting the inference "mode" being simulated, has three options: 
                1) Automatic Segmentation, denoted: 'IS_autoseg' 
                2) Interactive Initialisation: 'IS_inter_init'
                3) Interactive Editing: 'IS_inter_edit'
        
        class_configs: A dictionary containing the class label - class integer code mapping relationship being used.

        im: An interaction memory dictionary containing the set of interaction states. 
        Keys = Infer mode + optionally (for Edit) the inference iter num (1, ...).       

        Within each interaction state in IM:    
        
        prev_logits: A dictionary containing: {
                'paths': list of paths, to each individual logits map (HWD), in the same order as provided by output CHWD logits map}
                'metatensor': Non-modified (CHWD) metatensor/torch tensor that is forward-propagated from the prior output (CHWD).
                'meta_dict': Non-modified meta dictionary that is forward propagated.
                }
        prev_pred: A dictionary containing: {
                'path': path to the discretised map (HWD)}
                'metatensor': Non-modified metatensor/torch tensor that is forward-propagated from the prior output (1HWD).
                'meta_dict': Non-modified meta dictionary that is forward propagated.
                }

        prev_optional_memory: A dictionary containing any extra information that the application would like to forward propagate
        which is not currently provided.

        Refinement prompt information: See `<https://github.com/IS_Validate/blob/main/src/data/interaction_state_construct.py>`

        interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor] format 
            {'interactions':dict[prompt_type_str[list[torch.tensor] OR NONE ]], 
            'interactions_labels':dict[prompt_type_str[list[torch.tensor] OR NONE]],
            }
        interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
            (where each prompt spatial coord is represented as a sublist).  
            dict[prompt_type_str[class[list[list]] OR NONE]]


            -------------------------------------------------------------------------------------------------------

    Inference app must generate the output in a dict format with the following fields:

    NOTE: Checks will be put in place to ensure that image resolution, spacing, orientation will be matching & otherwise 
    the code will be non-functional.

        'logits': Dict which contains the following fields:

            'metatensor': MetaTensor or torch object, ((torch.float dtype)), multi-channel logits map (CHWD), where C = Number of Classes (channel first format)
        
            'meta_dict: Meta information in dict format,  ('affine must match the input-images' affine info)
        
        'pred': Dict which contains the following fields:
            metatensor: MetaTensor or torch tensor object ((torch.int dtype)) containing the discretised prediction (shape 1HWD)
            meta_dict: Meta information in dict format, which corresponds to the header of the prediction (affine array must match the input image's meta-info)

        NOTE: The meta dictionaries will be expected to contain a key:item pair denoted as "affine", containing the 
        affine array required for saving the segmentations in ITK format.
        
        NOTE: The affine must be a torch tensor.

    NOTE: These outputs must/should be stored/provided on cpu. 

    NOTE: Optional to include the "optional_memory" field also, for any extra arguments app would like to store in IM.
        if not required, put a None for the value of this item.
    '''
    def __init__(self, 
                infer_app: Callable, 
                args: dict):
        '''
        Inputs: 
    
        infer_app: Initialised inference application which can be called to process an input request (structure noted above).
        
        args: Dictionary containing the information required for performing the experiment, e.g.: 

            random_seed: The optional int denoting the seed being used for this instance of validation. Otherwise, it is
            None and there is no determinism.  
        
            config_labels_dict: Dictionary mapping class labels and integer codes.
            
            inf_prompt_procedure_type: String denoting the inference prompt generator type: Heuristic is the only 
            one with support currently.

            inf_init_prompt_config (and inf_edit_prompt_config): "use mode" specific prompt generation config 
            dictionaries for inference prompt generation.

            metric_init_prompt_config (and metric_edit_prompt_config): Same thing but for the metrics, currently not supported. 
            
            inference run configs: (e.g., modes, number of refinement iterations)
            
            metrics configs: metrics being computed, prompt generation configs for parameter-dependent metrics, etc.
            
            interaction memory configs: configs for how the interaction states will be stored to be passed through for
            the infer_app call: contains fields 'keep_init' and 'im_len' (former denotes the treatment of init interaction
            state, while im_len denotes the edit memory)

            NOTE: We currently do not accept deletion of the initialisation but retention of the edit iters, this 
            is a very atypical configuration.

                im_config = 
                    {'keep_init': bool,
                    'memory_len': int (this denotes the retained memory backwards, -1 denotes full memory, otherwise it 
                    denotes the memory retained relative to the "current" iter)
                    }  

            etc.

        TODO: Add a full exhaustive list of the dictionary fields.

        '''
        
        if not callable(self.infer_app):
            raise Exception('The inference app must be callable: an initialised class object with a __call__ function for performing inference fully end-to-end.')
        
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

    def set_seeds(self, seed: Union[int, None]): #, cuda_deterministic=True):
        if seed is None:
            print('We do not have a fixed seed!')
        
        else:
            warnings.warn('You have set a deterministic seed which will be re-initialised for each data instance, please check whether this is intended.')
            if isinstance(seed, int):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            else:
                raise TypeError('Seed must be an int if it is set for determinism.')
                    
        #TODO: Once containerisation is implemented, we can re-implement this functionality for instances where a DL model may be used for
        # prompt generation

        # if cuda_deterministic:
        #     cudnn.deterministic = True 
        #     cudnn.benchmark = False
        # else:
        #     cudnn.deterministic=False 
        #     cudnn.benchmark=True 

    def post_handlers_init(self):
        '''
        This function initialises the class objects which can be used for processing the outputs of calls to the 
        inference app, generating metrics, saving metrics etc.
        '''
        self.metrics_handler = MetricsHandler(
            dice_termination_threshold=self.args['dice_termination_thresh'],
            metrics_configs=self.args['metrics_class_configs'],
            metrics_savepaths=self.args['metrics_savepaths'],
            config_labels_dict=self.args['config_labels_dict']
        )

        self.output_processor = OutputProcessor(
            base_save_dir= self.args['base_write_dir'],
            config_labels_dict=self.args['config_labels_dict'],
            is_seg_tmp=self.args['is_seg_tmp'],
            save_prompts=self.args['save_prompts']
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
                sim_device=self.args['sim_device'],
                use_mem=False,
                prompt_configs=self.args['inf_init_prompt_config'],
                config_labels_dict=self.args['config_labels_dict']
            )
            self.inf_edit_generator = HeuristicInteractionState(
                sim_device=self.args['sim_device'],
                use_mem=self.args['use_mem_edit_generator'],
                prompt_configs=self.args['inf_edit_prompt_config'],
                config_labels_dict=self.args['config_labels_dict']
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
    def inf_im_handler(self,
                data_instance: dict, 
                infer_config: dict,
                im:Union[dict, None],
                prev_output_data: Union[dict, None],
                ):
        '''
        Function which handles the interaction memory dict for the input information. Takes the following args:

        data_instance - A dictionary containing the set of information with respect to the image, and ground truth, 
        required for the request generation + interaction state generation:

            Contains the following fields:

                'image': dict - A dictionary containing the following subfields
                    'path': path to the image 
                    'metatensor': Loaded MetaTensor in RAS orientation (pseudo-UI native domain)
                    'meta_dict': MetaTensor's meta_dict, contains the original affine array, and the pseudo-ui affine array
                
                'label': dict - A dictionary containing the sam subfields as the image! Not one-hot encoded for the MetaTensors!
            
            NOTE: KEY ASSUMPTION 1: The filename for both the image and ground truth will be the same. 
            NOTE: KEY ASSUMPTION 2: The ground truth will NOT be one-hot encoded. 

        infer_config: Dict - A dictionary containing two subfields:
            'mode': str - The mode that the inference call will be made for (Automatic Init, Interactive Init, Interactive Edit)
            'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
        
        im: Union[Dict, None] - An optional dictionary containing the existing interaction memory (None for inits)
        prev_output_data: Union[Dict, None] - An optional dictionary containing the post-processed output data from prior
        iteration's inference call.

        Returns:
        The updated interaction memory, with any post-processing implemented for cleanup (if activated)
        '''
        if not isinstance(data_instance, dict) or not data_instance:
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
                image=data_instance['image']['metatensor'], 
                mode=infer_config['mode'], 
                gt=data_instance['label']['metatensor'], 
                prev_output_data=prev_output_data, 
                im=None)} 


        elif infer_config['mode'].title() == 'Interactive Edit':
            im[f'Interactive Edit Iter {infer_config["edit_num"]}'] = self.inf_edit_generator(
                image=data_instance['image']['metatensor'],
                mode=infer_config['mode'],
                gt=data_instance['label']['metatensor'],
                prev_output_data=prev_output_data,
                im=im
            )
            #Here we implement the optional use of memory clipping in instances where memory concerns may exist.

            #We only implement this in the interactive edit mode since it necessitates that there be memory in the first place!

            im = im_cleanup(self.args['is_seg_tmp'], self.tmp_dir_path, self.args['im_config'], im, infer_config)
        
        return im 


    def app_output_processor(self,
                            data_instance: dict,
                            inf_req: dict,
                            output_data:dict,
                            inf_im:dict,
                            metric_im: Union[dict, None],
                            infer_call_config:dict, 
                            tracked_metrics:dict):
        '''
        Makes use of the output processor class. This will tie together several functionalities such as 
        reformatting the output data dictionary, writing the segmentations, computing the metrics etc.
        
        Inputs: 

        data_instance: Dict - A dictionary containing info related to the image, gt for the current data instance we 
        are evaluating on.

        inf_req: Dict - A dictionary containing the information provided for the input inference request.
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
        
        tracked_metrics: Dict - A dictionary containing the tracked metrics dictionary throughout the iterative refinement process.
        
        Returns: 

        processed_output_data : Dict - A dict containing the output of the inference call processed for future calls on the
        interaction state constructor class.

        updated_im_metric: Dict or None - An optional dict containing the tracked im for the metrics.

        updated_tracked_metrics: Dict - An updated dictionary containing the tracked metrics across the inference calls.

        terminate_early: Bool - A bool which is returned during the updating of the tracked metrics, which determines whether
        or not to terminate the refinement process early with respect to a termination criterion (currently: Dice = 1.0).
        '''
        
        if not isinstance(data_instance, dict) or not data_instance:
            raise Exception('The data instance must be a non-empty dictionary, should have been flagged earlier.')

        if not isinstance(inf_req, dict) or not inf_req:
            raise Exception('The inference request must be a dictionary which is non-empty.')
        
        if not isinstance(inf_im, dict) or not inf_im:
            raise Exception('The inference interaction memory must be a dictionary which is non-empty.')
        
        if not isinstance(metric_im, dict) and metric_im is not None:
            raise TypeError('The metric interaction memory should be a dict or a NoneType')
        
        if isinstance(metric_im, dict) and not metric_im:
            raise Exception('If the metric interaction memory exists, then it must not be empty')

        if not isinstance(infer_call_config, dict) or not infer_call_config:
            raise Exception('The inference call config must be a dictionary which is non-empty.')
        
        if not isinstance(tracked_metrics, dict):
            raise Exception('The tracked metrics must be a dictionary, even if empty for initialisation.')
        
        #First we call on the metric_im handler for generating any parameters required for metrics.
        updated_metric_im = self.metric_im_handler(inf_im=inf_im, metric_im=metric_im)
        
        #Then we call on the output processor, which checks that the app call provided a valid output (within a prescribed set of rules)
        # and reformats the data into that expected by future calls on the interaction state constructor. 
        processed_output_data = self.output_processor(input_request=inf_req, output_dict=output_data, tmp_dir=self.tmp_dir_path)
        
        #Then we update the tracked metrics.
        updated_tracked_metrics, terminate_early = self.metrics_handler.update_metrics(
            output_data=processed_output_data,
            data_instance=data_instance,
            tracked_metrics=tracked_metrics,
            im_inf=inf_im,
            im_metric=updated_metric_im,
            infer_call_info=infer_call_config
        )
        return processed_output_data, updated_metric_im, updated_tracked_metrics, terminate_early

    def infer_app_request_generator(self,
                            data_instance: dict,
                            infer_call_config: dict,
                            im: Union[dict, None], 
                            prev_output_data: Union[dict, None]):
        '''
        This function generates the app request (i.e. the input dictionary to the application) which is intended 
        to be called in the iterator.

        Each request comes with field containing the app_(sub)name also: I.e., Autosegmentation, Interactive Init, Interactive Edit. Users can provide three separate
        apps, or just repeat the same but it should be packaged in a manner such that the input request will be channeled appropriately for their requirements.
        
        We use the following convention, for Automatic Init: 'IS_autoseg', Interactive Init: 'IS_interactive_init', Interactive Edit: 'IS_interactive_edit'.

        This value will be stored under the "model" key in the input request. 


        Inputs:

        data_instance - A dictionary containing the set of information with respect to the image, and ground truth, 
        required for the request generation + interaction state generation:

            Contains the following fields:

                'image': dict - A dictionary containing the following subfields
                    'path': path to the image 
                    'metatensor': Loaded MetaTensor in RAS orientation (pseudo-UI native domain)
                    'meta_dict': MetaTensor's meta_dict, contains the affine array.
                
                'label': dict - A dictionary containing the same subfields as the image! Not one-hot encoded for the MetaTensors!
            
        
        infer_call_config: A dict providing info about the current infer call, contains
         
            mode - The mode in which the application is being used, therefore queried in the request, and the
        
            edit_num - The editing iteration number (1, ...) or NONE (for initialisation)
        
        im - (Optional) The currently existing inference interaction memory (for edit) or NoneType (for initialisations) 
        
        prev_output_data - (Optional) The post-processed output dictionary from the prior iteration of inference (for editing modes). 
        or NoneType. 

        Returns:

        request - The input request dictionary for input to the app inference call.
        im - The updated inference interaction memory dict for tracking.
        '''

        if not isinstance(data_instance, dict) or not data_instance:
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
            
            im = self.inf_im_handler(
                data_instance=data_instance, 
                infer_config=infer_call_config, 
                im=None, 
                prev_output_data=None)

            request = {
                'image': data_instance['image'],
                'model':'IS_autoseg', 
                'class_configs': self.args['configs_label_dict'],
                'im': im
                }
            
            return request, im

        elif infer_call_config['mode'].title() == 'Interactive Init':
            if prev_output_data is not None: #We choose an explicit check of Nonetype for the if statement
                raise TypeError('The previous output should not exist for initialisation')
            if infer_call_config['edit_num'] is not None:
                raise TypeError('The edit num in the infer call config dict should not exist for initialisation!')
            
            im = self.inf_im_handler(
                data_instance=data_instance, 
                infer_config=infer_call_config, 
                im=None, 
                prev_output_data=None) 


            request = {
                'image': data_instance['image'],
                'model': 'IS_inter_init', 
                'class_configs': self.args['configs_label_dict'],
                'im': im
                }
            return request, im 
        
        elif infer_call_config['mode'].title() == 'Interactive Edit':
            if prev_output_data is None:
                raise ValueError('There must be a dictionary containing the outputs of the prior inference call!')
            if infer_call_config['edit_num'] is None and not isinstance(infer_call_config['edit_num'], int):
                raise TypeError('The edit num in the infer call config dict should be an int!')
            
            im = self.inf_im_handler(
                data_instance=data_instance,
                infer_config=infer_call_config,
                im=im,
                prev_output_data=prev_output_data
            )

            request = {
                'image': data_instance['image'],
                'model': 'IS_inter_edit', 
                'class_configs': self.args['configs_label_dict'],
                'im':im
                }
            return request, im
        else:
            raise ValueError('The inference mode is invalid for app request generation!')
        
    def iterative_loop(self, data_instance:dict, tracked_metrics: dict):
        '''
        data_instance - A dictionary containing the set of information with respect to the image, and ground truth, 
        required for the request generation + interaction state generation:

            Contains the following fields:

                'image': dict - A dictionary containing the following subfields
                    'path': path to the image 
                    'metatensor': Loaded MetaTensor in RAS orientation (pseudo-UI native domain)
                    'meta_dict': MetaTensor's meta_dict, contains the original affine array, and the pseudo-ui affine array
                
                'label': dict - A dictionary containing the sam subfields as the image! Not one-hot encoded for the MetaTensors!
            
            NOTE: KEY ASSUMPTION 1: The filename for both the image and ground truth will be the same. 
            NOTE: KEY ASSUMPTION 2: The ground truth will NOT be one-hot encoded.

        tracked_metrics - An empty dictionary intended for tracking the metrics for each interaction state throughout the
        iterative refinement process.
        ''' 
        
        if not isinstance(data_instance, dict):
            raise TypeError('Expected data instance to be provided as a dictionary.')
        if not isinstance(tracked_metrics, dict):
            raise TypeError('The tracked metrics must be initialised as a dictionary')
        
        if tracked_metrics or not data_instance:
            raise Exception('Either the tracked metrics were non-empty, or the data_instance was empty.')

        infer_run_configs = self.args['infer_run_configs']
        
        if not isinstance(infer_run_configs, dict):
            raise TypeError('Inference run type configs must be presented in dict, with keys of "edit_bool" and "num_iters".')
        
        if not isinstance(infer_run_configs['edit_bool'],  bool):
            raise TypeError('edit_bool value must be a boolean in the inference run configs dict.')
        
        #We use the initialisation mode provided in the inference run config to initialise the model.
        if len({infer_run_configs['init'].title()} & {'Automatic Init', 'Interactive Init'}) != 1:

            #Generate the inference request and initialises the inference interaction memory:
            request, inf_im = self.infer_app_request_generator(
                data_instance=data_instance, 
                infer_call_config={'infer_mode': infer_run_configs['init'].title(), 'edit_num': None},
                im=None,
                prev_output_data=None
            )
            #Take the generated request dict, and pass it through the callable application.
            prev_output_data = self.infer_app(request)
            #This generates the non-processed output data.
        
            # We call on the output processor for reformatting/processing of the prev_output_data dictionary for 
            # future interaction state construction, writing of segs, and generation of metrics for tracking

            prev_output_data, metric_im, tracked_metrics, terminated_early = self.app_output_processor(
                data_instance=data_instance,
                inf_req=request, 
                output_data=prev_output_data,
                inf_im=inf_im,
                metric_im=None,
                infer_call_config={'infer_mode': infer_run_configs['init'].title(), 'edit_num': None},
                tracked_metrics=tracked_metrics
                )

            #We put a placeholder for handling the termination condition.
            if terminated_early:
                raise Exception('We do not yet have any handling for early convergence')
                print(f'Reached convergence already, terminating at {infer_run_configs['init'].title()}!')
                return tracked_metrics, terminated_early 
        else:
            raise KeyError('A supported initialisation mode was not selected')  
            

        #Now, we run the editing iterations, if applicable! 

        if infer_run_configs['edit_bool']:
            
            if not isinstance(infer_run_configs['num_iters'], int):
                raise TypeError('The num_iters value must be an int in the inference run configs if editing!')
            
            print('We are now performing iterative edits')

            for iter_num in range(1, infer_run_configs['num_iters'] + 1):

                #Generate the inference request and initialises the inference interaction memory:
                request, inf_im = self.infer_app_request_generator(
                    data_instance=data_instance, 
                    infer_call_config={'infer_mode': 'Interactive Edit', 'edit_num': iter_num},
                    im=inf_im,
                    prev_output_data=prev_output_data
                )
                #Take the generated request dict, and pass it through the callable application.
                prev_output_data = self.infer_app(request)
                #This generates the non-processed output data.
            
                # We call on the output processor for reformatting/processing of the prev_output_data dictionary for 
                # future interaction state construction, writing of segs, and generation of metrics for tracking
    
                prev_output_data, metric_im, tracked_metrics, terminated_early = self.app_output_processor(
                    output_data=prev_output_data,
                    inf_im=inf_im,
                    metric_im=metric_im, 
                    infer_call_config={'infer_mode': 'Interactive Edit', 'edit_num': iter_num},
                    tracked_metrics=tracked_metrics
                    )

                #We put a placeholder for handling the termination condition.
                if terminated_early:
                    raise Exception('We do not yet have any handling for early convergence')
                    print(f'Reached convergence already, terminating at Interactive Edit Iter {iter_num}!')
                    return tracked_metrics, terminated_early
        
        #We delete the inference interaction memory just to be safe so it has zero chance of leaking over into the next
        # data instance
        del inf_im 

        return tracked_metrics, terminated_early 
    
    def __call__(self, 
                data_instance:dict,
                tmp_dir_path:str):
        '''
        data_instance - A dictionary containing the set of information with respect to the image, and ground truth, 
        required for the request generation + interaction state generation:

            Contains the following fields:

                'image': dict - A dictionary containing the following subfields
                    'path': path to the image 
                    'metatensor': Loaded MetaTensor in RAS orientation (pseudo-UI native domain)
                    'meta_dict': MetaTensor's meta_dict, contains the original affine array, and the pseudo-ui affine array
                
                'label': dict - A dictionary containing the same subfields as the image! Not one-hot encoded for the MetaTensors!
            
            NOTE: KEY ASSUMPTION 1: The filename for both the image and ground truth will be the same. 
            NOTE: KEY ASSUMPTION 2: The ground truth will NOT be one-hot encoded. 
        
        tmp_dir_path - The path name for the current temporary directory initialised for the current data instance.
        '''

        if not isinstance(data_instance, dict) or not data_instance:
            raise Exception('The data instance must be a non-empty dictionary.')
        if not isinstance(tmp_dir_path, str) or not tmp_dir_path:
            raise Exception('The tmp_dir path must be a string and non-empty.')

        #Checking if the foreground for this instance is even non-empty... we will currently not be supporting this.
        if check_empty(data_instance['label']['metatensor'], self.args['config_labels_dict']['background']):
            raise Exception(f'The ground truth for the foreground cannot be empty, this functionality is not supported. Raised exception for {data_instance['image']['path']}')
        
        #Calling on the set_seeds function to re-initialise the seeds for each data instance (this ensures early
        #termination would not cause deterministic runs to vary across different models.)
        self.set_seeds(seed=self.args['random_seed'])
        
        #Re-assigning the tmp_dir_path attribute for each data instance. 
        self.tmp_dir_path = tmp_dir_path

        #Initialising a dictionary for tracking the metrics generated, not required for the im dictionaries as this
        #will be performed within the iterative loop.
        tracked_metrics = {}

        #Executing the iterative loop.
        tracked_metrics, terminated_early = self.iterative_loop(
            data_instance=data_instance,
            tracked_metrics=tracked_metrics)
        
        #Saving the final set of tracked metrics....

        self.metrics_handler.save_metrics(
            patient_name=os.path.split(data_instance['image']['path'])[1].split('.')[0],
            terminated_early=terminated_early,
            tracked_metrics=tracked_metrics
        )

        #Nothing is returned here, everything except final tmp_dir cleanup needs to be handled during the loop!
        #Final tmp_dir cleanup will occur in the script which calls this function.



if __name__ == '__main__':
    dummy = front_end_simulator()
    print('stop')