'''
This script is intended for simulating inference from the pseudo-front end, as part of the front-end to back-end setup of an end-to-end interactive seg. application. 
'''
from monai.transforms import Compose, ToDeviced 
from typing import Callable, Sequence, Union, Optional
import logging
import sys
import torch
import random
import numpy as np
import os
from src.data.interaction_state_construct import HeuristicInteractionState 
from src.data.interaction_memory_cleanup import im_cleanup

logger = logging.getLogger(__name__)

class front_end_simulator:
    '''
    This class serves as an "interface" for the pseudo-ui with operations such as: 
    
    Loading the imaging data in the "UI" domain,
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
        
        NOTE: Experimental config must pre-define the set of interaction states required with the following argument:
            im_config = 
            {'keep_init': bool,
            'memory_len': int (this denotes the retained memory backwards, -1 denotes full memory, otherwise it 
            denotes the memory retained relative to the "current" iter)
            }         

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
            {'interactions':dict[prompt_type_str[list[torch.tensor]]],
            'interactions_labels':dict[prompt_type_str[list[torch.tensor]]],
            }
        interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
            (where each prompt spatial coord is represented as a sublist).  
            dict[prompt_type_str[class[list[list]]]]


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
                infer_app, 
                args: dict):
        '''
        Inputs: 
    
        infer_app: Initialised inference application which can be called to process an input request.
        
        args: Dictionary containing the information required for performing the experiment, e.g.: 

            random_seed: The int denoting the seed being used for this instance of validation. 
        
            config_labels_dict: Dictionary mapping class labels and integer codes.
            
            inter_init_prompt_config (and inter_edit_prompt_config): "use mode" specific prompt generation config dictionary, 
            
            inference run configs: (e.g., modes, number of refinement iterations)
            
            metrics configs: metrics being computed, prompt generation configs for parameter-dependent metrics, etc.
            
            interaction memory configs: configs for how the interaction states will be stored to be passed through for
            the infer_app call: contains fields 'keep_init' and 'im_len' (former denotes the treatment of init interaction
            state, while im_len denotes the edit memory)

            etc.

        TODO: Add a full exhaustive list of the dictionary fields.

        '''
        
        self.infer_app = infer_app
        self.args = args

        self.init_seeds(seed=self.args['random_seed'])

    def init_seeds(self, seed): #, cuda_deterministic=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        #TODO: Once containerisation is implemented, we can re-implement this functionality for instances where a DL model may be used for
        # prompt generation

        # if cuda_deterministic:
        #     cudnn.deterministic = True 
        #     cudnn.benchmark = False
        # else:
        #     cudnn.deterministic=False 
        #     cudnn.benchmark=True 
    def app_output_processor(self, output_data, infer_call_config, metrics_dict):
        '''
        Makes use of the output processor class. This will tie together several functionalities such as 
        reformatting the output data dictionary, writing the segmentations, computing the metrics etc.
        
        output_data: Dict - A dictionary containing the pre-processed output dictionary from the inference app call.

        infer_config: Dict - A dictionary containing two subfields:
            'mode': str - The mode that the inference call was be made for (Automatic Init, Interactive Init, Interactive Edit)
            'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
        
        metrics_dict: Dict - A dictionary containing the tracked metrics.
        '''
        pass 

    def input_prompt_gen_initialiser(self):
        '''
        This function initialises the class objects which can generate the interaction states for use in inference.
        '''
        if self.args['prompt_procedure_type'].title() == 'Heuristic':
            # self.autoseg_state_generator = None  
            # The Autosegmentation state does not require any interaction state generators since they contain no 
            # interaction. All information for the autosegmentation state will be provided manually.

            self.inter_init_generator = HeuristicInteractionState(
                sim_device=self.args['sim_device'],
                use_mem=self.args['use_mem_generator'],
                prompt_configs=self.args['inter_init_prompt_config'],
                config_labels_dict=self.args['config_labels_dict']
            )
            self.inter_edit_generator = HeuristicInteractionState(
                sim_device=self.args['sim_device'],
                use_mem=self.args['use_mem_generator'],
                prompt_configs=self.args['inter_edit_prompt_config'],
                config_labels_dict=self.args['config_labels_dict']
            )
        else:
            raise ValueError('The selected prompt generation algorithm is not supported')
    
    def metric_prompt_gen_initialiser(self):
        '''
        Function intended for initialising the prompt generators for the metric computation.
        '''
        pass 

    def metric_im_handler(self):
        '''
        Function intended for interaction memory handling, but for instances where the generated data is used for metric
        computation.
        '''

    def input_im_handler(self,
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
        
        if infer_config['mode'].title() == 'Automatic Init':
            im = {'Automatic Init': None}

        elif infer_config['mode'].title() == 'Interactive Init':
            im = {'Interactive Init': self.inter_init_generator(
                image=data_instance['image']['metatensor'], 
                mode=infer_config['mode'], 
                gt=data_instance['label']['metatensor'], 
                prev_output_data=prev_output_data, 
                im=None)} 


        elif infer_config['mode'].title() == 'Interactive Edit':
            im[f'Interactive Edit Iter {infer_config["edit_num"]}'] = self.inter_edit_generator(
                image=data_instance['image']['metatensor'],
                mode=infer_config['mode'],
                gt=data_instance['label']['metatensor'],
                prev_output_data=prev_output_data,
                im=im
            )
            #Here we implement the optional use of memory clipping in instances where memory concerns may exist.

            im = im_cleanup(self.args['is_seg_tmp'], self.args['tmp_dir_path'], self.args['im_config'], im, infer_config)
        
        return im 

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
        
        im - (Optional) The currently existing interaction memory (for edit) or NoneType (for initialisations) 
        
        prev_output_data - (Optional) The post-processed output dictionary from the prior iteration of inference (for editing modes). 
        or NoneType. 

        Returns:

        request - The input request dictionary for input to the app inference call.
        im - The updated interaction memory dict for tracking.
        '''

        if infer_call_config['mode'].title() == 'Automatic Init':
            if prev_output_data is not None: #We choose an explicit check of Nonetype for the if statement
                raise ValueError('The previous output should not exist for initialisation')
            
            im = self.input_im_handler(
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
                raise ValueError('The previous output should not exist for initialisation')
            
            im = self.input_im_handler(
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
            
            im = self.input_im_handler(
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
        
    def iterative_loop(self, dataset):
        '''
        Dataset object which can be iterated through, performs a set of load transforms when called. 

        ''' 
        
        
        infer_run_configs = self.args['infer_run_configs']
        
        if type(infer_run_configs) != dict:
            raise TypeError('Inference run type configs must be presented in dict, with keys of "edit_bool" and "num_iters".')
        
        if type(infer_run_configs['edit_bool']) != bool:
            raise TypeError('edit_bool value must be a boolean in the inference run configs dict.')
        
        if infer_run_configs['edit_bool']:
            
            #We use the initialisation mode provided in the inference run config to initialise the model.
            if infer_run_configs['init'].title() == 'Automatic Init':
                pass 
                #TODO: Add the prompt generation and inference for the automatic initialisation.
            elif infer_run_configs['init'].title() == 'Interactive Init':
                pass 
                #TODO: Add the prompt generation and inference for the interactive initialisation.
            else:
                raise KeyError('A supported initialisation mode was not selected')  
            
            if type(self.args['infer_run_configs']['num_iters']) != int:
                raise TypeError('The variable for quantity of editing iterations was not an int type')
            
            for iter_num in range(self.args['infer_run_configs']['num_iters']): 
                #TODO: Add the prompt generation, inference and segmentation saving for the iterative segmentation process. 
                pass 
                # self.prompt_generation_handler()

        else:
            #Else, just the initialisation, either auto-seg or interactive init. Likely to be seldom used but just in case.
            if infer_run_configs['init'].title() == 'Automatic':
                pass 
                #TODO: Add the prompt generation, inference and segmentation saving for the auto initialisation ONLY runs.
            elif infer_run_configs['init'].title() == 'Interactive':
                pass
                #TODO: Add the prompt generation, inference and segmentation saving for the INTERACTIVE initialisation ONLY runs. 
            else:
                raise KeyError('A supported initialisation mode was not selected')
            
    def __call__(self):
        pass 


if __name__ == '__main__':
    dummy = front_end_simulator()
    print('stop')