'''
This script is intended for simulating inference from the pseudo-front end, as part of the front-end to back-end setup of an end-to-end interactive seg. application. 
'''
from monai.transforms import Compose, ToDeviced 
from typing import Callable, Sequence, Union, Optional
import logging
import sys
import torch
import os
from front_back_interact import front_back_processor
from src.data.interaction_state_construct import HeuristicInteractionState 


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

    NOTE: Orientation convention is always assumed to be RAS! 

        image: A dictionary containing a path & a pre-loaded (UI) metatensor object 
        {'path':image_path, 
        'metatensor':image_metatensor_obj
        'meta_dict': image_meta_dictionary}

        model: A string denoting the inference "mode" being simulated, has three options: 
                1) Automatic Segmentation, denoted: 'IS_autoseg' 
                2) Interactive Initialisation: 'IS_inter_init'
                3) Interactive Editing: 'IS_inter_edit'

        IM: A dictionary containing the set of interaction states. Keys = Inference iter num (0, 1, ...).
        
        NOTE: For initialisations, IM = None 

        NOTE: Experimental config must pre-define the set of interaction states required with the following argument:
            {'keep_init': bool,
            'memory_len': int
            }         

        Within each state in IM (NOTE: logits, prev_):    
        
        prev_logits: A dictionary containing: {
                'paths': list of paths, to each individual logits map (1HWD), in the same order as provided by output CHWD logits map}
                'metatensor': Non-modified metatensor/numpy array/torch tensor that is forward-propagated from the prior output (CHWD).
                'logits_meta_dict': Non-modified meta dictionary that is forward propagated.
                }
        prev_pred: A dictionary containing: {
                'path': path}
                'metatensor': Non-modified metatensor/numpy array/torch tensor that is forward-propagated from the prior output (1HWD).
                'pred_meta_dict': Non-modified meta dictionary that is forward propagated.
                }
        prev_optional_memory: A dictionary containing any extra information that the application would like to forward propagate
        which is not currently provided.

        Refinement prompt information: See `<https://github.com/ParhomEsmaeili/IS_Validate/blob/main/src/data/interaction_state_construct.py>`

        interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor] format 
            {'interactions':dict[prompt_type_str[list[torch.tensor]]],
            'interaction_labels':dict[prompt_type_str[list[torch.tensor]]],
            }
        interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
            (where each prompt spatial coord is represented as a sublist).  
            dict[prompt_type_str[class[list[list]]]]


    Inference app must generate the output in a dict format with the following fields:

    NOTE: Checks will be put in place to ensure that image resolution, spacing, orientation will be matching & otherwise 
    the code will be non-functional.

        logits: MetaTensor/numpy/torch object, multi-channel logits map (CHWD), where C = Number of Classes
        
        logits_meta_dict: Meta information in dict format,  ('affine must match the input-image metatensor's affine info)
        
        pred: MetaTensor/numpy/torch object containing the discretised prediction (shape 1HWD)
        pred_meta_dict: Meta information corresponding to the header of the prediction (must match the input image header)

    NOTE: Optional to include the "optional_memory" field also, for any extra arguments they would like to store in IM.

    '''
    def __init__(self, 
                infer_app, 
                args: dict):
        '''
        Inputs: 
    
        infer_app: Initialised inference application which can be called to process an input request.
        
        args: Dictionary containing the information required for performing the experiment, e.g.: 
        class_label_config dictionary, "use mode" specific prompt generation config dictionary, 
        inference run configs (e.g., modes, number of refinement iterations), metrics being computed, interaction memory storage,
        etc.

        TODO: Add a full exhaustive list of the dictionary fields.

        '''
        
        self.infer_app = infer_app
        self.args = args

        super().__init__()

    

  
    def prompt_gen_initialiser(self):
        '''
        This function initialises the class objects which can generate the interaction states for use in inference.
        '''
        if self.args['prompt_procedure_type'].title() == 'Heuristic':
            # self.autoseg_state_generator = None  
            # The Autosegmentation state does not require any interaction state generators since they contain no 
            # interaction. All information for the autosegmentation state will be provided with just an image input.

            self.inter_init_generator = HeuristicInteractionState(
                methods=self.args['inter_init_prompt_config'],
                config_labels_dict=self.args['config_labels_dict']
            )
            self.inter_edit_generator = HeuristicInteractionState(
                methods=self.args['inter_edit_prompt_config']
            )
        else:
            raise ValueError('The selected prompt generation algorithm is not supported')
        
    def app_request_generator(self,
                            inference_mode: str,
                            previous_output: Optional[dict] = None):
        '''
        This function generates the app request (i.e. the input dictionary to the application).

        Each request comes with field containing the app_(sub)name also: I.e., Autosegmentation, Interactive Init, Interactive Edit. Users can provide three separate
        apps, or just repeat the same but it should be packaged in a manner such that the input request will be channeled appropriately for their requirements.
        
        We use the following convention, for Automatic Init: 'IS_autoseg', Interactive Init: 'IS_interactive_init', Interactive Edit: 'IS_interactive_edit'.

        This value will be stored under the "App_Type" key in the input request. 


        Inputs:

        inference_mode - The mode in which the application is being used, therefore queried in the request.
        previous_output - (Optional) The output dictionary from the prior iteration of inference (for editing modes). 
        '''

        if inference_mode.title() == 'Automatic Init':
            if previous_output != None: #We choose an explicit check of Nonetype for the if statement
                raise ValueError('The previous output should not exist for initialisation')
            
            request = {'model': 'IS_autoseg', 'class_configs': self.args['class_label_configs']}

        elif inference_mode.title() == 'Interactive Init':
            if previous_output != None:
                raise ValueError('The previous output should not exist for initialisation')
            
            request = {'model': 'IS_inter_init', 'class_configs': self.args['class_label_configs']}

        elif inference_mode.title() == 'Interactive Edit':
            if previous_output == None:
                raise ValueError('There must be a dictionary containing the outputs of the prior inference call!')
            
            request = {'model': 'IS_inter_edit', 'class_configs': self.args['class_label_configs']}
        else:
            raise ValueError('The inference mode is invalid for app request generation!')

        raise NotImplementedError('The implementation for the app request generator is incomplete.')
    
        #TODO: 
        # Merge the base template for the request with the interaction state output into one dictionary
        # Merge the FSIM dictionary into the request also.
        return request 
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