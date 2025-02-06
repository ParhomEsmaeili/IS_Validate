'''
This script is intended for simulating inference from the pseudo-front end, as part of the front-end to back-end setup of an end-to-end interactive seg. application. 
'''
from monai.transforms import Compose, ToDeviced 
from typing import Callable, Sequence, Union, Optional
import logging
import sys
import torch
import os
from front_back_interact import front_back_checker
from src.data.interaction_state_construct import HeuristicInteractionState 


logger = logging.getLogger(__name__)

#Split input request formats to the "app backend" according to the format used for extracting the previous segmentation masks (e.g. path, or tensor based)
#Then need to split request formats according whether full end-to-end is provided or not by the app.


class front_end_simulator(front_back_checker):
    '''
    This class serves as the pseudo-ui with operations such as: 
    
    Loading the imaging data in the "UI" domain,
    Generating prompts in the "UI" domain, 
    Storing the interaction states through the iterative segmentation process,
    Saving the segmentation states for each image?


    etc.

    Inputs: 
    
    Initialised inference application which can be called using an input request.

    '''
    def __init__(self, 
                infer_app, 
                args: dict):
        
        self.infer_app = infer_app
        self.args = args

        super().__init__()

    

  
    def prompt_gen_initialiser(self):
        '''
        This function initialises the class objects which can generate the interaction states for use in inference.
        '''
        if self.args['prompt_procedure_type'].title() == 'Heuristic':
            # self.autoseg_state_generator = None  The Autosegmentation states do not require any interaction state generators since they contain no interaction.
            self.inter_init_generator = HeuristicInteractionState(
                mode='Interactive Init'
            )
            self.inter_edit_generator = HeuristicInteractionState(
                mode='Interactive Edit'
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
            
            request = {'model': 'IS_interactive_init', 'class_configs': self.args['class_label_configs']}

        elif inference_mode.title() == 'Interactive Edit':
            if previous_output == None:
                raise ValueError('There must be a dictionary containing the outputs of the prior inference call!')
            
            request = {'model': 'IS_interactive_edit', 'class_configs': self.args['class_label_configs']}
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