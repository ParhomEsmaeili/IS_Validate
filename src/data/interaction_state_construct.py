from typing import Union
from monai.data import MetaTensor 
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.prompt_generators.prompt_generators import HeuristicSpatialPromptGenerator
import logging 

logger = logging.getLogger(__name__)

class HeuristicInteractionState(HeuristicSpatialPromptGenerator):
        '''
        Class which can be used to take the image, Optional(prior segmentation information) and ground truth in order to generate an interaction state dictionary for each 
        interaction state throughout the iterative refinement process for any given prompt configuration provided:
        
        This can be dependent on use-mode, hence the class can be initialised for each use interactive use mode separately        
        according to the provided prompt-configuration set.

        NOTE: Assumption that will be made is that at least one prompt must be generated across the classes for
        interactive modes. Users would not otherwise interact with a system!
        
        Interaction state is defined as a dict which contains the following key:value pairs:
            
            Image - A dictionary separated by the datatype: 1) Path to the image, 2) MetaTensor of the image itself.
            
            Prompts (currently supports points, scribbles, bbox). The prompts and labels provided with
            two types of formats:

            1) list[torch] formats, key = 'interactions_torch_format' This is a dictionary with two subdicts:
        
                a) 'interactions":
                    Points: List of torch tensors each with shape 1 x N_dim (N_dims = number of image dimensions)
                    Scribbles: Nested list of scribbles (N_sp), each scribble is a list of torch tensors with shape [1 x N_dim] denoting the positional coordinate.
                    Bboxes: List of N_box torch tensors, each tensor is a 1 x 2*N_dim shape (Extrema of the bbox with order [i_min, j_min, k_min, i_max, j_max, k_max] where ijk = RAS convention) 
                
                b) 'interactions_labels'

                    Points_labels: List of torch tensors each with shape 1 (Values = class-integer code value for the given point: e.g. 0, 1, 2, 3...)
                    Scribbles_labels: List of torch tensors, each with shape 1 (Values = class-integer code value for the given point: e.g. 0, 1, 2, 3... )
                    Bboxes_labels: List of N_box torch tensors, each with shape 1 (Values = class-integer code value for the given point)

                "interactions" contains the prompts spatial coords info, and "interactions_labels" the corresponding
                labels for the prompts. 

            2) Currently supported Dict format. 
                Points: Dictionary of class separated (by class label name) nested list of lists, each with length N_dims. 
                Scribbles: Dictionary of class separated (by class label name) 3-fold nested list of lists: [Scribble Level List[Point Coordinate List[i,j,k]]]
                Bboxes: Dictionary of class separated (by class label name) 2-fold nested list of list [Box-level[List of length 2 * N_dim]]. 
                Each sublist denotes the extreme value of the points with order [i_min, j_min, k_min, i_max, j_max, k_max], where ijk=RAS convention. 
                
            NOTE: In instances where a prompt type is not at all being (or was not for a given iter) simulated, the corresponding values will be a Nonetype. 

            
            prev_pred: Info for the discretised prediction maps which the application has provided.
            prev_logits: Channelwise logit maps which correspond to that which the application has provided.
            
                Each contains a dictionary separated by the datatype format for representing them:
                    1) Path(s) to the file(s) 2) Original forward propagated form for the array 
                    3) Meta dictionary for the array
                    
            3) Extra input arguments that needs to be stored in memory for future inference passes, defined by the user's application! This will be pulled from the
                output_data dictionary generateed from the inference app call with the "optional_memory" key. 
                    
                E.g., prompt feature embeddings, image feature embeddings.

         
        '''
        
        def __init__(self,
                    device: str,
                    use_mem:bool, 
                    prompt_configs: dict,
                    config_labels_dict: dict,
                    ):

            '''
            inputs:

            device: The device which the prompt generation computations will be implemented on.
            
            use_mem: A bool denoting whether the interaction memory should be used as part of prompt generation (I.E is the "UI" 
            refreshing or accumulating prompts)

            prompt_configs: The configurations for the simulation methods (for each prompt type), the build params,
            and the prompt mixture params. 
            
            config_labels_dict: The dictionary mapping the class label and the integer-codes.

            '''
            super().__init__(
                            device=device,
                            use_mem=use_mem,
                            config_labels_dict=config_labels_dict,
                            sim_methods=prompt_configs['methods'], 
                            sim_build_params=prompt_configs['build_params'],
                            prompt_mixture_params=prompt_configs['mixture_params'])
            
            self.init_modes = ['Interactive Init'] 
            self.edit_modes = ['Interactive Edit']

        def __call__(
                    self, 
                    image: Union[torch.Tensor, MetaTensor],
                    infer_mode: str,
                    gt: Union[torch.Tensor, MetaTensor],
                    prev_output_data: Union[dict, None],
                    im: Union[dict, None]
                    ):
            '''
            Input:
            
            Image, in Torch tensor or MetaTensor format loaded in the native UI domain.
            
            Infer_mode, a string denoting the inference mode for which we are simulating prompts.

            GT, in Torch tensor or MetaTensor format loaded in the native UI domain. 
            
            Prev Output Data: An optional dictionary from the prior inference call output:
        
            Two related fields here - discretised segmentation and multi-channel (channel first) logits maps (background is a class).  

            im: An optional dictionary containing the set of retained interaction states (or NONE). 

            Returns:
            Interaction state dictionary for the current iteration for which the call has been made.
            '''


            generator_input_data = {
                'image': image,
                'gt': gt,
                'prev_output_data': prev_output_data,
                'im':im
                }
            
            #Generation of the prompts:
            if infer_mode.title() == 'Interactive Init' or infer_mode.title() == 'Interactive Edit':
                interaction_torch_format, interaction_labels_torch_format, interaction_dict_format = self.generate_prompt(generator_input_data)
            else:
                raise ValueError('The inference mode inputted is not valid!')
            
            #Here we perform a sleight of hand, to reformat the prev_output data for the interaction state defn.
            #since there will be no information regarding the prev_output_data for initialisation. 

            if infer_mode in self.init_modes:
                prev_output_data = {
                'logits':{
                    'paths': None,
                    'logits_metatensor': None,
                    'logits_meta_dict': None
                },
                'pred':{
                    'path': None,
                    'pred_metatensor': None,
                    'pred_meta_dict': None
                },
                'optional_memory': None
                }
            
            interaction_state_dict = {
                'interaction_torch_format':{
                    'interactions': interaction_torch_format,
                    'interactions_labels': interaction_labels_torch_format,
                },
                'interaction_dict_format': interaction_dict_format,
                'prev_logits':{
                    'paths': prev_output_data['logits']['path'],
                    'prev_logits_metatensor': prev_output_data['logits']['logits_metatensor'],
                    'prev_logits_meta_dict': prev_output_data['logits']['logits_meta_dict']
                },
                'prev_pred':{
                    'path': prev_output_data['pred']['path'],
                    'prev_pred_metatensor': prev_output_data['pred']['pred_metatensor'],
                    'prev_pred_meta_dict': prev_output_data['pred']['pred_meta_dict']
                },
                'prev_optional_memory': prev_output_data['optional_memory']
            }


            return interaction_state_dict 

# def interaction_state_metrics(self, metric_parametrisation):
#         '''
#         Function which takes the prompts, ground truth and generates the parametrisation for any parameter-dependent metrics specified in the experimental setup.
#         '''
#         pass  