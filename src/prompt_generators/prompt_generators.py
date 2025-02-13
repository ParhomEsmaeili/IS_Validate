from typing import Union, Callable, Sequence, Optional
from abc import abstractmethod
import torch
import os
import sys
import numpy as np
from monai.data import MetaTensor 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.prompt_generators.prompt_reformat_utils import PromptReformatter
from src.prompt_generators.build_heuristics import BuildHeuristic
import logging 

logger = logging.getLogger(__name__)

class BasicSpatialPromptGenerator(PromptReformatter):
    def __init__(self,
                device: str,
                config_labels_dict: dict,
                sim_methods: dict[str, dict],
                sim_build_params: Optional[dict[str, dict]] = None,
                prompt_mixture_params: Optional[dict[str,dict]] = None,
                ):
        '''
        Prompt generation class for generating the interactive spatial prompts for an interaction state.

        Inputs:

        device: A str denoting the cuda device (or cpu) being used for the prompt generation computations. 

        config_labels_dict: A dictionary mapping the class labels to the class integer codes. 

        sim_methods: Simulation methods used for generating each prompt. Must always provide for all prompt types.
        
        A nested dictionary first separated by the prompt type (e.g. points, scribbles, bbox).

        Within each prompt type, another dictionary contains key:value pairs containing the list of prompting strategies being used for prompt simulation. 
        This dictionary has the flexibility to permit combinations of strategies. 

        NOTE: Instances where a specific prompt type is not being used, will require that the value be a NoneType arg instead.
        The prompts generated will also be a NoneType argument for the given prompts also!

        



        (OPTIONAL) sim_build_params: A twice nested dictionary, for each prompt strategy within a prompt type it 
        contains a dictionary of build arguments for each corresponding strategy implemented. 

        Almost the same structure as prompt_methods, as they must correspond together!

        NOTE: Can also be a Nonetype, in which case there is no information required for building the prompters
        (e.g., parameter free/variable free, or if there is no prompt of that type being used)

        NOTE: Child classes where the prompt generation method may be the same across prompt types
        e.g. prompt generation D.L models, will necessitate that the prompt mixture argument denotes the strategy for
        cross-interaction (e.g., conditioning the model with a prompt prior, or passing all prompt-type requests concurrently 
        through). NOTE: However, prompt-type & method specific args can still be passed through in the build_params.
        





        (OPTIONAL)prompt_mixture: A twice nested dict denoting a strategy for mixing prompt simulation across intra
        and inter-prompting strategies. The mixture args are a dictionary for each corresponding pairing of cross-interactions.

        This prompt mixture arg will control whether/how prompt-methods will interact/condition one another during the 
        simulation. 

            Has structure: dict('inter_prompt':dict[tuple(prompt cross-interaction combinations), mixture_args/None], 
                                'intra_prompt': dict[prompt_type_str : dict[tuple(prompt_strategy cross-interaction combinations), mixture_args/None])
            
            Tuple can provide an immutable set of combinations

            NOTE: Downstream use will likely necessitate the use of set logic to verify combinations as the tuple 
            is immutable. Verification likely will entail the following: Generate set of potential combinations from
            prompt_types (or strategy), cross-reference with the corresponding dict item by converting key into set.


        Can optionally can be Nonetype (i.e., fully  independently assumed prompt generation) simulation of prompts (inter and intra-prompt strategy). 
        Hence they will be generated with no consideration of prompting intra and inter-dependencies outside of resolving redundancy conflicts (which should not happen).
    
        '''
        super().__init__(config_labels_dict)

        self.sim_methods = sim_methods
        self.sim_build_params = sim_build_params 
        self.prompt_mixture_params = prompt_mixture_params 

        self.prompt_types = list(sim_methods.keys()) 

        #Building the interactive prompter class, where the callback generates the prompts (and corresponding labels) in torch format.
        self.interactive_prompter = self.build_prompt_generator(device,
                                                                config_labels_dict,
                                                                self.sim_methods, 
                                                                self.sim_build_params, 
                                                                self.prompt_mixture_params)

        

    @abstractmethod
    def build_prompt_generator(self, device, prompt_methods, prompt_build_params, prompt_mixture_params): 
        pass
    
    
    def reformat_to_dict(self, torch_format_prompts: dict, torch_format_labels: dict):
        '''
        This function converts the torch format prompts into a dictionary format. Assumed convention:
    
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
        
        NOTE: In instances where a prompt type is not at all being (or was not for a given iter) simulated, the corresponding values will be a Nonetype. 

        Returns:

        Currently supported Dict format:
         
            Points: Dictionary of class separated (by class label name) nested list of lists, each with length N_dims. 
            Scribbles: Dictionary of class separated (by class label name) 3-fold nested list of lists: [Scribble Level List[Point Coordinate List[i,j,k]]]
            Bboxes: Dictionary of class separated (by class label name) 2-fold nested list of list [Box-level[List of length 2 * N_dim]]. 
            Each sublist denotes the extreme value of the points with order [i_min, j_min, k_min, i_max, j_max, k_max], where ijk=RAS convention. 
        
        NOTE: In instances where a prompt type is not at all being (or was not for a given iter) simulated, the corresponding values will be a Nonetype. 

        '''

        #We reformat the prompt coord & label information into a dictionary format:
         
        prompt_dict_format = dict()

        for prompt_type in self.prompt_types:
            prompts = torch_format_prompts[prompt_type]
            prompts_labels = torch_format_labels[prompt_type]

            if prompts is None or prompts_labels is None:
                prompt_dict_format[prompt_type] = None 
            else:
                prompt_dict_format[prompt_type] = self.reformat_prompts(prompt_type, prompts, prompts_labels)    

        return prompt_dict_format 
        
    def generate_prompt(self, data: dict):
        
        '''
        Inputs: 

        data: Dictionary which contains the following fields:

        image: Torch tensor OR Metatensor containing the image in the native image domain (no pre-processing applied other than re-orientation in RAS)
        gt: Torch tensor OR Metatensor containing the ground truth map in RAS orientation, but otherwise in the native image domain (no pre-processing other than RAS re-orientation).
        
        prev_output_data: (NOTE: OPTIONAL) output dictionary from the inference call which has been post-processed 
        in the pseudo-ui front-end.
       
        Two relevant fields for prompt generation contained are the: 
            pred: A dictionary containing 3 subfields:
                1) "path": Path to the prediction file (Not Relevant)
                And two relevant subfields
                2) "pred_metatensor" A Metatensor or torch tensor containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "pred_meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
            
            logits:
                1) "paths": List of paths to the prediction file (Not Relevant)
                And two potentially relevant subfields
                2) "logits_metatensor" A Metatensor or torch tensor containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "logits_meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
             
        '''
        
        if not isinstance(data['gt'], torch.Tensor) and not isinstance(data['gt'], MetaTensor):
            raise TypeError('The ground truth must belong to the torch tensor, or monai MetaTensor datatype')

        if not isinstance(data['image'], torch.Tensor) and not isinstance(data['image'], MetaTensor):
            raise TypeError('The image must belong to the torch tensor, or monai MetaTensor datatype')
        
        if not data['prev_output_data'] is None:
            #We run this check for the instance where the "prev_output_data" exists (i.e. for refinement iters)
            if not isinstance(data['pred']['pred_metatensor'], torch.Tensor) and not isinstance(data['pred']['pred_metatensor'], MetaTensor):
                raise TypeError('The previous pred must belong to the torch tensor, or monai MetaTensor datatype')
            
            if not isinstance(data['logits']['logits_metatensor'], torch.Tensor) and not isinstance(data['logits']['logits_metatensor'], MetaTensor):
                raise TypeError('The previous logits must belong to the torch tensor, or monai MetaTensor datatype')
        
        p_torch_format, plabels_torch_format = self.interactive_prompter(data) 
        p_dict_format = self.reformat_to_dict(p_torch_format, plabels_torch_format)

        return p_torch_format, plabels_torch_format, p_dict_format
    


class HeuristicSpatialPromptGenerator(BasicSpatialPromptGenerator):
    def __init__(self,
                device: str,
                config_labels_dict: dict[str, int],
                sim_methods:dict[str,dict], 
                sim_build_params:Optional[dict[str, dict]] = None,
                prompt_mixture_params:Optional[dict[str,dict]]=None):
        
        super().__init__(
                        device=device,
                        config_labels_dict=config_labels_dict,
                        sim_methods=sim_methods, 
                        sim_build_params=sim_build_params,
                        prompt_mixture_params=prompt_mixture_params
                        ) 
    
    def build_prompt_generator(self,
                               device: str, 
                               config_labels_dict:dict[str, int], 
                               sim_methods: dict, 
                               sim_build_params: Union[dict, None], 
                               prompt_mixture_params:  Union[dict, None]):
        
        return BuildHeuristic(
                            device=device,
                            config_labels_dict=config_labels_dict, 
                            heuristics=sim_methods, 
                            heuristic_params=sim_build_params,
                            heuristic_mixtures=prompt_mixture_params)

if __name__=='__main__':
    generator = HeuristicSpatialPromptGenerator(sim_methods={'points':['uniform_random'], 'scribbles':None, 'bbox':None},
                                    config_labels_dict={'tumour':1, 'background':0},
                                    sim_build_params=None, #{'points':None, 'scribbles':None, 'bbox':None},
                                    prompt_mixture_params=None)
    generator.generate_prompt({'gt':torch.ones([128,128,128]), 'image':torch.ones([128,128,128])})