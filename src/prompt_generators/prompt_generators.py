from typing import Union, Callable, Sequence, Optional
from abc import abstractmethod
import torch
from src.data.prompt_reformat_utils import PromptReformatter
import logging 

logger = logging.getLogger(__name__)

class BasicSpatialPromptGenerator(PromptReformatter):
    def __init__(self,
                sim_methods: dict[str, dict],
                config_labels_dict: dict,
                sim_build_params: Optional[dict[str, dict]] = None,
                prompt_mixture_params: Optional[dict[str,dict]] = None,
                ):
        '''
        Prompt generation class for generating the interactive spatial prompts for an interaction state.

        Inputs:

        sim_methods: Simulation methods for each prompt.
        
        A nested dictionary first separated by the prompt type (e.g. points, scribbles, bbox).

        Within each prompt type, the dictionary contains key:value pairs containing the list of prompting strategies being used for prompt simulation. 
        This dictionary has the flexibility to permit combinations of strategies. 

        config_labels_dict: A dictionary mapping the class labels to the class integer codes. 

        (OPTIONAL) prompt_build_params: A nested dictionary with key:value pairs containing the build arguments for each corresponding strategy implemented. 
        Same structure as prompt_methods

        Can also be a Nonetype, in which case there is no information required for building the prompters (e.g., parameter free/variable free)

        (OPTIONAL)prompt_mixture: A nested dict denoting a strategy for mixing prompt simulation across intra and inter-prompting strategies.

            Has structure: dict('inter_prompt':dict[prompt cross-interaction permutations, mixture_args/None], 
                                'intra_prompt': dict[prompt_type_str : dict[prompt_method cross-interaction permutations str, mixture_args/None])

        Can optionally can be Nonetype (i.e., fully  independent) simulation of prompts (inter and intra-prompt strategy). Hence they will be generated with no consideration
        of prompting inter-dependencies.
    
        '''
        super().__init__(config_labels_dict)

        self.sim_methods = sim_methods
        self.sim_build_params = sim_build_params 
        self.prompt_mixture_params = prompt_mixture_params 

        self.prompt_types = list(sim_methods) 

        #Building the interactive prompter class, where the callback generates the prompts (and corresponding labels) in torch format.
        self.interactive_prompter = self.build_prompt_generator(self.sim_methods, self.sim_build_params, self.prompt_mixture_params)

        

    @abstractmethod
    def build_prompt_generator(self, prompt_methods, prompt_build_params, prompt_mixture_params) -> dict[str, list[torch.Tensor]]: 
        pass
    
    
    def reformat_to_dict(self, torch_format_prompts: dict, torch_format_labels):
        '''
        This function converts the torch format prompts into a dictionary format. Assumed convention:

        1) Currently supported list[torch] formats:
        
            Points: List of torch tensors each with shape 1 x N_dim (N_dims = number of image dimensions)
            Point Labels: List of torch tensors each with shape 1 (Values = class-integer code value for the given point: e.g. 0, 1, 2, 3...)
            Scribbles: Nested list of scribbles (N_sp), each scribble is a list of torch tensors with shape [1 x N_dim] denoting the positional coordinate.
            Scribbles Labels: List of torch tensors, each with shape 1 (Values = class-integer code value for the given point: e.g. 0, 1, 2, 3... )
            Bbox: List of N_box torch tensors shaped [1 x 2 * N_dim] (Extreme points of the bbox with order [i_min, j_min, k_min, i_max, j_max, k_max] where ijk = RAS convention) 
            Bbox Labels: List of N_box torch tensors, each with shape 1 (Values = class-integer code value for the given point)
            Label config: Dictionary which denotes the class-integer code mapping.
        
        2) Currently supported Dict format 
            Points: Dictionary of class separated (by class label name) nested list of lists, each with length N_dims. 
            Scribbles: Dictionary of class separated (by class label name) 3-fold nested list of lists: [Scribble Level List[Point Coordinate List[i,j,k]]]
            Bbox: Dictionary of class separated (by class label name) 3-fold nested list of list [Box-level[List of length 2 * N_dim, with each sublist of length N_dims]]. 
            Each sublist denotes the extreme value of the points with order [i_min, j_min, k_min, i_max, j_max, k_max], where ijk=RAS convention. 
            Label Config: Dictionary which denotes the class-integer code mapping.
        '''

        #We reformat the prompt coord & label information:
         
        prompt_dict_format = dict()

        for prompt_type in self.prompt_types:
            prompts = torch_format_prompts[prompt_type]
            prompts_labels = torch_format_labels[prompt_type]

            prompt_dict_format[prompt_type] = self.reformat_prompts(prompt_type, prompts, prompts_labels)    
    
        
    def generate_prompt(self, data: dict):
        
        '''
        Inputs: 

        data: Dictionary which contains the following fields:

        gt: Torch tensor OR Metatensor containing the ground truth map in RAS orientation, but otherwise in the native image domain (no pre-processing other than RAS re-orientation).
        image: Torch tensor OR Metatensor containing the image in the native image domain (no pre-processing applied other than re-orientation in RAS)
        prev_seg: Optionally a Metatensor containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
        '''
        
        p_torch_format, plabels_torch_format = self.interactive_prompter(data) 
        p_dict_format = self.reformat_to_dict(p_torch_format, plabels_torch_format)

        return p_torch_format, plabels_torch_format, p_dict_format
    


class HeuristicSpatialPromptGenerator(BasicSpatialPromptGenerator):
    def __init__(self, 
                prompt_methods:dict[str,dict], 
                config_labels_dict: dict[str, int], 
                *args, 
                **kwargs):
        
        super().__init__(sim_methods=prompt_methods, 
                        config_labels_dict=config_labels_dict,
                        *args, 
                        **kwargs) 
    
    def build_prompt_generator(self, 
                               prompt_methods: dict, 
                               prompt_build_params: Union[dict, None], 
                               prompt_mixture_params:  Union[dict, None]):
        pass 