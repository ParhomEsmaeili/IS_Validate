from typing import Union, Callable, Sequence, Optional
from abc import abstractmethod
import torch
import os
import sys
import numpy as np
from monai.data import MetaTensor 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt_generators.reformatting_utils.prompt_reformat_utils import PromptReformatter
from prompt_generators.heuristics.build_heuristics import BuildHeuristic
import logging 

logger = logging.getLogger(__name__)

class BasicSpatialPromptGenerator(PromptReformatter):
    def __init__(self,
                sim_device: torch.device,
                semantic_id_dict: dict,
                use_mem: bool,
                sim_methods: dict,
                sim_build_params: dict,
                cascade_config: Union[dict, None],
                prompter_class_type: str,
                output_conversion: dict = None
                ):
        '''
        Prompt generation class for generating the interactive spatial prompts for an interaction state.

        Inputs:

        sim_device: A torch.device obj being used for the prompt generation computations. 

        semantic_id_dict: A dictionary mapping the class labels to the class integer codes. 

        use_mem: A bool determining whether prompt generation will make use of any interaction memory as part of conditioning the current state of simulation. 

        sim_methods: Simulation methods used for generating each prompt. Must always provide for all prompt types.
        
        A dictionary first separated by the prompt type (e.g. points, scribbles, bbox).

        Within each prompt type contains the list of prompting strategies being used for prompt simulation. 
        This has the flexibility to permit combinations of strategies. 

        NOTE: Instances where a specific prompt type is not being used, will require that the value be a NoneType arg instead.
        The prompts generated will also be a NoneType argument for the given prompts also!

        

        (REQUIRED) sim_build_params: A twice nested dictionary, for each prompt strategy within a prompt type it 
        contains a dictionary of build arguments for each corresponding strategy implemented. 

        Almost the same structure as prompt_methods, as they must correspond together!

        NOTE: For each method it must contain information about handling at the method-level.
        
        NOTE: Can also be a Nonetype if there is no prompt of that type being used.

        NOTE: Child classes where the prompt generation method may be the same across prompt types
        e.g. prompt generation D.L models, may  necessitate that the cascade config denotes the strategy for
        cross-interaction (e.g., conditioning the model with a prompt prior, or passing all prompt-type requests concurrently 
        through). NOTE: However, prompt-type/method specific args must still be passed through in the build_params.
    

        (OPTIONAL) cascade_config: An arbitrarily nested dict denoting a strategy for handling prompt simulation 
        e.g. class-level, intra-prompt level and inter-prompting level strategies etc.
        
    
        This cascade config will control whether/how prompt-methods will interact/condition one another 
        during the simulation. 

            For example, a structure may look like: 
        
        dict(
            'class_level': dict of mixture args/None
        
            'inter_prompt':dict[tuple(prompt cross-interaction combinations), mixture_args/None], 
                            
            'intra_prompt': dict[prompt_type_str : dict[tuple(prompt_strategy cross-interaction combinations), mixture_args/None])
        
            NOTE: Tuples can provide an immutable set of combinations

            NOTE: Downstream use will likely necessitate the use of set logic to verify combinations as the tuple 
            is immutable. Verification likely will entail the following: Generate set of potential combinations from
            prompt_types (or strategy), cross-reference with the corresponding dict item by converting key into set.


        Can optionally be fully Nonetype (i.e., will perform a default behaviour)

        prompter_class_type: The class type to be used for prompt generation. This may provide some basic structure for
        prompt generation which acts as a skeleton to be filled out with prior arguments when generating the prompts.
        '''
        super().__init__(semantic_id_dict)

        self.sim_device = sim_device
        self.semantic_id_dict = semantic_id_dict 
        self.use_mem = use_mem 
        self.sim_methods = sim_methods
        self.sim_build_params = sim_build_params 
        self.cascade_config = cascade_config 
        self.prompter_class_type = prompter_class_type
        self.output_conversion = output_conversion if output_conversion else None

        self.prompt_types = list(sim_methods.keys()) 

        #Building the interactive prompter class, where the callback generates the prompts (and corresponding labels) in torch format.
        
        self.interactive_prompter = self.build_prompt_generator() 

    @abstractmethod
    def build_prompt_generator(self): #, sim_device, prompt_methods, sim_build_params, cascade_config): 
        pass
    
    
    def output_reformatting(self, torch_format_prompts: dict, torch_format_labels: dict):
        '''
        This function converts the torch format prompts into a dictionary format. Assumed convention:
    
        a) 'interactions":
            Points: List of torch tensors each with shape 1 x N_dim (N_dims = number of image dimensions)
            Scribbles: List of scribbles, each scribble is a torch tensor with shape [N_s.p x N_dim] denoting the 
            positional coordinate. NOTE: N_s.p is highly dependent on the scribble length which can vary between scribbles! 
            Bboxes: List of N_box torch tensors, each tensor is a 1 x 2*N_dim shape (Extrema of the bbox with order [i_min, j_min, k_min, i_max, j_max, k_max] where ijk = RAS convention) 
            #BBox coordinates indicate the grid position (even at a sub-voxel level) for the extrema of the box.
            Ergo: [0] = the edge of the image array, not the centre of the first voxel.
            Lassos: List of N_lasso torch tensors, each lasso is a torch tensor with shape [N_lasso.p x N_dim]. This outlines a set of points which
            describe the lasso. Downstream algorithms will be required to determine the enclosed region from this set.
        b) 'interactions_labels'

            Points_labels: List of torch tensors each with shape 1 (Values = class-integer code value for the given point: e.g. 0, 1, 2, 3...)
            Scribbles_labels: List of torch tensors, each with shape 1 (Values = class-integer code value for the given point: e.g. 0, 1, 2, 3... )
            Bboxes_labels: List of N_box torch tensors, each with shape 1 (Values = class-integer code value for the given point)
            Lassos_labels: List of N_lasso torch tensors, each with shape 1 (Values = class-integer code value for the given lasso)
        "interactions" contains the prompts spatial coords info, and "interactions_labels" the corresponding
        labels for the prompts. 
        
        NOTE: In instances where a prompt type is not at all being (or was not for a given iter) simulated,
        the corresponding values will be a Nonetype. 

        Returns:

        Currently supported Dict format:
         
            Points: Dictionary of class separated (by class label name) nested list of lists, each with length N_dims. 
            Scribbles: Dictionary of class separated (by class label name) 3-fold nested list of lists: [Scribble Level List[Point Coordinate List[i,j,k]]]
            Bboxes: Dictionary of class separated (by class label name) 2-fold nested list of list [Box-level[List of length 2 * N_dim]]. 
            Each sublist denotes the extreme value of the points with order [i_min, j_min, k_min, i_max, j_max, k_max], where ijk=RAS convention. 
            Lassos: Dictionary of class separated (by class label name) 3-fold nested list of lists: [Lasso Level List[Point Coordinate List[i,j,k]]].
        NOTE: In instances where a prompt type is not at all being (or was not for a given iter) simulated, 
        the corresponding values will be a Nonetype. 

        '''

        #We reformat the prompt coord & label information into a dictionary format:
         
        prompt_dict_format = dict()

        for prompt_type in self.prompt_types:
            prompts, prompts_labels = torch_format_prompts[prompt_type], torch_format_labels[f'{prompt_type}_labels']

            if prompts is None or prompts_labels is None:
                prompt_dict_format[prompt_type] = None 
            else:
                reformatted_torch_format_prompts, reformatted_torch_format_labels = self.ensure_torch_datatype(prompts, prompts_labels)
                prompt_dict_format[prompt_type] = self.reformat_to_dict(
                    prompt_type=prompt_type,
                    prompts=reformatted_torch_format_prompts, 
                    prompts_labels=reformatted_torch_format_labels
                    )    
                torch_format_prompts[prompt_type] = reformatted_torch_format_prompts
                torch_format_labels[f'{prompt_type}_labels'] = reformatted_torch_format_labels
                
        return torch_format_prompts, torch_format_labels, prompt_dict_format 
        
    def generate_prompt(self, data: dict):
        
        '''
        Inputs: 

        data: Dictionary which contains the following fields:

        image: Torch tensor OR Metatensor containing the image in the native image domain (no pre-processing applied other than re-orientation in RAS)
        gt: Torch tensor OR Metatensor containing the ground truth map in RAS orientation, but otherwise in the native image domain (no pre-processing other than RAS re-orientation).
        
        prev_output_data: (NOTE: OPTIONAL) output dictionary from the inference call which has been post-processed 
        in the processor called in the pseudo-ui front-end.
       
        Two relevant fields for prompt generation contained are the: 
            pred: A dictionary containing 2 subfields:
                1) "metatensor" A Metatensor or torch tensor (1HW(D)) containing the previous segmentation in the native image domain 
                2) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
            
            probs:
                1) "metatensor" A Metatensor or torch tensor (CHW(D)) containing the previous segmentation in the native image domain 
                2) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
        
        im: (NOTE: OPTIONAL, NoneType for initialisation mode.) the interaction memory dictionary as defined in
        `<https://github.com/IS_Validate/blob/main/src/front_back_interactor/pseudo_ui.py>`   
        '''
        
        if not isinstance(data['gt'], torch.Tensor) and not isinstance(data['gt'], MetaTensor):
            raise TypeError('The ground truth must belong to the torch tensor, or monai MetaTensor datatype')

        if not isinstance(data['image'], torch.Tensor) and not isinstance(data['image'], MetaTensor):
            raise TypeError('The image must belong to the torch tensor, or monai MetaTensor datatype')
        
        if not data['prev_output_data'] is None:
            #We run this check for the instance where the "prev_output_data" exists (i.e. for refinement iters)
            if not isinstance(data['prev_output_data']['pred']['metatensor'], torch.Tensor) and not isinstance(data['prev_output_data']['pred']['metatensor'], MetaTensor):
                raise TypeError('The previous pred must belong to the torch tensor, or monai MetaTensor datatype')
            
            if not isinstance(data['prev_output_data']['probs']['metatensor'], torch.Tensor) and not isinstance(data['prev_output_data']['probs']['metatensor'], MetaTensor):
                raise TypeError('The previous probs must belong to the torch tensor, or monai MetaTensor datatype')
        
        if data['im'] is not None:
            if not isinstance(data['im'], dict):
                raise TypeError('The interaction memory must be a dict if it is not a NoneType')
            
        p_torch_format, plabels_torch_format = self.interactive_prompter(data)
        p_torch_format, plabels_torch_format, p_dict_format = self.output_reformatting(p_torch_format, plabels_torch_format)

        return p_torch_format, plabels_torch_format, p_dict_format
    


class HeuristicSpatialPromptGenerator(BasicSpatialPromptGenerator):
    def __init__(self,
                sim_device: torch.device,
                use_mem: bool,
                semantic_id_dict: dict[str, int],
                sim_methods:dict, 
                sim_build_params:dict,
                cascade_config:Union[dict, None],
                prompter_class_type: str,
                output_conversion: Union[dict, None]):
        
        super().__init__(
                        sim_device=sim_device,
                        use_mem=use_mem,
                        semantic_id_dict=semantic_id_dict,
                        sim_methods=sim_methods, 
                        sim_build_params=sim_build_params,
                        cascade_config=cascade_config,
                        output_conversion=output_conversion,
                        prompter_class_type=prompter_class_type
                        )  

    def build_prompt_generator(self):
        
        return BuildHeuristic(
                            sim_device=self.sim_device,
                            use_mem=self.use_mem,
                            semantic_id_dict=self.semantic_id_dict, 
                            heuristics=self.sim_methods,
                            heuristic_params=self.sim_build_params,
                            cascade_config=self.cascade_config,
                            heuristic_class_type=self.prompter_class_type,
                            output_conversion=self.output_conversion,
                            )

if __name__=='__main__':
    generator = HeuristicSpatialPromptGenerator(sim_methods={'points':['uniform_random'], 'scribbles':None, 'bbox':None, 'lassos':None},
                                    semantic_id_dict={'tumour':1, 'background':0},
                                    sim_build_params=None,
                                    cascade_config=None)
    generator.generate_prompt({'gt':torch.ones([128,128,128]), 'image':torch.ones([128,128,128])})