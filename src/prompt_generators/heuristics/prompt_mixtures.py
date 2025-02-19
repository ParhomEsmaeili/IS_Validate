import torch
import numpy as np
import os
from os.path import dirname as up
import sys
from abc import abstractmethod
sys.path.append(up(up(up(up(os.path.abspath(__file__))))))
from src.prompt_generators.heuristics.prompt_bases import PointBase, ScribbleBase, BboxBase 
'''
This file contains the prompt mixture generation classes.

Required arguments:

use_mem: Bool - Denotes whether front-End IM memory is used for handling prompt placement (e.g., is IM used to inform prompt generation)
config_labels_dict: Dict - Denotes the class-label to class-integer code mapping.
device: str - Denotes the device which the prompt generation will be performed on.
heur_fn_dict: Dict - Prompt-type separated dictionary containing the set of heuristics fncs being used:
    
    Each field contains:
    Optional Dict separated by heuristic name containing the most basic abstract function for the given heuristic: 
    {
        heuristic_name_1: heur_fnc_1
        ...
    }
    If prompt-type is not being used then NoneType.

build_params: Optional(Dict) can also be none-type if no params needed (UNLIKELY and BAD) - 
    
    Prompt-type separated dictionary containing the heuristic arguments for each heuristic in the heur_fn_dict.
    Contains the parameters for how the heuristic is to be called wrt methods in the base classes: for handling
    challenges in the prompt generation process.


        Possible heuristic arguments include: 

        Multi-class related variables, e.g.:

        How classes may interact, 
        Parametrisations for picking which subcomponents, etc.

        Multi-component related variables, e.g.:
        
        Quantity of disconnected components to place prompts in,
        Component priority list method, etc. 

        General prompt-level specific parametrisations for valid-prompts, e.g.:

        Probabilistic sampling related params,
        Points budget specific params, 
        Point distribution specific parameters, 
        Scribbles specific params,
        Transforms related params (e.g. breaking scribbles), etc.

        AND/OR

        General prompt-specific parametrisations for non-valid prompts, e.g.:
        
        Probabilistic sampling related params for executing the heuristic samples mistakenly. 
        Prompt generation specific parameters for executing mistaken heuristics.

heuristic_mixtures_args: 
    
    An optional dictionary which contains information about arguments pertaining to mixing prompt generation strategies together
    intra and inter-prompt. E.g.:

    Can include arguments related specifically to fusing prompts together in the actual generation method.

    
    It can also permit for arguments pertaining to how they should be combined together, e.g.:

        Probabilistic measures where we may want to run a bernoulli trial or multinomial trial for deciding which 
        prompt heuristics to use in a mixture method for a given instance (i.e. the flexibility to not use all prompts or prompt methods at ALL times for stress testing). 

--------------------------------------------------------------------------------------------------------------------


General Call Inputs:

All of the following are in pseudo-ui native image space:

data: This is a dictionary which contains the following information:

    gt: Metatensor, or torch tensor containing the ground truth mask.
    img: Metatensor, or torch tensor containing the image.

    prev_output_data: (Optional) Dictionary containing the information from the prior inference calls 
    OR NONETYPE (for init modes).

    Contains the following fields:
        pred: A dict, which contains a subfield "metatensor", with a 1HWD Metatensor, or torch tensor containing the prediction mask from the prior inference call.
        logits: A dict, which contains a subfield "metatensor", with a CHWD Metatensor, or torch tensor for the logits map from the prior inference call.

    'im': (Optional) Dictionary containing the interaction memory from prior iterations of interaction. 
    
generated_prompts: Dict - contains field for each given prompt type:

    'prompt' (e.g. points/scribbles/bboxes):
    
    An empty list (OR NONETYPE for skippable prompts) for the spatial prompts for the current iteration.

generated_prompt_labels: Dict - contains field for each given prompt type;

    prompt_labels (e.g. points_labels/scribbles_labels/bboxes_labels): 
    
    A empty list (OR NONETYPE for empty/skippable prompts) for the labels of the spatial prompts for the current iteration.
    
Outputs:

generated_prompts:
    Amended dict with the additional prompts from the simulation.

generated_prompt_labels: 
    Amended dict with the additional labels from the simulation. 
'''

###################################################################################################################
#Here we place the classes which wrap together the abstracted functions in a manner which handles all the headache 
#about parametrisations, multi-class, multi-component etc. 

class BaseMixture(PointBase, ScribbleBase, BboxBase):

    def filter_empty(self, prompts, prompts_labels, ptype):
        '''
        Function which filters out the empty prompt type lists into NoneTypes.
        '''
        
        if prompts[ptype] == [] or prompts_labels[ptype] == []:
            prompts[ptype] = None 
            prompts_labels[ptype] = None

        return prompts, prompts_labels

class BasicValidOnlyMixture(BaseMixture): 
    def __init__(self, use_mem):
        self.use_mem = use_mem 

class BasicMistakesMixture(BaseMixture):
    pass 

class PseudoMixture(BasicValidOnlyMixture):
    '''
    This function implements the process of iterating through the prompt-gen heuristics fns to simulate prompts.
    Intended for heuristics implementations which do not have a mixture model (inter-prompt or intra-prompt).

    This function also will implement no drop-out of prompts, etc. I.e. no parametrisations are needed for the 
    "wrapper"-like functions for probabilistic trials to determine whether a heuristic is used.
    '''
    def __init__(
            self,
            config_labels_dict: dict,
            device: str,
            heur_fn_dict: dict,
            build_args: dict,
            mixture_args: dict = None,
            use_mem: bool = False,
            ):
        
        super().__init__(
            config_labels_dict=config_labels_dict,
            device=device,
            use_mem=use_mem
        )
        self.configs_labels_dict = config_labels_dict
        self.device = device
        self.heur_fn_dict = heur_fn_dict
        self.heur_build_args = build_args
        self.mixture_args = mixture_args
        self.use_mem = use_mem

        
        #Priority list for the independent fusion strategy, it bins the prompt types into distinct groups of priority
        #each sublist has items equal in priority.
        self.indep_fusion_priority_list = [['bbox'], ['scribbles', 'points']]

    def independent_iterator(self, valid_ptypes, data, generated_prompts, generated_prompt_labels):
        
        #We iterate through the priority list, bbox goes first as it is only capable of grounding a segmentation.
        for sublist in self.indep_fusion_priority_list:

        #We shuffle the valid_ptypes list randomly within the priority list bracket for prompt diversity.
        #(e.g., downstream apps may not necessarily treat scribble points, and standard points the same in their model)

            #We create a random permutation of integers from 0 to len(sublist) - 1.
            indices = torch.randperm(range(len(sublist))).to(int) 
            #shuffled sublist:
            shuffled_sublist = [sublist[i] for i in indices]

            # for heur in ptype_heurs:
            #         #Here we will simulate the prompts for the given prompt type, across classes in task,
            #         #according to the heuristics + build provided. 
            #         # 
            #         # If there are no empty voxels left for prompt placement, we have an error raised, and skip.

            #         try:
            #             generated_prompts, generated_prompt_labels = heur(data, generated_prompts, generated_prompt_labels, self.heuristic_params[ptype][heur])
                    
            #         except:
            #             continue 
    def __call__(self, valid_ptypes, data, generated_prompts, generated_prompt_labels):
        pass 

    
#Mixture registry is for mixture models in which inter and intra-prompt interactions occur during the prompt generation process 
#we also include pseudomixture (this is where there is absolutely no interaction between prompt generation) for tidyness.

mixture_class_registry = {
    'pseudo_mixture': PseudoMixture,
    # 'bbox_constrained_point': bbox_constrained_point
}