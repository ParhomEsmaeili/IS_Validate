import torch
import numpy as np
import os
import sys
from abc import abstractmethod

'''
This file contains all of the prompt generation classes for each of the heuristics that may be used as part of the 
prompt simulation call. 


For initialisation: 

Possible heuristic arguments include: 

Multi-class related variables (e.g. how they may interact, parametrisations for picking which subcomponents), 
multi-component related variables (e.g. quantity of disconnected components to place prompts in, priority list method?), 

General prompt-specific parametrisations (e.g. quantity of points, distribution specific parameters [e.g. ]
or the range if it is not deterministic/fixed,
budget for the scribbles) for valid prompts.

General prompt-specific parametrisations for non-valid prompts (e.g. probability of executing the heuristic which
samples a mistaken point). 


It should also permit for probabilistic measures potentially where we may want to run a bernoulli trial or
multinomial trial for deciding which prompt heuristics to use in a mixture method for a given instance 
(i.e. the flexibility to not use all prompts or prompt methods at ALL times for stress testing). 


Also intended for performing any checks required on the generated prompts (e.g. removing repeats, checking at least
one prompt is placed)






General Inputs:

All of the following are in pseudo-ui native image space:

data: This is a dictionary which contains the following information:

    gt: Metatensor, or torch tensor containing the ground truth mask.
    img: Metatensor, or torch tensor containing the image.

    prev_output_data: (Optional) Dictionary containing the information from the prior inference calls 
    OR NONETYPE (for init modes).

    Contains the following fields:
        pred: A dict, which contains a subfield "pred_metatensor", with a 1HWD Metatensor, or torch tensor containing the prediction mask from the prior inference call.
        logits: A dict, which contains a subfield "logits_metatensor", with a CHWD Metatensor, or torch tensor for the logits map from the prior inference call.

    
generated_prompts: Contains field for each given prompt type:

    'prompt' (e.g. points/scribbles/bboxes):
    
    An empty list (OR NONETYPE for skippable prompts) for the spatial prompts for the current iteration.

generated_prompt_labels: Contains field for each given prompt type;

    prompt_labels (e.g. points_labels/scribbles_labels/bboxes_labels): 
    
    A empty list (OR NONETYPE for empty/skippable prompts) for the labels of the spatial prompts for the current iteration.
    
Outputs:

generated_prompts:
    Amended dict with the additional prompts from the simulation.

generated_prompt_labels: 
    Amended dict with the additional labels from the simulation. 
'''

#################################################################################################################

#Here we place the abstracted functions which just perform the implementation according to the input args only.

#These funcs should be able to generate errors in instances where there are no longer any remaining voxels for prompt placement
#more specifically relevant to the fine granular interaction mechanisms. 

#NOTE: This is not the same as there being no voxels for correction from the start! 

def uniform_random():
    pass 

###################################################################################################################
#Here we place the classes which wrap together the abstracted functions in a manner which handles all the headache about
#parametrisations, multi-class, multi-component etc. 

class BaseMixture:
    
    def filter_empty(self, prompts, prompts_labels, ptype):
        #We convert empty prompt lists to NoneTypes:
        
        if prompts[ptype] == [] or prompts_labels[ptype] == []:
            prompts[ptype] = None 
            prompts_labels[ptype] = None

        return prompts, prompts_labels

class BasicValidOnlyMixture(BaseMixture):
    pass  

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
            build_args: dict,
            heur_fn_dict: dict,
            device: str):
        
        self.configs_labels_dict = config_labels_dict
        self.heur_build_args = build_args
        self.heur_fn_dict = heur_fn_dict
        self.device = device 

        #Priority list for the independent fusion strategy, it bins the prompt types into distinct groups of priority
        #each sublist has items equal in priority.
        self.indep_fusion_priority_list = [['bbox'], ['scribbles', 'points']]

    def independent_iterator(self, valid_ptypes, data, generated_prompts, generated_prompt_labels):
        
        #We iterate through the priority list, bbox goes first as it is only capable of grounding a segmentation.
        for sublist in self.indep_fusion_priority_list:

        #We shuffle the valid_ptypes list randomly within the priority list bracket for prompt diversity.
        #(e.g., downstream apps may not necessarily treat scribble points, and standard points the same in their model)

            shuffle(valid_ptypes) 

            for heur in ptype_heurs:
                    #Here we will simulate the prompts for the given prompt type, across classes in task,
                    #according to the heuristics + build provided. 
                    # 
                    # If there are no empty voxels left for prompt placement, we have an error raised, and skip.

                    try:
                        generated_prompts, generated_prompt_labels = heur(data, generated_prompts, generated_prompt_labels, self.heuristic_params[ptype][heur])
                    
                    except:
                        continue 
    def __call__(self, valid_ptypes, data, generated_prompts, generated_prompt_labels):
        pass 

        

# class bbox_constrained_point:
#     pass 

base_fncs_registry = {
    'points':{
    'uniform_random': uniform_random
    }
}

#Mixture registry is for mixture models in which inter and intra-prompt interactions occur during the prompt generation process 
#we also include pseudomixture (this is where there is absolutely no interaction between prompt generation) for tidyness.

mixture_class_registry = {
    'pseudo_mixture': PseudoMixture,
    # 'bbox_constrained_point': bbox_constrained_point
}