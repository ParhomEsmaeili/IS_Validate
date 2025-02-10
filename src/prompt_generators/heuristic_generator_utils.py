import torch
import numpy as np
import os
import sys

'''
This file contains all of the prompt generation functions for each of the heuristics that may be used as part of the 
prompt simulation call. 


Possible heuristic arguments include: probability of execution, multi-class related variables (e.g. how they may interact),
multi-component related variables, etc, probability of executing a heuristic which samples a mistaken point.

General Inputs:

All of the following are in pseudo-ui native image space:

data: This is a dictionary which contains the following information:
gt: Metatensor, or torch tensor or numpy array containing the ground truth mask.
img: Metatensor, or torch tensor, or numpy array containing the image.

prev_output_data: (Optional) Dictionary containing the information from the prior inference calls 
OR NONETYPE (for init modes).

Contains the following fields:
    pred: A dict, which contains a subfield "metatensor", with a 1HWD Metatensor, or torch tensor containing the prediction mask from the prior inference call.
    logits: A dict, which contains a subfield "metatensor", with a CHWD Metatensor, or torch tensor for the logits map from the prior inference call.

    
generated_prompts: Contains field for each given prompt type:

    prompt: A list of the currently existing placed spatial prompts for the current iteration.

generated_prompt_labels: Contains field for each given prompt type;

    prompt_labels: A list of the currently existing labels for the placed spatial prompts for the current iteration.
    heuristic_params: A dictionary containing all of the relevant parameters required for the heuristics used.

Outputs:

generated_prompts:
    Amended dict with the additional prompts from the simulation.

generated_prompt_labels: 
    Amended dict with the additional labels from the simulation. 
    
OR

Error raised for instances where simulation must be skipped.
'''

def uniform_random(probability_of_execution):
    probability_of_execution = 1
    pass 

###################################################################################################################

def bbox_constrained_point():
    pass 

base_registry = {
    'points':{
    'uniform_random': uniform_random
    }
}

#Mixture registry is for mixture models in which prompt types interact during the prompt generation process 
mixture_registry = {
    'point_bbox_constrainer': bbox_constrained_point
}