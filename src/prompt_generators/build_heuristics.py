
'''
This file is intended to wrap the heuristic-based generator utilities into a callable class. Primary intention is to
maintain a tidy codebase, and to ensure that a mixture of prompt generation can be organised.

E.g., it should permit for mixing together the prompts after the fact, it should permit for mixtures of the prompts during
generation, and it should also permit for probabilistic measures potentially where we may want to run a bernoulli trial or
multinomial trial for deciding which prompt heuristics to use for a given instance (i.e. the flexibility to not use 
all prompt methods at ALL times. Also intended for performing any checks required on the generated prompts (e.g. removing repeats?))
'''
import os
import sys
from src.prompt_generators.heuristic_generator_utils import base_registry, mixture_registry
from random import shuffle 

class BuildHeuristic:
    def __init__(self, 
                config_labels_dict,
                heuristics,
                heuristic_params,
                heuristic_mixtures 
                ):
        self.config_labels_dict = config_labels_dict
        self.heuristics = heuristics 
        self.heuristic_params = heuristic_params
        self.heuristic_mixtures = heuristic_mixtures 

        self.initialise_heuristics()
        self.prompt_priority_list = [['bbox'], ['scribbles', 'points']]

    def initialise_heuristics(self):


        if not self.heuristic_mixtures:
            #Here we will initialise the heuristics for fully-independent prompt generation methods.
            heur_fn_dict = dict()

            for prompt_type, heuristics in self.heuristics.items():
                prompt_heur_fns = dict() 
                
                if heuristics: #If heuristic is not a NoneType/I.e. if config exists
                    for heuristic in heuristics: 
                        prompt_heur_fns[heuristic] = base_registry[prompt_type][heuristic]
                else:
                    prompt_heur_fns = None 
                
                heur_fn_dict[prompt_type] = prompt_heur_fns
            return heur_fn_dict 
        else:
            pass 

            #TODO: Implement the code for prompt mixture methods.

    def independent_iterate_heuristics(self, data: dict):
        '''
        This function implements the process of iterating through the prompt-gen heuristics to simulate prompts, while 
        checking for redundancies between prompts which are assumed to be VALID. 
        
        The exception here is potentially for instances where we are simulating user mistakes. Intended for 
        heuristics implementations which do not have a mixture model (inter-prompt or intra-prompt).

        '''

        #We initialise the dictionary containing the prompts across the prompt types:
        generated_prompts = dict()
        generated_prompt_labels = dict()
        for ptype in [j for js in self.prompt_priority_list for j in js]:
            generated_prompts[ptype] = []

            generated_prompt_labels[f'{ptype}_labels'] = []
        
        #We then populate the prompts dictionary:

        for p_type_sublist in self.prompt_priority_list:
            valid_ptypes = [i for i in p_type_sublist if i in self.heuristics]
            #We extract the prompt types that are being used in the experiment.
            
            #We shuffle the valid_ptypes list randomly within the priority list bracket for prompt diversity.
            shuffle(valid_ptypes)
            for ptype in valid_ptypes:
            
                ptype_heurs = self.heuristics[ptype]
                #We shuffle the heuristics fns randomly for prompting diversity also. 

                for heur in ptype_heurs:
                    #Here we will simulate the prompts for the given prompt type, across classes,
                    # according to the heuristics provided. If there are no empty voxels left for prompt placement,
                    #we have an error raised, and skip.

                    try:
                        generated_prompts, generated_prompt_labels = heur(data, generated_prompts, generated_prompt_labels, self.heuristic_params[ptype][heur])
                    
                    except:
                        continue 
        return generated_prompts, generated_prompt_labels

    def __call__(self, data):
        
        if not self.heuristic_mixtures:
            prompts_torch_format, prompt_labels_torch_format = self.independent_iterate_heuristics(data)
            
        return prompts_torch_format, prompt_labels_torch_format 