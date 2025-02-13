
'''
This file is intended to wrap the heuristic-based generator utilities into a callable class. Primary intention is to
maintain a tidy codebase, and to ensure that a mixture of prompt generation can be organised.

It permits calling functions for mixing together the prompts after the fact, and for heuristics which implement mixtures
of the prompts during generation. 

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
        
        '''
        Heuristic Prompt Generation builder. This class makes use of the simulation arguments, and constructs a simulation
        class which can be called for generation of prompts and labels in the list[torch] format. 

        Inputs:

        config_labels_dict: A dictionary mapping the class labels to the class integer codes. 

        sim_methods: Dict - Simulation methods used for generating each prompt (currently: [points, scribbles, bbox]). 
        
        Must always provide for all prompt types. A nested dictionary first separated by the prompt type:

        Within each prompt type, another dictionary contains key:value pairs containing the list of prompting strategies being used for prompt simulation. 
        This dictionary has the flexibility to permit combinations/lists of strategies. 

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
        Hence they will be generated with no consideration of prompting intra and inter-dependencies.
        '''
        
        self.config_labels_dict = config_labels_dict
        self.heuristics = heuristics 
        self.heuristic_params = heuristic_params
        self.heuristic_mixtures = heuristic_mixtures 

        self.heuristic_fn_dict = self.initialise_heuristics()
        #Priority list for the independent fusion strategy, it bins the prompt types into distinct groups of priority
        #each sublist has items equal in priority.
        self.indep_fusion_priority_list = [['bbox'], ['scribbles', 'points']]

    def initialise_heuristics(self):
        '''
        This function will initialise the heuristics to be used, such that they can be called for prompt generation.
        
        Returns:
        
        heur_fns_dict: The dictionary containing the function calls necessary for prompt generation.

        For instances with no heuristic mixtures, this consists of prompt-type separated nested dict with subdicts 
        separated by the heuristic type.

        TODO: For instances with heuristic mixtures, the structure is currently undefined.
        '''

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
            raise NotImplementedError('Implement the code for initialising prompt mixture methods.')
            '''
            Info: 

            self.heuristic_mixtures: A twice nested dict denoting a strategy for mixing prompt simulation across intra
            and inter-prompting strategies. The mixture args are a dictionary for each corresponding pairing of cross-interactions.

            This prompt mixture arg will control whether/how prompt-methods will interact/condition one another during the 
            simulation. 

                Has structure: dict('inter_prompt':dict[tuple(prompt cross-interaction combinations), mixture_args/None], 
                                    'intra_prompt': dict[prompt_type_str : dict[tuple(prompt_strategy cross-interaction combinations), mixture_args/None])
                
                Tuple can provide an immutable set of combinations

                NOTE: Downstream use will likely necessitate the use of set logic to verify combinations as the tuple 
                is immutable. Verification likely will entail the following: Generate set of potential combinations from
                prompt_types (or methods), cross-reference with the corresponding item by converting key into set.
            '''

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
        for ptype in [j for js in self.indep_fusion_priority_list for j in js]:
            generated_prompts[ptype] = []

            generated_prompt_labels[f'{ptype}_labels'] = []
        
        #We then populate the prompts dictionary:

        for p_type_sublist in self.indep_fusion_priority_list:
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

        '''
        Inputs:

        data: A dictionary containing the following fields: 

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
        
                

        Returns: 

        Both outputs are in the list[torch] format denoted in: `<https://github.com/ParhomEsmaeili/IS_Validate/blob/main/src/data/interaction_state_construct.py>` 

        prompts_torch_format: dict - A dictionary, separated by the prompt-type, which contains the prompt spatial information
        for the selected prompt types in the prompt generation config.  

        prompts_labels_torch_format: dict - A dictionary, separated by the prompt type, which contains the prompt
        labels for the corresponding prompts (or NoneTypes for the empty prompts!). 
        '''
        if not self.heuristic_mixtures:
            prompts_torch_format, prompts_labels_torch_format = self.independent_iterate_heuristics(data)
        
        elif self.heuristic_mixtures:

            raise NotImplementedError('The heuristic mixture strategy has not yet been implemented')
        return prompts_torch_format, prompts_labels_torch_format 