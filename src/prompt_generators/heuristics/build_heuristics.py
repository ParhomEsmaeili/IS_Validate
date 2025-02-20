import os
from os.path import dirname as up
import sys
sys.path.append(up(up(up(up(os.path.abspath(__file__))))))

from typing import Union 
import torch
from src.prompt_generators.heuristics.prompt_mixtures import mixture_class_registry
from src.prompt_generators.heuristics.heuristic_prompt_utils.heuristic_utils_registry import base_registry

class BuildHeuristic:

    def __init__(self,
                sim_device: torch.device,
                use_mem:bool,
                config_labels_dict:dict,
                heuristics:dict,
                heuristic_params: Union[dict, None],
                heuristic_mixtures: Union[dict, None] 
                ):
        
        '''
        Heuristic Prompt Generation builder. This class makes use of the simulation arguments, and constructs a simulation
        class which can be called for generation of prompts and labels in the list[torch] format. 

        Inputs:

        sim_device: torch.device - The device which the computations will be implemented on for gpu (or cpu) processing.

        use_mem: A bool - Denotes whether the interaction memory dictionary will be used so that stored memory
        is being retained/used to filter the error regions for prompt generation.

        config_labels_dict: A dictionary mapping the class labels to the class integer codes. 

        heuristics: Dict - Simulation methods used for generating each prompt (currently: [points, scribbles, bbox]). 
        
        Must always provide for all prompt types. A nested dictionary first separated by the prompt type:

        Within each prompt type, another dictionary contains key:value pairs containing the list of prompting strategies being used for prompt simulation. 
        This dictionary has the flexibility to permit combinations/lists of strategies. 

        NOTE: Instances where a specific prompt type is not being used (or was skipped over in simulation), will 
        require that the value be a NoneType arg instead. The prompts generated will also be a NoneType argument 
        for the given prompts also!

        


        
        (OPTIONAL) heuristic_params: A twice nested dictionary, for each prompt strategy within a prompt type it 
        contains a dictionary of build arguments for each corresponding strategy implemented. 

        Almost the same structure as prompt_methods, as they must correspond together!

        NOTE: Must also contain information about handling the iterative loop (e.g. sorting components strategy, 
        otherwise assumed to be fully according to the default.)

        NOTE: Can also be a Nonetype, in which case there is no information required for building the prompters
        (e.g., parameter free/variable free [very unlikely], or if there is no prompt of that type being used)

        NOTE: Child classes where the prompt generation method may be the same across prompt types
        e.g. prompt generation D.L models, will necessitate that the prompt mixture argument denotes the strategy for
        cross-interaction (e.g., conditioning the model with a prompt prior, or passing all prompt-type requests concurrently 
        through). NOTE: However, prompt-type & method specific args can still be passed through in the build_params.
        





        (OPTIONAL)heuristic_mixture: A twice nested dict denoting a strategy for mixing prompt simulation across intra
        and inter-prompting strategies. The mixture args are a dictionary for each corresponding pairing of cross-interactions.

        This heuristic mixture arg will control whether/how prompt-methods will interact/condition one another during the 
        simulation. 

            Has structure: dict('inter_prompt':dict[tuple(prompt cross-interaction combinations), mixture_args/None], 
                                'intra_prompt': dict[prompt_type_str : dict[tuple(prompt_strategy cross-interaction combinations), mixture_args/None])
            
            Tuple can provide an immutable set of combinations

            NOTE: Downstream use will likely necessitate the use of set logic to verify combinations as the tuple 
            is immutable. Verification likely will entail the following: Generate set of potential combinations from
            prompt_types (or strategy), cross-reference with the corresponding dict item by converting key into set.


        Can optionally can be Nonetype (i.e., fully  independently assumed prompt generation) simulation of prompts (inter and intra-prompt strategy). 
        Hence they will be generated with no consideration of prompting intra and inter-dependencies (outside of the sampling region being filled)
        '''
        self.sim_device = sim_device 
        self.use_mem = use_mem 
        self.config_labels_dict = config_labels_dict
        self.heuristics = heuristics 
        self.heuristic_params = heuristic_params
        self.heuristic_mixtures = heuristic_mixtures 

        self.heuristic_caller = self.initialise_heuristics()

        #Checking that at least one prompt is being used:

        if not any(list(self.heuristics.values())):
            raise ValueError('There must be at least one prompt type which is not a NoneType for simulation.')

    def at_least_one_prompt(self, 
                            generated_prompts:dict, 
                            generated_prompts_labels:dict):
        '''
        This function will check whether at least one prompt type has a prompt, otherwise it will raise an exception.

        Requirement: Any empty list should have been converted to a NoneType. We will check for this! 
        '''

        #Checking for empty lists to raise an exception about code elsewhere.

        if [] in generated_prompts.values() or [] in generated_prompts_labels.values():
            raise ValueError('Any empty prompt lists or prompt labels lists must be replaced with a NoneType for the value')
        
        #Checking that at least one valid prompt type has something populating it

        # I.e., if it went through the exception handling for every valid prompt type then it would be only NoneTypes throughout.
        
        #NOTE: This is insufficient outside of just refinement prompt simulation, as bbox may be present even without
        #refinement at each iter! Needs consistent refinement! 

        if not all(generated_prompts.values()) or not all(generated_prompts_labels.values()):
            raise ValueError('There must be at least one prompt!')
        
    def initialise_heuristics(self):
        '''
        This function will initialise the heuristics to be used, such that they can be called for prompt generation.
        
        Returns:
        
        heur_caller: An initialised class which can be used to generate the prompts: 

        For instances with no heuristic mixtures, this consists of prompt-type separated nested dict with subdicts 
        separated by the heuristic type.

        TODO: For instances with heuristic mixtures, the structure is currently undefined.
        '''

        if not self.heuristic_mixtures:
            #Here we will initialise the heuristics for fully-independent prompt generation methods (pseudo-mixture).
            heur_fn_dict = dict()

            for prompt_type, heuristics in self.heuristics.items():
                prompt_heur_fns = dict() 
                
                if heuristics: #If heuristics is not a NoneType/I.e. if config exists for a prompt type
                    for heuristic in heuristics: 
                        prompt_heur_fns[heuristic] = base_registry[prompt_type][heuristic]
                else:
                    prompt_heur_fns = None 
                
                heur_fn_dict[prompt_type] = prompt_heur_fns

            return mixture_class_registry['pseudo_mixture'](
                config_labels_dict=self.config_labels_dict,
                sim_device=self.sim_device,
                use_mem=self.use_mem,
                build_args=self.heuristic_params,
                mixture_args=self.heuristic_mixtures,
                heur_fn_dict=heur_fn_dict,                                
                )
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

    def extract_prompts(self, data: dict):
        '''
        This function initialises then populates the generated prompts and labels using the heuristic caller.

        Also calls on the output checker, to ensure that there is at least one prompt simulated.

        Inputs:

        data: This is a dictionary which contains the following information:
            gt: Metatensor, or torch tensor containing the ground truth mask.
            img: Metatensor, or torch tensor containing the image.

            prev_output_data: (Optional) Dictionary containing the information from the prior inference calls 
            OR NONETYPE (for init modes).

            Contains the following fields:
                pred: A dict, which contains a subfield "metatensor", with a 1HWD Metatensor, or torch tensor 
                containing the discretised prediction mask from the prior inference call.
                logits: A dict, which contains a subfield "metatensor", with a CHWD Metatensor, or torch tensor 
                containing the logits map from the prior inference call.

        '''

        #We initialise the dictionary containing the prompts and labels across the prompt types:
        generated_prompts = dict()
        generated_prompt_labels = dict()
    
        for ptype, heurs in self.heuristics.items():
            
            valid_ptypes = []
            #Populate the list of valid (used/configured) prompt types according to the heuristics dict. 
            
            #checks whether the heur methods are nonetypes by default.
            if heurs is not None:
                valid_ptypes.append(ptype)
                
                #Initialising the lists for the valid ptypes 
                generated_prompts[ptype] = []
                generated_prompt_labels[f'{ptype}_labels'] = []

            elif heurs is None:
                #If nonetype, then the generated prompts will be a nonetype for that prompt-type.
                generated_prompts[ptype] = None
                generated_prompt_labels[f'{ptype}_labels'] = None

        #We populate the prompts:
        generated_prompts, generated_prompt_labels = self.heuristic_caller(valid_ptypes, data, generated_prompts, generated_prompt_labels)
                
        #We check that the output was generated correctly/with a valid output.
        self.at_least_one_prompt(generated_prompts, generated_prompt_labels)

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
                2) "metatensor" A Metatensor or torch tensor containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
            
            logits:
                1) "paths": List of paths to the prediction file (Not Relevant)
                And two potentially relevant subfields
                2) "metatensor" A Metatensor or torch tensor containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
        
        im: Optional dictionary containing the interaction memory from the prior interaction states.      

        Returns: 

        Both outputs are in the list[torch] format denoted in: `<https://github.com/IS_Validate/blob/main/src/data/interaction_state_construct.py>` 

        prompts_torch_format: dict - A dictionary, separated by the prompt-type, which contains the prompt spatial information
        for the selected prompt types in the prompt generation config.  

        prompts_labels_torch_format: dict - A dictionary, separated by the prompt type, which contains the prompt
        labels for the corresponding prompts (or NoneTypes for the empty prompts!). 
        '''
        if not self.heuristic_mixtures:
            prompts_torch_format, prompts_labels_torch_format = self.extract_prompts(data)
        
        elif self.heuristic_mixtures:
            raise NotImplementedError('The heuristic mixture strategy has not yet been implemented')
    
        return prompts_torch_format, prompts_labels_torch_format 