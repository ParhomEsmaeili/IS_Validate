import os
from os.path import dirname as up
import sys
sys.path.append(up(up(up(up(os.path.abspath(__file__))))))
import torch
from typing import Union
import sys 
sys.path.append(up(up(up(os.path.abspath(__file__)))))
from src.prompt_generators.heuristics.prompt_mixtures import mixture_class_registry
from src.prompt_generators.heuristics.heuristic_prompt_utils.heuristic_utils_registry import base_registry

class BuildHeuristic:

    def __init__(self,
                sim_device: torch.device,
                use_mem:bool,
                semantic_id_dict:dict,
                heuristics:dict,
                heuristic_params: dict,
                cascade_config: Union[dict, None],
                heuristic_class_type: str,
                output_conversion: dict = None
                ):
        
        '''
        Heuristic Prompt Generation builder. This class makes use of the simulation arguments, and constructs a simulation
        class which can be called for generation of prompts and labels in the list[torch] format. 

        Inputs:

        sim_device: torch.device - The device which the computations will be implemented on for gpu (or cpu) processing.

        use_mem: A bool - Denotes whether the interaction memory dictionary will be used so that stored memory
        is being retained/used to filter the error regions for prompt generation.

        semantic_id_dict: A dictionary mapping the class labels to the class integer codes. 

        
        
        (REQUIRED) heuristics: Dict - Simulation methods used for generating each prompt (currently: [points, scribbles, bbox]). 
        
        Must always provide for all prompt types. A dictionary first separated by the prompt type:

        Within each prompt type, containing the list of prompting strategies being used for prompt simulation. 
        This dictionary allows the flexibility to permit combinations/lists of strategies. 

        NOTE: Instances where a specific prompt type is never being used, will require that the value be a NoneType 
        arg instead. The prompts generated will also be a NoneType argument for the given prompts also! 

        NOTE: If any prompt drop-out methods are used, it is possible for an output to be a NoneType even if the 
        input arg for the prompt type is not.

        (REQUIRED) heuristic_params: A twice-nested dictionary of build arguments per prompt strategy.
        Same structure as heuristics. For valid prompt types this must be non-None.

        (OPTIONAL) cascade_config: The cascade toggling configuration dict (e.g. class_level).
        May optionally be None for default cascade behaviour.

        It is expected to have the same structure as the cascade's toggling dict:

            {
                'class_level': ... | None,
                'inter_prompt_level': ... | None,
                'intra_prompt_level': ... | None,
            }

        The intra_heur_level key is appended internally from heuristic_params.
        '''

        self.sim_device = sim_device 
        self.use_mem = use_mem 
        self.semantic_id_dict = semantic_id_dict
        self.heuristics = heuristics
        self.heuristic_params = heuristic_params
        self.cascade_config = cascade_config or {}
        self.heuristic_class_type = heuristic_class_type
        self.output_conversion = output_conversion if output_conversion else None
        self.free_form_prompts = ['points', 'scribbles']
        self.partition_prompts = ['bboxes', 'lassos'] 

        self.heuristic_caller = self.initialise_heuristics()

        #Checking that at least one prompt is being used:

        if not any(list(self.heuristics.values())):
            raise ValueError('There must be at least one prompt type which is not a NoneType for simulation.')

    def at_least_one_prompt(self, 
                            generated_prompts:dict, 
                            generated_prompts_labels:dict, 
                            data: dict):
        '''
        This function will check whether at least one prompt type has a prompt, otherwise it will raise an exception.

        Requirement: Any empty list should have been converted to a NoneType. We will check for this! 

        inputs: 
        
        generated_prompts: A prompt type separated dict containing the list of prompts.
        generated_prompts_labels: A prompt type separated dict containing the list of prompts corresponding labels.
        data: The data dictionary which was passed into the call operation, contains info about the prev_output_data.
        '''

        #Checking for empty lists to raise an exception about code elsewhere.

        if [] in generated_prompts.values() or [] in generated_prompts_labels.values():
            raise ValueError('Any empty prompt lists or prompt labels lists must be replaced with a NoneType for the value')

        #Free-form prompts (points, scribbles) can always generate from any voxel, so failure means something is wrong.
        #Partition prompts (bboxes, lassos) can legitimately fail — no raise needed.

        free_form_configured = any(
            self.heuristics.get(ptype) is not None
            for ptype in self.free_form_prompts
        )

        if free_form_configured:
            if not any(generated_prompts.values()):
                raise ValueError('At least one prompt must be generated if free-form prompts are configured! If there is a candidate region then' \
                'there should always be prompts generated if free form prompts are configured! ')
         

        
        
    def initialise_heuristics(self):
        '''
        This function will initialise the heuristic call to be used, such that it can be called for prompt 
        generation. It takes the abstract heuristics from the registry, and places them into a nested dictionary 
        split by prompt type and then split by heuristic type.
        
        Returns:
        
        heur_caller: An initialised class which can be used to generate the prompts.
        '''

        intra_heur = self.heuristic_params
        if not isinstance(intra_heur, dict):
            raise ValueError(
                'heuristic_params must be a dict (per-ptype heuristic build arguments)'
            )

        for prompt_type, heuristics in self.heuristics.items():
            if heuristics:
                if not intra_heur.get(prompt_type):
                    raise ValueError(
                        f'heuristic_params missing or empty for configured '
                        f'prompt type "{prompt_type}"'
                    )
                for heuristic in heuristics:
                    if not intra_heur[prompt_type].get(heuristic):
                        raise ValueError(
                            f'heuristic_params["{prompt_type}"] missing or empty '
                            f'for heuristic "{heuristic}"'
                        )

        heur_fn_dict = dict()

        for prompt_type, heuristics in self.heuristics.items():
            prompt_heur_fns = dict() 
            
            if heuristics:
                for heuristic in heuristics: 
                    prompt_heur_fns[heuristic] = base_registry[prompt_type][heuristic]
            else:
                prompt_heur_fns = None 
            
            heur_fn_dict[prompt_type] = prompt_heur_fns

        full_cascade = dict(self.cascade_config)
        full_cascade['intra_heur_level'] = intra_heur

        return mixture_class_registry[self.heuristic_class_type](
            args={
                'semantic_id_dict': self.semantic_id_dict,
                'sim_device': self.sim_device,
                'use_mem': self.use_mem,
                'cascade_config': full_cascade,
                'heur_fn_dict': heur_fn_dict,
                'output_conversion': self.output_conversion,
            })

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
                probs: A dict, which contains a subfield "metatensor", with a CHWD Metatensor, or torch tensor 
                containing the probs map from the prior inference call.

        '''

        generated_prompts, generated_prompt_labels = self.heuristic_caller(data)
                
        #We check that the output was generated correctly/with a valid output.
        self.at_least_one_prompt(generated_prompts, generated_prompt_labels, data)

        return generated_prompts, generated_prompt_labels

    def __call__(self, data):

        '''
        Inputs:

        data: A dictionary containing the following fields: 

        image: Torch tensor OR Metatensor containing the image in the native image domain (no pre-processing applied other than re-orientation in RAS)
        gt: Torch tensor OR Metatensor containing the ground truth map in RAS orientation, but otherwise in the native image domain (no pre-processing other than RAS re-orientation).
        
        prev_output_data: (NOTE: OPTIONAL, is NONE otherwise) output dictionary from the inference call which has been post-processed 
        in the pseudo-ui front-end.
       
        Two relevant fields for prompt generation contained are the: 
            pred: A dictionary containing 3 subfields:
                1) "path": Path to the prediction file (Not Relevant)
                And two relevant subfields
                2) "metatensor" A Metatensor or torch tensor (1HW(D)) containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
            
            probs:
                1) "paths": List of paths to the prediction file (Not Relevant)
                And two potentially relevant subfields
                2) "metatensor" A Metatensor or torch tensor (CHW(D)) containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
        
        im: Optional dictionary containing the interaction memory from the prior interaction states.      

        Returns: 

        Both outputs are in the list[torch] format denoted in: src/data/interaction_state_construct.py 

        prompts_torch_format: dict - A dictionary, separated by the prompt-type, which contains the prompt spatial information
        for the selected prompt types in the prompt generation config.  

        prompts_labels_torch_format: dict - A dictionary, separated by the prompt type, which contains the prompt
        labels for the corresponding prompts (or NoneTypes for the empty prompts!). 
        '''
        prompts_torch_format, prompts_labels_torch_format = self.extract_prompts(data)
    
        return prompts_torch_format, prompts_labels_torch_format 