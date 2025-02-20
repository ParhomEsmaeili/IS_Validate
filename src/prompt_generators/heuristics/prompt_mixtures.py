from typing import Union 
import warnings 
import itertools
import torch
from monai.data import MetaTensor 
# from monai.transforms import 
import copy 
import re
# import numpy as np
import os
from os.path import dirname as up
import sys
from abc import abstractmethod
sys.path.append(up(up(up(up(os.path.abspath(__file__))))))
from src.prompt_generators.heuristics.prompt_bases import PointBase, ScribbleBase, BboxBase
from src.utils.dict_utils import extractor, dict_path_modif
from src.prompt_generators.heuristics.spatial_utils.component_extraction import get_label_ccp, extract_connected_components

'''
This file contains the prompt mixture generation classes.

Required arguments:

use_mem: Bool - Denotes whether front-End IM memory is used for handling prompt placement (e.g., is IM used to inform prompt generation)
config_labels_dict: Dict - Denotes the class-label to class-integer code mapping.
sim_device: torch.device - Denotes the device which the prompt generation will be performed on.
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
    def __init__(self, sim_device: torch.device, config_labels_dict: dict):
        self.spatial_prompts = ['points', 'scribbles', 'bboxes']
        self.sim_device = sim_device
        self.config_labels_dict = config_labels_dict 

    def filter_empty(self,
            prompts:Union[list[torch.Tensor], None], 
            prompts_lbs: Union[list[torch.Tensor], None] 
            ):
        '''
        Function which performs checks on the prompts and prompts labels to ensure that they match one another, and
        then evaluates how to filter the values according to what the prompts and prompt labels contain (or lack-thereof)
        '''
        if not prompts and not prompts_lbs: #In instances where list is empty, or where it is NoneType both eval as False.
            prompts = None
            prompts_lbs = None 
        elif bool(prompts) ^ bool(prompts_lbs): #Logical XOR, if they do not match then there is an error!
            raise Exception('One of the prompts and prompt labels evaluated as a False, whereas the other was True. Mismatch.')
        elif prompts & prompts_lbs: #If both are true-types that are not empty.
            if len(prompts) != len(prompts_lbs):
                raise Exception('Non-matching quantity of prompts and prompts labels')
            else:
                pass #Made explicit for readability.
        
        return prompts, prompts_lbs 
            
    def filter_empty_dict(self, 
            prompts_dict: dict[str, Union[list[torch.Tensor], None]], 
            prompts_lbs_dict: dict[str, Union[list[torch.Tensor], None]], 
            ):
        '''
        Function which filters out the empty prompt types in output dicts into NoneTypes.

        inputs:

        prompts: A dictionary, separated by prompt types containing the lists of prompt inputs.
        '''

        if not set(prompts_dict) == set(prompts_lbs_dict):
            raise Exception('The prompts dict and the prompts labels dict did not contain the same prompt types')
                
        for prompt_type in prompts_dict.keys():
            p_list = prompts_dict[prompt_type]
            p_lbs_list = prompts_lbs_dict[prompt_type]

            try:
                p_updated, p_lbs_updated = self.filter_empty(p_list, p_lbs_list)
            except:
                Exception(f'Error for prompt type: {prompt_type}')
            prompts_dict.update({prompt_type, p_updated})
            prompts_lbs_dict.update({prompt_type, p_lbs_updated})

        return prompts_dict, prompts_lbs_dict
    
    def spatial_dtype_checker(self, prompt_list: Union[list[torch.Tensor], None]):
        '''
        Function which checks and converts input prompts containing spatial coordinates, to ensure that they are 
        torch.int32 datatypes.  

        inputs: 
        
        prompt_list: An optional list of torch tensors each denoting a set of spatial coordinates (size is not relevant).
        If NoneType then ignores. 

        returns: 

        prompt_list: A list of torch tensors but in the correct datatype (torch.int32) or NoneType for unused instances.
        '''

        if not prompt_list: #If empty list, or NoneType, just pass through.
            pass #Explicit logic provided here.... for debugging.
        else:
            for idx, tensor in enumerate(prompt_list):
                if tensor.is_complex():
                    raise Exception('No complex numbers should be possible.')
                elif tensor.is_floating_point():
                    warnings.warn('The spatial coordinates should be provided as a torch int type.')
                    prompt_list[idx] = tensor.to(dtype=torch.int32)
                else:
                    continue #We pass through if the tensor is a torch int-type.

        return prompt_list     
        
    def reformat_im(self, im: Union[dict, None], ptypes: tuple):
        '''
        Function which creates a deepcopy of the interaction memory prompt information and places them on the 
        selected simulation device picked at config. 
        
        It takes the entire set of interaction states (as any non-desirable memory states should be discarded in the 
        front-end im-handler). For each prompt type it then merges the prompts across the states (with no further operation).

        Prompts extracted are provided in the ordering of the interaction memory.

        Downstream functions may be required for implementing any handling of redundancies (e.g. copies), 
        or forward propagation of prompts (e.g. bbox) etc.

        NOTE: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        NOTE: The underlying assumption for this function is that the each instance of a prompt is provided as a torch
        TENSOR. As such, a set of prompts for a given state must only be a list of torch tensors.
        
        inputs:

        im: An optional dictionary (None when empty, e.g. for initialisation) containing the full list of IM which 
        may be used for prompt generation.

        The only relevant field here is: 'interaction_torch_format', which contains two subfields:

        interactions: A dictionary, separated by prompt type, containing the list of torch tensors with the corresponding prompt info.
        interactions_labels: A dictionary, separated by prompt type, containing the list of torch tensors with the 
        corresponding class-integer code for the prompts.

        p_types: A tuple of strings denoting the prompt types. Must match with the keys of each state in the im.

        returns: 

        None, None (if no IM) OR ordered_prompts, ordered_prompts_labels (each is a dict split by ptype with a list of torch tensors on device). 
        '''
        if not im:
            return None, None
        else:
            im_copy = copy.deepcopy(im)
            
            ordered_prompts_dict = dict.fromkeys(ptypes, [])
            ordered_prompts_lb_dict = dict.fromkeys(ptypes, [])

            if 'Interactive Init' in im_copy.keys():
                init_state = im_copy.pop('Interactive Init') 
                #We first extract the interactive init dict. 
                cartesian_prod = ptypes  #tuple(itertools.product(('Interactive Init',), ptypes))
                
                for ptype in cartesian_prod:
                    p = extractor(init_state, ('interaction_torch_format', 'interactions', ptype))
                    p_lb = extractor(init_state,('interaction_torch_format', 'interactions_labels', ptype))

                
                    assert type(p) == list and type(p_lb) == list 
                    #Put a check for each item in the list
                    assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p])
                    assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p_lb]) 

                    #We then merge into the ordered prompts dicts.
                    ordered_prompts_dict[ptype].extend([p_item.to(device=self.sim_device) for p_item in p])
                    ordered_prompts_lb_dict[ptype].extend([p_item.to(device=self.sim_device) for p_item in p_lb])
                
            #We then extract from the interactive edits, in order. If the remaining state keys are empty, then the 
            # cartesian product will return an empty tuple so that the iterator will just terminate.

            #Orders the remaining keys according to the edit iter num:
            edit_names_list = list(im_copy.keys())
            edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))
            
            cartesian_prod = tuple(itertools.product(edit_names_list, ptypes))
        
           #Extracts using all of the tuple combinations between state names and prompt types. 

            for state_name, ptype in cartesian_prod:
                #Extract
                p = extractor(im_copy, (state_name, 'interaction_torch_format', 'interactions', ptype))
                p_lb = extractor(im_copy, (state_name, 'interaction_torch_format', 'interactions_labels', ptype))
                
                assert type(p) == list and type(p_lb) == list 
                #Put a check for each item in the list
                assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p])
                assert all([type(p_item) == torch.Tensor or type(p_item) == MetaTensor for p_item in p_lb]) 

                #We then merge into the ordered prompts dicts.
                ordered_prompts_dict[ptype].extend([p_item.to(device=self.sim_device) for p_item in p])
                ordered_prompts_lb_dict[ptype].extend([p_item.to(device=self.sim_device) for p_item in p_lb])
            
        
            for ptype in ptypes:
                if len(ordered_prompts_dict[ptype]) != len(ordered_prompts_lb_dict[ptype]):
                    raise Exception(f'Mismatch between quantity of prompts and prompt labels for {ptype}')
                
                if all([i is None for i in ordered_prompts_dict[ptype]]) ^ all([i is None for i in ordered_prompts_lb_dict[ptype]]):
                    raise Exception[f'Mismatch between whether all prompts and all prompt labels in memory are NoneTypes for {ptype}']
                
                elif all([i is None for i in ordered_prompts_dict[ptype]]) ^ all([i is None for i in ordered_prompts_lb_dict[ptype]]):
                    ordered_prompts_dict[ptype] = None 
                    ordered_prompts_lb_dict[ptype] = None 

            del im_copy 

            return ordered_prompts_dict, ordered_prompts_lb_dict 
        

    
class BasicValidOnlyMixture(BaseMixture): 
    def __init__(self, use_mem: bool, **kwargs):
        self.use_mem = use_mem
        super().__init__(**kwargs)

    def rm_intra_prompt_spat_repeats(self, ordered_prompts_dict, ordered_prompts_lb_dict):
        '''
        Lazy implementation of removing intra-prompt type repeats for spatial prompts (which should not really occur
        if interaction memory is being used for prompt generation conditioning, which is the only common instance 
        where this function would be used anyways. 
        
        Does not make any considerations of putting placeholders etc for retaining the structure.

        NOTE: Alternative use case: It can also be called to remove any repeats for prompts (e.g. bboxes) which
        may occur if its sampled at every iteration. ASSUMPTION: Bbox is static.

        NOTE: Assumption is that all prompts are provided as a list of torch tensors. 
        
        NOTE: Second assumption, is that this will only implemented on intra-prompt. Hence, checks will not be imp-
        lemented across prompts. I.e., for verifying that scribble coords do not overlap with point coordinates.
        
        NOTE: Third assumption is that given that this is only for valid prompt placements (according to GT), hence
        why we are capable of filtering in the chronological order for prompt placement. Otherwise, it would cause 
        confusion.

        inputs: 

        ordered_prompts_dict: A dictionary, split by ptype, which has been extracted from interaction memory.
        ordered_prompts_lbs_dict: A dictionary, split by ptype, which has been extracted from interaction memory.

        '''
        
        
        deleter = lambda l, id_to_del : [i for j, i in enumerate(l) if j not in id_to_del]

        for ptype, plist in ordered_prompts_dict.items():

            #Any removal must occur on both the prompts and the prompt labels, so we save indices of repeats.
            removal_indices = {i for i, v1 in enumerate(plist) if not any(torch.all(v1 == v2) for v2 in plist[:i])}

            new_plist = deleter(plist, removal_indices)
            new_plb_list = deleter(ordered_prompts_lb_dict[ptype], removal_indices)

            ordered_prompts_dict[ptype] = new_plist
            ordered_prompts_lb_dict[ptype] = new_plb_list 

        return ordered_prompts_dict, ordered_prompts_lb_dict 

    def init_sampling_regions(self, pred: Union[None, Union[torch.Tensor, MetaTensor]], gt: Union[torch.Tensor, MetaTensor]):
        '''
        Function which initialises the prompt sampling regions. Returns a dict of regions first split by error region
        and gt. Within each of these, is a subdict separated by class containing a tuple of disconnected components 
        in order of component size. 
        
        '''
        #if pred is None, then we just return the gt.
        if not pred:
            if not gt.device == self.sim_device:
                warnings.warn('The gt mask must be placed on the input device')
                gt = gt.to(device=self.sim_device)
            #We then split the gt into a list of components for iteration. 
            for i in 
            get_label_ccp(gt)
            return gt 
        else:
            pass 
        #Place both pred and gt on device and in int32 dtype. 

    def update_error_region():
        pass 
        #This should generalise to instances where prompts are placed outside of the current error region so it
        #can be used for multi-component problems?
        
        
class BasicMistakesMixture(BaseMixture):
    def rm_repeats(self, ordered_prompts_dict, ordered_prompts_lb_dict):

        '''
        Should be an extension of the intraprompt method implemented in the valid only mixtures method. 

        NOTE: 
        Analogous function would need to invert the chronological order in IM.
        And it would also require cross-prompt filtering also, as class labels may flip and change as the error region would 
        not get blocked out according to the prior prompts! 
        
        This was not a concern for the valid only mixture because we would be continuously modifying the error region map!

        ''' 

class PseudoMixture(BasicValidOnlyMixture):
    '''
    This function implements the most basic process of iterating through the prompt-gen heuristics fns to simulate prompts.
    Intended for heuristics implementations which do not have a mixture model (inter-prompt or intra-prompt).

    This function also will implement no drop-out of prompts, etc. I.e. no parametrisations are needed for the 
    "wrapper"-like functions for probabilistic trials to determine whether a heuristic is used.
    '''
    def __init__(
            self,
            config_labels_dict: dict,
            sim_device: torch.device,
            heur_fn_dict: dict,
            build_args: dict,
            mixture_args: dict = None,
            use_mem: bool = False,
            ):
        
        super().__init__(
            use_mem=use_mem,
            config_labels_dict=config_labels_dict,
            sim_device=sim_device,
        )
        self.configs_labels_dict = config_labels_dict
        self.sim_device = sim_device
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