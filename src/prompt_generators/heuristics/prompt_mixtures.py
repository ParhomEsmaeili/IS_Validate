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
from src.prompt_generators.heuristics.spatial_utils.update_binary_mask import update_binary_mask
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

build_params: Optional(Dict) can also be none-type if no params needed. 
    
    NOTE: NONE should only be used for the initial prototype (as there will always be actual arguments required.) 
    
    Prompt-type separated dictionary containing the heuristic arguments for each heuristic in the heur_fn_dict.
    Contains the parameters for how the heuristic is to be called wrt methods in the base classes: for handling
    challenges in the prompt generation process.


        Possible heuristic arguments include: 

        Multi-class related variables, e.g.:

        How classes may interact affect the prompt generation process within a given heuristic.

        Multi-component related variables, e.g.:
        
        Quantity of disconnected components to place prompts in,
        Component priority list method, or Parametrisations for picking which subcomponents, etc. 

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

    def shuffle_list(self, input_list: list, sort_criterion: str = None):
        '''
        Can be used to sort a list of prompts types within a priority list, or to sort a list of heuristics    
        ''' 
        if sort_criterion is None:
            return components 
        elif sort_criterion.title() == 'Random':
            #We create a random permutation of integers from 0 to len(sublist) - 1.
            indices = torch.randperm(range(len(input_list))).to(int) 
            return [input_list[i] for i in indices]
        else:
            raise NotImplementedError('The list sort criterion provided was not supported')
    
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

                    temp_p = copy.deepcopy(ordered_prompts_dict[ptype])
                    temp_p.extend([p_item.to(device=self.sim_device) for p_item in p])
                    
                    temp_plb = copy.deepcopy(ordered_prompts_lb_dict[ptype])
                    temp_plb.extend([p_item.to(device=self.sim_device) for p_item in p_lb])

                    ordered_prompts_dict[ptype] = temp_p
                    ordered_prompts_lb_dict[ptype] = temp_plb 
                
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
                temp_p = copy.deepcopy(ordered_prompts_dict[ptype])
                temp_p.extend([p_item.to(device=self.sim_device) for p_item in p])
                
                temp_plb = copy.deepcopy(ordered_prompts_lb_dict[ptype])
                temp_plb.extend([p_item.to(device=self.sim_device) for p_item in p_lb])

                ordered_prompts_dict[ptype] = temp_p 
                ordered_prompts_lb_dict[ptype] = temp_plb
        
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

    def init_sample_regions_components(
                            self, 
                            pred: Union[None, Union[torch.Tensor, MetaTensor]], 
                            gt: Union[torch.Tensor, MetaTensor],
                            ):
        '''
        Function which initialises the prompt sampling regions on a component basis. Returns a dict of regions 
        for gt split by class and components, and also the same for the error regions, but split according to the
        gt and preds.

        For instances where the error region is not desired (initialisation), the error region subdict is None.
    

        Motivation: Class specific challenges wrt ordering the components. E.g. if we had a huge organ and a small
        organ both oversegmented into background, the strategy of only using false negatives would bias to the huge 
        organ. Similarly, there may be variation in component sizes within a class.

        Resolution- 
        
        For the error region: 
        
        Split into class, then split each false negative region by the predicted class (this would be a
        false positive of that class). Then into a list of components (torch tensors) within that predicted class.
        
        Then, apply a sorting algorithm during the iterative looping for each heuristic.

        For the gt: Split into class, then into a list of components (torch tensors) within the class. 



        inputs: 
        
        pred - Optional (torch.Tensor or MetaTensor) discretised pred (not one-hot encoded)
        gt - torch.Tensor or MetaTensor which is discrete and not one-hot encoded.

        returns:
        
        sampling_regions_dict: A dictionary, contains two fields:

        'gt': Contains a dictionary which is structured as: dict[class_str, list of torch.Tensors]
        
        'error_regions': Contains a nested dictionary which is structured as:

        dict[class_str denoting the GT of error voxel, dict[class_str denoting the prediction, list of torch.Tensors]]
        
        This dictionary chunks up the false negative error region in the following manner: It splits it by the 
        ground truth class of the error voxels, it then splits it according to the predicted class, it then splits it
        into a list of components from the error voxels extracted from this intersection.  

        NOTE: For any instances where a gt class is empty, or the error region is empty for the given config path, then 
        it will be an empty list []. 
        '''
        raise NotImplementedError('Needs to be checked for debugging again, add a logical check e.g. that gts sum to quantity of voxels')
        sampling_regions_dict = dict() 
        
        #First we implement the gt extraction since this is always required. 

        #Place gt on device and in int32 dtype. 

        if not gt.device == self.sim_device:
            warnings.warn('The gt mask must be placed on the sim device')
            gt = gt.to(dtype=torch.int32, device=self.sim_device)
        
        sampling_regions_dict['gt'] = dict.fromkeys(self.config_labels_dict.keys(), [])

        #We then split the gt, by class and into a list of components for each class. 
        for label, value  in self.config_labels_dict.items():
            #We split gt by label. 
            gt_temp = torch.where(gt == value, 1, 0).to(dtype=torch.int32, device=self.sim_device)
            components_list, _ = get_label_ccp(gt_temp) 
            if components_list == []:
                warnings.warn(f'Class {label} was empty in gt.')
            tmp = copy.deepcopy(sampling_regions_dict['gt'][label]) 
            tmp.extend(components_list)
            sampling_regions_dict['gt'][label] = tmp 

        if all([masks == [] for label, masks in sampling_regions_dict['gt'].items() if label.title() != 'Background']):
            raise Exception('All of the foreground classes in the ground truth cannot be empty.')
            
        #if pred is None, then we just return the gts.
        
        if pred is None:
            sampling_regions_dict['error_regions'] = None 
            return sampling_regions_dict
        else: 
            #Place pred on device and in int32 dtype. 

            if not pred.device == self.sim_device:
                warnings.warn('The pred mask must be placed on the sim device')
                pred = pred.to(dtype=torch.int32, device=self.sim_device)
            
            #Find the false negative error region. 
            error_map_bool = torch.where(pred != gt, 1, 0).to(dtype=torch.int32, device=self.sim_device)
            
            #Create the error regions dict: 
            err_regions_dict = dict() 

            for l1, v1 in self.config_labels_dict.items():
                #Splitting into classes according to gt (i.e. voxels where an error occured and where the gt class exists)                
                temp_gt = torch.where(gt == v1, 1, 0).to(dtype=torch.int32, device=self.sim_device)
                split_by_gt = error_map_bool * temp_gt 
                
                err_regions_dict[l1]  = dict() 

                for l2, v2 in {key:val for key, val in self.config_labels_dict.items() if key != l1}.items():    
                    #Splitting into classes according to predicted class that do not belong to.
                    
                    #NOTE: We use where key != l1 because error would not occur if the pred was the same as the gt label.

                    #split by pred gives us the map where the gt = v1 but pred = v2
                    temp_pred = torch.where(pred == v2, 1, 0).to(dtype=torch.int32, device=self.sim_device)
                    split_by_pred = split_by_gt * temp_pred 

                    #Splitting into the list of components
                    components_list, _ = get_label_ccp(split_by_pred)

                    err_regions_dict[l1][l2] = components_list 

            sampling_regions_dict['error_regions'] = err_regions_dict 
        
            return sampling_regions_dict
    
    
    def init_sample_regions_no_components(
                            self, 
                            pred: Union[None, Union[torch.Tensor, MetaTensor]], 
                            gt: Union[torch.Tensor, MetaTensor],
                            ):
        '''
        Very basic function which initialises the prompt sampling regions without a per-component basis. 
        Returns a dict of  regions for gt split by class, and also the same for the error regions.

        It is not intended for sophisticated behaviours, e.g. where one must consider the per-component basis, or the
        scale of a class. 

        For instances where the error region is not desired (initialisation), the error region subdict is None.
        Otherwise, it just treats the entirety of the error region as a single mask for each class it belongs to.

        inputs: 
        
        pred - Optional (torch.Tensor or MetaTensor) discretised pred (not one-hot encoded)
        gt - torch.Tensor or MetaTensor which is discrete and not one-hot encoded.

        returns:
        
        sampling_regions_dict: A dictionary, contains two fields:

        'gt': Contains a dictionary which is structured as: dict[class_str, torch.Tensor]
        
        'error_regions': Contains a nested dictionary which is structured as:

        dict[class_str denoting the GT of error voxel, torch.Tensor]]
        
        This dictionary chunks up the false negative error region in the following manner: It splits it by the 
        ground truth class of the error voxels, it then splits it according to the predicted class. 

        NOTE: For any instances where a gt class is empty, or the error region is empty for the given config path, then 
        it will be Nonetype!  
        '''
        sampling_regions_dict = dict() 
        
        #First we implement the gt extraction since this is always required. 

        #Place gt on device and in int32 dtype. 

        if not gt.device == self.sim_device:
            warnings.warn('The gt mask must be placed on the sim device')
            gt = gt.to(dtype=torch.int32, device=self.sim_device)
        
        sampling_regions_dict['gt'] = dict.fromkeys(self.config_labels_dict.keys(), None)

        #We then split the gt, by class for each class. 
        for label, value  in self.config_labels_dict.items():
            #We split gt by label. 
            gt_temp = torch.where(gt == value, 1, 0).to(dtype=torch.int32, device=self.sim_device) 
            if gt_temp.sum() == 0:
                warnings.warn(f'Class {label} was empty in gt.')
                sampling_regions_dict['gt'][label] = None 
            else:
                sampling_regions_dict['gt'][label] = gt_temp 

        if all([masks is None for label, masks in sampling_regions_dict['gt'].items() if label.title() != 'Background']):
            raise Exception('All of the foreground classes in the ground truth cannot be empty.')
        
        if torch.all(torch.stack(list(sampling_regions_dict['gt'].values())).sum(dim=0) == torch.ones_like(gt)):
            print('GT all GOOD!')
        else:
            raise Exception('GT maps did not merge to a tensor of ones')
        #if pred is None, then we just return the gts.
        
        if pred is None:
            sampling_regions_dict['error_regions'] = None 
            return sampling_regions_dict
        else: 
            #Place pred on device and in int32 dtype. 

            if not pred.device == self.sim_device:
                warnings.warn('The pred mask must be placed on the sim device')
                pred = pred.to(dtype=torch.int32, device=self.sim_device)
            
            #Find the false negative error region. 
            error_map_bool = torch.where(pred != gt, 1, 0).to(dtype=torch.int32, device=self.sim_device)
            
            #Create the error regions dict: 
            err_regions_dict = dict() 

            for l1, v1 in self.config_labels_dict.items():
                #Splitting into classes according to gt (i.e. voxels where an error occured and where the gt class exists)                
                temp_gt = torch.where(gt == v1, 1, 0).to(dtype=torch.int32, device=self.sim_device)
                #We recompute because calling from gt above, NoneType would break.
                split_by_gt = error_map_bool * temp_gt 
                
                if split_by_gt.sum() == 0:
                    err_regions_dict[l1] = None 
                else: 
                    err_regions_dict[l1] = split_by_gt 


            sampling_regions_dict['error_regions'] = err_regions_dict 

            if torch.all(torch.stack(list(sampling_regions_dict['error_regions'].values())).sum(dim=0) <= 1):
                print('No error regions are overlapping!')
            else:
                raise Exception('There are overlapping error regions')
            return sampling_regions_dict
        
    def update_error_region(self, region_mask, refinement_prompts: list[torch.Tensor]):
        '''
        This is a function which updates a region mask according to a set of refinement prompts. It will convert any 
        coords with 1s at the coordinates to zeroes.

        This can be incorporated into an approach for multi-component handling, multi-class handling and also for 
        handling different prompt types.

        In particular, it can handle points and scribbles. It assumes that the prompts are provided as a list of
        tensors with shape N x N_dim (N = 1 for points, and N_s.p for scribbles). It will raise an exception if
        the spatial dimensions are larger than N_dims. 

        inputs:

        region_mask: A binary mask with N_dim spatial dims denoting an error region with values of 1, everywhere else is zero. 
        refinement_prompts: A list of prompts N x N_dim for updating the region mask.

        '''
        if not all([prompt.shape[1] == region_mask.dim() for prompt in refinement_prompts]):
            raise Exception('The number of spatial dimensions of all input prompts must match the number of spatial dimensions of the mask')
        
        if refinement_prompts == []:
            warnings.warn('The prompts are empty')
            return region_mask
        
        if region_mask.device != self.sim_device:
            region_mask.to(device=self.sim_device)

        
        for coords in refinement_prompts:
            region_mask = update_binary_mask(region_mask, coords)    
        return region_mask
    
    def sort_components(self, components: Union[list[Union[torch.Tensor, MetaTensor]], None], sort_criterion: str = None):
        '''
        This is a function that will sort a list of mask components according to a sort criterion. 

        inputs: 

        components - list: A list of components, can optionally also be an empty list.
        sort_criterion - str: A sorting criterion.
        '''

        if sort_criterion is None:
            return components 
        elif sort_criterion.title() == 'Random':
            #We create a random permutation of integers from 0 to len(sublist) - 1.
            indices = torch.randperm(range(len(components))).to(int) 
            return [components[i] for i in indices]
        elif sort_criterion.title() == 'Component Sum':
            #This is on the basis of the sum of the torch tensors (i.e. the size of the component)
            _, indices = torch.sort(torch.stack(components).sum(list(range(1, len(torch.stack(components).shape)))))
            return [components[i] for i in indices]


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

class PrototypePseudoMixture(BasicValidOnlyMixture):
    '''
    This function implements the most basic process of iterating through the prompt-gen heuristics fns to 
    simulate prompts.

    Intended as a prototype for heuristics implementations which do not have a mixture model (inter-prompt or 
    intra-prompt) or any heuristics arguments at all. Any of the relevant heuristics arguments will be temporarily 
    hard-coded for this prototype:
    
    E.g., number of points, number of boxes, number of scribbles etc per class and per heuristic, toggling off the
    components handling. Toggling off the use-mem for determining whether im is used for prompt generation. No 
    toggling of the drop-out of prompts. No toggling of the order in which prompts are generated (just does it randomly)
    
    The only intra/inter-prompt interactions is the removal of coordinates for sampling refinement prompts.

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
        self.prompt_priority_list = [['bbox'], ['scribbles', 'points']]
    
    def toggle_components(self):
        pass 

    def iterator(self, valid_ptypes,, generated_prompts, generated_prompt_labels):
        
        #We iterate through a priority list, bbox goes first as it is only capable of grounding a segmentation.

        #This priority list is provided because we are sampling without replacement (mostly just relevant for the
        # refinement prompts). 
         
        for sublist in self.prompt_priority_list:


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
            #         # If there are no empty voxels left for prompt placement, we have an empty list, just skip.

            
    def __call__(self, valid_ptypes, data, tracked_prompts, tracked_prompts_lbs):
        '''
        Function which calls on the methods for implementing the prompt generation process. 

        inputs: 

        valid_types: A list of prompt types which have a heuristic provided (i.e. for which simulation could occur)
        
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
            
            logits:
                1) "paths": List of paths to the prediction file (Not Relevant)
                And two potentially relevant subfields
                2) "metatensor" A Metatensor or torch tensor (CHW(D)) containing the previous segmentation in the native image domain (no pre-processing applied other than re-orientation in RAS) 
                3) "meta_dict" A dict containing (at least) the affine matrix for the image, containing native image domain relevant knowledge.
        
        im: Optional (or NoneType) dictionary containing the interaction memory from the prior interaction states.      


        tracked_prompts: A dictionary, split by prompt type, which contains the initialised dict according to the 
        set of valid prompt types (i.e. those for which a prompt can be simulated). Contains empty lists for valid,
        and NoneTypes for invalid prompt types.

        tracked_prompts_lbs: Same as tracked_prompts, except for the prompts' labels. Split by {prompt_type}_labels.
        '''

        if self.use_mem:
            #Extract the interaction memory.
            im = data['im']

            if not im:
                raise Exception('If using interaction memory, then it requires interaction memory available! Received nonetype')
            
            raise NotImplementedError('Not implemented the handling of interaction memory') 
            #TODO: If using memory then add the functionality for extracting prior prompts and reformatting etc.
        else:
            if data['prev_output_data'] is None:
                print('We have no prior output data, please check that this is an initialisation!')
                pred = None 
            else:
                print('We have prior output data, please check that this is an editing iteration')
                pred = data['prev_output_data']['pred']['metatensor'][0, :]
                pred = pred.to(dtype=torch.int32, device=self.sim_device)

                if not isinstance(pred, torch.Tensor) or not isinstance(gt, MetaTensor):
                    raise TypeError('The pred needs to be a torch tensor or a Monai MetaTensor')            

            gt = data['gt'][0, :]
            gt = gt.to(dtype=torch.int32, device=self.sim_device)

            if not isinstance(gt, torch.Tensor) or not isinstance(gt, MetaTensor):
                raise TypeError('The gt needs to be a torch tensor or a Monai MetaTensor')
            
            #Extracts a dict with fields 'gt' and 'error_regions'. Both class separated dicts.
            sampling_regions_dict = self.init_sample_regions_no_components(pred, gt)
		    # No need to adjust the tracked prompts or tracked prompts labels variable!
            
        toggle_components = False 
        toggle_class = False 
        toggle_priority = False 

#Mixture registry is for classes which wrap together the prompt generation process with complex inter/intra-prompt relationships.
#  
#we also include basic_pseudomixture (this is where there is absolutely no interaction between prompt generation) 
#and no complexities about how components or classes are handled at all.

mixture_class_registry = {
    'prototype_pseudo_mixture': PrototypePseudoMixture,
    # 'bbox_constrained_point': bbox_constrained_point
}