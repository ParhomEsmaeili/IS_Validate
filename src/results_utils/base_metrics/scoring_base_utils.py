import torch
import numpy as np 
from monai.data import MetaTensor 
from typing import Union 
from surface_distance import compute_surface_distances 
from surface_distance.metrics import compute_surface_dice_at_tolerance
import copy 
import warnings 
import os 
import sys
import gc 
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from version_handling import monai_version 


class BaseScoringWrapper:
    '''
    Assumption is that the class integer codes are in order, and starting from 0, with increments of 1 per class. 
    Class code = 0 refers to background semantic class ALWAYS.

    NOTE: Metrics configs can be used directly for a base metric, or it can be used as part of a metric config which 
    uses a base metric (e.g., a human-intent metric). However, the base component within the "wrapped" metric 
    must have the configs structured in the same capacity as a base metric would. 

    Input: 
    
    Metrics_configs: A nested dictionary separated by base metric type fields:

    'base_metric_type_1': metric_type_1_args
    'base_metric_type_2': metric_type_2_args 

    '''
    def __init__(
            self,
            calc_device: torch.device, 
            metrics_configs:dict, 
            config_labels_dict:dict):

        self.calc_device = calc_device
        self.metrics_configs = metrics_configs  
        self.config_labels_dict = config_labels_dict 
        self.metric_classes = {}

        self.metric_initialiser_map = {
            'Dice':self.init_dice,
            'NSD': self.init_NSD,
            }
            # 'Error Rate':self.init_error_rate}
        
        for metric_type in self.metrics_configs.keys():
            self.metric_initialiser_map[metric_type]()

    def init_dice(self): 

        self.metric_classes['Dice'] = Dice(
            calc_device=self.calc_device,
            ignore_empty=self.metrics_configs['Dice']['ignore_empty'],
            include_background=self.metrics_configs['Dice']['include_background_metric'],
            include_per_class_scores=self.metrics_configs['Dice']['include_per_class_scores'],
            config_labels_dict=self.config_labels_dict
            )
    def init_NSD(self):

        self.metric_classes['NSD'] = NSD(
            calc_device=torch.device('cpu'), #NSD is not a GPU metric, it is a CPU metric, so we just use CPU for now.
            #TEMPORARY. 
            ignore_empty=self.metrics_configs['NSD']['ignore_empty'],
            include_background=self.metrics_configs['NSD']['include_background_metric'],
            include_per_class_scores=self.metrics_configs['NSD']['include_per_class_scores'],
            tolerance_mms=self.metrics_configs['NSD']['tolerance_mms'], #this is the tolerance used in the metric, not the spacing for the image. 
            config_labels_dict=self.config_labels_dict       
            )
        
    def __call__(self,  
                image_masks: tuple, 
                pred: Union[torch.Tensor, MetaTensor], 
                gt: Union[torch.Tensor, MetaTensor]):
        '''
        Inputs:

        image_masks: A dictionary containing two fields:

        'cross_class_mask': A torch tensor or monai metatensor denoting the mask being used.
        'per_class_masks': A dictionary separated by classes denoting each mask for each class. 
        '''

        if not isinstance(image_masks, tuple):
            raise Exception["Score generation failed as the weightmap masks were not in a tuple format"]
        
        if not isinstance(image_masks[0], torch.Tensor) and not isinstance(image_masks[0], MetaTensor):
            raise TypeError("Score generation failed because the cross-class mask was not a torch.Tensor or MetaTensor")
        
        if not isinstance(image_masks[1], dict):
            raise TypeError("Score generation failed because the per-class masks were not presented in a class-separated dict.")

        if not isinstance(pred, torch.Tensor) and not isinstance(pred, MetaTensor):
            raise TypeError("Score generation failed as the prediction was not a torch.Tensor or a monai MetaTensor")
        
        if not isinstance(gt, torch.Tensor) and not isinstance(gt, MetaTensor):
            raise TypeError("Score generation failed as the gt was not a torch.Tensor or a monai MetaTensor")

        
        return {metric_type:self.metric_classes[metric_type](image_masks, pred, gt) for metric_type in self.metrics_configs.keys()} 
    

class Dice:

    def __init__(self,
        calc_device: torch.device,
        ignore_empty: bool,
        include_background: bool,
        include_per_class_scores: bool,
        config_labels_dict: dict[str, int]
    ):
        self.calc_device = calc_device
        self.ignore_empty = ignore_empty
        self.include_background = include_background
        self.include_per_class_scores = include_per_class_scores
        self.config_labels_dict = config_labels_dict

    def dice_score(self,
            cross_class_mask: Union[torch.Tensor, MetaTensor], 
            per_class_masks: dict[str, Union[torch.Tensor, MetaTensor]],
            pred: Union[torch.Tensor, MetaTensor], 
            gt: Union[torch.Tensor, MetaTensor]):
        
        '''
        Dice-score computer, it implements two modes for computing dice score, on a per-class basis and by summing across all of the classes prior to 
        computing the overall Dice Score. Latter is not necessarily viable for large class-imbalances.

        inputs: 
        
        cross_class_mask - torch tensor or monai metatensor for voxel-weighting the multiclass dice score computation. 
        This cannot be a zeroes tensor. 

        Assumed to be HW(D). 

        per_class_masks - class-separated dictionary of torch tensors or monai metatensors for voxel-weighting the dice score computations.
        These all cannot be a zeroes tensor. 

        Assumed to each be HW(D)

        pred, gt - torch tensors or monai metatensors for the pred and gt.

        Assumed to each be HW(D) (not one-hot encoded)

        '''

        #For multi-class (or binary class where self-include background is TRUE)

        #We weight/mask this according to the class-separated image_masks (BINARISED).  
        if len(self.config_labels_dict) > 2:
            raise NotImplementedError('Optimal multi-class dice score computation not yet implemented for class-imbalance \n'
            'handling.')
        cross_class = self.dice_score_multiclass(pred, gt, cross_class_mask) 

        if self.include_per_class_scores:
            per_class_scores = dict() 
            
            pred =pred.to(device=self.calc_device, dtype=torch.uint8)
            gt = gt.to(device=self.calc_device, dtype=torch.uint8)

            for class_label, class_integer_code in self.config_labels_dict.items():

                if not self.include_background:
                    if class_label.title() == 'Background':
                        continue 
        
                # class_sep_pred = torch.where(pred == class_integer_code, 1, 0)
                # class_sep_gt = torch.where(gt == class_integer_code, 1, 0)


                # per_class_scores[class_label] = self.dice_score_per_class(
                #     torch.where(pred == class_integer_code, 1, 0).to(dtype=torch.uint8),
                #     torch.where(gt == class_integer_code, 1, 0).to(dtype=torch.uint8),
                #     #TODO: check whether this mask is going to get stuck in cuda memory.
                #     per_class_masks[class_label].to(device=self.calc_device, dtype=torch.uint8)
                # ).to(device='cpu', dtype=torch.float64) 

                per_class_scores[class_label] = self.dice_score_per_class(
                    pred == class_integer_code,
                    gt == class_integer_code,
                    image_mask=per_class_masks[class_label].to(device=self.calc_device, dtype=torch.bool)
                )
                assert type(per_class_scores[class_label]) == torch.Tensor and per_class_scores[class_label].size() == torch.Size([1]), 'Generation of dice score failed because the score for a specific class was not a torch tensor of size 1'
        
                pred = pred.to(device='cpu', dtype=torch.bool) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
                gt = gt.to(device='cpu', dtype=torch.bool) #Just to be careful, we don't need this anymore, so we can delete it to save memory.        
                # gc.collect() #Functionally is doing nothing other than slowing us down.
                torch.cuda.empty_cache() #Just to be careful, we don't need this anymore, so we can delete it to save memory.

            assert type(cross_class) == torch.Tensor and cross_class.size() == torch.Size([1]), 'Generation of dice score failed because the cross-class score was not a torch tensor of size 1'
        
        else:
            per_class_scores = None 

        return (cross_class, per_class_scores)
        
    
        
    def dice_score_per_class(self, class_sep_pred, class_sep_gt, image_mask):
        '''
        Class_sep_pred: torch tensor, bool type. 
        Class_sep_gt: torch tensor, bool type.
        image_mask: torch tensor, bool type. 
        '''
        # y_o = torch.sum(torch.where(class_sep_gt > 0, 1, 0) * image_mask)
        # y_hat_o = torch.sum(torch.where(class_sep_pred > 0, 1, 0) * image_mask)
        y_o = torch.sum(class_sep_gt & image_mask)
        y_hat_o = torch.sum(class_sep_pred & image_mask)
        torch.cuda.empty_cache() # Clearing cache to minimise VRAM accumulation.
        denom = y_o + y_hat_o
        # weighted_pred = class_sep_pred * image_mask 
        # weighted_gt = class_sep_gt * image_mask 

        # intersection = torch.sum(torch.masked_select(image_mask, class_sep_pred * class_sep_gt > 0))
        intersection = torch.sum(
            torch.masked_select(image_mask, class_sep_pred & class_sep_gt)
        )
        denom = denom.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
        intersection = intersection.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
        
        torch.cuda.empty_cache() # Clearing cache to minimise VRAM accumulation.        
        if y_o > 0: #in this case we can always calculate a dice score because denom will not be zero.
            return torch.tensor([(2 * intersection)/(denom)])
        
        if self.ignore_empty:
            #If y_o was not  >0 (i.e, the gt region was under consideration was not empty) then if ignore_empty, we just return a nan value
            return torch.tensor([float("nan")])
        if denom <=0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor([1.0])
        #else:
        return torch.tensor([0.0])
    
    def dice_score_multiclass(self, pred, gt, image_mask):

        '''
        image mask here is the cross-class image mask which is binarised.
        '''
        pred = pred.to(device=self.calc_device, dtype=torch.uint8) 
        #Ensuring that the pred is on the correct device and we can also assume pred is a discrete mask with relatively small
        #values so uint8 is sufficient. 
        gt = gt.to(device=self.calc_device, dtype=torch.uint8)
        image_mask = image_mask.to(device=self.calc_device, dtype=torch.uint8)
        #For now we assume that the image mask is a binarised mask (e.g. a binary window, if even that) so it is also uint8.
        if not self.include_background:
            #then background not included
            y_o = torch.sum(torch.where(gt > 0, 1, 0) * image_mask)
            y_hat_o = torch.sum(torch.where(pred > 0, 1, 0) * image_mask)
        else:
            #when background is included, it includes all voxels..
            y_o = torch.sum(torch.ones_like(gt) * image_mask)
            y_hat_o = torch.sum(torch.where(pred > 0, 1, 0) * image_mask)

        #Free the memory used by the calculation
        torch.cuda.empty_cache() 

        if y_o > 0:
            
            intersection = 0
            
            
            for class_label, class_code in self.config_labels_dict.items():
                if not self.include_background:
                    if class_label.title() == 'Background':
                        continue 
                

                # pred_channel = torch.where(pred == class_code, 1, 0) 
                
                # gt_channel = torch.where(gt == class_code, 1, 0) 


                #NOTE: For weightmaps which are not a tensor of ones (and non binarised maps) the voxel values
                # must have already been weighted by the corresponding values in the image mask, so that when we sum
                # it already contains the weight.

                # intersection += torch.sum(torch.masked_select(image_mask, pred_channel * gt_channel > 0 )) 
                              
                intersection += torch.sum(
                    torch.masked_select(
                        image_mask, 
                        (pred == class_code) & (gt == class_code))
                ) #More memory efficient way of computing the intersection without generating intermediate full tensors. 
                torch.cuda.empty_cache() #Emptying the cache to minimise VRAM usage.

            pred = pred.to(device='cpu', dtype=torch.bool) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
            gt = gt.to(device='cpu', dtype=torch.bool) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
            image_mask = image_mask.to(device='cpu', dtype=torch.bool) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
            
            intersection = intersection.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
            y_o = y_o.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
            y_hat_o = y_hat_o.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
            #Now we can compute the dice score.
            # gc.collect() Doesn't do anything here. No loose variables to clean up.
            torch.cuda.empty_cache()
            return torch.tensor([(2.0 * intersection) / (y_o + y_hat_o)])
        
        
        if self.ignore_empty:
            #If we ignore empty, then just return a nan value if there was nothing in the "gt".
            return torch.tensor([float("nan")])
        denom = y_o + y_hat_o
        
        denom = denom.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
        y_o = y_o.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
        y_hat_o = y_hat_o.to(device='cpu', dtype=torch.float64) #Just to be careful, we don't need this anymore, so we can delete it to save memory.
        # gc.collect() #Not needed as we have no variables we are cleaning up.
        torch.cuda.empty_cache()

        if denom <= 0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor([1.0])
        return torch.tensor([0.0])

    def __call__(self, 
                image_masks, 
                pred, 
                gt):
                                                                #cross_class, per_class
        (cross_class_score, per_class_scores) = self.dice_score(image_masks[0], image_masks[1], pred, gt)

        return {"cross_class_scores":cross_class_score, "per_class_scores":per_class_scores}    
    

class NSD: 
    #Normalised Surface Dice Metric calculation:

    # Borrows heavily from the Deepmind implementation, but wrapped for our purposes to accept upstream initialisation parameters, needs to extend to multi-class semantic
    #seg. 
    
    def __init__(self,
        calc_device: torch.device,
        #calc_device: torch.device, #Not currently going to do much as we will hardcode cpu as the underlying function which has been
        # wrapped cannot be executed on gpu. But kept for consistency with the BaseScoringWrapper interface.
        ignore_empty: bool,
        include_background: bool,
        include_per_class_scores: bool,
        config_labels_dict: dict[str, int],
        tolerance_mms: list[float] 
        #temporarily we will assume the same tolerance is to be used across all classes, which will be reasonable in binary semantic seg. 
        #should ideally be overridden on a target by target basis, or some reasonable measure across an instance-by-instance basis.

        #Tolerance mms is a list of tolerance values, which can be length 1 if we have a single tolerance
        # or length > 1 if we want to generate scores for a set of different tolerances. 
    ):  
        self.calc_device = calc_device
        self.ignore_empty = ignore_empty
        self.include_background = include_background
        self.include_per_class_scores = include_per_class_scores
        self.config_labels_dict = config_labels_dict
        self.tolerance_mms = tolerance_mms 
 
    def NSD_calc(
            self, 
            pred: Union[torch.Tensor, MetaTensor], 
            gt: MetaTensor):
        
        '''
        Calculates surface distance on a class by class basis (unlike a simple overlap counting metric like dice it cannot reasonably be computed in a multi-class manner
        without performing an averaging over semantic classes)

        pred, gt - torch tensors or monai metatensors for the pred and gt.

        Assumed to each be HW(D) (not one-hot encoded)
        ''' 
        
        per_class_scores = dict() 
            
        for class_label, class_integer_code in self.config_labels_dict.items():
            if class_label.title() == 'Background': 
                #Background will never have an actual meaningful interpretation, it is a "stuff" class with no structure associated, 
                # and is not particularly relevant, so typically we will not include it in the NSD computation.
                #However, if the user has specified to include it, then we will compute the NSD for it, but it is not recommended.
                if not self.include_background:
                    continue
                else:
                    warnings.warn('Background class is not that meaningful a class for surface dice computation, recommended to not include it')

            # class_sep_pred = torch.where(pred == class_integer_code, 1, 0)
            # class_sep_gt = torch.where(gt == class_integer_code, 1, 0)

            # per_class_scores[class_label] = self.NSD_binary(torch.where(pred == class_integer_code, 1, 0), torch.where(gt == class_integer_code, 1, 0))
            
            per_class_scores[class_label] = self.NSD_binary(pred == class_integer_code, gt == class_integer_code)
            #Minimising the memory usage by not storing the class_sep_pred and class_sep_gt tensors, as they are not needed after the NSD_binary computation.
        
        #Compute the cross-class score, dummy method for now is to just average over the classes.        
        if self.ignore_empty:
            #if ignoring empty "gt" then we filter the nan values in the cross-class computation, as it would correspond to instances where there was no, 
            # "gt"/reference annotation and/or no prediction either. 

            # filtered_scores = [i for i in per_class_scores.values() if not torch.isnan(i)]            
            # if len(filtered_scores) == 0:        
            #     cross_class = torch.tensor([float("nan")]) #There was no class for which there was a non nan-type metric, so just return nan.
            # else:
                # cross_class = torch.nanmean(torch.cat(list(per_class_scores.values()))) #We just compute the nan-mean.
                # cross_class = torch.tensor([float(cross_class)])
                # if torch.isnan(cross_class):
                #     raise Exception('Nan generated by cross-class scores on the separate NSD scores for each class, despite applying a filter on the nans s.t. all classes being empty is correctly handled')

            #Instead of prior method, we will just use nanmean's capability to return a 
            #nan if there is no non-nan value. 
            
            #First we create a 2d array of scores across the tolerance values and the classes.
            scores_array = torch.stack(list(per_class_scores.values())) #This will be a tensor of size (num_classes, num_tolerances)
            #Then we will compute torch.nanmean across the class axis.
            cross_class = torch.nanmean(scores_array, dim=0) #This will be a tensor of size (num_tolerances)
            assert cross_class.size() == torch.Size([len(self.tolerance_mms)]), 'Generation of NSD has failed because the cross-class score was not a torch tensor of size equal to the number of tolerances being evaluated'
        
        else:
            #if not ignore empty then we will always have a numeric value for the per-class scores, so we can just compute the mean across all of the classes.
            # cross_class = torch.mean(torch.cat(list(per_class_scores.values())))
            cross_class = torch.mean(torch.stack(list(per_class_scores.values())), dim=0) 
            #This will be a tensor of size (num_tolerances)
            # cross_class = torch.tensor([float(cross_class)]) #just in case it isn't a floating point value (it should be)
        
        if not self.include_per_class_scores:
            per_class_scores = None 
            warnings.warn('Strongly recommended to include per-class scores for NSD as it is not a simple counting metric, each target has its own -shape- it is recommended to examine per-class scores. \n')

        if len(self.tolerance_mms) == 1:
            assert type(cross_class) == torch.Tensor and cross_class.size() == torch.Size([1]), 'Generation of NSD has failed because the cross-class score was not a torch tensor of size 1'
        else:
            assert type(cross_class) == torch.Tensor and cross_class.size() == torch.Size([len(self.tolerance_mms)]), 'Generation of NSD has failed because the cross-class score was not a torch tensor of size equal to the number of tolerances being evaluated'

        return (cross_class, per_class_scores)
        
    def NSD_binary(self, class_sep_pred, class_sep_gt):
        '''
        Function which performs the calculation of the Normalised Surface Dice on a pair of binary masks (could be for any pairing, semantic or even downstream... instance)
        
        
        return: A torch tensor of size len(self.tolerance_mms) containing the NSD scores for each of the
        tolerance values being used in the metric computation.
        '''
        #Just to be careful because of weirdness about MetaTensor behaviours across MONAI versions we abuse deepcopy... temporary hacky method... so we don't break
        #any underlying obj... 

        if isinstance(class_sep_pred, MetaTensor):
            if monai_version == '1.4.0':
                #Version for other venvs
                pred_np = class_sep_pred.array #copy.deepcopy(class_sep_pred.array) 
            elif monai_version == '0.9.0':
                #Version for the segvol_venvs:
                pred_np = class_sep_pred.data.numpy() #copy.deepcopy(class_sep_pred.data.numpy())
            else:
                raise Exception('Unknown MONAI version.')             
        elif isinstance(class_sep_pred, torch.Tensor):
            if monai_version == '1.4.0':
                #Version for other venvs:
                pred_np = class_sep_pred.numpy() #copy.deepcopy(class_sep_pred.numpy())
            elif monai_version == '0.9.0':            
                #Version for the segvol_venv:
                pred_np = class_sep_pred.numpy() #copy.deepcopy(class_sep_pred.numpy())
            else:
                raise Exception('Unknown MONAI version')
        if isinstance(class_sep_gt, MetaTensor):
            if monai_version == '1.4.0':
                #Version for other venvs:
                gt_np = class_sep_gt.array  #copy.deepcopy(class_sep_gt.array)
            elif monai_version == '0.9.0':
                #Version for the segvol_venv:
                gt_np = class_sep_gt.data.numpy() #copy.deepcopy(class_sep_gt.data.numpy())
            else:
                raise Exception('Unknown MONAI version') 
        else:
            raise Exception('Need the image spacing, therefore the underlying image spacing must be available from the reference annotation.')

        #We put some pre-checks here to handle the cases where ground truth is empty such that we can raise some warnings, even though the imported function will
        #be able to handle the case where the ground truth is empty but the prediction is not empty, or both are empty. Also, we want the flexibility for not ignoring empty
        #ground truths, so we can return a perfect score of 1.0 if both are empty, or a score of 0.0 if the prediction is not empty but the ground truth is empty.
        #Although, it is highly recommended to use ignore_empty=True, as otherwise it can lead to a lot of confusion in the results.
        if gt_np.sum() == 0: 
            if self.ignore_empty:
                #If we ignore empty, then just return a nan value if there was nothing in the "gt". 
                warnings.warn('There is a "ground truth" which has no foreground for a given target, please consider checking whether this is intended or reasonable.')
                return torch.tensor([float("nan")] * len(self.tolerance_mms)) #If we have multiple tolerances, we return a nan value for each of the tolerances.
            # Unlike a Dice Overlap score, there is no simple interpretation to computing the surface dice in the absence of a reference annotation, but we can still
            # put something reasonable here if the prediction is also empty, then we can return a perfect score of 1.0. Highly non-recommended to not use 
            # ignore_empty=False, for this reason, as it will lead to a lot of confusion in the results.
            else:
                if pred_np.sum() == 0:
                    warnings.warn('Ignore_empty is False, but there is a "ground truth" which has no foreground for a given target, but the prediction is also empty, so returning a perfect score of 1.0.')
                    return torch.tensor([1.0] * len(self.tolerance_mms)) #If we have multiple tolerances, we return a perfect score for each of the tolerances.
                else:
                    warnings.warn('Ignore_empty is False, but there is a "ground truth" which has no foreground for a given target, but the prediction is not empty, so returning a score of 0.0.')
                    return torch.tensor([0.0] * len(self.tolerance_mms)) #If we have multiple tolerances, we return a score of 0.0 for each of the tolerances.



        val_dom_affine = copy.deepcopy(class_sep_gt.meta['affine']) 
        #extracting the affine (in the domain of the validation, i.e. after orientation to RAS) to obtain the image spacing    
        if val_dom_affine is not None:
            if val_dom_affine.shape[0] != 4: 
                raise NotImplementedError('We do not yet provide handling for non 3D-annotation domains')
            dim = val_dom_affine.shape[0] - 1
            _m_key = (slice(-1), slice(-1))
            im_spacing = np.linalg.norm(val_dom_affine[_m_key] @ np.eye(dim), axis=0)
        #spacing_mm is the voxel-level image-spacing itself.
        # surface_distances = compute_surface_distances(mask_gt=gt_np.astype(bool), mask_pred=pred_np.astype(bool), spacing_mm=im_spacing)
        #tolerance_mm is the tolerance in mm for the surface dice.

        if len(self.tolerance_mms) == 1:
            surface_dice = compute_surface_dice_at_tolerance(
                surface_distances=compute_surface_distances(mask_gt=gt_np.astype(bool), mask_pred=pred_np.astype(bool), spacing_mm=im_spacing), 
                tolerance_mm=self.tolerance_mms[0])
            
            # del surface_distances #just to be careful, we don't need this anymore, so we can delete it to save memory.

            if np.isnan(surface_dice): #in this case the pred and gt must both be empty.
                if gt_np.sum() != 0 or pred_np.sum() != 0:
                    raise Exception('Somehow a nan generated for surface dice despite not having empty masks for both pred and gt')
                return torch.tensor([float("nan")])
            else:
                return torch.tensor([float(surface_dice)]) #just in case it isn't a floating point value (it should be) 
        else:
            surface_dice_scores = []
            surface_distances=compute_surface_distances(mask_gt=gt_np.astype(bool), mask_pred=pred_np.astype(bool), spacing_mm=im_spacing) 
            for tolerance_mm_val in self.tolerance_mms:
                
                surface_dice = compute_surface_dice_at_tolerance(
                    surface_distances=surface_distances,tolerance_mm=tolerance_mm_val)
                
                #We unwrap the prior function to store the surface distances in memory such that we can run multiple tolerances.
            
                if np.isnan(surface_dice): #in this case the pred and gt must both be empty.
                    if gt_np.sum() != 0 or pred_np.sum() != 0:
                        raise Exception('Somehow a nan generated for surface dice despite not having empty masks for both pred and gt')
                    else:
                        surface_dice_scores.append(float("nan"))
                else:
                    # return torch.tensor([float(surface_dice)]) #just in case it isn't a floating point value (it should be) 
                    surface_dice_scores.append(float(surface_dice))

            #Now we convert the list of floats into a torch tensor.
            surface_dice_scores = torch.tensor(surface_dice_scores)
            return surface_dice_scores

    def __call__(self, 
                image_masks, 
                pred, 
                gt):
        
                                                                #cross_class, per_class
        (cross_class_score, per_class_scores) = self.NSD_calc(pred, gt)
        #Here we will process the cross class and per class scores if we have multiple
        #tolerances.

        if len(self.tolerance_mms) > 1:
            cross_class_score = {idx: torch.tensor([cross_class_score[idx]]) for idx in range(len(self.tolerance_mms))}
            per_class_scores = {class_label: {idx: torch.tensor([per_class_scores[class_label][idx]]) for idx in range(len(self.tolerance_mms))} for class_label in per_class_scores.keys()}

        return {"cross_class_scores":cross_class_score, "per_class_scores":per_class_scores}


if __name__ == '__main__':
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
    from monai.data import MetaTensor 
    import torch 
    input_dict = {'label':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTs/BraTS2021_00002.nii.gz'}
    load_transf = [LoadImaged(keys=['label'], image_only=False), EnsureChannelFirstd(keys=['label'])]
    loaded_im_dict = Compose(load_transf)(input_dict)

    im_metatensor = loaded_im_dict['label'] #MetaTensor(x=torch.from_numpy(loaded_im_dict['label']), meta={'affine':torch.from_numpy(loaded_im_dict['label_meta_dict']['affine'])})
    scoring_class = BaseScoringWrapper(
        metrics_configs=
        {
            'NSD': {
                'ignore_empty':True,
                'include_background_metric':False, 
                'include_per_class_scores':True,                
                'tolerance_mm':1
            }
        },
        config_labels_dict={'background':0, 'tumour':1}
    )  
    # NSD_calc_class = NSD(ignore_empty=True, include_background=False, include_per_class_scores=True, config_labels_dict={'background':0, 'tumour':1}, tolerance_mm=1) 
    #output = NSD_calc_class(None, im_metatensor[0], im_metatensor[0])
    output = scoring_class((torch.zeros([100,100,100]), dict()), im_metatensor[0], im_metatensor[0])
    print('fin')