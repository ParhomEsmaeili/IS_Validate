import torch 
from monai.data import MetaTensor 
from typing import Union 

class BaseScoringWrapper:
    '''
    Assumption is that the class integer codes are in order, and starting from 0, with increments of 1 per class. 
    Class code = 0 refers to background ALWAYS.

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
            metrics_configs:dict, 
            config_labels_dict:dict):

        self.metrics_configs = metrics_configs  
        self.config_labels_dict = config_labels_dict 
        self.metric_classes = {}

        self.metric_initialiser_map = {
            'Dice':self.init_dice, 
            'Error Rate':self.init_error_rate}
        
        for metric_type in self.metrics_configs.keys():
            self.metric_initialiser_map[metric_type]

    def init_dice(self): 

        self.metric_classes['Dice'] = DiceScore(
                ignore_empty=self.metrics_configs['Dice']['ignore_empty'],
                include_background=self.metrics_configs['Dice']['include_background'],
                include_per_class_scores=self.metrics_configs['Dice']['include_per_class_scores'],
                config_labels_dict=self.config_labels_dict
            )
    
    def init_error_rate(self):
   
        self.metric_classes['Error Rate'] = ErrorRate(
            ignore_empty=self.metrics_configs['Error Rate']['ignore_empty'],
            include_background=self.metrics_configs['Error Rate']['include_background'],
            include_per_class_scores=self.metrics_configs['Error Rate']['include_per_class_scores'],
            config_labels_dict=self.config_labels_dict
        ) 
    def __call__(self,  
                image_masks: dict, 
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
            raise TypeError("Score generation failed because the cross-class weightmap was not a torch.Tensor or MetaTensor")
        
        if not isinstance(image_masks[1], dict):
            raise TypeError("Score generation failed because the per-class weightmaps were not presented in a class-separated dict.")

        if not isinstance(pred, torch.Tensor) and not isinstance(pred, MetaTensor):
            raise TypeError("Score generation failed as the prediction was not a torch.Tensor or a monai MetaTensor")
        
        if not isinstance(gt, torch.Tensor) and not isinstance(gt, MetaTensor):
            raise TypeError("Score generation failed as the gt was not a torch.Tensor or a monai MetaTensor")

        
        return {metric_type:self.metric_classes[metric_type](image_masks, pred, gt) for metric_type in self.metrics_configs.keys()} 
    

class DiceScore:

    def __init__(self,
        ignore_empty: bool,
        include_background: bool,
        include_per_class_scores: bool,
        config_labels_dict: dict[str, int]
    ):
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
        Multi-class generalisable dice-score computer, it implements the summing across all of the classes prior to 
        computing the overall Dice Score.

        inputs: 
        
        cross_class_mask - torch tensor or monai metatensor for weighting the multi-class computation. 
        This cannot be a zeroes tensor. 

        Assumed to be HW(D). 

        per_class_masks - class-separated dictionary of torch tensors or monai metatensors for weighting the multiclass computation.
        These all cannot be a zeroes tensor. 

        Assumed to each be HW(D)

        pred, gt - torch tensors or monai metatensors for the pred and gt.

        Assumed to each be HW(D) (not one-hot encoded)

        '''

        #For multi-class (or binary class where self-include background is TRUE)

        #We weight this according to the class-separated image_masks (BINARISED).  

        cross_class = self.dice_score_multiclass(pred, gt, cross_class_mask) 

        if self.include_per_class_scores:
            per_class_scores = dict() 
            
            for class_label, class_integer_code in self.config_labels_dict.items():

                if not self.include_background:
                    if class_label.title() == 'Background':
                        continue 
        
                class_sep_pred = torch.where(pred == class_integer_code, 1, 0)
                class_sep_gt = torch.where(gt == class_integer_code, 1, 0)


                per_class_scores[class_label] = self.dice_score_per_class(class_sep_pred, class_sep_gt, per_class_masks[class_label])
            

            assert type(cross_class) == torch.Tensor and cross_class.size() == torch.Size([1]), 'Generation of dice score failed because the cross-class score was not a torch tensor of size 1'
        
        else:
            per_class_scores = None 

        return (cross_class, per_class_scores)
        
    
        
    def dice_score_per_class(self, class_sep_pred, class_sep_gt, image_mask):

        y_o = torch.sum(torch.where(class_sep_gt > 0, 1, 0) * image_mask)
        y_hat_o = torch.sum(torch.where(class_sep_pred > 0, 1, 0) * image_mask)
        
        denom = y_o + y_hat_o
        # weighted_pred = class_sep_pred * image_mask 
        # weighted_gt = class_sep_gt * image_mask 

        intersection = torch.sum(torch.masked_select(image_mask, class_sep_pred * class_sep_gt > 0))

        if y_o > 0:
            return torch.tensor([(2 * intersection)/(denom)])
        
        if self.ignore_empty or denom.isnan():
            #If we ignore empty or the click set was empty then just return a nan value
            return torch.tensor([float("nan")])
        
        if denom <=0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor([1.0])
        
        #else:
        return torch.tensor([0.0])
    
    def dice_score_multiclass(self, pred, gt, image_mask, dict_class_codes):

        '''
        image mask here is the cross-class image mask which is binarised.
        '''
        if not self.include_background:
            #then background not included
            y_o = torch.sum(torch.where(gt > 0, 1, 0) * image_mask)
            y_hat_o = torch.sum(torch.where(pred > 0, 1, 0) * image_mask)
        else:
            #when background is included, it includes all voxels..
            y_o = torch.sum(torch.ones_like(gt) * image_mask)
            y_hat_o = torch.sum(torch.ones_like(pred) * image_mask)

        if y_o > 0:
            
            intersection = 0
            
            
            for class_label, class_code in dict_class_codes.items():
                
                if not self.include_background:
                    if class_label.title() == 'Background':
                        continue 
                

                pred_channel = torch.where(pred == class_code, 1, 0) 
                
                gt_channel = torch.where(gt == class_code, 1, 0) 


                #NOTE: For weightmaps which are not a tensor of ones (and non binarised maps) the voxel values have 
                # must have already been weighted by the corresponding values in the image mask, so that when we sum
                # it already contains the weight.

                intersection += torch.sum(torch.masked_select(image_mask, pred_channel * gt_channel > 0 )) 
                              

            return torch.tensor([(2.0 * intersection) / (y_o + y_hat_o)])
        
        
        if self.ignore_empty or denorm.isnan():
            #If we ignore empty or the click set was empty, then just return a nan value
            return torch.tensor([float("nan")])
        
        denorm = y_o + y_hat_o
        if denorm <= 0:
            #If we do not ignore empties, then return a value of 1 if the denom is <=0 (i.e. when both are empty) to indicate full coverage...
            return torch.tensor([1.0])
        return torch.tensor([0.0])

    def __call__(self, 
                image_masks, 
                pred, 
                gt):
        
        
        (cross_class_score, per_class_scores) = self.dice_score(image_masks['cross_class_mask'], image_masks['per_class_masks'], pred, gt)
        
        return {"cross_class_scores":cross_class_score, "per_class_scores":per_class_scores}    
    
class ErrorRate:
    def __init__(self):
        raise NotImplementedError('Requires refactor to move class-level arguments into the class init. Also to use the include_per_class_scores thing?')

    
    def error_rate(self, ignore_empty, include_background, include_per_class_scores, image_masks, pred, gt, dict_class_codes):
        
        #This is multi-class generalisable.        
        
        #We assume the prediction and gt are discrete.

        #We assume that the image mask contains the information about which voxels are being used for computing the error rate (it may also captures a weight map).
        assert type(ignore_empty) == bool, 'Error rate generation failed because ignore_empty was not a bool'
        assert type(include_background) == bool, 'Error rate generation failed because include_background was not a bool'
        assert type(include_per_class_scores) == bool, 'Error rate generation failed because include_per_class_scores parameter was not a bool'
        assert type(image_masks) == tuple, 'Error rate generation failed because image_masks were not presented in a tuple consisting of a cross-class mask and a class separated dict of masks'
        assert type(image_masks[0]) == torch.Tensor, 'Error rate generation failed because the cross-class mask was not a torch.Tensor'
        assert type(image_masks[1]) == dict, 'Error rate generation failed because the per-class mask parameter was not a class-separated dict.'

    
        # class_separated_pred = dict() 

        # class_separated_gt = dict()

        #Unlike the dice score computation, the include background metric parameter is not used for the cross class score. Any information pertaining to that 
        # should be encoded into the image mask[0]. I.e. the image mask contains the weighting for all of the voxels under consideration. 
        
        _, _, error_rate_cross_class = self.extract_error_rate_info(ignore_empty, pred, gt, image_masks[0])
        

        if include_per_class_scores:

            per_class_scores = dict()
            for class_label, class_code in dict_class_codes.items():

                if not include_background:
                    if class_label.title() == "Background":
                        continue 

                class_separated_pred = torch.where(pred == class_code, 1, 0)
                class_separated_gt = torch.where(gt == class_code, 1, 0)
            
                (per_class_weighted_errors, per_class_weighted_denom, per_class_error_rate) = self.extract_error_rate_info(ignore_empty, class_separated_pred, class_separated_gt, image_masks[1][class_label])

                
                per_class_scores[class_label] = per_class_error_rate

        else:
            per_class_scores = None 
        
        return (error_rate_cross_class, per_class_scores)
        
    def extract_error_rate_info(self, ignore_empty, pred, gt, image_mask):
        
        disjoint = torch.ones_like(pred) - torch.where(pred == gt, 1, 0) 

        #applying the image mask weightings to these error voxels

        weighted_errors = torch.sum(disjoint * image_mask)

        #computing the denominator (the weighting of the gt voxels) from the gt mask, the image mask should implicitly capture the set of voxels being
        # examined, so we can just sum over the gt.... 

        weighted_denom = torch.sum(image_mask) 


        error_rate = self.error_rate_comp(ignore_empty, weighted_errors, weighted_denom)


        return (weighted_errors, weighted_denom, error_rate)
    
    def error_rate_comp(self, ignore_empty, weighted_errors, weighted_denom):

        if weighted_denom > 0:
            #In this case, there were some voxels for this class which had been modified.
            error_rate = torch.tensor([weighted_errors/weighted_denom])
            assert float(error_rate) <= 1.0

            return error_rate 

        if ignore_empty or weighted_denom.isnan():
            #if ignore empty or the mask was empty (due to the click set OR due to the changed voxels set.)
            return torch.tensor([float("nan")])
        
        if weighted_denom <= 0:
            #If there are no changed voxels (this is when the denom would = 0), then just return an error rate of 0.
            return torch.tensor([float(0)])


    def __call__(self, ignore_empty, include_background, include_per_class_scores, image_masks, pred, gt, dict_class_codes):
        

        # (cross_class_score, per_class_scores) = self.error_rate(ignore_empty, image_mask, pred, gt, num_classes)

        (cross_class_score, per_class_scores) = self.error_rate(ignore_empty, include_background, include_per_class_scores, image_masks, pred, gt, dict_class_codes)
        
        return {"cross_class_scores":cross_class_score, "per_class_scores":per_class_scores} 