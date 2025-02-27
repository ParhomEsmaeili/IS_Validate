import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from monai.data import MetaTensor 
from typing import Union 

from src.results_utils.metric_save_util import write_to_csvs
from src.results_utils.base_metrics.scoring_base_utils import BaseScoringWrapper

class MetricsHandler:
    def __init__(
        self,
        dice_termination_threshold: float,
        metrics_configs: dict[str, dict],
        metrics_savepaths: dict[str, dict[str, Union[str, None, dict]]],
        config_labels_dict: dict[str, int]

    ):
        '''
        Class which is used for computing and saving quantitative metrics. 
        
        inputs: 

        metrics_configs: A nested dictionary - Contains the information about each metric under consideration necessary
        for initialising its corresponding computation class.

        Format - metric_type: dict of args.

        metrics_savepaths: A nested dictionary - Contains the savepaths to each csv file being used for saving metrics.
        
        Format - metric_type: Nested Dict of metric type specific paths containing the paths for cross-class and 
        per-class (Optional) scores for the given metric. CSVs are pre-initialised in the runscript.

        '''
        self.termination_thresh = dice_termination_threshold 
        self.metrics_configs = metrics_configs
        self.metrics_savepaths = metrics_savepaths 
        self.config_labels_dict = config_labels_dict 

        #Note, each metric must contain a corresponding savepath!

        if not set(self.metrics_configs) == set(self.metrics_savepaths):
            raise Exception('The metrics configs and metrics savepaths must have the exact same keys')

        self.supported_metrics = {
            'base':{'Dice'},
            'base_relative':set(),#{'Consistent Dice Improvement'},
            'human_centric':set(),
            'human_centric_relative':set(),
            'annotation_budget':set()
        }

        #Checking that all the selected metric types are supported.
        all_metrics = set()
        for val in self.supported_metrics.values():
            all_metrics |= val 
     
        if not set(self.metrics_configs) <= all_metrics:
            raise Exception('The selected metrics in the metrics configs are not all supported.')
        
        #Dividing the selected metrics into their corresponding subtype.
        self.base_metrics = self.supported_metrics['base'] & set(self.metrics_configs)
        self.base_relative_metrics = self.supported_metrics['base_relative'] & set(self.metrics_configs)
        self.human_centric_metrics = self.supported_metrics['human_centric'] & set(self.metrics_configs)
        self.human_centric_relative_metrics = self.supported_metrics['human_centric_relative'] & set(self.metrics_configs)
        self.budget_metrics = self.supported_metrics['annotation_budget'] & set(self.metrics_configs)

        
        self.init_base_metrics()
        #TODO: Populate the funcs for the others:
        # self.init_base_relative_metrics() 
        # self.init_human_centric_metrics()
        # self.init_budget_metrics()

    def output_checking(self, metric_type, metric_subdict):
        '''

        Metric_type: A string denoting the metric the scores belong to. 

        Metric subdict contains two fields: 

        cross_class_score: A torch tensor or a metatensor.
        per_class_scores: A class separated dict of torch tensors or metatensors OR a Nonetype.

        '''
        
        if not isinstance(metric_subdict['cross_class_score'], torch.Tensor) and not isinstance(metric_subdict['cross_class_score'], MetaTensor):
            raise TypeError(f'The cross class score provided for {metric_type} was not a torch tensor or a metatensor')

        if not isinstance(metric_subdict['per_class_scores'], dict) and not metric_subdict['per_class_scores'] is None:
            raise TypeError('The per class scores provided for {metric_type} were not a dict or a NoneType')
        
        if isinstance(metric_subdict['per_class_scores'], dict):
            for key, val in metric_subdict.items():
                if not isinstance(val, torch.Tensor) and not isinstance(val, MetaTensor):
                    raise TypeError(f'The key {key} in per class scores in {metric_type} was not a Torch Tensor or a MetaTensor')

        return True 
    
    def extract_spatial_dims(self, input):
        '''
        Function which extracts the spatial dimensions of the input, assumed to be CHW(D). Returns it in torch.int32 dtype.
        '''
        return input[0,:].to(dtype=torch.int32)
    
    def init_base_metrics(self):
        #Extracting the metric configs for the base metric types ONLY.
        base_metrics_configs = {key:val for key, val in self.metrics_configs.items() if key in self.base_metrics}
        self.base_computer = BaseScoringWrapper(
            metrics_configs=base_metrics_configs,
            config_labels_dict=self.config_labels_dict
        )


    def init_base_relative_metrics(self):
        pass 
    
    def generate_base_masks(self, tensor):
        '''
        Basic function which generates a tensor of ones (with shape matching input tensor) across the entirety of the 
        spatial dimensions for the base metrics (i.e. without masks applied). It generates this for both the 
        cross_class_mask and the per_class_masks.
        '''
        return {
            'cross_class_mask':torch.ones_like(tensor, dtype=torch.int32),
            'per_class_masks': {class_lab:torch.ones_like(tensor,dtype=torch.int32) for class_lab in self.config_labels_dict.keys()}
        }
    
    def exec_base_metrics(self, 
                        pred:Union[torch.Tensor, MetaTensor], 
                        gt:Union[torch.Tensor, MetaTensor],
                        tracked_metrics:dict,
                        infer_call_info:dict):
        '''
        This is a function which computes the metrics using the base metrics wrapper, and updates the tracked metrics dictionary.

        It also returns a boolean depending on whether a termination condition has been reached using dice score.
        '''
        image_masks = self.generate_base_masks(pred)
        metric_output = self.base_computer(
            image_masks = image_masks,
            pred = pred,
            gt = gt,
        )
        
        #Checking the metric output by metric_type:
        if all([self.output_checking(metric_type, subdict) for metric_type, subdict in metric_output.items()]):
            print('Output datatypes for the metrics are correct')

        #Metric output is provided as a dictionary separated by the metric type, and then by the infer mode, and then 
        #by cross-class and per-class scores (or NoneType)

        for metric_type, metrics_dict in metric_output.items():

            if infer_call_info['mode'].title() != 'Interactive Edit':
                tracked_metrics[metric_type] = {infer_call_info['mode'].title():metrics_dict}
            else:
                tracked_metrics[metric_type] = {f'{infer_call_info["mode"].title()} Iter {infer_call_info["edit_num"]}':metric_output}

        #We implement an early-stopping check for termination based on overall Dice overlap score.

        if float(metric_output['Dice']['cross_class_scores']) >= self.termination_thresh:
            return tracked_metrics, True 
        
        else:
            return tracked_metrics, False 
        
    def exec_base_relative_metrics(self):
        pass 
    
    def update_metrics(
        self,
        output_data:dict,
        data_instance: dict,
        tracked_metrics: dict,
        im_inf: Union[dict, None], #Should never be None for this, really..
        im_metric: Union[dict, None], #Could be Nonetype for this if there are no parametrisation requirements.
        infer_call_info: dict,
        ):
        #The output data must have been post-processed and checked to ensure that the output of the user is valid
        #prior to metric computation. 
        
        extracted_pred = self.extract_spatial_dims(output_data['pred']['metatensor'])
        extracted_gt = self.extract_spatial_dims(data_instance['gt']['metatensor'])

        tracked_metrics, terminate_bool = self.exec_base_metrics(extracted_pred, extracted_gt, tracked_metrics, infer_call_info)
        
        #TODO: Add implementation for other metrics.

        return tracked_metrics, terminate_bool

    def save_metrics(
        self,
        patient_name: str, 
        terminated_early: bool,
        tracked_metrics: dict
        ):

        '''
        Function which is intended for when the iterative loop is complete, and where we will just be saving the 
        tracked metrics.

        Inputs:

        patient_name: A string (extracted from the loaded data_instance on the pseudo-ui front-end) denoting the name
        of the patient. 

        terminated_early: A bool denoting whether the iterative refinement process terminated early due to the segmentation
        quality reaching an adequate level controlled by the threshold in the class initialisation. 

        tracked_metrics: A dictionary containing the tracked metrics for the given data instance across the iterative
        refinement process. 

            Tracked metrics will be this exemplar structure:
            {
                'Dice':{
                       '__ Init': __, {
                        'cross_class_scores': ___,
                        'per_class_scores': dict() OR NoneType
                          }
            
                       'Interactive Edit Iter __': {} or NoneType
                      },
                'xyz':{
                      },
                'xyz_editing_only_metric':{
                  }

        '''

        if terminated_early:
            raise NotImplementedError('No implementation for handling the tracked metrics when the process terminated early')
        else:
            write_to_csvs(patient_name=patient_name, csv_paths=self.metrics_savepaths, tracked_metrics=tracked_metrics)  
