import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from monai.data import MetaTensor 
from typing import Union 

from src.results_utils.metric_save_util import save_csv
from src.results_utils.base_metrics.scoring_base_utils import BaseScoringWrapper

class MetricComputer:
    def __init__(
        self,
        dice_termination_threshold: float,
        metrics_configs: dict[str, dict],
        metrics_savepaths: dict[str, str],
        class_configs_dict: dict[str, int]

    ):
        '''
        Class which is used for computing and saving quantitative metrics. 
        
        inputs: 

        metrics_configs: A dictionary - Contains the information about each metric under consideration necessary
        for initialising its corresponding computation class.

        Format - metric_type: dict of args.

        metrics_savepaths: A dictionary - Contains the savepaths to each csv file being used for saving metrics.
        '''
        self.termination_thresh = dice_termination_threshold 
        self.metrics_configs = metrics_configs
        self.metrics_savepaths = metrics_savepaths 
        self.class_configs_dict = class_configs_dict 

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
            class_configs_dict=self.class_configs_dict
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
            'per_class_masks': {class_lab:torch.ones_like(tensor,dtype=torch.int32) for class_lab in self.class_configs_dict.keys()}
        }
    def exec_base_metrics(self, 
                        pred:Union[torch.Tensor, MetaTensor], 
                        gt:Union[torch.Tensor, MetaTensor],
                        tracked_metrics:dict,
                        infer_call_info:dict):
        
        image_masks = self.generate_base_masks(pred)
        metric_output = self.base_computer(
            image_masks = image_masks,
            pred = pred,
            gt = gt,
        )
        #Metric output is provided as a dictionary separated by the metric type.
        if infer_call_info['mode'].title() != 'Interactive Edit':
            tracked_metrics[infer_call_info['mode']] = metric_output
        else:
            tracked_metrics[f'{infer_call_info["mode"]} Iter {infer_call_info["edit_num"]}'] = metric_output

        if float(tracked_metrics['Dice']) >= self.termination_thresh:
            return tracked_metrics, True 
        
        else:
            return tracked_metrics, False 
        
    def exec_base_relative_metrics(self):
        pass 

    def __call__(self,
                output_data:dict,
                data_instance: dict,
                tracked_metrics: dict,
                im: dict,
                infer_call_info: dict,
                write_metrics: bool):
        
        if not write_metrics:
            extracted_pred = self.extract_spatial_dims(output_data['pred']['metatensor'])
            extracted_gt = self.extract_spatial_dims(data_instance['gt']['metatensor'])

            tracked_metrics, terminate_bool = self.exec_base_metrics(extracted_pred, extracted_gt, tracked_metrics, infer_call_info)
            
            return tracked_metrics, terminate_bool
        else:
            #In this situation we presume that the iterative loop is complete, and that we will just be saving the 
            #tracked metrics.

            #Tracked metrics will have the following structure:
            # {
            #     '__ Init': {
            #         'Dice': __ 
            #         'xyz': __
            #     }
            #     'Interactive Edit Iter __':{
            #         'Dice': __
            #         'xyz': __ 
            #         'xyz editing only metric': __ 
            #     }
            #   etc.
            # }
            pass 