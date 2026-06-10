import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
from monai.data import MetaTensor 
from typing import Union 
import warnings 
import gc
from src.results_utils.metric_save_util import write_to_csvs
from src.results_utils.base_metrics.scoring_base_utils import BaseScoringWrapper

class MetricsHandler:
    def __init__(
        self,
        calc_device: torch.device, 
        early_termination_criterion: dict, #dice_termination_thresh: float,
        metrics_configs: dict[str, dict],
        metrics_savepaths: dict[str, dict[str, Union[str, None, dict]]] | None,
        semantic_id_dict: dict[str, int]

    ):
        '''
        Class which is used for computing and saving quantitative metrics. 
        
        inputs: 

        metrics_configs: A nested dictionary - Contains the information about each metric under consideration necessary
        for initialising its corresponding computation class.

        Format - metric_type: dict of args.

        metrics_savepaths: 
        A nested dictionary (or NoneType if skipping over saving metrics) - Contains the savepaths to each csv file being used for saving metrics.
        
        Format - metric_type: Nested Dict of metric type specific paths containing the paths for cross-class and 
        per-class (Optional) scores for the given metric. CSVs are pre-initialised in the runscript.

        '''
        self.calc_device = calc_device
        self.early_termination_criterion = early_termination_criterion 
        # self.termination_thresh = dice_termination_threshold 
        self.metrics_configs = metrics_configs
        self.metrics_savepaths = metrics_savepaths 
        self.semantic_id_dict = semantic_id_dict 

        #Note, each metric must contain a corresponding savepath!
        if self.metrics_savepaths != None:
            if not set(self.metrics_configs) == set(self.metrics_savepaths):
                raise Exception('The metrics configs and metrics savepaths must have the exact same keys')
        
        self.supported_metrics = {
            'base':{'Dice', 'NSD'},
            # 'base_relative':set(),#{'Consistent Dice Improvement'},
            # 'human_centric':set(),
            # 'human_centric_relative':set(),
            # 'annotation_budget':set()
        }

        #Checking that all the selected metric types are supported.
        all_metrics = set()
        for val in self.supported_metrics.values():
            all_metrics |= val 
     
        if not set(self.metrics_configs) <= all_metrics:
            raise Exception('The selected metrics in the metrics configs are not all supported.')
        
        #Checking that the early-termination criterion metric is in the metrics configs, otherwise
        #we will not be able to perform the early termination check.
        if self.early_termination_criterion['metric'] not in self.metrics_configs:
            raise Exception('The early termination criterion metric must be included in the selected metrics in the metrics configs, otherwise we will not be able to perform the early termination check.')
        if 'threshold' not in self.early_termination_criterion:
            raise Exception('The early termination criterion config must contain a field called threshold which denotes the threshold at which the process will be terminated early.')
            #For now our only termination criterion will be based on a score -> in the future this may
            #evolve into something more dependent on a characteristic of the segmentation e.g., connected-ness.
        
        #Dividing the selected metrics into their corresponding subtype.
        self.base_metrics = self.supported_metrics['base'] & set(self.metrics_configs)
        # self.budget_metrics = self.supported_metrics['annotation_budget'] & set(self.metrics_configs)

        
        self.init_base_metrics()
        #NOTE: Populate based off time-estimates.
        # self.init_budget_metrics()

    def output_checking(self, metric_type, metric_subdict):
        '''

        Metric_type: A string denoting the metric the scores belong to. 

        Metric subdict contains two fields: 

        cross_class_scores: A torch tensor or a metatensor.
        per_class_scores: A class separated dict of torch tensors or metatensors OR a Nonetype.

        '''
        if 'multiple_parameter_values' in self.metrics_configs[metric_type]:
            if not isinstance(metric_subdict['cross_class_scores'], dict) and not isinstance(metric_subdict['cross_class_scores'], dict):
                raise TypeError(f'The cross class score provided for {metric_type} was not a torch tensor or a metatensor')

            if not isinstance(metric_subdict['per_class_scores'], dict) and not metric_subdict['per_class_scores'] is None:
                raise TypeError(f'The per class scores provided for {metric_type} were not a dict or a NoneType')
            
            if isinstance(metric_subdict['per_class_scores'], dict):
                for key, val in metric_subdict['per_class_scores'].items():
                    if not isinstance(val, dict) and not isinstance(val, dict):
                        raise TypeError(f'The key {key} in per class scores in {metric_type} was not a Torch Tensor or a MetaTensor')

        else:
            if not isinstance(metric_subdict['cross_class_scores'], torch.Tensor) and not isinstance(metric_subdict['cross_class_scores'], MetaTensor):
                raise TypeError(f'The cross class score provided for {metric_type} was not a torch tensor or a metatensor')

            if not isinstance(metric_subdict['per_class_scores'], dict) and not metric_subdict['per_class_scores'] is None:
                raise TypeError(f'The per class scores provided for {metric_type} were not a dict or a NoneType')
            
            if isinstance(metric_subdict['per_class_scores'], dict):
                for key, val in metric_subdict['per_class_scores'].items():
                    if not isinstance(val, torch.Tensor) and not isinstance(val, MetaTensor):
                        raise TypeError(f'The key {key} in per class scores in {metric_type} was not a Torch Tensor or a MetaTensor')

        return True 
    
    def extract_spatial_dims(self, input):
        '''
        Function which extracts the spatial dimensions of the input, assumed to be CHW(D). Returns it in torch.int64 dtype in a channel-split list.
        '''
        return [input[i,:].to(dtype=torch.uint8) for i in range(input.shape[0])]
    
    def init_base_metrics(self):
        #Extracting the metric configs for the base metric types ONLY.
        base_metrics_configs = {key:val for key, val in self.metrics_configs.items() if key in self.base_metrics}
        self.base_computer = BaseScoringWrapper(
            calc_device=self.calc_device, 
            metrics_configs=base_metrics_configs,
            semantic_id_dict=self.semantic_id_dict
        )


    # def init_base_relative_metrics(self):
    #     pass 
    
    def generate_base_masks(self, tensor):
        '''
        Basic function which generates a tensor of ones (with shape matching input tensor) across the entirety of the 
        spatial dimensions for the base counting metrics (i.e. without masks applied). It generates this for both the 
        cross_class_mask and the per_class_masks.
        '''
        return (
            torch.ones_like(tensor, dtype=torch.uint8), #'cross_class_mask':
            {class_lab:torch.ones_like(tensor,dtype=torch.uint8) for class_lab in self.semantic_id_dict.keys()} #'per_class_masks': 
        )
    
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
                tracked_metrics[metric_type].update({f'{infer_call_info["mode"].title()} Iter {infer_call_info["edit_num"]}':metrics_dict})

        #We implement an early-stopping check for termination based on overall Dice overlap score.
        early_termination = self.check_early_termination(metric_output)
        # if float(metric_output['Dice']['cross_class_scores']) >= self.termination_thresh:
        #     return tracked_metrics, True 
        # else:
        #     return tracked_metrics, False 
        return tracked_metrics, early_termination 
    
    def check_early_termination(self, metric_output):
        '''
        Function which checks whether the early termination criterion has been met based on the provided metric output. 

        For now, this is just based on a single metric score exceeding a threshold, but in the future this could be extended to be more complex and dependent on multiple factors. 
        '''
        if float(metric_output[self.early_termination_criterion['metric']]['cross_class_scores']) >= self.early_termination_criterion['threshold']:
            return True 
        else:
            return False
        
    def update_metrics(
        self,
        output_data:dict,
        data_instance: dict,
        tracked_metrics: dict,
        im_inf: dict,
        infer_call_info: dict,
        ):
        if output_data is None:
            for metric_type in self.metrics_configs:
                if metric_type not in tracked_metrics:
                    tracked_metrics[metric_type] = {}

                if infer_call_info['mode'].title() != 'Interactive Edit':
                    key = infer_call_info['mode'].title()
                else:
                    key = f"{infer_call_info['mode'].title()} Iter {infer_call_info['edit_num']}"

                entry = {'cross_class_scores': 'could_not_generate_prompt'}

                if self.metrics_configs[metric_type].get('include_per_class_scores', False):
                    per_class = {}
                    for class_lb in self.semantic_id_dict:
                        if class_lb.title() == 'Background' and not self.metrics_configs[metric_type].get('include_background_metric', True):
                            continue
                        per_class[class_lb] = 'could_not_generate_prompt'
                    entry['per_class_scores'] = per_class

                tracked_metrics[metric_type][key] = entry

            return tracked_metrics, True

        #The output data must have been post-processed and checked to ensure that the output of the user is valid
        #prior to metric computation. 
        
        #Modified to be fully in-line, so that garbage collection is not necessary due to intermediate variables.
        tracked_metrics, terminate_bool = self.exec_base_metrics(
            self.extract_spatial_dims(output_data['pred']['metatensor'])[0], 
            self.extract_spatial_dims(data_instance['eval_label']['metatensor'])[0], 
            tracked_metrics, 
            infer_call_info
            )
        
        torch.cuda.empty_cache()
        return tracked_metrics, terminate_bool

    def save_metrics(
        self,
        case_name: str, 
        empty_foreground: bool,
        terminated_early: bool,
        temporary_iter_lims:tuple[int],
        tracked_metrics: dict
        ):

        '''
        Function which is intended for when the iterative loop is complete, and where we will just be saving the 
        tracked metrics.

        Inputs:

        case_name: A string (extracted from the loaded data_instance on the pseudo-ui front-end) denoting the name
        of the case. 

        empty_foreground: A bool denoting whether the foreground was empty for the given data instance, this is required for
        cases where the experiment was configured to assess algorithm behaviour when requesting inference for an empty foreground
        (i.e. sim_empty_fg_automatic config set to True).

        terminated_early: A bool denoting whether the iterative refinement process terminated early due to the segmentation
        quality reaching an adequate level. 

        temporary_iter_lims: A tuple for temporarily handling early termination, for bottom and upper iteration limit for padding the tracked metrics. 

        tracked_metrics: A dictionary containing the tracked metrics for the given data instance across the iterative
        refinement process. 

            Tracked metrics will be this exemplar structure:
            {
                'Dice':{
                       '__ Init': __, {
                        'cross_class_scores': ___,
                        'per_class_scores': dict() OR NoneType
                          }
            
                       'Interactive Edit Iter __': same thing: cross class scores: ___, or per_class_scores: {} or NoneType
                      },
                'xyz':{
                      },
                'xyz_editing_only_metric':{
                  }

        '''
        if type(temporary_iter_lims[1]) != int:
            raise Exception('The upper iteration limit must be an integer denoting the final iteration to be saved/padded to.')
         
        if temporary_iter_lims[1] == 0:
            #If the upper limit is 0 then we are not performing editing iteration, so no need to provide handling.
            print('The upper budget limit of the iterations is 0 as we only do init, so we will not need to provide any special handling for additional iterations')
        else:
            if empty_foreground:
                #In this case we will pad the tracked metrics with "empty_foreground" denoting that the foreground was empty 
                # and so we could not perform any editing iterations.
                warnings.warn('The foreground was empty and the sim_empty_fg_automatic config was set to True, so if performing interactive edits we will pad the tracked metrics with a string denoting empty foreground')
                for metric_type, metric_subdict in tracked_metrics.items():
                    if temporary_iter_lims[0] != 0:
                        raise Exception('If the foreground is empty, the bottom iteration limit must be 0, as there are no edits possible.')

                    for iter_num in range(temporary_iter_lims[0] + 1, temporary_iter_lims[1] + 1):
                        #We will be starting at the first iteration after the stopping point, and by default we always have the init, so 
                        #we will only be padding the interactive edit iterations.
                        
                        #First we pad the cross-class scores:
                        metric_subdict['Interactive Edit Iter ' + str(iter_num)] = dict()

                        if 'multiple_parameter_values' in self.metrics_configs[metric_type]:
                            metric_subdict['Interactive Edit Iter ' + str(iter_num)]['cross_class_scores'] = dict()
                            parameter_name = self.metrics_configs[metric_type]['multiple_parameter_values']
                            for parameter_idx in self.metrics_configs[metric_type][parameter_name]:
                                metric_subdict['Interactive Edit Iter ' + str(iter_num)]['cross_class_scores'][parameter_idx] = 'empty_foreground'
                        else:
                            metric_subdict['Interactive Edit Iter ' + str(iter_num)]['cross_class_scores'] = 'empty_foreground'
                        
                        
                        #Then we pad the per-class scores, if include_per_class_scores config is True
                        if self.metrics_configs[metric_type]['include_per_class_scores']:
                            if 'multiple_parameter_values' in self.metrics_configs[metric_type]:
                                metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'] = dict()
                                parameter_name = self.metrics_configs[metric_type]['multiple_parameter_values']
                                for class_lb in self.semantic_id_dict.keys():
                                    if class_lb.title() == 'Background' and not self.metrics_configs[metric_type]['include_background_metric']:
                                        continue 
                                    metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'][class_lb] = dict()
                                    for parameter_idx in self.metrics_configs[metric_type][parameter_name]:
                                        metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'][class_lb][parameter_idx] = 'empty_foreground'
                            else:
                                metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'] = dict()
                                for class_lb in self.semantic_id_dict.keys():
                                    if class_lb.title() == 'Background' and not self.metrics_configs[metric_type]['include_background_metric']:
                                        continue 
                                    #We will pad the per-class scores with a string denoting empty foreground.
                                    metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'][class_lb] = 'empty_foreground'
            
            elif terminated_early: #Terminated early but not because of a fully empty foreground.
                warnings.warn('The process terminated early, given that determining the early termination is a design choice that is not fully covered,'
                ' the current implementation will pad with a string denoting early termination')
                # raise NotImplementedError('No implementation for handling the tracked metrics when the process terminated early')

                for metric_type, metric_subdict in tracked_metrics.items():
                    if type(temporary_iter_lims[0]) != int:
                        raise Exception('If the process terminated early, the bottom iteration limit must be an integer denoting the last completed iteration.')
                    if not temporary_iter_lims[1] > temporary_iter_lims[0]:
                        raise Exception('The upper iteration limit must be greater than the bottom iteration limit when handling \n' \
                        'early termination. Because, either the process did not terminate early (in which case no padding is needed), \n' \
                        'or the upper limit must be greater than the last completed iteration to allow for padding.')
                     
                    for iter_num in range(temporary_iter_lims[0] + 1, temporary_iter_lims[1] + 1):
                        #We will be starting at the first iteration after the stopping point, and by default we always have the init, so 
                        #we will only be padding the interactive edit iterations.
                        
                        #First we pad the cross-class scores:
                        metric_subdict['Interactive Edit Iter ' + str(iter_num)] = dict()

                        if 'multiple_parameter_values' in self.metrics_configs[metric_type]:
                            metric_subdict['Interactive Edit Iter ' + str(iter_num)]['cross_class_scores'] = dict()
                            parameter_name = self.metrics_configs[metric_type]['multiple_parameter_values']
                            for parameter_idx in self.metrics_configs[metric_type][parameter_name]:
                                metric_subdict['Interactive Edit Iter ' + str(iter_num)]['cross_class_scores'][parameter_idx] = 'terminated_early'
                        else:
                            metric_subdict['Interactive Edit Iter ' + str(iter_num)]['cross_class_scores'] = 'terminated_early'

                        #Then we pad the per-class scores, if include_per_class_scores config is True
                        if self.metrics_configs[metric_type]['include_per_class_scores']:
                                if 'multiple_parameter_values' in self.metrics_configs[metric_type]:
                                    metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'] = dict()
                                    parameter_name = self.metrics_configs[metric_type]['multiple_parameter_values']
                                    for class_lb in self.semantic_id_dict.keys():
                                        if class_lb.title() == 'Background' and not self.metrics_configs[metric_type]['include_background_metric']:
                                            continue 
                                        metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'][class_lb] = dict()
                                        for parameter_idx in self.metrics_configs[metric_type][parameter_name]:
                                            metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'][class_lb][parameter_idx] = 'terminated_early'  
                                else:
                                    metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'] = dict()
                                    for class_lb in self.semantic_id_dict.keys():
                                        if class_lb.title() == 'Background' and not self.metrics_configs[metric_type]['include_background_metric']:
                                            continue 
                                        #We will pad the per-class scores with a string denoting early termination.
                                        metric_subdict['Interactive Edit Iter ' + str(iter_num)]['per_class_scores'][class_lb] = 'terminated_early'
            else:
                if temporary_iter_lims[0] != None:
                    raise Exception('If the process did not terminate early, the bottom iteration limit must be a NoneType. There is \n' \
                    'no need to pad the tracked metrics.')
                # print('The process did not terminate early, so we will save the tracked metrics as they are without modification')
        
        if self.metrics_savepaths == None:
            raise Exception('The metrics savepaths were not properly initialised, so we cannot save the metrics. Please check the metrics savepath initialisation.')
        write_to_csvs(
            case_name=case_name, 
            csv_paths=self.metrics_savepaths, 
            tracked_metrics=tracked_metrics,
            metrics_configs=self.metrics_configs)  
