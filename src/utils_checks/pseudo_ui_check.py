#Module containing the checkers for the pseudo-ui class.
from typing import Union
import torch 
from monai.data import MetaTensor 


def check_empty(tensor: MetaTensor, bg_val):
    '''
    Basic function which checks whether an input tensor has any foreground voxel. This is determined according to
    the bg val provided.. 
    '''
    if not isinstance(tensor, MetaTensor):
        raise TypeError('The input tensor can only be a Monai metatensor')
    
    return torch.all(tensor == bg_val)

def check_config_labels(config_labels_dict: dict):
    '''
    Basic function which implements some checks on the config labels dictionary.
    '''
    if not isinstance(config_labels_dict, dict) or not config_labels_dict:
        raise TypeError('Config label-code mapping must be provided as a non-empty dict.')
        
    def check_ints(config_labels: dict):
        if not all(isinstance(k, str) for k in config_labels.keys()):
            raise TypeError("All keys in config_labels_dict must be strs")
        if not all(isinstance(k, int) for k in config_labels.values()):
            raise TypeError("All the values in config_labels_dict must be ints")
        
    def check_bg(config_labels: dict):
        if 'background' not in config_labels.keys():
            raise KeyError('The background class must be in the config labels dict.')
        else:
            if config_labels['background'] != 0:
                raise ValueError('The value of the background class must be zero.')
    
    def check_fg_empty(config_labels_dict):
        if not set(config_labels_dict.keys()).difference({'background'}): #Empty set evaluates as False.
            raise Exception('There must be at least one foreground class!')
    

    check_ints(config_labels_dict)
    check_bg(config_labels_dict)
    check_fg_empty(config_labels_dict)


