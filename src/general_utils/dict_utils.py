from typing import Union 
import re
import os
import json 

def extractor(x: dict, y: tuple[Union[tuple,str, int]]): 
    '''
    This general purpose function is adapted from a lambda function which iterates through the dict using the tuple, 
    where the order/index denotes the depth. 
    
    Once tuple is empty it stops (also for NoneType it stops, i.e. just returns the current dict.) and returns the
    item from the provided tuple path. 

    Inputs: 
        x - A dictionary which is populated and will be extracted from
        y - A tuple consisting of the iterative path through the nested dict.
    '''
    if not y:
        return x
    else:
        if not isinstance(y, tuple):
            raise TypeError('y - path in dict must be a tuple')
        
        if y: #If y exists and we are iterating through it still:
            if not isinstance(x, dict):
                raise TypeError('x must be a dictionary')
            if x == {}:
                raise ValueError('The input dict must be populated otherwise we cannot extract anything.')
            else:
                return extractor(x[y[0]], y[1:]) 
            

def dict_path_modif(x: dict, y: tuple[Union[tuple, str, int]], item):
    '''
    This function takes a dictionary (can be nested), and a tuple path, and replaces the value at the given tuple path
    with the provided item.

    Requires that all keys prior to the final must exist. 

    NOTE: Use of a tuple with length 1 requires that the tuple be inserted as (path, ) 

    inputs:
    x - A dictionary which is populated and will have the item modified.
    y - A tuple path, which can consist of specific immutable datatypes (e.g. tuple, str, int) supported by dicts.
    item - Any item, which is being placed in dictionary x, with path y. 

    '''
    if not isinstance(x, dict):
        raise TypeError('The input arg for dict must be a dictionary')
    if not isinstance(y, tuple):
        raise TypeError('The input arg for the tuple path must be a tuple')
    # if not item:
    #     raise ValueError('The item cannot be a NoneType')  #NOTE: If an item is not provided then error is raised when calling func.
    # if x == {}:
    #     raise ValueError('The input dict must be populated.')

    #Recursively iterating through the dictionary using the tuple path.
    if len(y) > 1:
        x[y[0]] = dict_path_modif(x[y[0]], y[1:], item)
        return x 
    elif len(y) == 1:
        x[y[0]] = item
        return x  
    else:
        raise ValueError('The input tuple must at least be length 1.')
    



def dict_path_create(x: dict, y: tuple[Union[tuple, str, int]], item):
    '''
    This function takes a dictionary, and a tuple path, and creates a value at the given tuple path
    with the provided item. This is a generalisation of dict_path_modif.

    NOTE: Use of a tuple with length 1 requires that the tuple be inserted as (path, ) 

    inputs:
    x - A dictionary which may not necessarily be populated and will have the item inserted or modified.
    y - A tuple path, which can consist of specific immutable datatypes (e.g. tuple, str, int) supported by dicts.
    item - Any item, which is being placed in dictionary x, with path y. 

    '''
    if not isinstance(x, dict):
        raise TypeError('The input arg for dict must be a dictionary')
    if not isinstance(y, tuple):
        raise TypeError('The input arg for the tuple path must be a tuple')
    # if not item:
    #     raise ValueError('The item cannot be a NoneType')  #NOTE: If an item is not provided then error is raised when calling func.
    # if x == {}:
    #     raise ValueError('The input dict must be populated.')

    #Recursively iterating through the dictionary using the tuple path.
    if len(y) > 1:
        try:
            x[y[0]] = dict_path_create(x[y[0]], y[1:], item)
        except:
            x[y[0]] = dict()
            x[y[0]] = dict_path_create(x[y[0]], y[1:], item)
        return x 
    elif len(y) == 1:
        x[y[0]] = item
        return x  
    else:
        raise ValueError('The input tuple must at least be length 1.')
    
def extract_config(path, name):
    #Function which extracts configs dicts from json or txt files. Takes the path to the file, and the name of the specific config desired.

    if not os.path.exists(path):
        raise Exception(f'The path {path} was not a valid one. Please check.')    

    #Loading the file:
    with open(path) as f:
        configs_registry = json.load(f)
        config = configs_registry[name]

    return config 


def dict_deep_equals(dict1, dict2):
    """
    Recursively checks if two dictionaries are exactly equal at all depths.
    
    Args:
        dict1: First dictionary to compare
        dict2: Second dictionary to compare
    
    Returns:
        tuple: (bool indicating if equal, list of differences found)
    
    Supported types:
        - dict: Recursively compared by keys and values
        - list, tuple: Recursively compared element-wise (must be same type)
        - Primitives (int, str, float, bool, None): Compared by equality
    """
    differences = []
    
    def _compare(val1, val2, path=""):
        """Recursively compare values with type checking"""
        # Type mismatch
        if type(val1) != type(val2):
            msg = f"Type mismatch at {path or 'root'}: {type(val1).__name__} vs {type(val2).__name__}"
            differences.append(msg)
            return
        
        # Dict comparison (recurse into nested dicts)
        if isinstance(val1, dict):
            for key in val1:
                if key not in val2:
                    differences.append(f"Key '{key}' missing in second dict at {path or 'root'}")
                else:
                    new_path = f"{path}.{key}" if path else key
                    _compare(val1[key], val2[key], new_path)
            
            for key in val2:
                if key not in val1:
                    differences.append(f"Extra key '{key}' in second dict at {path or 'root'}")
        
        # List/Tuple comparison (recurse into nested iterables)
        elif isinstance(val1, (list, tuple)):
            if len(val1) != len(val2):
                msg = f"Length mismatch at {path or 'root'}: {len(val1)} vs {len(val2)}"
                differences.append(msg)
            else:
                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    _compare(v1, v2, new_path)
        
        # Primitive types (int, str, float, bool, None, etc.)
        else:
            if val1 != val2:
                msg = f"Value mismatch at {path or 'root'}: {val1!r} != {val2!r}"
                differences.append(msg)
    
    _compare(dict1, dict2)
    return len(differences) == 0, differences


def sort_infer_calls(infer_call_names):
    '''
    This function sorts the inference call names, and outputs them in a tuple format such that they are immutable.
    '''
    if not isinstance(infer_call_names, set):
        raise TypeError('The input infer call names must be a set data structure, please convert it to a set before passing it through!')
    if len(infer_call_names) < 1:
        raise Exception(f'At least one infer mode call subdict is required for metrics to be saved!')
    
    #We do not assume that the inference call names (or dict they were taken from) were ordered correctly, 
    # even if it is unlikely to be incorrectly ordered.

    infer_call_names_order = []
    #Check if there is an initialisation: if so, place that first. 
    init_modes  = {'Automatic Init', 'Interactive Init'}

    if init_modes & infer_call_names:
        #If the set is not empty
        if len(init_modes & infer_call_names) > 1:
            raise Exception('Cannot have two conflicting initialisation modes')
        else:
            infer_call_names_order.extend(init_modes & infer_call_names)
        
    #We already implemented a check to ensure that the infer call names are not empty! 

    #Therefore, we just sort and append according to the iteration num of the edit iter. First finding asymmetric
    #set diff.

    edit_names_list = list(infer_call_names.difference(init_modes))
    #Sorting this list.
    edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))
    
    #Extending the infer call ordered list. 
    #
    infer_call_names_order.extend(edit_names_list) 
    
    #Returning it as a tuple so that it is immutable.

    return tuple(infer_call_names_order)

