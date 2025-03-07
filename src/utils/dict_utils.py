from typing import Union 

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