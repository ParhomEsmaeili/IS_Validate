import tempfile 
import os 
import torch
import numpy as np
import logging
import warnings 
from typing import Optional, Union

'''
    This class OR module needs to be able to process the app output in the following capacities:
    Needs to modify the output data dictionary such that it matches expected structure of the interaction state constructors
    Needs to implement the writers, for writing the pred, and logits, then adding the paths for each of these correspondingly to 
    the corresponding written files in the expected structure of the interaction state constructor. 
'''

class OutputProcessor:
    '''
    Class which initialises the output processing class.
    '''
    def __init__(
      self,
      save_dir:str,
      config_labels_dict:dict[str,int],
      perm_seg:bool = True,
      save_prompts:bool = False,
    ):

        self.save_dir = save_dir 
        self.config_lbs = config_labels_dict 
        self.perm_seg = perm_seg
        self.save_prompt = save_prompts

        #List of fields in the output dictionary that must be on cpu. Each item is in tuple format, index=depth of dict.
        self.cpu_fields = [('logits'), ('logits_meta_dict', 'affine'), ('pred'), ('pred_meta_dict')] 
    
    def extractor(self, x: dict, y: tuple[Union[tuple,str, int]]): 
        '''
        This general purpose function is adapted from a lambda function which iterates through the dict using the tuple, 
        where the order/index denotes the depth. 
        
        Once tuple is empty it stops (also for NoneType it stops, i.e. just returns the current dict.) and returns the
        item from the provided tuple path. 

        Inputs: 
            x - A dictionary which is populated and will be extracted from
            y - A tuple consisting of the iterative path through the nested dict.
        '''

        if not isinstance(y, tuple):
            raise TypeError('y - path in dict must be a tuple')
        
        if y: #If y exists and we are iterating through it still:
            if not isinstance(x, dict):
                raise TypeError('x must be a dictionary')
            if x == {}:
                raise ValueError('The input dict must be populated otherwise we cannot extract anything.')
            else:
                return self.extractor(x[y[0]], y[1:]) 
        else:
            return x  
         
    def dict_path_modif(self, x: dict, y: tuple[Union[tuple, str, int]], item):
        '''
        This function takes a dictionary (can be nested), and a tuple path, and replaces the value at the given tuple path
        with the provided item.

        Optionally can be used to populate a dictionary also according to a tuple path.
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
            x[y[0]] = self.dict_path_modif(x[y[0]], y[1:], item)
            return x 
        elif len(y) == 1:
            x[y[0]] = item
            return x  
        else:
            raise ValueError('The input tuple must at least be length 1.')

             
        # return x
        
    def check_device(self, 
                    data_dict: dict, 
                    data_info: tuple[Union[str, tuple, int]]):

        item = self.extractor(data_dict, data_info)

        if item.get_device() != -1:
            warnings.warn(f'Careful, the dict: \n {data_dict} \n has item at path: \n {data_info} \n which should be stored on cpu device.')
            #If not on cpu, place it on cpu.
            item = item.to(device='cpu')
            # data_dict.update({key:data_dict[key].to(device='cpu')})
            data_dict = self.dict_path_modif(data_dict, data_info, item)

            # Debug check.
            if self.extractor(data_dict, data_info).get_device() != -1:
                raise Exception(f'The output field {data_info} was not correctly processed to be placed on cpu during output processing')
            
        return data_dict 

    def check_integrity(self,
                        reference_dict: dict,
                        reference_info: Union[tuple, None], 
                        data_dict:dict,
                        data_info:Union[tuple, None],
                        fields:tuple[str]):
        '''
        Function which checks the integrity of a data dict through comparison to a reference dictionary.

        Inputs:
        
        reference_dict: A dictionary containing data which the data_dict will be cross-examined against.
        
        reference_info: An optional tuple containing the keys for reaching the reference data being used for cross-examination.
        Index = depth of the dictionary.
        NOTE:(if None, then it defaults to the dict being the reference data)

        data_dict: A dictionary containing data which will be cross-examined by the reference dictionary.
        data_info: A tuple containing the keys for reaching the data being used for cross examination. Index=dict depth.
        NOTE: If None, then it defaults to the data_dict being the data)

        fields: A tuple of strings denoting the points of comparison being used (e.g. spatial_res, meta_info)
        '''

        lambdas = {
        'check_spatial_res' : lambda x,y : x.shape  == y.shape, # x, y = Channel-first tensors.
        'check_num_dims' : lambda x,y : x.ndim == y.ndim, # x, y = Tensors 
        'check_num_channel' : lambda x,y : x.shape[0] == len(y), # x = Channel-first tensor, y = class labels dict (inclusive of background).
        'check_torch_match': lambda x,y : torch.all(x == y), # x and y = Torch tensors which match in spatial dims.
        'check_npy_match': lambda x,y: np.all(x == y) # x and y = npy arrays which match in spatial dims.
        } 

        #The check_num_dims lambda function is intended to be used for checking that the #of dims match (standard
        #intended use case is for checking if a tensor is channel-first by comparison to one that is,
        # but can be used to check num_spatial dims too)

        try: 
            x = self.extractor(reference_dict, reference_info)
        except:
            raise Exception['The reference dict did not have the right structure, or the tuple provided did not']
        try:
            y = self.extractor(data_dict, data_info)
        except:
            raise Exception['The data dict did not have the right structure, or the tuple provided did not.']
        
        #Testing true for all the provided fields:
        # if not all([lambdas[field](x,y) for field in fields]):
        #     raise Exception['One of the tests failed']
        for field in fields:
            if not lambdas[field](x,y):
                raise Exception(f'Failed test: {field}, for reference: \n {reference_info} \n in \n {reference_dict} \n against data: \n {data_info} in \n {data_dict}')

    def check_output(self, input_data, output_data):
        '''
        Function which checks whether the output data provided logits, segs, meta info etc, are on cpu.

        Moves any offending tensors onto cpu.

        Function also performs checks for the integrity of the outputs.



        Function which checks whether the output data (logits, pred, logits_meta_dict, pred_meta_dict) match the 
        requirements with respect to the image/tensor resolution/size, the metadata. 
        
        The metadata will be assessed through comparison of the affine array in the meta_dict to that which was 
        provided in the input request.
        '''


if __name__ == '__main__':
    check_class = OutputProcessor('dummy', 'dummy', 'dummy', 'dummy')

    #Setting some dummy variables.
    testing_dict = {'hi':{'hey':1, 'hello':2, 'bye':3}}
    testing_tuple = ('hi', 'hey')
    testing_tuple_len1 = ('hi',)

    #Testing the item extraction function from a dict using a tuple path:
    print(check_class.extractor(testing_dict, testing_tuple))
    print(check_class.extractor(testing_dict, testing_tuple_len1))
    # print(check_class.extractor({}, testing_tuple))

    #Testing the dict path modification function:
    print(check_class.dict_path_modif(testing_dict, testing_tuple, 40))
    print(check_class.dict_path_modif(testing_dict, testing_tuple_len1, 40))