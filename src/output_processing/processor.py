import tempfile 
import os 
import torch
import numpy as np
import logging
import warnings 
from typing import Optional, Union

logger = logging.getLogger(__name__)

'''
    This class OR module needs to be able to process the app output in the following capacities:
    Needs to modify the output data dictionary such that it matches expected structure of the interaction state constructors
    Needs to implement the writers, for writing the pred, and logits, then adding the paths for each of these correspondingly to 
    the corresponding written files in the expected structure of the interaction state constructor. 
'''

# NOTE: We use ITK for image saving, hence we convert out RAS orientated tensors into LPS orientation for ITK.

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
        self.class_configs_dict = config_labels_dict 
        self.perm_seg = perm_seg
        self.save_prompt = save_prompts

        #List of paths in the output dictionary that must be on cpu. Each item is in tuple format, index=depth of dict.
        self.check_cpu_info = [('logits'), ('logits_meta_dict', 'affine'), ('pred'), ('pred_meta_dict', 'affine')] 
       
        #Dictionary containing the reference dict (paths), output_dict (paths) and the corresponding checks being examined.

        self.check_integrity_info  = {
            #Checking the number of dims for logits and pred. Must be channelfirst CHW(D) and match the quantity 
            #provided in the input image (which must be loaded as a channel first). Also checking the spatial resolution
            #of the output HW(D) against the input.
            'check_logits':{
                'reference_name':('reference', 'class_configs_dict'),
                'reference_paths':(('image','metatensor'), None),
                'output_paths': (('logits','metatensor'), ('logits', 'metatensor')),
                'checks': (('check_num_dims','check_spatial_res'), ('check_num_channel',)),
                },
            'check_pred':{
                'reference_name':('reference',),
                'reference_paths':(('image','metatensor'),),
                'output_paths': (('pred','metatensor'),),
                'checks': (('check_num_dims','check_spatial_res'),),
            },
            #Checking that the pred/logits meta information (affine array only!) matches that of the input request.
            'check_pred_meta':{
                'reference_name':('reference',),
                'reference_paths':(('image', 'meta_dict', 'affine'),),
                'output_paths':(('pred', 'meta_dict', 'affine'),),
                'checks': (('check_torch_match',),),
            },
            'check_logits_meta':{
                'reference_name':('reference',),
                'reference_paths':(('image', 'meta_dict', 'affine'),),
                'output_paths':(('logits', 'meta_dict', 'affine'),),
                'checks': (('check_torch_match',),),
            } 

            
        }
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
                    return self.extractor(x[y[0]], y[1:]) 

         
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
        
    def check_device(self, 
                    data_dict: dict, 
                    data_path: tuple[Union[str, tuple, int]]):

        item = self.extractor(data_dict, data_path)

        if item.get_device() != -1:
            warnings.warn(f'Careful, the dict: \n {data_dict} \n has item at path: \n {data_path} \n which should be stored on cpu device.')
            #If not on cpu, place it on cpu.
            item = item.to(device='cpu')
            # data_dict.update({key:data_dict[key].to(device='cpu')})
            data_dict = self.dict_path_modif(data_dict, data_path, item)

            # Debug check.
            if self.extractor(data_dict, data_path).get_device() != -1:
                raise Exception(f'The output field {data_path} was not correctly processed to be placed on cpu during output processing')
            
        return data_dict 

    def check_integrity(self,
                        reference_dict: dict,
                        reference_path: Union[tuple, None], 
                        data_dict:dict,
                        data_path:Union[tuple, None],
                        checks:tuple[str]):
        '''
        Function which checks the integrity of a data dict through comparison to a reference dictionary.

        Inputs:
        
        reference_dict: A dictionary containing data which the data_dict will be cross-examined against.
        
        reference_info: An optional tuple containing the path of keys for reaching the reference data being used for cross-examination.
        Index = depth of the dictionary.
        NOTE:(if None, then it defaults to the dict being the reference data)

        data_dict: A dictionary containing data which will be cross-examined by the reference dictionary.
        data_info: A tuple containing the keys for reaching the data being used for cross examination. Index=dict depth.
        NOTE: If None, then it defaults to the data_dict being the data)

        checks: A tuple of keys denoting the points of comparison being used (e.g. spatial_res, meta_info)
        '''

        lambdas = {
        'check_spatial_res' : lambda x,y : x.shape[1:]  == y.shape[1:], # x, y = Channel-first tensors.
        'check_num_dims' : lambda x,y : x.ndim == y.ndim, # x, y = Tensors 
        'check_num_channel' : lambda x,y : x.shape[0] == len(y), # x = Channel-first tensor, y = class labels dict (inclusive of background).
        'check_torch_match': lambda x,y : torch.all(x == y), # x and y = Torch tensors which match in spatial dims.
        'check_npy_match': lambda x,y: np.all(x == y) # x and y = npy arrays which match in spatial dims.
        } 

        #The check_num_dims lambda function is intended to be used for checking that the #of dims match (standard
        #intended use case is for checking if a tensor is channel-first by comparison to one that is,
        # but can be used to check num_spatial dims too)

        try: 
            x = self.extractor(reference_dict, reference_path)
        except:
            raise Exception['The reference dict did not have the right structure, or the tuple provided did not']
        try:
            y = self.extractor(data_dict, data_path)
        except:
            raise Exception['The data dict did not have the right structure, or the tuple provided did not.']
        
        #Testing true for all the provided fields:
        # if not all([lambdas[field](x,y) for field in fields]):
        #     raise Exception['One of the tests failed']
        for check in checks:
            if not lambdas[check](x,y):
                raise Exception(f'Failed test: {check}, for reference: \n {reference_path} \n in \n {reference_dict} \n against data: \n {data_path} in \n {data_dict}')

    def check_output(self, reference_data, output_data):
        '''
        Function which checks whether the output data provided logits, segs, meta info etc, are on cpu. Moves any 
        offending tensors onto cpu.

        Function also performs checks for the integrity of the outputs (logits, pred, logits_meta_dict, 
        pred_meta_dict) match the requirements with respect to input request.

        Performs the following checks:

        Matching image size/resolution.
        Matching image metadata: this will be assessed through comparison of the affine array in the meta_dicts to 
        that which was provided in the input request's affine array in the image meta dictionary.

        '''
        #Performing the checks that the pred, logits, and their meta dict's affine array are on cpu. (or placing them on cpu)
        for field in self.check_cpu_info:
            output_data = self.check_device(output_data, field)
        
        #Performing the checks that the pred, logits, and their meta info match the "pseudo-UI" domain's 
        #expected requirements.

        for item, info in self.check_integrity_info.keys():
            logger.info(f'Performing integrity check on item: \n {item}')
            #We then iterate through each of the items
            for idx, checks_subtuple in enumerate(item['checks']):

                if info['reference'] == 'reference':
                    self.check_integrity(reference_data, info['reference_paths'][idx], output_data, info['output_paths'][idx], checks_subtuple)
                elif info['reference'] == 'class_configs_dicts':
                    self.check_integrity(self.class_configs_dict, info['reference_paths'][idx], output_data, info['output_paths'][idx], checks_subtuple)

    def reformat_output(self, output_data):
        
        '''This function is intended for filling out the strings for the segmentation paths in the output data for 
        forward propagation. '''
        pass 

    def __call__(input_request, output_dict):
        pass 

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