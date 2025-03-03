# import tempfile
import shutil  
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import numpy as np
from monai.data import MetaTensor 
import logging
import warnings 
from typing import Optional, Union
from src.utils.dict_utils import extractor, dict_path_modif
from src.write_utils.post import WriteOutput


# logger = logging.getLogger(__name__)

'''
    This class needs to be able to process the app output in the following capacities:
    Needs to modify the output data dictionary such that it matches expected structure of the interaction state constructors
    Needs to implement the writers, for writing the pred, and logits, then adding the paths to the output data for each
    of these corresponding written files, in the expected structure of the interaction state constructor. 
'''

class OutputProcessor:
    '''
    Class which initialises the output processing class. Takes as initialisation args:

    base_save_dir: Str - The abspath for the base directory in which all of the metric results and segmentations will be saved
    config_labels_dict: Dict - The class-integer code mapping.
    is_seg_tmp: Bool - A boolean denoting whether the predicted segmentations should be saved as temporary files or permanent files.
    save_prompts: Bool - A boolean denoting whether the input prompts should be saved permanently. 

    '''
    def __init__(
      self,
      base_save_dir:str,
      config_labels_dict:dict[str,int],
      is_seg_tmp:bool = False,
      save_prompts:bool = False, 
    ):

        self.base_save_dir = base_save_dir 
        self.config_labels_dict = config_labels_dict 
        self.is_seg_tmp = is_seg_tmp
        self.save_prompts = save_prompts

        #List of paths in the output dictionary that must be on cpu. Each item is in tuple format, index=depth of dict.
        self.check_cpu_info = [('logits','metatensor'), ('logits', 'meta_dict', 'affine'), ('pred','metatensor'), ('pred','meta_dict', 'affine')] 
       
        #Dictionary containing the reference dict (paths), output_dict (paths) and the corresponding checks being examined.

        self.check_integrity_info  = {
            #Checking the number of dims for logits and pred. Must be channelfirst CHW(D) and match the quantity 
            #provided in the input image (which must be loaded as a channel first). Also checking the spatial resolution
            #of the output HW(D) against the input. Also checking that if the returned obj is a MetaTensor, then the affine must match that of the image.
            'check_logits':{
                'reference_name':('reference', 'config_labels_dict'),
                'reference_paths':(('image','metatensor'), None),
                'output_paths': (('logits','metatensor'), ('logits', 'metatensor')),
                'checks': (('check_num_dims','check_spatial_res', 'check_meta_affine'), ('check_num_channel',)),
                },
            'check_pred':{
                'reference_name':('reference',),
                'reference_paths':(('label','metatensor'),),
                'output_paths': (('pred','metatensor'),),
                'checks': (('check_num_dims','check_spatial_res', 'check_meta_affine'),),
            },
            #Checking that the pred/logits meta dict item (affine array only!) matches that of the input request.
            'check_pred_meta_dict':{
                'reference_name':('reference',),
                'reference_paths':(('image', 'meta_dict', 'affine'),),
                'output_paths':(('pred', 'meta_dict', 'affine'),),
                'checks': (('check_metadict_affine',),),
            },
            'check_logits_meta_dict':{
                'reference_name':('reference',),
                'reference_paths':(('image', 'meta_dict', 'affine'),),
                'output_paths':(('logits', 'meta_dict', 'affine'),),
                'checks': (('check_metadict_affine',),),
            }
        }

        self.pred_writer = WriteOutput(ref_key='label', result_key='pred', dtype=np.int32, compress=False, invert_orient=True, file_ext='.nii.gz')
        self.logits_writer = WriteOutput(ref_key='image', result_key='logits', dtype=np.float32, compress=False, invert_orient=True, file_ext='.nii.gz')

        if self.save_prompts:
            raise NotImplementedError('No class provided for saving prompts')

        #Dict of info regarding the dict-paths for each filepath being placed after the segmentations have been saved.
        self.reformat_dict_info = {
            'logits': ('logits','paths'), 
            'pred': ('pred','path'),
        }        
    def check_device(self, 
                    data_dict: dict, 
                    data_path: tuple[Union[str, tuple, int]]):

        item = extractor(data_dict, data_path)

        if item.get_device() != -1:
            warnings.warn(f'Careful, the dict: \n {data_dict} \n has item at path: \n {data_path} \n which should be stored on cpu device.')
            #If not on cpu, place it on cpu.
            item = item.to(device='cpu')
            # data_dict.update({key:data_dict[key].to(device='cpu')})
            data_dict = dict_path_modif(data_dict, data_path, item)

            # Debug check.
            if extractor(data_dict, data_path).get_device() != -1:
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
        helper_lambdas = {
        'torch_conversion': lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else (x if isinstance(x, torch.Tensor) else None) #Presumes that it can only be torch tensor or numpy array.
        }
        lambdas = {
        'check_spatial_res' : lambda x,y : x.shape[1:]  == y.shape[1:], # x, y = Channel-first tensors.
        'check_num_dims' : lambda x,y : x.ndim == y.ndim, # x, y = Tensors 
        'check_num_channel' : lambda x,y : x.shape[0] == len(y), # x = Channel-first tensor, y = class labels dict (inclusive of background).
        'check_meta_affine': lambda x,y: torch.all(helper_lambdas['torch_conversion'](x.meta['affine']) == helper_lambdas['torch_conversion'](y.meta['affine'])) if isinstance(y, MetaTensor) else True, #x = MetaTensor, y = MetaTensor or torch Tensor
        'check_metadict_affine': lambda x,y: torch.all(helper_lambdas['torch_conversion'](x) == helper_lambdas['torch_conversion'](y)), #x,y are either torch tensors or np arrays
        } 

        #The check_num_dims lambda function is intended to be used for checking that the #of dims match (standard
        #intended use case is for checking if a tensor is channel-first by comparison to one that is,
        # but can be used to check num_spatial dims too)

        try: 
            x = extractor(reference_dict, reference_path)
        except:
            raise Exception['The reference dict did not have the right structure, or the tuple provided did not']
        try:
            y = extractor(data_dict, data_path)
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

        Matching image size/resolution, implicitly checks if the image is channel-first by comparing image size and resolution of the 1: dimensions.
        Matching image metadata: this will be assessed through comparison of the affine array in the meta_dicts to 
        that which was provided in the input request's affine array in the image meta dictionary.

        '''
        #Performing an explicit check that the optional memory item is provided, even if it is a NoneType. Not required
        #for the other fields as they have an expected structure, and any deviations will be flagged through exceptions being 
        #thrown.

        try:
            output_data['optional_memory']
        except:
            warnings.warn('The optional_memory key must be included in the output dictionary, even as a NoneType')
            output_data['optional_memory'] = None 

        #Performing the checks that the pred, logits, and their meta dict's affine array are on cpu. (or placing them on cpu)
        for field in self.check_cpu_info:
            output_data = self.check_device(output_data, field)
        
        #Performing the checks that the pred, logits, and their meta info match the "pseudo-UI" domain's 
        #expected requirements.

        for item, info in self.check_integrity_info.keys():
            print(f'Performing integrity check on item: \n {item}')
            #We then iterate through each of the items
            for idx, checks_subtuple in enumerate(item['checks']):

                if info['reference'] == 'reference':
                    self.check_integrity(reference_data, info['reference_paths'][idx], output_data, info['output_paths'][idx], checks_subtuple)
                elif info['reference'] == 'config_labels_dict':
                    self.check_integrity(self.config_labels_dict, info['reference_paths'][idx], output_data, info['output_paths'][idx], checks_subtuple)

    def reformat_output(self, output_data: dict, pred_path:str, logits_paths:list[str]):
        
        '''
        This function is intended for filling out the fields containing the strings for the segmentation paths in 
        the output data for forward propagation. 
        
        inputs:

        output_data: A dictionary containing the outputs of the prior inference call. 
        
        pred_path: A string denoting the path to the discretised prediction of the prior inference call.
        
        logits_paths: A list of strings denoting the paths to the channel-unrolled logits maps from the prior inference call.
        (in the same order as was provided by the inference call)

        returns: 
        
        output_data with pred_path and logits_paths inserted into the output_data dict in the "pred" and "logits" subdicts.
        '''
        for key, val in self.reformat_dict_info.items():
            if key.title() == 'Logits':
                output_data = dict_path_modif(output_data, val, logits_paths)
            elif key.title() == 'Pred':
                output_data = dict_path_modif(output_data, val, pred_path)
            else:
                raise KeyError('Reformatter info dictionary contained an unsupported key')
        return output_data

    def write_maps(
            self, 
            input_req:dict , 
            output_dict: dict, 
            inf_call_config: dict,
            tmp_dir: str):
        '''
        This function is intended for writing the maps (seg and logits) from the output data to permanent or temp files.
        Also provides the paths to the corresponding files as outputs.

        Inputs:
            input_req: Dict - Dictionary containing the input request for the inference call, contains the information
            regarding the name of the data instance in question, and also the information about the image for performing any necessary re-orientation for writing etc.

            output_dict: Dict - Dictionary containing the prior iteration after having been passed through the output checker
            (also ensures that the corresponding arrays will be on cpu).

            inf_call_config: Dict - A dictionary containing two subfields:
                'mode': str - The mode that the inference call was be made for (Automatic Init, Interactive Init, Interactive Edit)
                'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
            
            tmp_dir: Str - A string denoting the path to the temporary directory. 
        '''

        #First we write the discretised prediction map

        #Call the writer which must output the tempfile path
        tmp_path = self.pred_writer(inf_req=input_req, output_data=output_dict, tmp_dir=tmp_dir)

        if not self.is_seg_tmp:
            img_filename = os.path.split(input_req['image']['path'])[1]
            infer_config_dir = f'{inf_call_config["mode"]} Iter {inf_call_config["edit_num"]}' if inf_call_config['mode'].title() != 'Interactive Edit' else inf_call_config['mode'].title() 
            pred_path = os.path.join(self.base_save_dir, 'segmentations', infer_config_dir, img_filename) 
            
            shutil.move(tmp_path, pred_path)
        else:
            pred_path = tmp_path 

          
        #Now we write the logits maps to a set of tempfiles. 
        logits_paths = self.logits_writer(inf_req=input_req, output_data=output_dict, tmp_dir=tmp_dir)

        return pred_path, logits_paths
    
    def __call__(self, input_request, output_dict, infer_call_config, tmp_dir):
        '''
        Function wraps together the post-processing steps required for checking and writing the segmentations and logits maps.
        and the output dictionary.
        
        input_request: Dict - The request dictionary that was used for performing the inference call.
        
        output_dict: Dict - The returned dictionary from the inference call.

        infer_call_config: Dict - A dictionary containing two subfields:
            'mode': str - The mode that the inference call was made for (Automatic Init, Interactive Init, Interactive Edit)
            'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
        '''
        try:
            self.check_output(input_request, output_dict)
        except:
            Exception('Checking the output data failed due to the aforementioned error')
        try:
            pred_path, logits_paths = self.write_maps(input_req=input_request, output_dict=output_dict, inf_call_config=infer_call_config, tmp_dir=tmp_dir)
        except:
            Exception('Writing the segmentation maps failed due to the aforementioned error')
        try:
            output_dict = self.reformat_output(output_data=output_dict, pred_path=pred_path, logits_paths=logits_paths)
        except:
            Exception('Reformatting the output data dictionary failed due to the aforementioned error')
        
        return output_dict
    
if __name__ == '__main__':

    pass 


    # check_class = OutputProcessor('dummy', 'dummy', 'dummy', 'dummy')

    # #Setting some dummy variables.
    # testing_dict = {'hi':{'hey':1, 'hello':2, 'bye':3}}
    # testing_tuple = ('hi', 'hey')
    # testing_tuple_len1 = ('hi',)

    # #Testing the item extraction function from a dict using a tuple path:
    # print(check_class.extractor(testing_dict, testing_tuple))
    # print(check_class.extractor(testing_dict, testing_tuple_len1))
    # # print(check_class.extractor({}, testing_tuple))

    # #Testing the dict path modification function:
    # print(check_class.dict_path_modif(testing_dict, testing_tuple, 40))
    # print(check_class.dict_path_modif(testing_dict, testing_tuple_len1, 40))

    # check_class.check_output({}, {})