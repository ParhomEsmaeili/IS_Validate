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
from src.general_utils.dict_utils import extractor, dict_path_modif
from src.write_image_utils.post import WriteOutput
# from monai_version_hack import write_segmentation

# logger = logging.getLogger(__name__)

'''
    This class needs to be able to process the app output in the following capacities:
    Needs to modify the output data dictionary such that it matches expected structure of the interaction state constructors
    Needs to implement the writers, for writing the pred, and probs, then adding the paths to the output data for each
    of these corresponding written files, in the expected structure of the interaction state constructor. 
'''

class OutputProcessor:
    '''
    Class which initialises the output processing class. Takes as initialisation args:

    base_save_dir: Str - The abspath for the base directory in which all of the segmentations will be saved
    config_labels_dict: Dict - The class-integer code mapping.
    is_seg_tmp: Bool - A boolean denoting whether the predicted segmentations should be saved as temporary files or permanent files.
    save_prompts: Bool - A boolean denoting whether the input prompts should be saved permanently. 
    write_segmentation: Bool - A temporary hack which overrides the is_seg_tmp flag to bypass the IO operations entirely. 

    '''
    def __init__(
      self,
      base_save_dir:str,
      config_labels_dict:dict[str,int],
      is_seg_tmp:bool = False,
      save_prompts:bool = False, 
      write_segmentation: bool = False
    ):

        self.base_save_dir = base_save_dir 
        self.config_labels_dict = config_labels_dict 
        self.is_seg_tmp = is_seg_tmp
        self.save_prompts = save_prompts
        self.write_segmentation = write_segmentation

        #List of paths in the output dictionary that must be on cpu. Each item is in tuple format, index=depth of dict.
        
        # NOTE: These are checks which do not require a reference for validating the integrity of the output data/i.e., they are
        # static checks based on the output data structure and types.

        #For ensuring that the output data is on cpu device.
        self.check_device_info = [('probs','metatensor'), ('probs', 'meta_dict', 'affine'), ('pred','metatensor'), ('pred','meta_dict', 'affine')] 
        #Checking that the probs and preds are torch objects.
        self.check_obj_type_info = [('probs','metatensor'), ('pred','metatensor')]
        
        #NOTE: These are checks which are intended to ensure that the output data matches expected requirements with respect to
        # some reference data, i.e. it is dynamic according to the experimental configuration or the API request.

        #Dictionary containing the reference dict (paths), output_dict (paths) and the corresponding checks being examined.
        self.check_integrity_info  = {
            #Checking the number of dims for probs and pred. Must be channelfirst CHW(D) and match the quantity 
            #provided in the input image (which must be loaded as a channel first). Also checking the spatial resolution
            #of the output HW(D) against the input. 
            
            #Also checking that if the returned obj is a torch Tensor ONLY! NOTE: Big apologies for any confusion that arises from this
            # I just didn't have the time to refactor all the variable namings to account for removing MetaTensors from the API, 
            # so I just added comments to clarify.

            # Checking that the affine information (the only currently supported meta-information) must match that of the 
            # reference image.
            'check_probs':{
                'reference_name':('reference', 'config_labels_dict'),
                'reference_paths':(('image','metatensor'), None),
                'output_paths': (('probs','metatensor'), ('probs', 'metatensor')),
                #NOTE: check_meta_affine is DEPRECATED since MetaTensor is no longer used at the API level. We exclusively use the metadict to
                # 'checks': (('check_num_dims','check_spatial_res', 'check_meta_affine'), ('check_num_channel',)),
                'checks': (('check_num_dims','check_spatial_res'), ('check_num_channel',)),
                },
            'check_pred':{
                'reference_name':('reference',),
                'reference_paths':(('image','metatensor'),),
                'output_paths': (('pred','metatensor'),),
                #NOTE: check_meta_affine is DEPRECATED since MetaTensor is no longer used at the API level. We exclusively use the metadict to
                # 'checks': (('check_num_dims','check_spatial_res', 'check_meta_affine'),),
                'checks': (('check_num_dims','check_spatial_res'),),
            },
            #Checking that the pred/probs meta dict item (affine array only!) matches that of the input request.
            #
            'check_pred_meta_dict':{
                'reference_name':('reference',),
                'reference_paths':(('image', 'meta_dict', 'affine'),),
                'output_paths':(('pred', 'meta_dict', 'affine'),),
                'checks': (('check_metadict_affine',),),
            },
            'check_probs_meta_dict':{
                'reference_name':('reference',),
                'reference_paths':(('image', 'meta_dict', 'affine'),),
                'output_paths':(('probs', 'meta_dict', 'affine'),),
                'checks': (('check_metadict_affine',),),
            }
        }

        self.pred_writer = WriteOutput(ref_key='image', result_key='pred', dtype=np.int32, compress=False, invert_orient=True, file_ext='.nii.gz')
        self.probs_writer = WriteOutput(ref_key='image', result_key='probs', dtype=np.float32, compress=False, invert_orient=True, file_ext='.nii.gz')

        if self.save_prompts:
            raise NotImplementedError('No class provided for saving prompts')

    def check_device(self, 
                    data_dict: dict, 
                    data_path: tuple[Union[str, tuple, int]]):

        item = extractor(data_dict, data_path)

        if item.get_device() != -1:
            warnings.warn(f'Careful, the output dict has item at path: \n {data_path} \n which should be stored on cpu device.')
            #If not on cpu, place it on cpu.
            item = item.to(device='cpu')
            # data_dict.update({key:data_dict[key].to(device='cpu')})
            data_dict = dict_path_modif(data_dict, data_path, item)

            # Debug check.
            if extractor(data_dict, data_path).get_device() != -1:
                raise Exception(f'The output field {data_path} was not correctly processed to be placed on cpu during output processing')
            
        return data_dict 
    
    def check_obj_type(self, 
                    data_dict: dict, 
                    data_path: tuple[Union[str, tuple, int]]):
        item = extractor(data_dict, data_path)
        if not type(item) == torch.Tensor:
            raise TypeError(f'Expected torch.Tensor ONLY in at path: {data_path}, but got {type(item)}')
        
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

        #Lambdas for performing checks and/or providing helper functions.
        helper_lambdas = {
        'torch_conversion': lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else (x if isinstance(x, torch.Tensor) else None) #Presumes that it can only be torch tensor or numpy array.
        }
        lambdas = {
        'check_spatial_res' : lambda x,y : x.shape[1:]  == y.shape[1:], # x, y = Channel-first tensors.
        'check_num_dims' : lambda x,y : x.ndim == y.ndim, # x, y = Tensors 
        'check_num_channel' : lambda x,y : len(x) == y.shape[0], # x = class labels dict (inclusive of background), y = Channel-first tensor
        #We make a slight adjustment to the affine check because an exact equivalence check can fail due to floating point precision even
        # though the affine matrices are equivalent. We use torch.isclose to check that the values are close enough.
        
        # NOTE: check_meta_affine is DEPRECATED since MetaTensor is no longer used at the API level. We exclusively use the metadict to
        # cross-reference the meta information. 
        # 'check_meta_affine': lambda x,y: torch.all(torch.isclose(helper_lambdas['torch_conversion'](x.meta['affine']), helper_lambdas['torch_conversion'](y.meta['affine']))) if isinstance(y, MetaTensor) else True,        
        'check_metadict_affine': lambda x,y: torch.all(torch.isclose(helper_lambdas['torch_conversion'](x), helper_lambdas['torch_conversion'](y)))
        #It is assumed that the affine information is provided as torch tensor or numpy array.
        } 

        #The check_num_dims lambda function is intended to be used for checking that the #of dims match (standard
        #intended use case is for checking if a tensor is channel-first by comparison to one that is,
        # but can be used to check num_spatial dims too)

        #The x variable in the lambdas are for the reference information, the y variable is for the output
        #data which is being cross-referenced for correctness.
        try: 
            x = extractor(reference_dict, reference_path)
        except:
            raise Exception['The reference dict did not have the right structure, or the tuple provided did not']
        try:
            y = extractor(data_dict, data_path)
        except:
            raise Exception['The data dict did not have the right structure, or the tuple provided did not.']
        
        #Testing true for all the provided fields:
        for check in checks:
            if not lambdas[check](x,y):
                raise Exception(f'Failed test: {check}, for reference: \n {reference_path} \n in \n {reference_dict} \n against data: \n {data_path} in \n {data_dict}')

    def check_output(self, reference_data, output_data):
        '''
        Function which checks whether the output data provided probs, segs, meta info etc, are on cpu. Moves any 
        offending tensors onto cpu.

        Function also performs checks for the integrity of the outputs (probs, pred, probs_meta_dict, 
        pred_meta_dict) match the requirements with respect to input request.

        Performs the following checks:

        Matching image size/resolution, implicitly checks if the image is channel-first by comparing image size and resolution of the 1: dimensions.
        Matching image metadata: this will be assessed through comparison of the affine array in the meta_dicts to 
        that which was provided in the input request's affine array in the image meta dictionary.

        '''

        #Static Checks:

        #Performing the checks that the pred, probs, and their meta dict's affine array are on cpu. (or placing them on cpu)
        for field in self.check_device_info:
            output_data = self.check_device(output_data, field)
        
        #Performing the checks that the pred, probs are torch objects.
        for field in self.check_obj_type_info:
            output_data = self.check_obj_type(output_data, field)

        #####################################################################################################################

        #Dynamic checks (i.e., experiment or API request dependent):

        #Performing the checks that the pred, probs, and their meta info match the "pseudo-UI" domain's 
        #expected requirements.

        for check, check_info in self.check_integrity_info.items():
            print(f'Performing integrity check: \n {check}')
            #We then iterate through each of the items
            for idx, checks_subtuple in enumerate(check_info['checks']):

                if check_info['reference_name'][idx] == 'reference':
                    self.check_integrity(reference_data, check_info['reference_paths'][idx], output_data, check_info['output_paths'][idx], checks_subtuple)
                elif check_info['reference_name'][idx] == 'config_labels_dict':
                    self.check_integrity(self.config_labels_dict, check_info['reference_paths'][idx], output_data, check_info['output_paths'][idx], checks_subtuple)
                else:
                    raise KeyError('Error, the string denoting the reference name is not one of the supported ones.')

    def write_maps(
            self, 
            data_instance:dict,
            case_name: str, 
            output_dict: dict, 
            inf_call_config: dict,
            tmp_dir: str):
        '''
        This function is intended for writing the maps (seg and probs) from the output data to permanent or temp files.
        Also provides the paths to the corresponding files as outputs.

        Inputs:
            data_instance: Dict - Dictionary containing the data instance which provides information about the image for performing any necessary re-orientation 
            for writing etc.

            case_name: str - A string denoting the name of the case/image. Disentangled from the data instance for anonymisation. 
            
            output_dict: Dict - Dictionary containing the prior iteration after having been passed through the output checker
            (also ensures that the corresponding arrays will be on cpu).

            inf_call_config: Dict - A dictionary containing two subfields:
                'mode': str - The mode that the inference call was be made for (Automatic Init, Interactive Init, Interactive Edit)
                'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
            
            tmp_dir: Str - A string denoting the path to the temporary directory. 
        '''

        #First we write the discretised prediction map

        #Call the writer which must output the tempfile path
        tmp_path = self.pred_writer(data_instance=data_instance, output_data=output_dict, tmp_dir=tmp_dir)

        if not self.is_seg_tmp:
            # img_filename = os.path.split(input_req['image']['path'])[1]
            infer_config_dir = f'{inf_call_config["mode"]} Iter {inf_call_config["edit_num"]}' if inf_call_config['mode'].title() == 'Interactive Edit' else inf_call_config['mode'].title() 
            pred_path = os.path.join(self.base_save_dir, 'segmentations', infer_config_dir, case_name + '.nii.gz') 
            
            shutil.move(tmp_path[0], pred_path)
        else:
            pred_path = tmp_path[0] 

          
        #Now we write the probs maps to a set of tempfiles. 
        probs_paths = self.probs_writer(data_instance=data_instance, output_data=output_dict, tmp_dir=tmp_dir)

        return pred_path, probs_paths
    
    def __call__(self, data_instance, case_name, output_dict, infer_call_config, tmp_dir):
        '''
        Function wraps together the post-processing steps required for checking and writing the segmentations and probs maps.
        and the output dictionary.
        
        data_instance: Dict - The dictionary that represents the case instance which was used for generating the input data.

        case_name: Str - The name (without extension) of the case name/image name. 

        output_dict: Dict - The returned dictionary from the inference call.

        infer_call_config: Dict - A dictionary containing two subfields:
            'mode': str - The mode that the inference call was made for (Automatic Init, Interactive Init, Interactive Edit)
            'edit num': Union[int, None] - The current edit's iteration number or None for initialisations.
        '''
        
        self.check_output(data_instance, output_dict)
            
        if self.write_segmentation: #HACK: hacky fix to bypass the segmentation io writing limitations.
            pred_path, probs_paths = self.write_maps(data_instance=data_instance, case_name=case_name, output_dict=output_dict, inf_call_config=infer_call_config, tmp_dir=tmp_dir)
        else:
            #HACK: We put a dummy in here so that the code won't break when trying to perform cleanup operations.
            #We use raw bytes to be extremely fast, faster to do this than refactor the cleanup code for now. 
            with open(os.path.join(tmp_dir, 'dummy_pred.bin'), 'wb') as f:
                f.write(b'dummy!')
            pred_path = os.path.join(tmp_dir, 'dummy_pred.bin') 
            probs_paths = []
            for class_lb in self.config_labels_dict.keys():
                with open(os.path.join(tmp_dir, f'dummy_prob_{class_lb}.bin'), 'wb') as f:
                    f.write(b'dummy')
                probs_paths.append(os.path.join(tmp_dir, f'dummy_prob_{class_lb}.bin'))

        return {'pred':pred_path, 'probs':probs_paths}
    