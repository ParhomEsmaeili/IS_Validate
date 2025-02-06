from monai.transforms import Compose, Orientationd, EnsureChannelFirstd, LoadImaged, ToDeviced 
from monai.data import Dataset, DataLoader
import torch 
import logging 

logger = logging.getLogger(__name__)
def dataset_generator(self, data_dict):
    '''
    This function handles the construction of a dataset object for iterating through.
    '''
    load_transforms = [
        LoadImaged(keys=['image', 'label'], reader="ITKReader", dtype=torch.float32, image_only=False),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ToDeviced(keys=("image", "label"), device=self.args.device)
    ]
    return Dataset(data_dict, load_transforms)
    
