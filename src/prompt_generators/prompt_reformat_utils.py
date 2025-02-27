import logging
import torch
from typing import Union 

logger = logging.getLogger(__name__)

class PromptReformatter:

    def __init__(self,
                class_config_dict: dict):
        # logger.info('Initialising the prompt reformatter')

        self.class_config = class_config_dict 

    def points_reformat(self, prompts, labels):
        '''
        Reformatter which reformats from the list[tensor] format to the class separated dictionary format.
        
        Inputs: 

        prompts: List of points which are each assumed to be of shape: 1 x N_dim
        labels: List of labels for each of the points, which are each assumed to be of shape: 1. 

        Outputs:

        prompt_reformat: Class separated dict, with each class containing a nested list of points, with each point 
        being represented as a list of coordinates.

        ''' 
        prompt_reformat = dict()
        point_lambda = lambda  p : p[0].clone().detach().to(int).tolist()

        for class_label, class_code in self.class_config.items():
            #We use prompt[0] to be able to extract the spatial coords axis directly.
            prompt_reformat[class_label] = [point_lambda(prompt) for idx, prompt in enumerate(prompts) if labels[idx] == class_code]
        
        return prompt_reformat 
    
    def scribble_reformat(self, prompts, labels):
        '''
        Function which reformats scribbles from the list[torch]-type format (with a separate set of labels) into a dict format
        in which the scribble sets are separated by class, but where the spatial coords are encoded as lists.

        Input:
        
        prompts: Scribbles, which are assumed to be provided as a list of tensors, each tensor is N_s.p x N_dim torch tensors
        encoding the spatial coordinates for the points in the scribble set.

        labels: Scribbles labels, which are assumed to be provided as a list of torch tensors with shape 1 denoting the
        class code.

        Output:
         
        prompt_reformat: Class separated dict, where each class contains a 3-fold nested list: Scribble-Level[Points[i,j,k]] for each class.
        '''

        prompt_reformat = dict() 

        scribble_lambda = lambda prompt : [p.clone().detach().to(int).tolist() for p in prompt]

        for class_label, class_code in self.class_config.items():
            scribbles_list = [scribble_lambda(prompt) for idx, prompt in enumerate(prompts) if labels[idx] == class_code]
            prompt_reformat[class_label] = scribbles_list 

        return prompt_reformat 
    
    def bbox_reformat(self, prompts, labels):
        '''
        Function which reformats the bboxes from list[torch]-type format into the class-separated dict type format.

        Inputs:

        prompts: Bboxes, provided as a list of individual bboxes. Each bbox is represented by a torch tensor with shape 1 x 2 * N_dims.
        with values denoting the extreme values for the bounding box in i_min, i_max, j_min, j_max, k_min, k_max format.

        labels: List of torch tensors denoting the label of the bboxes. 

        prompt_reformat: A class separated dict, with each class having a nested list of structure:
            [bbox-level[i_min, i_max, j_min, j_max, k_min, k_max]]
        '''
        prompt_reformat = dict() 
        bbox_lambda = lambda b : b[0].clone().detach().to(int).tolist() 
        #Same as the point implementation, as this is effectively a set of extreme values.

        for class_label, class_code in self.class_config.items():
            prompt_reformat[class_label] = [bbox_lambda(prompt) for idx, prompt in enumerate(prompts) if labels[idx] == class_code]
        
        return prompt_reformat 
    
    def reformat_prompts(self, prompt_type:str, prompts: Union[list[torch.Tensor], None], prompts_labels: Union[list[torch.Tensor], None]):
        
        if prompts is None and prompts_labels is None:
            return None
        elif type(prompts) != type(prompts_labels):
            raise Exception('The prompts and the prompts labels must be the same datatype (i.e. both lists of tensors, or both Nonetypes)')
        else:
            if prompt_type.title() == 'Points':
                return self.points_reformat(prompts, prompts_labels)
            elif prompt_type.title() == 'Scribbles':
                return self.scribble_reformat(prompts, prompts_labels)
            elif prompt_type.title() == 'Bboxes':
                return self.bbox_reformat(prompts, prompts_labels)
            else:
                raise NotImplementedError('The currently selected prompt type is not supported')

if __name__ == '__main__':
    config_labels_dict = {'tumor':1, 'background':0}

    #Testing basic set ups where each each class has a prompt provided for all prompts.
    points_1 = [torch.Tensor([[1,2,3]]).to(dtype=torch.int32), torch.Tensor([[4,5,6]]).to(dtype=torch.int32)]
    scribbles_1 = [torch.stack([torch.Tensor([1,2,3]).to(dtype=torch.int32), torch.Tensor([4,5,6]).to(dtype=torch.int32)]), 
                   torch.stack([torch.Tensor([7,8,9]).to(dtype=torch.int32), torch.Tensor([10,11,12]).to(dtype=torch.int32)]), torch.stack([torch.Tensor([13,14,15]).to(dtype=torch.int32)])]
    #We introduce variation in the scribble length to ensure that it is compatible with variations in scribbles,
    #not required for other prompts since those have a fixed definition for each instance.

    bboxs_1 = [torch.Tensor([[1,2,3,4,5,6]]).to(dtype=torch.int32),
               torch.Tensor([[7,8,9,10,11,12]]).to(dtype=torch.int32)
            ]
    

    points_lb_1 = [torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([0]).to(dtype=torch.int32)]
    scribbles_lb_1 = [torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([0]).to(dtype=torch.int32), torch.Tensor([1]).to(dtype=torch.int32)] 
    bboxs_lb_1 = [torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([0]).to(dtype=torch.int32)] 

    #Basic setup where not every class has a prompt simulated for each prompt type. 
    points_2 = points_1
    scribbles_2 = scribbles_1 
    bboxs_2 = bboxs_1 

    points_lb_2 = [torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([1]).to(dtype=torch.int32)]
    scribbles_lb_2 = [torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([1]).to(dtype=torch.int32)]
    bboxs_lb_2 = [torch.Tensor([1]).to(dtype=torch.int32), torch.Tensor([1]).to(dtype=torch.int32)] 

    #Basic setup where we set Nonetypes for some prompt types (i.e. no prompt simulation)
    points_3 = points_1
    scribbles_3 = None 
    bboxs_3 = None 

    points_lb_3 = points_lb_1
    scribbles_lb_3 = None 
    bboxs_lb_3 = None 


    # Running the tests/debugging.
    reformatter_class = PromptReformatter(class_config_dict=config_labels_dict)

    print(reformatter_class.reformat_prompts('points', points_1, points_lb_1))
    print(reformatter_class.reformat_prompts('points', points_2, points_lb_2))
    
    print(reformatter_class.reformat_prompts('scribbles', scribbles_1, scribbles_lb_1))
    print(reformatter_class.reformat_prompts('scribbles', scribbles_2, scribbles_lb_2))
    
    print(reformatter_class.reformat_prompts('bboxes', bboxs_1, bboxs_lb_1))
    print(reformatter_class.reformat_prompts('bboxes', bboxs_2, bboxs_lb_2))
    
    
    print(reformatter_class.reformat_prompts('points', points_3, points_lb_3))
    print(reformatter_class.reformat_prompts('scribbles', scribbles_3, scribbles_lb_3))
    print(reformatter_class.reformat_prompts('bboxes', bboxs_3, bboxs_lb_3))
    
    # print(reformatter_class.reformat_prompts('granularity', points_1, points_lb_1))
