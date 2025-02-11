import logging
import torch 
logger = logging.getLogger(__name__)

class PromptReformatter:

    def __init__(self,
                class_config_dict: dict):
        logger.info('Initialising the prompt reformatter')

        self.class_config = class_config_dict 

    def points_reformat(self, prompts, labels):
        
        prompt_reformat = dict()

        for class_label, class_code in self.class_config.items():
            prompt_reformat[class_label] = [prompts[class_pos_idx].clone().detach().tolist() for class_pos_idx in torch.argwhere(points_label[batch_idx] == integer_code)]

    def scribble_reformat(self, prompts, labels):
        pass 
            
    def bbox_reformat(self, prompts, labels):
        pass 
        
    
    def reformat_prompts(self, prompt_type, prompts: list[torch.Tensor], prompts_labels: list[torch.Tensor]):
        
        if prompt_type.title() == 'Points':
            return self.points_reformat(prompts, prompts_labels)
        elif prompt_type.title() == 'Scribbles':
            return self.scribble_reformat(prompts, prompts_labels)
        elif prompt_type.title() == 'Bboxes':
            return self.bbox_reformat(prompts, prompts_labels)


if '__name__' == '__main__':
    points = [torch.Tensor([[1,2,3]]), torch.Tensor([[4,5,6]])]
    scribbles = [[torch.Tensor([[1,2,3]]), torch.Tensor([[4,5,6]])], [torch.Tensor([[7,8,9]]), torch.Tensor([[10,11,12]])]]
    bbox = [[torch.Tensor()], []]