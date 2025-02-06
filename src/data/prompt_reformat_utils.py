import logging
logger = logging.getLogger(__name__)

class PromptReformatter:

    def __init__(self,
                class_config_dict: dict):
        logger.info('Initialising the prompt reformatter')

        self.class_config = class_config_dict 

    def points_reformat(self, prompts, labels):
        pass 

    def scribble_reformat(self, prompts, labels):
        pass 
        # for class_label, class_code in class_label_configs.items():
        # [input[class_pos_idx].clone().detach().tolist() for class_pos_idx in torch.argwhere(points_label[batch_idx] == integer_code)]    
    def bbox_reformat(self, prompts, labels):
        pass 
        
    
    def reformat_prompts(self, prompt_type, prompts: list, prompts_labels: list):
        
        if prompt_type.title() == 'Points':
            self.points_reformat(prompts, prompts_labels)
        elif prompt_type.title() == 'Scribble':
            self.scribble_reformat(prompts, prompts_labels)
        elif prompt_type.title() == 'Bbox':
            self.bbox_reformat(prompts, prompts_labels)
