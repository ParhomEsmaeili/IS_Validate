import os
from os.path import dirname as up
import sys 
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))

from src.prompt_generators.heuristics.heuristic_prompt_utils.point import (
    uniform_random as point_uniform_random,
    center as point_center
)
#TODO:
# from src.prompt_generators.heuristics.heuristic_prompt_utils.scribble import (
#     #Add scribble functions here later
    
# )

from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox import (
    bbox_from_binary_mask,
)

# from src.prompt_generators.heuristics.heuristic_prompt_utils.lasso import (
#     #Add lasso functions here later
# )

'''
This file contains a registry of functions which can be used for heuristics based prompt generation.

'''

base_registry = {
    'points':{
    'uniform_random': point_uniform_random,
    'center': point_center,
    },
    'scribbles':{
    'uniform_random': lambda x, y: y, 
    'skeletonise': lambda x,y: y
    },
    'bboxes':{
    'extrema': bbox_from_binary_mask,
    }
}