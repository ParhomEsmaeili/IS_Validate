import os
from os.path import dirname as up
import sys 
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))

from src.prompt_generators.heuristics.heuristic_prompt_utils.point import (
    uniform_random as point_uniform_random,
    center as point_center
)

'''
This file contains a registry of functions which can be used for heuristics based prompt generation.

'''

base_registry = {
    'points':{
    'uniform_random': point_uniform_random,
    'center': point_center,
    },
    'scribbles':{
    'skeletonise': lambda x,y: y
    },
    'bboxes':{
    'jitter': lambda x,y: y
    }
}