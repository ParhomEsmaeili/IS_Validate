'''
This script is intended for running any app planners, if required, for a given application. Similar in concept to the use of nnU-Net's heuristic planner! 
'''

import importlib
import os
import json
# import torch
import sys
import argparse

class build_app_planner:
    def __init__(self, app_planner_path: str, init_py: bool, app_planner_args):
        '''
        Inputs:
        
        app_planner_path: Str: The relative path from "input_application" to the directory which contains the "run_planner.py" in order to perform any planning required.
        init_py: Bool: which dictates whether the build folder has an __init__.py file or not (for handling imports).
        app_planner_args: Dict: The dict which contains the input arguments required for the planner running implementation. 
        '''
        self.build_planner_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'input_application', app_planner_path)
        
        if not os.path.exists(self.build_planner_dir):
            raise ValueError('The relative path provided does not exist.')
        
        if init_py:
        
            MODULE_PATH = os.path.join(self.build_planner_dir, "__init__.py")
            MODULE_NAME = "run_planner"
            
            spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module 
            spec.loader.exec_module(module)
            
            from run_planner import execute_planner

        else:
            spec = importlib.util.spec_from_file_location("runner.app", os.path.join(self.build_planner_dir, 'run_planner.py'))
            foo = importlib.util.module_from_spec(spec)
            sys.modules["runner.app"] = foo
            spec.loader.exec_module(foo)
            execute_planner = foo.execute_planner

        self.planner_func = execute_planner 

    #Template requests?


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_planner_path', type=str, default = 'Sample_TEST/src/build_app_planner', help='The directory from the parent folder for the repository which contains the testing data')
    parser.add_argument('--init_bool', action='store_true', help='Bool which dictates whether the planner build folder contains an __init__ file')
    parser.add_argument('--planner_args_path', type=str, help='Path to the json format which contains all of the input arguments required for the planner script')

    args = parser.parse_args()
    
    run_planner_class = build_app_planner(args.app_planner_path, args.init_bool)
    run_planner_class.planner_func(json.loads(args.planner_args_path))

if __name__ == '__main__':

    main()

    print('\n Planner execution finished')
    