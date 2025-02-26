'''This script takes the existing build scripts implemented by the user, and packages it into a structure where it performs analogous to calling on an application
We roughly adhere to a MONAILabel-like structure.


'''
import importlib
import os
import json
# import torch
import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# import argparse

class build_app:
    def __init__(self, build_app_path: str, init_py: bool):
        '''
        Inputs:
        
        Build_app_path: Str The relative path from "input_application" to the directory which contains "load_app.py" which will initialise the application being used 
        for performing inference
        Init_py: Bool which dictates whether the build folder has an __init__.py file or not (for handling imports).
        '''
        self.build_app_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'input_application', build_app_path)
        
        if not os.path.exists(self.build_app_dir):
            raise ValueError('The provided path does not exist')
        
        if init_py:
        
            MODULE_PATH = os.path.join(self.build_app_dir, "__init__.py")
            MODULE_NAME = "load_app"
            
            spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module 
            spec.loader.exec_module(module)
            
            from load_app import build_app

        else:
            spec = importlib.util.spec_from_file_location("load.app", os.path.join(self.build_app_dir, 'load_app.py'))
            foo = importlib.util.module_from_spec(spec)
            sys.modules["load.app"] = foo
            spec.loader.exec_module(foo)
            build_app = foo.build_app

        self.build_func = build_app


    def build_application(self, build_app_args: dict):
        '''
        Build_app_args: Dict: The arguments required for  initialising the inference application.
        '''

        return self.build_func(build_app_args)

    #Template requests?


if __name__ == '__main__':
    #Debugging.
    build_class = build_app('Sample_TEST/src_validate/build_app', False)
    print(build_class.build_func())
  