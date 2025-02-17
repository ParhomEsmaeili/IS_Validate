import os
import sys
from typing import Union 

def file_cleanup(path:str):
    '''
    This function is intended to clean up/delete any files if they are no longer required to prevent 
    clutter. Checks whether the path exists first.

    Supports the use of singular path only.
    '''

    if isinstance(path, str):
        #In this circumstance, it is a singular path.
        if os.path.exists(path):
            os.remove(path)
        else:
            raise ValueError('The path did not exist')
    else:
        raise TypeError('The path in file cleanup was not a str')

def temp_dir_cleanup(temp_dir_path:str):
    '''
    This function is intended to clean up/delete a folder after the execution of the validation to 
    prevent clutter. Typically intended for the temp file.

    Checks whether the temp dir exists prior to execution.
    '''
    raise NotImplementedError('Potentially not required.')

def tempfiles_cleanup(tmp_dir: str, 
                    paths: Union[list[str], str]):
    
    if isinstance(paths, str):
        #Checking if path is in the temporary dir, we do not want to delete anything outside of the temporary dir.
        if tmp_dir == os.path.abspath(os.dirname(paths)): 
            file_cleanup(paths) 
        else:
            raise ValueError('Attempting to delete a temp file outside of the defined temp directory')
    
    elif isinstance(paths, list):
        for path in paths:
            if isinstance(path, str):
                #Checking if path is in the temporary dir, we do not want to delete anything outside of the temporary dir.
                if tmp_dir != os.path.abspath(os.dirname(path)): 
                    file_cleanup(path) 
                else:
                    raise ValueError('Attempting to delete a temp file outside of the defined temp directory')
            else:
                raise TypeError('Each path in a list of paths must be a str')
    else:
        raise TypeError('The paths must be in a list, or a singular path str')    