import os
import sys
import tempfile 
from typing import Union 
import warnings 

def file_cleanup(path:str):
    '''
    This function is intended to clean up/delete any files if they are no longer required to prevent 
    clutter. Does not checks whether the path exists first, as there may be instances where a file was deleted ahead
    of time..

    Supports the use of singular path only.
    '''

    if isinstance(path, str):
        #In this circumstance, it is a singular path.
        # if os.path.exists(path):
        #     os.remove(path)
        # else:
        #     raise ValueError('The path did not exist')
        try:
            os.remove(path)
        except:
            warnings.warn('Attempted to perform file cleanup for a file that did not exist. Potentially check that there are no issues')
    else:
        raise TypeError('The path in file cleanup was not a str')

def temp_dir_cleanup(temp_dir:tempfile.TemporaryDirectory):
    '''
    This function is intended to delete a tempdirectory object (which creates a folder on init) after the execution 
    of the validation for each data instance to prevent clutter. 

    Checks whether the temp dir exists prior to execution.
    '''
    
    if not temp_dir:
        raise Exception('The temp_dir obj must exist!')
    else:
        if not os.path.exists(temp_dir.name):
            raise Exception('Tried to run tempdir cleanup with a tempdir which did not exist.')
        else: 
            temp_dir.cleanup()
        
def selected_tempfiles_cleanup(tmp_dir: str, 
                    paths: Union[list[str], str]):
    '''
    Function is intended for cleanup of specific files provided at the given set of paths within the temp dir.
    '''
    if isinstance(paths, str):
        #Checking if path is in the temporary dir, we do not want to delete anything outside of the temporary dir.
        if tmp_dir == os.path.abspath(os.path.dirname(paths)): 
            file_cleanup(paths) 
        else:
            raise ValueError('Attempting to delete a temp file outside of the defined temp directory')
    
    elif isinstance(paths, list):
        for path in paths:
            if isinstance(path, str):
                #Checking if path is in the temporary dir, we do not want to delete anything outside of the temporary dir.
                if tmp_dir != os.path.abspath(os.path.dirname(path)): 
                    file_cleanup(path) 
                else:
                    raise ValueError('Attempting to delete a temp file outside of the defined temp directory')
            else:
                raise TypeError('Each path in a list of paths must be a str')
    else:
        raise TypeError('The paths must be in a list, or a singular path str')    