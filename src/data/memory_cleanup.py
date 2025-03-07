import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.cleanup import selected_tempfiles_cleanup
from typing import Union 
import warnings 
#NOTE: This is compatible with NoneType interaction states also! It just deletes the state entirely.

def memory_cleanup(
    is_seg_tmp: bool,
    tmp_dir: str,
    im_config: dict,
    im: dict,
    paths: dict,
    infer_config: dict):
    '''
    Function which takes the existing interaction memory dictionary, and cleans up according to the definition of 
    the interaction memory config.

    Only intended to be called during the editing iterations.

    Input:

    is_seg_tmp: bool - Bool which denotes whether the segmentations are being saved permanently or temporarily
    tmp_dir: path - The absolute path to the temp directory which the temporarynamedfiles will be saved to. 

    im_config: The dictionary of configurations which denote how the interaction memory will be clipped.
    im: dict - Interaction memory dictionary, with the individual interaction states denoted by the keys.
    paths: dict - Paths dictionary with the individual interaction states that they correspond to denoted by the keys. 
    infer_config: dict - The current inference config which interaction was generated for, contains two fields 1) Mode (str), 2) Edit iter num (int)

    Output: 

    im: The cleaned up interaction memory dictionary, according to the inference memory config provided.
    paths: The cleaned up paths memory dictionary according to the inference memory config provided. 
    '''

    if not isinstance(infer_config['edit_num'], int):
        raise TypeError('The editing number must always be provided as an int (for edits), we do not call cleanup after the initialisation! Any initialisation cleanup occurs at edit iter 1!')
    if not isinstance(infer_config['mode'], str):
        raise TypeError('The inference mode must be provided as a str')
    if infer_config['mode'] != 'Interactive Edit':
        raise ValueError('The mode must be interactive edit for the cleanup function')

    im_keys = list(im.keys()) 

    if infer_config['edit_num'] == 1:

        #This variable handles whether the initialisation is always being retained, this is treated independent of
        #the editing iterations due to the keys used. 
        if not im_config['keep_init']:
            warnings.warn('Are you sure you want to delete the initialisation information? This may contain information about grounded prompts.')

            deletion_set = set([mode for mode in im_keys if 'Init' in mode])
            if len(deletion_set) != 1:
                raise ValueError('The deletion set can only be 1 long for the handling of initialisation!')
            
            #NOTE: Could comment out: We do not absolutely need to perform this operation below, why? 
            # Because if we delete edit states in im_memory it would eventually go through and deletes the tempfiles using the 
            
            # for key in deletion_set:
            #     logits_tmp_paths = im[key]['prev_logits']['paths'] 
            #     selected_tempfiles_cleanup(tmp_dir=tmp_dir, paths=logits_tmp_paths)
                        
            #     #We then run cleanup of the temp files stored for the segs, according to the flag which handles whether segs are tmp or permanent.
            #     if is_seg_tmp:
            #         seg_tmp_path = im[key]['prev_pred']['path']
            #         selected_tempfiles_cleanup(tmp_dir=tmp_dir, paths=seg_tmp_path)

            del im[tuple(deletion_set)[0]]
            del paths[tuple(deletion_set)[0]]

        if im_config['im_len'] <= 0:
            raise Exception('Cannot delete the interactions of the current interaction state prior to inference call, if you want no prompt conditioning then implement use_mem=False')
    elif infer_config['edit_num'] > 1:

        #This variable handles the quantity of editing iteration states kept in memory. -1 denotes full retention, 
        #any value > 0  indicates the memory length (prior to the final edit interaction state, inclusive of the prior state).

        edit_memory_len = im_config['im_len']
        
        if edit_memory_len == -1:
            #In this circumstance, the -1 flag denotes that full memory length is being used.
            if not im_config['keep_init']:
                raise Exception('Not permitted to delete initialisation states and leave the editing states, little reason to do this.')

            #We make this explicit for comprehension. 
            print('No interaction memory being cleared!')

        else: 
            
            warnings.warn('Are you sure you wanted to delete edit states in im_memory?')
                
            edit_iter_num = infer_config['edit_num'] #Variable denoting which editing iteration this cleanup was called at.
            
            if edit_memory_len <= 0:
                raise Exception('Cannot be removing the interaction information of the current iteration state prior to the inference call, if you want no prompt conditioning then implement use_mem=False')
            
            #Denotes the upper bound (exclusive) for removal of the memory states (inclusive of the current state).
            upper_bound_iter_clip = edit_iter_num - edit_memory_len + 1
            #Eqn: Iter num (start current state) - memory len + 1 (because we are inclusive of the current state in our memory length, e.g. with memory len 2, at iter 5, it should only retain 5,4. Hence upper bound is 4 exclusive)
                
            if upper_bound_iter_clip <= 1:
                print(f'Cannot clear yet, edit state memory retention length is {edit_memory_len} and current iter is {edit_iter_num}')
                #In instances where memory length including current iter = memory length param, 
                # upper bound = 1. Hence for upper_bound <= 1, keep all.

            else:
                deletion_set = set([f'Interactive Edit Iter {i}' for i in range(1, upper_bound_iter_clip)]) 
                #Deletion set for all potential iters required to be deleted according to upper bound (non inclusive)
                
                for key in deletion_set & set(im):
                    #First we run cleanup of the temp files stored for the logits. 
                    logits_tmp_paths = paths[key]['prev_logits']['paths']

                    selected_tempfiles_cleanup(tmp_dir=tmp_dir, paths=logits_tmp_paths)
                    
                    #We then run cleanup of the temp files stored for the segs, according to the flag.

                    if is_seg_tmp:
                        seg_tmp_path = paths[key]['prev_pred']['path']
                        selected_tempfiles_cleanup(tmp_dir=tmp_dir, paths=seg_tmp_path)

                    #Then we delete the dictionary entry. 
                    del im[key]
                    del paths[key]
                
    else:
        raise ValueError('The edit iter num must be greater than 0 (i.e. only should be called for cleanup on edit iters)')
    return im, paths 
