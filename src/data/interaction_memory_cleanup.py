import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.cleanup import tempfiles_cleanup, temp_dir_cleanup
from typing import Union 

def im_cleanup(
    is_seg_tmp: bool,
    tmp_dir: str,
    im_config: dict,
    im: dict,
    infer_config: dict):
    '''
    Function which takes the existing interaction memory dictionary, and cleans up according to the definition of 
    the interaction memory config.

    Only intended to be called during the editing iterations.

    Input:

    is_seg_tmp: bool - Bool which denotes whether the segmentations are being saved permanently or temporarily
    tmp_dir: path - The absolute path to the temp directory which the temporarynamedfiles will be saved to. 

    im: dict - Interaction memory dictionary, with the individual interaction states denoted by the keys.
    infer_config: dict - Inference config, contains two fields 1) Mode (str), 2) Edit iter num (int)

    Output: 

    im: The cleaned up interaction memory dictionary, according to the interaction memory config provided.
    '''

    if not isinstance(infer_config['edit_num'], int):
        raise TypeError('The editing number must always be provided as an int')
    if not isinstance(infer_config['mode'], str):
        raise TypeError('The inference mode must be provided as a str')
    if infer_config['mode'] is not 'Interactive Edit':
        raise ValueError('The mode must be interactive edit for the cleanup function')

    im_keys = list(im.keys()) 

    if infer_config['edit_num'] == 1:

        #This variable handles whether the initialisation is always being retained, this is treated independent of
        #the editing iterations due to the keys used. 
        if not im_config['keep_init']:
            init_mode_key = [mode for mode in im_keys if 'Init' in mode]
            #NOTE: We do not need to run any cleanup for the saved tmp files because any deletion will be handled by the 
            #subsequent edit iters.
            del im[init_mode_key]
    
    elif infer_config['edit_num'] > 1:

        #This variable handles the quantity of editing iteration states kept in memory. -1 denotes full retention, 
        #any value > 0  indicates the memory length (prior to the final edit interaction state, inclusive of the prior state).

        edit_memory_len = im_config['im_len']
        
        if edit_memory_len == -1:
            #In this circumstance, the -1 flag denotes that full memory length is being used.
            
            #We make this explicit for comprehension. 
            pass

        else: 

            edit_iter_num = infer_config['edit_num'] #Variable denoting which editing iteration this cleanup was called at.

            #Denotes the upper bound (exclusive) for removal of the memory states.
            upper_bound_iter_clip = edit_iter_num - edit_memory_len #Eqn: Iter num - 1 (start from prior state) - memory len + 1 (+1 because inclusive of the prior state)
            
            if upper_bound_iter_clip <= 1:
                pass 
                #In instances where full memory length prior to current iter = memory length param, 
                # upper bound = 1. Hence for upper_bound <= 1, keep all.

            else:
                deletion_set = set([f'Interactive Edit Iter {i}' for i in range(1, upper_bound_iter_clip)]) 
                #Deletion set for all potential iters required to be deleted according to upper bound (non inclusive)

                for key in deletion_set & set(im):
                    #First we run cleanup of the temp files stored for the logits. 
                    logits_tmp_paths = im[key]['prev_logits']['paths']

                    tempfiles_cleanup(tmp_dir=tmp_dir, paths=logits_tmp_paths)
                    
                    #We then run cleanup of the temp files stored for the segs, according to the flag.

                    if is_seg_tmp:
                        seg_tmp_path = im[key]['prev_pred']['path']
                        tempfiles_cleanup(tmp_dir=tmp_dir, paths=seg_tmp_path)

                    #Then we delete the dictionary entry. 
                    del im[key]
                
    else:
        raise ValueError('The edit iter num must be greater than 0 (i.e. only should be called for cleanup on edit iters)')
    return im 
