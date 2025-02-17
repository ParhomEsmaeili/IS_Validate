import logging
import os
import time
import torch
import shutil
import numpy as np
import pandas

def save_csv(args, logger, patient_list,
             loss, loss_nsd,
             ):
    save_predict_dir = os.path.join(args.save_base_dir, 'csv_file')
    if not os.path.exists(save_predict_dir):
        os.makedirs(save_predict_dir)

    df_dict = {'patient': patient_list,
               'dice': loss,
               'nsd': loss_nsd,
               }

    df = pandas.DataFrame(df_dict)
    df.to_csv(os.path.join(save_predict_dir, 'prompt_' + str(args.num_prompts)
                           + '_' + str(args.save_name) + '.csv'), index=False)
    logger.info("- CSV saved")

