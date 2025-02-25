import argparse
import json
import logging 
import sys 
import os 
import datetime 
sys.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


def set_parse():
    # %% set up parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rel_dataset_path', type=str, default='Task10_colon')
    args = parser.parse_args()
    return args


def main():
    pass 





# parser = argparse.ArgumentParser()
# parser.add_argument('-tdp', '--test_data_path', type=str, help='The directory from the parent folder for the repository which contains the testing data')
# parser.add_argument('-ckpt', '--checkpoint_path', type=str)
# parser.add_argument('--output_dir', type=str, default='./')
# parser.add_argument('--task_name', type=str, default='test_amos')
# parser.add_argument('--skip_existing_pred', action='store_true', default=False)
# parser.add_argument('--save_image_and_gt', action='store_true', default=False)
# parser.add_argument('--sliding_window', action='store_true', default=False)

# # parser.add_argument('--image_size', type=int, default=256)
# parser.add_argument('--crop_size', type=int, default=128)
# parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('-nc', '--num_clicks', type=int, default=5)
# parser.add_argument('-pm', '--point_method', type=str, default='default')
# parser.add_argument('-dt', '--data_type', type=str, default='Ts')

# # parser.add_argument('--threshold', type=int, default=0)
# parser.add_argument('--dim', type=int, default=3)
# parser.add_argument('--split_idx', type=int, default=0)
# parser.add_argument('--split_num', type=int, default=1)
# # parser.add_argument('--ft2d', action='store_true', default=False)
# parser.add_argument('--seed', type=int, default=2023)

# args = parser.parse_args()

# ''' parse and output_dir and task_name '''
# args.output_dir = join(args.output_dir, args.task_name)
# args.pred_output_dir = join(args.output_dir, "pred")
# os.makedirs(args.output_dir, exist_ok=True)
# os.makedirs(args.pred_output_dir, exist_ok=True)
# args.save_name = join(args.output_dir, "dice.py")
# print("output_dir set to", args.output_dir)

# SEED = args.seed
# print("set seed as", SEED)
# torch.manual_seed(SEED)
# np.random.seed(SEED)

# # if torch.cuda.is_available():
# #     torch.cuda.init()


# # if __name__ == 

