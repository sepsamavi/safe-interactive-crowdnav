import os
import logging
import sys
import argparse

from RL_test import test

curr_dir = os.getcwd()

parser = argparse.ArgumentParser(description='Override Default Values')
parser.add_argument('--save_dir', type=str, default=os.path.join(curr_dir, "test_logs"), help='location of environment config file')

parser.add_argument('--model_name', type=str, default= "sarl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_3_hmin_3_hmax_3")
parser.add_argument('--version_num', type=int, default=5)
parser.add_argument('--trained_steps', type=int, default=4000)
parser.add_argument('--save_name', type=str, default=None,
                    help='Override default save name which stems from model name.')

args = parser.parse_args()

# configure logging and device
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

stdout_handler = logging.StreamHandler(sys.stdout)
log_file_path = os.path.join(curr_dir, 'logs')
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
i=0
logfile_name = 'debug_log_{:}.log'.format(i)
while os.path.exists(os.path.join(log_file_path, logfile_name)):
    i+=1
    logfile_name = 'debug_log_{:}.log'.format(i)
file_handler = logging.FileHandler(os.path.join(log_file_path, logfile_name), mode='w')
# set logging config for both handlers. Set the level for stdout to INFO and the level for file to DEBUG
stdout_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)
# set the format for both handlers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s, %(levelname)s: %(message)s', handlers=[stdout_handler, file_handler],
                    datefmt='%Y-%m-%d %H:%M:%S')

model_expansion = f"{args.model_name}_{args.version_num}/{args.model_name}_{args.trained_steps}_steps"
model_path = os.path.join(curr_dir, f"logs/{model_expansion}")

# grab configs from model
env_config_file = os.path.join(curr_dir, f"logs/{args.model_name}_{args.version_num}/env.config")
policy_config_file = os.path.join(curr_dir, f"logs/{args.model_name}_{args.version_num}/sarl_policy.config")

test(args.save_dir, args.model_name, model_path, env_config_file, policy_config_file, False)