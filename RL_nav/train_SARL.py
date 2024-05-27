import os
import argparse

from RL_train import train

curr_dir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='Override Default Values')
parser.add_argument('--env_config', type=str, default=os.path.join(curr_dir, "configs/env.config"),
                    help='location of environment config file')
parser.add_argument('--policy_config', type=str, default=os.path.join(curr_dir, "configs/sarl_policy.config"),
                    help='location of SB3 policy training config file')
parser.add_argument('--load_imit_file', type=bool, default=False,
                    help='Set true to use existing imitation learning file')

args = parser.parse_args()

train(args.env_config, args.policy_config, args.load_imit_file)