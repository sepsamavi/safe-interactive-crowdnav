import configparser
import os
import argparse

import gym
from RL_nav.SB3_models.DQNGeneral import DQNGeneral
from stable_baselines3.common.env_checker import check_env
import torch

from RL_nav.imitation_learning.trainer import Trainer
from RL_nav.imitation_learning.memory import ReplayMemory
from RL_nav.imitation_learning.explorer import Explorer

from crowd_sim_plus.envs.utils.robot_plus import Robot
from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from RL_nav.configs.policy import SB3_policy
from RL_nav.SB3_models import  SARL, SARLNetwork, RGLNetwork, RGL

curr_dir = os.path.dirname(os.path.realpath(__file__))

def train(env_config_file, policy_config_file, load_imit_file):
	# read config files
	env_config = configparser.ConfigParser()
	env_config.read(env_config_file)
	policy_config = configparser.RawConfigParser()
	policy_config.read(policy_config_file)

	# environment set up
	env = gym.make('CrowdSimPlus-v0')
	env.configure(env_config, policy_config=policy_config)

	# robot is just there to move, the action of robot comes from SB3 policy
	robot = Robot(env_config, 'robot')
	env.set_robot(robot)

	# policy set up
	policy = policy_factory[env_config.get('robot', 'policy')]()
	policy.configure(policy_config, env_config)
	robot.set_policy(policy)

	SB3_vars = SB3_policy(env, policy_config, curr_dir)
	policy_kwargs = SB3_vars.get_policy_kwargs()
	checkpoint_callback = SB3_vars.get_custom_callback(env_config_file, policy_config_file)

	# model training
	if SB3_vars.model_name == "sarl":
		init_weights=False
		model_to_init = SARLNetwork(**policy_kwargs)
		env.set_imitation_learning(True)

		init_weights = imitation_learning(model_to_init, policy, SB3_vars.model_name, policy_config, SB3_vars, robot, env, load_imit_file)
		policy_kwargs["SB3_vars"].init_weights = init_weights
		robot.set_policy(policy)
		env.set_imitation_learning(False)
		model = DQNGeneral(SARL, env, verbose=1, gamma=SB3_vars.gamma, exploration_fraction=SB3_vars.exploration_fraction, policy_kwargs=policy_kwargs, learning_rate=SB3_vars.learning_rate, tensorboard_log=SB3_vars.tensorboard_log, buffer_size=10000)

		# policy.set_attention(True)
	elif SB3_vars.model_name == 'rgl':
		model_to_init = RGLNetwork(**policy_kwargs)
		env.set_imitation_learning(True)

		init_weights = imitation_learning(model_to_init, policy, SB3_vars.model_name, policy_config, SB3_vars, robot, env, load_imit_file)
		policy_kwargs["SB3_vars"].init_weights = init_weights
		robot.set_policy(policy)
		env.set_imitation_learning(False)
		model = DQNGeneral(RGL, env, verbose=1, gamma=SB3_vars.gamma, exploration_fraction=SB3_vars.exploration_fraction, policy_kwargs=policy_kwargs, learning_rate=SB3_vars.learning_rate, tensorboard_log=SB3_vars.tensorboard_log, buffer_size=10000)

	else:
		raise ValueError("Model name not recognized")

	policy.set_model(model)
	model.learn(total_timesteps=policy_config.getint('rl', 'total_timesteps'), callback=checkpoint_callback)

def imitation_learning(model_to_init, policy, model_name, policy_config, SB3_vars, robot, env, load_imit_file):
	# load relevant params:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	capacity = policy_config.getint('train', 'capacity')
	model = model_to_init.to(torch.device(device))
	# configure trainer and explorer
	memory = ReplayMemory(capacity)

	il_weight_folder  = os.path.join(curr_dir, f"il_weights")
	if not os.path.exists(il_weight_folder):
		os.makedirs(il_weight_folder)

	il_weight_file = os.path.join(il_weight_folder,f"{model_name}.pth")

	batch_size = policy_config.getint('trainer', 'batch_size')
	trainer = Trainer(model, memory, device, batch_size)
	explorer = Explorer(env, robot, device, memory, policy.gamma)
	il_episodes = policy_config.getint('imitation_learning', 'il_episodes')
	il_policy = policy_config.get('imitation_learning', 'il_policy')
	il_epochs = policy_config.getint('imitation_learning', 'il_epochs')
	il_learning_rate = policy_config.getfloat('imitation_learning', 'il_learning_rate')
	trainer.set_learning_rate(il_learning_rate)
	if robot.visible:
		safety_space = 0
	else:
		safety_space = policy_config.getfloat('imitation_learning', 'safety_space')
	il_policy = policy_factory[il_policy]()
	il_policy.multiagent_training = policy.multiagent_training
	il_policy.safety_space = safety_space
	robot.set_policy(il_policy)

	if load_imit_file is False:
		explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
		trainer.optimize_epoch(il_epochs)
		torch.save(trainer.model.state_dict(), il_weight_file)
		explorer.update_target_model(model)

	return il_weight_file