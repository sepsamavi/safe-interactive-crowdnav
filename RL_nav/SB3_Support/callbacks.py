import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import shutil

import gym
import numpy as np

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

class CustomCallback(BaseCallback):
    """
    Docmentation of SB3 Custom Class: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

    """
    def __init__(
        self,

        # checkpoint saving
        save_freq: int,
        save_path: str,

        # hyperparameter storage
        env_config_path: str,
        policy_config_path: str,

        # checkpoint saving kwargs
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,

        # Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

        self.env_config_path = env_config_path
        self.policy_config_path = policy_config_path


        self.rate_rewards = ("Collision", "WallCollision", "Frozen", "Danger", "Progress", "AngularSmoothness", "LinearSmoothness")
        self.over_time_rewards = ("ReachGoal", "Timeout")
        
        # to calculate rates of this things during and per an episode:
        self.rate_of_occurence_during_episode = {}
        for reward_type in self.rate_rewards:
            self.rate_of_occurence_during_episode[reward_type] = np.array([0,0])
        
        # will use a rolling average
        self.rate_of_occurence_over_episodes = {}
        for reward_type in self.over_time_rewards:
            self.rate_of_occurence_over_episodes[reward_type] = 0
        


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.
        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_training_start(self) -> None:

        # Store all hyperparameters with model checkpoints
        # Store key hyperparameters with tensorboard
        shutil.copy(self.env_config_path, self.save_path)
        shutil.copy(self.policy_config_path, self.save_path)

        # hparam_dict = {
        #     "algorithm": self.model.__class__.__name__,
        #     "learning rate": self.model.learning_rate,
        #     "gamma": self.model.gamma,
        # }
        # # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # # Tensorbaord will find & display metrics from the `SCALARS` tab
        # metric_dict = {
        #     "rollout/ep_len_mean": 0,
        #     "train/value_loss": 0,
        # }
        # self.logger.record(
        #     "hparams",
        #     HParam(hparam_dict, metric_dict),
        #     exclude=("stdout", "log", "json", "csv"),
        # )


    def _on_rollout_start(self) -> None:

        pass

    def _on_step(self) -> bool:
        self.logger.dump(self.num_timesteps)
        # model saves
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            # if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            #     # If model has a replay buffer, save it too
            #     replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
            #     self.model.save_replay_buffer(replay_buffer_path)
            #     if self.verbose > 1:
            #         print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            # if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            #     # Save the VecNormalize statistics
            #     vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
            #     self.model.get_vec_normalize_env().save(vec_normalize_path)
            #     if self.verbose >= 2:
            #         print(f"Saving model VecNormalize to {vec_normalize_path}")

        # extra tensorboard logging
        # eps_complete = self.locals["infos"][0]["info"]["Done"].val
        eps_complete = self.locals["infos"][0]["Done"].val
        for key, value in self.locals["infos"][0].items():
            if key=="Done" or key == "episode" or key == "terminal_observation":
                continue
            # record raw values
            self.logger.record(key, value.val)

            if key in self.rate_rewards:
                if key != "Progress":
                    if value.val == 0:
                        self.rate_of_occurence_during_episode[key] += np.array([0,1])
                    else:
                        self.rate_of_occurence_during_episode[key] += np.array([1,1])
                else:
                    # For Progress we only want to keep track of positive progress
                    if value.val > 0:
                        self.rate_of_occurence_during_episode[key] += np.array([1,1])
                    else:
                        self.rate_of_occurence_during_episode[key] += np.array([0,1])
                
                occ_rate = self.rate_of_occurence_during_episode[key][0]/self.rate_of_occurence_during_episode[key][1]
                self.logger.record(f"{key}_rate_occ_during_eps", occ_rate)



            if eps_complete:
                if key in self.rate_rewards:
                    self.logger.record(f"{key}_rate_occ_per_eps", occ_rate)
                    # reset it after:
                    self.rate_of_occurence_during_episode[key] = np.array([0,0])

                if key in self.over_time_rewards:
                    update_val = 1 if value.val !=0 else 0
                    self.rate_of_occurence_over_episodes[key] = self.rate_of_occurence_over_episodes[key]*0.95 + update_val*0.05



        # could potentially log videos here

        return True

    def _on_rollout_end(self) -> None:

        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass