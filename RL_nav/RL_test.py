import argparse
import configparser
import os
import logging
import sys
import pandas as pd

import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from crowd_sim_plus.envs.utils.robot_plus import Robot, RobotFullKnowledge
from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from RL_nav.configs.policy import SB3_policy


curr_dir = os.getcwd()


phase="test"

def test(save_stats_path, model_name, model_path, env_config_file, policy_config_file, MPC_test):
    summary_df = pd.DataFrame(columns=["Name", "test_case", "num_collisions", "nav_time", "test_case_success", "coll_freq", "frozen_freq", "ID", "campc_config/mpc_env/hum_model", "Created", "num_frozen"])

    # read config files
    env_config = configparser.ConfigParser()
    env_config.read(env_config_file)
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    # create environment
    env = gym.make('CrowdSimPlus-v0')
    env.set_detailed_reward_return(True)
    env.configure(env_config, policy_config=policy_config)

    if MPC_test is False:
        robot = Robot(env_config, 'robot')
        env.set_robot(robot)
        policy = policy_factory[env_config.get('robot', 'policy')]()
        policy.configure(policy_config, env_config)
        robot.set_policy(policy)
        model = DQN.load(model_path, env=env)
        policy.set_attention(True)
        policy.set_model(model)

    else:
        policy = policy_factory["campc"]()
        policy_config = configparser.RawConfigParser()
        policy_config.read(os.path.join(curr_dir,"configs/campc.config"))
        policy.configure(policy_config)

        robot = RobotFullKnowledge(env_config, 'robot')
        robot.set_policy(policy)
        env.set_robot(robot)
        env.set_model("campc")
        policy.set_env(env)

    num_humans_test = [env_config.getint('sim', 'human_num')]

    num_tests = 500

    test_sim = env.test_sim #env.train_val_sim
    save_dir = os.path.join(curr_dir, "test_logs")
    if MPC_test is False:
        save_stats_path=  os.path.join(save_dir, f"sim_{test_sim}_model_{model_name}")
    else:
        save_stats_path=  os.path.join(save_dir, f"sim_{test_sim}_model_MPC")

    # right now only learning one environment at a time
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    else:
        i = 1
        while True:
            temp_save_stats_path = f"{save_stats_path}_{i}"
            if not os.path.exists(temp_save_stats_path):
                os.makedirs(temp_save_stats_path)
                save_stats_path = temp_save_stats_path
                break
            else:
                i += 1

    for num_human in num_humans_test:
        # metric tracking
        success_rate, times_to_end, human_collision_rates, wall_collision_rates, frozen_rates, discomfort_rates = [], [], [], [], [], []

        smallest_time_to_end = 10**10
        largest_time_to_end = 0
        random_sample = np.random.randint(3,num_tests-3)

        for i in range(num_tests):
            # we can add another loop for different testing scenarios like hallway, open space, etc. if we decide to
            test_case = i
            obs, static_obs = env.reset(phase="test", test_case=test_case, testing_human_num=num_human, return_stat=True)
            done = False
            logging.info("[RL_test] Starting test number: {:}".format(i))
            logging.debug("[RL_test] obs: {:}".format(obs))
            # metric tracking
            num_time_steps, success, num_human_collisions, num_wall_collisions, num_frozen, num_discomfort = 0, 0, 0, 0, 0, 0

            while done is False:
                if MPC_test is False:
                    action, _states = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = robot.act(obs, env.static_obstacles)

                logging.debug("[RL_test] action: {:}".format(action))
                obs, rewards, done, info = env.step(action)

                # metric tracking
                if info["ReachGoal"].val != 0:
                    success = 1

                if info["Collision"].val != 0:
                    num_human_collisions += 1
                elif info["WallCollision"].val != 0:
                    num_wall_collisions += 1

                if info["Danger"].val != 0:
                    num_discomfort += 1

                if info["Frozen"].val != 0:
                    num_frozen += 1

                num_time_steps += 1

            time_to_end = env.global_time

            success_rate.append(success)
            times_to_end.append(time_to_end)
            human_collision_rates.append(num_human_collisions)
            wall_collision_rates.append(num_wall_collisions)
            frozen_rates.append(num_frozen/num_time_steps)
            discomfort_rates.append(num_discomfort/num_time_steps)

            logging.info("[RL_test] done test: {:}. Success = {:}, Nav Time: {:.3f}, Num collisions: {:}, Num frozen: {:}".format(i, success, time_to_end, num_human_collisions, num_frozen))

            new_row = {
                "Name": "",
                "test_case": i,
                "num_collisions": num_human_collisions,
                "nav_time": time_to_end,
                "test_case_success": success,
                "coll_freq": num_human_collisions / (time_to_end * env.time_step),
                "frozen_freq": num_frozen / (time_to_end * env.time_step),
                "ID": "",
                "campc_config/: mpc_env/hum_model": "",
                "Created": "",
                "num_frozen": num_frozen
            }

            summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
            summary_df.to_csv(os.path.join(save_stats_path, "summary.csv"))

            if time_to_end < smallest_time_to_end:
                smallest_time_to_end = time_to_end
                video_file = os.path.join(save_stats_path, f"best_{num_human}_humans.mp4" )
                env.render(mode = 'video', output_file=video_file)
            elif time_to_end > largest_time_to_end:
                largest_time_to_end = time_to_end
                video_file = os.path.join(save_stats_path, f"worst_{num_human}_humans.mp4" )
                env.render(mode = 'video', output_file=video_file)

            if i >= random_sample - 3 and i <= random_sample + 3:
                video_file = os.path.join(save_stats_path, f"random{i}_{num_human}_humans.mp4" )
                env.render(mode = 'video', output_file=video_file)


        times_to_end=np.array(times_to_end)
        success_rate = np.array(success_rate)
        human_collision_rates = np.array(human_collision_rates) #*100
        wall_collision_rates = np.array(wall_collision_rates) # *100
        frozen_rates = np.array(frozen_rates)
        discomfort_rates = np.array(discomfort_rates)


        avg_success = np.mean(success_rate)
        avg_time_to_end = np.mean(times_to_end)
        std_time_to_end = np.std(times_to_end)

        avg_human_collision_rate = np.mean(human_collision_rates)
        std_human_collision_rate = np.std(human_collision_rates)

        avg_wall_collision_rate = np.mean(wall_collision_rates)
        std_wall_collision_rate = np.std(wall_collision_rates)

        avg_frozen_rate = np.mean(frozen_rates)
        std_frozen_rate = np.std(frozen_rates)

        avg_discomfort_rate = np.mean(discomfort_rates)
        std_discomfort_rate = np.std(discomfort_rates)

        with open(os.path.join(save_stats_path, f"{num_human}_humans.txt" ), 'w') as f:
            f.write(f'Success Rate: {avg_success:.2f}\n')
            f.write(f'Time to End: {avg_time_to_end:.2f}, {std_time_to_end:.2f}\n')
            f.write(f'Human Collision Rate: {avg_human_collision_rate:.2f}, {std_human_collision_rate:.2f}\n')
            f.write(f'Wall Collision Rate: {avg_wall_collision_rate:.2f}, {std_wall_collision_rate:.2f}\n')
            f.write(f'Frozen Rate: {avg_frozen_rate:.2f}, {std_frozen_rate:.2f}\n')
            f.write(f'Discomfort Rate: {avg_discomfort_rate:.2f}, {std_discomfort_rate:.2f}\n')

        with open(os.path.join(save_stats_path, "env_test.config" ), 'w') as f:
            env_config.write(f)
