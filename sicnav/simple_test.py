from copy import deepcopy
import logging
import argparse
import configparser
import os
import pickle
import torch
import numpy as np
import gym

from policy.policy_factory import policy_factory
from crowd_sim_plus.envs.utils.robot_plus import Robot, RobotFullKnowledge
from crowd_sim_plus.envs.policy.orca import ORCA

from crowd_sim_plus.envs.crowd_sim_plus import *

def get_scenario_name(args, env_config_file, policy_config, env, robot):
    hum_policy = env.config.get('humans','policy')
    env_id = env.test_sim + '_N_{:}'.format(args.num_humans)
    if 'sfm' in hum_policy:
        env_id = 'sfm_hums_' + env_id
    if args.policy == 'campc':
        method_name = 'sicnav' if policy_config.get('mpc_env', 'hum_model') == 'orca_casadi_kkt' else 'mpc-cvmm'
        if 'sicnav' in method_name and not robot.policy.priviledged_info:
            method_name += '-np'
        elif 'sicnav' in method_name and robot.policy.priviledged_info:
            method_name += '-p'
        save_dir = os.path.join(os.getcwd(), 'results_'+method_name, env_id)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        limhor_text = 'na' if policy_config.getint('mpc_env', 'orca_kkt_horiz') == 0 else '{:02d}'.format(policy_config.getint('mpc_env', 'orca_kkt_horiz'))
        scenario_name = method_name+'_{:}_{:}_robrad_{:}_K_{:}_Ko_{:}_hmodel_{:}'.format(env_id, robot.policy.ref_type, int(policy_config.getfloat('mpc_env', 'rob_rad_buffer')*100), robot.policy.horiz, limhor_text, policy_config.get('mpc_env', 'hum_model'))
    elif 'dwa' in args.policy:
        save_dir = os.path.join(os.getcwd(), 'results_dwa', env_id)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        scenario_name = 'dwa_{:}_{:}'.format(env_id, args.num_humans)
    elif 'orca_plus' in args.policy:
        save_dir = os.path.join(os.getcwd(), 'results_orca_plus', env_id)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        scenario_name = 'orca_plus_{:}_{:}'.format(env_id, args.num_humans)
    else:
        scenario_name = 'null'


    return scenario_name, env_id, save_dir


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    # configurations for envs and policies
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default=None)

    # this will replace the value in the env_config file
    parser.add_argument('--num_humans', type=int, default=-1, help='NB Overrides value found in --env_config file')

    # For saving and loading experiment results
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--hallway', default=False, action='store_true')
    parser.add_argument('--hallway_opdir', default=False, action='store_true')
    parser.add_argument('--hallway_static', default=False, action='store_true')
    parser.add_argument('--hallway_bottleneck', default=False, action='store_true')
    parser.add_argument('--hallway_squeeze', default=False, action='store_true')
    args = parser.parse_args()

    env_config_file = args.env_config
    policy_config_file = args.policy_config

    # configure logging and device
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('[TEST PHASE] Using device: %s', device)

    # configure environment
    env_config = configparser.RawConfigParser()
    logging.info('[TEST PHASE] env_config_file: {:}'.format(env_config_file))
    env_config.read(env_config_file)
    if args.num_humans != -1:
        env_config.set('sim', 'human_num', args.num_humans)
    else:
        args.num_humans = env_config.getint('sim', 'human_num')

    env_order = gym.make('CrowdSimPlus-v0')
    env = env_order.unwrapped # needed

    env.configure(env_config)
    args.env_config = env_config_file
    args.policy_config = policy_config_file

    if args.square:
        env.test_sim = 'square_crossing'
    elif args.circle:
        env.test_sim = 'circle_crossing'
    elif args.hallway:
        env.test_sim = 'hallway'
    elif args.hallway_opdir:
        env.test_sim = 'hallway_opdir'
    elif args.hallway_static:
        env.test_sim = 'hallway_static'
    elif args.hallway_bottleneck:
        env.test_sim = 'hallway_bottleneck'
    elif args.hallway_squeeze:
        env.test_sim = 'hallway_squeeze'
    else:
        env.test_sim = env_config.get('sim', 'test_sim')

    # configure policy
    if args.policy is not None:
        policy = policy_factory[args.policy]()
        env_config.set('robot', 'policy', args.policy)
    else:
        policy = policy_factory[env_config.get('robot', 'policy')]()
        args.policy = env_config.get('robot', 'policy')
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    policy.configure(policy_config)


    # make robot and set robot in environment
    if args.policy == 'distnav' or args.policy == 'campc' and policy.priviledged_info:
        robot = RobotFullKnowledge(env_config, 'robot')
    else:
        robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # set policy details
    policy.set_phase('test')
    policy.set_device(device)

    if not robot.visible:
        logging.warn('[TEST PHASE] Setting robot to be invisable to simulated humans! Why?! You might as well just use a dataset of real human trajectories if the simulated human cannot see the robot either...')

    if args.policy == 'dwa':
        policy.time_step = env_config.getfloat('env', 'time_step')
        policy.configure_dwa(policy_config, env_config)

    policy.set_env(env)
    robot.print_info()

    scenario_name, env_id, save_dir = get_scenario_name(args, env_config_file, policy_config, env, robot)

    # Get a test case randomly
    if args.test_case is None:
        viz_test_case = np.random.choice(env.case_capacity['test'])
        args.test_case = viz_test_case
    else:
        viz_test_case = args.test_case

    # Set the name
    if args.policy == 'campc':
        tc_name = '{:}_tc_{:}'.format(scenario_name, args.test_case)
        video_name = '{:}.mp4'.format(tc_name)
        pickle_name = '{:}.pkl'.format(tc_name)
        video_file = os.path.join(save_dir, video_name)
    else:
        tc_name ='{:}_tc_{:}'.format(scenario_name, args.test_case)
        video_name = '{:}.mp4'.format(tc_name)
        pickle_name = '{:}.pkl'.format(tc_name)
        video_file = os.path.join(save_dir, video_name)


    ob, static_obs = env.reset('test', viz_test_case, return_stat=True)

    logging.info('[TEST PHASE] About to start test case %i.', viz_test_case)
    done = False

    states = []
    actions_array = []
    min_dist = []
    collision_times = []
    collision_cases = []
    wall_collision_times = []
    wall_collision_cases = []
    frozen_times = []
    danger_log = 0
    coll_log = 0
    wall_coll_log = 0
    frozen_log = 0
    per_step_summary = {
        'step': [0],
        'time': [0.0],
        'danger': [0],
        'coll': [0],
        'wall_coll': [0],
        'frozen': [0],
        'min_dist': [np.inf],
    }
    while not done:
        states.append(robot.get_joint_state(ob, static_obs))
        # if we use the load argument, generate the actions from the previously collected dataset

        action = robot.act(ob, static_obs)

        actions_array.append(action)
        ob, _, done, info = env.step(action)


        if isinstance(action, ActionRot):
            disp_vel = action.v
        elif isinstance(action, ActionXY):
            disp_vel = np.sqrt(action.vx**2 + action.vy**2)
        logging.info('[TEST PHASE] Policy time: {:.2f}, v: {:.3f} => displacement: {:.3f}'.format(env.global_time, disp_vel, disp_vel*env.time_step))

        coll_step = 0
        dang_step = 0
        wall_coll_step = 0
        froz_step = 0
        if info["Danger"].val != 0:
            # min_dist.append(info.min_dist)
            min_dist.append(info["Danger"].min_dist)
            danger_log += 1
            dang_step = 1

        if info["Collision"].val != 0:
            collision_cases.append(viz_test_case)
            collision_times.append(env.global_time)
            coll_log += 1
            coll_step = 1

        if info["WallCollision"].val != 0:
            wall_collision_cases.append(viz_test_case)
            wall_collision_times.append(env.global_time)
            wall_coll_log += 1
            wall_coll_step = 1

        if info["Frozen"].val != 0:
            frozen_times.append([viz_test_case, env.global_time])
            frozen_log += 1
            froz_step = 1

        useful_info = {}
        for key_str in info.keys():
            useful_info[key_str] = info[key_str].val
        per_step_summary['step'].append(env.global_time_step)
        per_step_summary['time'].append(env.global_time)
        per_step_summary['danger'].append(dang_step)
        per_step_summary['coll'].append(coll_step)
        per_step_summary['wall_coll'].append(wall_coll_step)
        per_step_summary['frozen'].append(froz_step)
        per_step_summary['min_dist'].append(info["Danger"].min_dist)
    # END WHILE LOOP
    states.append(robot.get_joint_state(ob, static_obs))

    if useful_info['Timeout'] != 0 and useful_info['ReachGoal'] == 0:
        test_case_success = 0
    elif useful_info['Timeout'] == 0 and useful_info['ReachGoal'] != 0:
        test_case_success = 1
    else:
        test_case_success = -1
        logging.warn('Test case %i is neither timeout nor reach goal. Check the log file for more info.', viz_test_case)

    logging.info('[TEST PHASE] It takes %.2f seconds to finish. Final status is %s', env.global_time,  'Reached Goal' if test_case_success == 1 else 'Timeout' if test_case_success == 0 else 'Unknown')

    if robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('[TEST PHASE] Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))


    if args.policy == 'campc':
        campc_data = {
            'all_x_val' : deepcopy(robot.policy.all_x_val),
            'all_u_val' : deepcopy(robot.policy.all_u_val),
            'all_x_guess' : deepcopy(robot.policy.all_x_guess),
            'all_x_goals' : deepcopy(robot.policy.all_x_goals),
            'all_u_guess' : deepcopy(robot.policy.all_u_guess),
            'all_u_goals' : deepcopy(robot.policy.all_u_goals),
            'all_debug_text' : deepcopy(robot.policy.all_debug_text),
            'mpc_sol_succ' : deepcopy(robot.policy.mpc_sol_succ),
            'calc_times' : deepcopy(robot.policy.calc_times),
            'solver_summary' : deepcopy(robot.policy.solver_summary),
            'mpc_sol_succ_freq': np.sum(np.array(robot.policy.mpc_sol_succ, dtype=np.int32)) / len(robot.policy.mpc_sol_succ),
        }
    else:
        campc_data = {}
        campc_plots = None

    summ_dict = { 'test_case' : viz_test_case,
                'test_case_success' : test_case_success,
                'num_steps' : env.tot_env_steps,
                'nav_time' : env.global_time,
                'num_collisions' : coll_log,
                'num_wall_collisions' : wall_coll_log,
                'num_frozen' : frozen_log,
                'num_too_close' : danger_log,
                'coll_freq': coll_log / env.tot_env_steps,
                'wall_coll_freq': wall_coll_log / env.tot_env_steps,
                'frozen_freq': frozen_log / env.tot_env_steps,
                'too_close_freq' : danger_log / env.tot_env_steps,
                'video_file': video_file,
                'campc/campc_data': campc_data,
                }
    logging.info('test_case_success: {:}'.format(summ_dict['test_case_success']))
    logging.info('num_steps: {:}'.format(summ_dict['num_steps']))
    logging.info('nav_time: {:}'.format(summ_dict['nav_time']))
    logging.info('num_collisions: {:}'.format(summ_dict['num_collisions']))
    logging.info('num_wall_collisions: {:}'.format(summ_dict['num_wall_collisions']))
    logging.info('num_frozen: {:}'.format(summ_dict['num_frozen']))
    logging.info('num_too_close: {:}'.format(summ_dict['num_too_close']))
    logging.info('coll_freq: {:}'.format(summ_dict['coll_freq']))
    logging.info('wall_coll_freq: {:}'.format(summ_dict['wall_coll_freq']))
    logging.info('frozen_freq: {:}'.format(summ_dict['frozen_freq']))
    logging.info('too_close_freq: {:}'.format(summ_dict['too_close_freq']))
    logging.info('video_file: {:}'.format(summ_dict['video_file']))

    with open(os.path.join(save_dir, pickle_name), 'wb') as f:
        pickle.dump(summ_dict, f)

    env.render('video', video_file)

    if video_file is None:
        plt.show()


    print('done')


if __name__ == '__main__':
    main()
