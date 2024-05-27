import logging
import itertools
from copy import deepcopy

import gym
from gym import spaces
import matplotlib.lines as mlines
import numpy as np

# matplotlib imports
from matplotlib import cm as cmap
from matplotlib import animation
import matplotlib.pyplot as plt
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import json

from matplotlib import patches
from numpy.linalg import norm
from crowd_sim_plus.envs.utils.human_plus import Human
from crowd_sim_plus.envs.utils.info_plus import *
from crowd_sim_plus.envs.utils.utils_plus import point_to_segment_dist, closest_point_on_segment, intersection_of_vec_line_and_2p_line, closest_point_on_segment_extended, check_point_on_line_seg, find_intersection_of_static_obstacles, closest_distance_between_line_segments
from crowd_sim_plus.envs.utils.robot_plus import get_vizable_robocentric
from crowd_sim_plus.envs.utils.state_plus import *
from crowd_sim_plus.envs.utils.action import ActionXY, ActionRot

class CrowdSimPlus(gym.Env):

    """CrowdSimPlus is an extension of CrowdSim (https://github.com/vita-epfl/CrowdNav) that additionally simulates impenetrable static obstacles formulated as line segments. CrowdSimPlus also allows for more complex state, observation, and action spaces, and reward functions that are compatible with Stable Baselines 3.
    """

    def __init__(self):
        super().__init__()
        # additional configurations
        # reward function
        self.global_time_step = None
        self.freezing_penalty = None
        # simulation configuration
        self.config = None
        self.rect_width = None
        self.rect_height = None
        self.human_observability = False
        # static obstacles:
        self.static_obstacles = []
        # humans
        self.humans = []

        self.action_space = None
        self.observation_space = None
        self.detailed_reward_return = False
        self.imitation_learning = False

        # total number of time steps that the environment has been updated
        self.tot_env_steps = 0


    def configure(self, config, occlusion_override=False, occlusion_val=False, policy_config=None):
        """_summary_

        :param config: _description_
        :param occlusion_override: _description_, defaults to False
        :param occlusion_val: _description_, defaults to False
        :param policy_config: _description_, defaults to None
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        # SB3 integration
        self.SB3 = config.getboolean('env', "SB3", fallback=False)
        if self.SB3 is True:
            assert policy_config is not None
            self.policy_config = policy_config
            self.SB3_model = policy_config.get('rl', 'model')
            self.robot_input_size = policy_config.getint(self.SB3_model, 'robot_input_size')
            self.human_input_size = policy_config.getint(self.SB3_model, 'human_input_size')

        self.occlusion = occlusion_val if occlusion_override else config.getboolean('env', 'occlusion', fallback=False)
        if self.occlusion:
            raise NotImplementedError
        # From crowd sim:
        self.config = config
        self.time_limit = config.getfloat('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')

        # reward configuration
        supported_reward_types = set(["success_reward", "collision_penalty", "discomfort_penalty_factor", "discomfort_dist", "progress_factor", "timeout", "freezing_penalty", "wall_collision_penalty", "angular_smoothness_factor", "linear_smoothness_factor"])
        # get reward values from env config value
        # get which rewards to support from policy config
        rewards = dict(config.items('reward'))
        self.rewards = {key:float(value) for key, value in rewards.items()}
        if self.detailed_reward_return is not True:
            if self.SB3:
                model_rewards = policy_config.get(self.SB3_model, 'rewards')
                model_rewards = model_rewards.split(',')
                model_rewards = [reward.strip() for reward in model_rewards]
                model_rewards = set(model_rewards)
            else:
                model_rewards = supported_reward_types
            illegal_reward_keys = []
            for reward_type, value in self.rewards.items():
                if reward_type not in supported_reward_types:
                    illegal_reward_keys.append(reward_type)
                    logging.warn(f"Illegal Key Type of: {reward_type} found!")
            [self.rewards.pop(key) for key in illegal_reward_keys]

            self.rewards = {k:v for k,v in self.rewards.items() if k in model_rewards}

        if "discomfort_dist" in self.rewards and "discomfort_penalty_factor" in self.rewards:
            self.rewards["discomfort"] = True
        else:
            # We need some form of discomfort dist when initializing humans in the scene
            self.rewards["discomfort_dist"] = 0.2
            self.rewards["discomfort"] = False

        # For non-RL testing, we want the following reward values to have non-zero values such that they can be detected
        if self.SB3 is False:
            if "timeout" not in self.rewards:
                self.rewards["timeout"] = -1.0
            if "success_reward" not in self.rewards:
                self.rewards["success_reward"] = 1.0
            if "collision_penalty" not in self.rewards:
                self.rewards["collision_penalty"] = -1.0
            if "wall_collision_penalty" not in self.rewards:
                self.rewards["wall_collision_penalty"] = -1.0
            if "freezing_penalty" not in self.rewards:
                self.rewards["freezing_penalty"] = -1.0

        assert (len(self.rewards) > 1)

        if self.config.get('humans', 'policy') == 'orca' or self.config.get('humans', 'policy') == 'orca_plus':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
        elif self.config.get('humans', 'policy') == 'sfm':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                            'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        if self.randomize_attributes:
            logging.info("[CrowdSimPlus] Randomize human's radius and preferred speed")
        else:
            logging.info("[CrowdSimPlus] Not randomize human's radius and preferred speed")
        logging.info('[CrowdSimPlus] Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('[CrowdSimPlus] Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        # My additions
        self.rect_width = config.getfloat('sim', 'rect_width')
        self.rect_height = config.getfloat('sim', 'rect_height')
        self.starts_moving = config.getint('sim', 'starts_moving', fallback=0)
        logging.info('[CrowdSimPlus] Hallway rect width: {}, rect height: {}'.format(self.rect_width, self.rect_height))

        # SB3 training additions
        self.human_num = config.getint('sim', 'human_num')
        logging.info('[CrowdSimPlus] human number: {}'.format(self.human_num))
        # if training with different number of humans is specificed we override
        if self.SB3:
            self.max_human_num = config.getint('sim', 'max_human_num')
            try:
                self.training_schema = json.loads(config.get('sim', 'training_schema'))
                self.training_schema = list(np.arange(self.training_schema[0], self.training_schema[1] + 1))
            except:
                self.training_schema=[self.human_num]
            self.training_schema_rot = 0
            self.num_choices = len(self.training_schema)


            # action space set up
            self.robot_holonomic = config.getboolean('robot', 'holonomic')
            self.speed_samples = config.getint('robot', 'speed_samples')
            self.rotation_samples = config.getint('robot', 'rotation_samples')
            self.rotation_bound_per_second = config.getfloat('robot', 'rotation_bound_per_second', fallback=180)
            self.rotation_bound = self.rotation_bound_per_second * self.time_step
            self.vpref = config.getfloat('robot', 'vpref')

            self.action_space_map = self.build_action_space(self.vpref, self.speed_samples, self.rotation_samples, self.robot_holonomic, self.rotation_bound)
            self.num_actions = len(self.action_space_map)
            self.action_space = spaces.Discrete(self.num_actions)

            # observation space set up
            self.observation_space = self.build_observation_space(self.max_human_num, self.robot_input_size, self.human_input_size)
        else:
            self.training_schema=[self.human_num]
            self.training_schema_rot = 0
            self.num_choices = len(self.training_schema)

    def build_observation_space(self, num_humans, robot_input_size, human_input_size):
        """_summary_

        :param num_humans: _description_
        :param robot_input_size: _description_
        :param human_input_size: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """

        if self.SB3_model == "sarl" or self.SB3_model == "rgl":
            # Note that I do not need the done flag in either the current or propogated states for SARL.
            # If current state is done, we won't need to execute the value function on propogated states
            # If future state is done, that's okay.
            d={}
            # Base state S_t
            d[f'robot_state'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(robot_input_size,), dtype = np.float32)
            for i in range(num_humans):
                d[f'human_state_{i}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(human_input_size,), dtype=np.float32)

            # Future R and states S_t+1 for all actions
            for action_num in range(self.num_actions):
                d[f'robot_state_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(robot_input_size,), dtype = np.float32)
                for i in range(num_humans):
                    d[f'human_state_{i}_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(human_input_size,), dtype=np.float32)
                d[f'reward_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

            # Number of humans in scene (used for occlusions)
            d['num_humans'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int8)

            observation_space=gym.spaces.Dict(d)
            return observation_space
        elif self.SB3_model == "rgl_multistep":
            d={}
            # Base state S_t
            d[f'robot_state'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(robot_input_size,), dtype = np.float32)
            for i in range(num_humans):
                d[f'human_state_{i}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(human_input_size,), dtype=np.float32)

            # Future R and states S_t+1 for all actions
            for action_num in range(self.num_actions):
                d[f'robot_state_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(robot_input_size,), dtype = np.float32)
                for i in range(num_humans):
                    d[f'human_state_{i}_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(human_input_size,), dtype=np.float32)
                d[f'reward_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

            # d = 2. Forecast all possible future states and rewards for 2 time steps
            for key, value in deepcopy(d).items():
                if 'robot_state_' in key:
                    first_action = key.split('_')[-1]
                    for action_num in range(self.num_actions):
                        d[f'robot_state_{first_action}_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(robot_input_size,), dtype = np.float32)
                        for i in range(num_humans):
                            d[f'human_state_{i}_{first_action}_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(human_input_size,), dtype=np.float32)
                        d[f'reward_{first_action}_{action_num}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

            # Number of humans in scene (used for occlusions)
            d['num_humans'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int8)
            observation_space=gym.spaces.Dict(d)
            return observation_space
        elif self.SB3_model == "qsarl":
            d={}
            # Base state S_t
            d[f'robot_state'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(robot_input_size,), dtype = np.float32)
            for i in range(num_humans):
                d[f'human_state_{i}'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(human_input_size,), dtype=np.float32)

            # Number of humans in scene (used for occlusions)
            d['num_humans'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int8)

            observation_space=gym.spaces.Dict(d)
            return observation_space
        else:
            raise NotImplementedError


    def build_action_space(self, v_pref, speed_samples, rotation_samples, holonomic, rotation_bound):
        """ A holonomic robot refers to vx,vy commands. Here a "not holonomic" robot refers to point-turn robot which is controlled by v, theta.

        :param v_pref: _description_
        :param speed_samples: _description_
        :param rotation_samples: _description_
        :param holonomic: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """

        if not holonomic:
            speeds = [(np.exp((i + 1) / speed_samples) - 1) / (np.e - 1) * v_pref for i in range(speed_samples)]
            # convert rotation_bound to radians
            rotation_bound_rads = rotation_bound * np.pi / 180
            rotations = np.linspace(-rotation_bound_rads, rotation_bound_rads, rotation_samples, endpoint=False)

            action_space = [ActionRot(0, 0)]
            for rotation, speed in itertools.product(rotations, speeds):
                action_space.append(ActionRot(speed, rotation))

            action_space_map = {idx:action for idx, action in enumerate(action_space)}
            return action_space_map

        else:
            logging.error("Holonomic robot not implemented for SB3!")
            raise NotImplementedError

    def set_robot(self, robot):
        self.robot = robot

    def set_human_observability(self, human_observability):
        """_summary_

        :param human_observability: _description_
        """
        self.human_observability = human_observability


    def set_human_num(self, human_num):
        """_summary_

        :param human_num: _description_
        """
        self.human_num = human_num


    def generate_static_obstacles(self, rule, static_obstacles=None):
        """_summary_

        :param rule: _description_
        :param static_obstacles: _description_, defaults to None
        :return: _description_
        """
        if static_obstacles is None:
            self.static_obstacles = []
            if rule == 'hallway_static' or rule == 'hallway_static_with_back' or rule == "hallway_bottleneck" or rule == "hallway_squeeze":

                circle_radius = self.circle_radius
                self.door_y_max = circle_radius - self.robot.radius * 2.0
                self.door_y_min = -circle_radius + self.robot.radius * 2.0
                self.door_x_mid = 0.0
                door_y_mid_max = self.door_y_max + (self.door_y_min - self.door_y_max) * 0.40
                self.door_y_mid_max = door_y_mid_max
                door_y_mid_min = self.door_y_max + (self.door_y_min - self.door_y_max) * (1.0 - 0.40)
                self.door_y_mid_min = door_y_mid_min
                self.door_width = 0.5 * self.rect_width if rule == "hallway_squeeze" else 1.0
                door_x_left = self.door_x_mid - self.door_width / 2.0
                door_x_left_mid = door_x_left + ((-self.rect_width * 0.5) - door_x_left) * 0.75
                door_x_right = self.door_x_mid + self.door_width / 2.0
                door_x_right_mid = door_x_right + (self.rect_width * 0.5 - door_x_right) * 0.75

                if rule == "hallway_squeeze":
                    y_mid = 0
                    self.static_obstacles = [
                        # left wall
                        [(-self.rect_width * 0.5, -self.circle_radius*2.5), (door_x_left , y_mid)], #0
                        [(door_x_left , y_mid), (-self.rect_width * 0.5, self.circle_radius*2.5)], #0
                        # right wall
                        [(self.rect_width * 0.5, -self.circle_radius*2.5), (door_x_right, y_mid)], #1
                        [(door_x_right, y_mid), (self.rect_width * 0.5, self.circle_radius*2.5)], #1
                    ]
                else:
                    self.static_obstacles = [
                        # left wall
                        [(-self.rect_width * 0.5, -self.rect_height), (-self.rect_width * 0.5, self.rect_height)], #0
                        # right wall
                        [(self.rect_width * 0.5, -self.rect_height), (self.rect_width * 0.5, self.rect_height)], #1
                    ]
                    if 'hallway_static' in rule:
                        hallway_door = [
                            #left side
                            [(-self.rect_width * 0.5, self.door_y_min), (door_x_left_mid, self.door_y_min)], #6
                            [(door_x_left_mid, self.door_y_min), (door_x_left, door_y_mid_min)], #7
                            [(door_x_left, door_y_mid_min), (door_x_left, door_y_mid_max)], #8
                            [(door_x_left, door_y_mid_max), (door_x_left_mid, self.door_y_max)], #9
                            [(door_x_left_mid, self.door_y_max), (-self.rect_width * 0.5, self.door_y_max)], #10
                            #   right side
                            [ (self.rect_width * 0.5,  self.door_y_min), (door_x_right_mid , self.door_y_min)], #11
                            [(door_x_right_mid , self.door_y_min), (door_x_right , door_y_mid_min)], #12
                            [(door_x_right , door_y_mid_min), (door_x_right , door_y_mid_max)], #13
                            [(door_x_right, door_y_mid_max), (door_x_right_mid, self.door_y_max)], #14
                            [(door_x_right_mid, self.door_y_max), (self.rect_width * 0.5, self.door_y_max)], #15
                        ]
                    elif rule == "hallway_bottleneck":
                        y_mid = 0
                        hallway_door = [
                            [(-self.rect_width * 0.5, y_mid),(door_x_left , y_mid)], #2
                            [(door_x_right, y_mid),(self.rect_width * 0.5, y_mid)], #3
                        ]

                    self.static_obstacles += hallway_door

                    if rule == "hallway_static_with_back":
                        self.static_obstacles += [[(-self.rect_width * 0.5, -self.rect_height*0.5), (self.rect_width * 0.5, -self.rect_height*0.5)], #18
                                                [(-self.rect_width * 0.5, self.rect_height*0.5), (self.rect_width * 0.5, self.rect_height*0.5)],] #19

            elif rule == "hallway":
                self.static_obstacles = [
                    # left wall
                    [(-self.rect_width * 0.5, -self.rect_height), (-self.rect_width * 0.5, self.rect_height)], #0
                    # right wall
                    [(self.rect_width * 0.5, -self.rect_height), (self.rect_width * 0.5, self.rect_height)], #1

                ]

            elif rule == "rectangle":
                self.static_obstacles = [
                    # left wall
                    [(-self.rect_width * 0.5, -self.rect_height * 0.5), (-self.rect_width * 0.5, self.rect_height * 0.5)], #0
                    # right wall
                    [(self.rect_width * 0.5, -self.rect_height * 0.5), (self.rect_width * 0.5, self.rect_height * 0.5)], #1

                    # bottom wall
                    [(-self.rect_width * 0.5, -self.rect_height * 0.5), (self.rect_width * 0.5, -self.rect_height * 0.5)],
                    # top wall
                    [(-self.rect_width * 0.5, self.rect_height * 0.5), (self.rect_width * 0.5, self.rect_height * 0.5)],
                ]
            elif rule == "left_wall":
                self.static_obstacles = [
                    # left wall
                    [(-self.rect_width * 0.5, -self.rect_height * 1000), (-self.rect_width * 0.5, self.rect_height * 1000)], #0
                ]

        else:
            self.static_obstacles = static_obstacles

        return self.static_obstacles


    def generate_random_human_position(self, human_num, rule, rng):
        """
        Generate human position according to certain rule, extend CrowdSim to include more rules.
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human(rng))
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human(rng))
        elif rule == 'hallway' or rule == 'hallway_static' or rule == 'hallway_bottleneck' or rule == 'hallway_squeeze' or rule == "rectangle" or rule == "hallway_static_with_back" or rule == "left_wall" or rule == "no_walls":
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_hallway_human(rng))
                if rule == 'hallway_bottleneck' and self.humans[i].policy.name == 'sfm':
                    self.humans[i].policy.is_bottleneck = True
        else:
            raise ValueError("Rule doesn't exist")


    def generate_circle_crossing_human(self, rng):
        """Generates one circle crossing human

        :param rng: random number generator
        :return: a Human object
        """
        human = Human(self.config, 'humans', self.human_observability, env=self)
        if self.randomize_attributes:
            # human.sample_random_attributes()
            human.v_pref = rng.uniform(0.5, 1.5)
        while True:
            angle = rng.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (rng.random() - 0.5) * human.v_pref
            py_noise = (rng.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.rewards["discomfort_dist"]
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human


    def generate_square_crossing_human(self, rng):
        """Generates one square crossing human

        :param rng: random number generator
        :return: a Human object
        """
        human = Human(self.config, 'humans', self.human_observability)
        if self.randomize_attributes:
            # human.sample_random_attributes()
            human.v_pref = rng.uniform(0.5, 1.5)
        if rng.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = rng.random() * self.square_width * 0.5 * sign
            py = (rng.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = rng.random() * self.square_width * 0.5 * -sign
            gy = (rng.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human


    def generate_hallway_human(self, rng):
        human = Human(self.config, 'humans', self.human_observability, env=self)
        if not (self.config.get('humans', 'policy') == 'orca_plus' or self.config.get('humans', 'policy') == 'sfm'):
            raise RuntimeError("In hallway scenarios, human policy must be orca_plus or sfm, no other human policies supported due to static obstacles")
        effective_rect_height = self.rect_height
        while True:
            if self.randomize_attributes:
                # if sampling random attributes, only set v_pref randomly, not radius
                human.v_pref = rng.uniform(0.5, 1.5)

            # Decide whether the human is travelling up or down (y dir)
            if rng.random() < 0.15:
                dir_sign = 1
            else:
                dir_sign = -1

            # Decide whether the human is walking on the left or on the right (x dir)
            prob_right = 0.8
            if dir_sign > 0:
                right_num = prob_right
            else:
                right_num = 1 - prob_right

            # whether the human is walking on the right or on the left
            if rng.random() < right_num:
                wor_sign = -1 # walk on the right (perspective of y:up)
            else:
                wor_sign = 1 # walk on the left (perspective of y:up)
                        # i.e. walk on the right (perspective of y:down)

            prob_cross = 0.3
            # if the human is on the left, switch with high probability, else don't
            if rng.random() < right_num:
                prob_cross = 1 - prob_cross

            # whether the human crosses right to left or vice versa or not
            if rng.random() < prob_cross:
                cross_sign = -wor_sign
            else:
                cross_sign = wor_sign

            px = (rng.random()) * 0.5 * wor_sign * (self.rect_width - human.radius * 2)
            py = (rng.random()) * 0.25 * dir_sign * self.circle_radius * (effective_rect_height - human.radius * 2)
            collide = False
            for agent in [self.robot]:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.rewards['discomfort_dist']:
                    collide = True
                    break
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius:
                    collide = True
                    break
            if not collide:
                for line in self.static_obstacles:
                    closest_dist = np.abs(point_to_segment_dist(line[0][0], line[0][1], line[1][0], line[1][1], px, py))
                    if closest_dist < (human.radius + 0.01):
                        collide = True
                        break

            if collide:
                effective_rect_height *= 1.1
                continue

            gx = (rng.random()) * 0.5 * cross_sign * (self.rect_width - human.radius * 2)
            gy = (rng.random()) * 0.5 * -dir_sign * self.circle_radius * (effective_rect_height - human.radius * 2)# + -dir_sign * (self.rect_height - human.radius * 2)
            collide = False
            for agent in [self.robot] + self.humans:
            # for agent in [self.robot]:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius:
                    collide = True
                    break
            if not collide:
                for line in self.static_obstacles:
                    closest_dist = np.abs(point_to_segment_dist(line[0][0], line[0][1], line[1][0], line[1][1], gx, gy))
                    if closest_dist < (human.radius):
                        collide = True
                        break
            if not collide:
                break
            else:
                effective_rect_height *= 1.1
        # changed s.t. humans are initialized pointing to their goal.
        human.set(px, py, gx, gy, 0, 0, np.arctan2(gy-py, gx-px))
        return human



    def reset(self, phase='test', test_case=None, return_stat=False, testing_human_num=-1, SB3_override=False):
        """Set px, py, gx, gy, vx, vy, theta for robot and humans

        :param phase: _description_, defaults to 'test'
        :param test_case: _description_, defaults to None
        :param return_stat: _description_, defaults to False
        :param testing_human_num: _description_, defaults to -1
        :param SB3_override: _description_, defaults to False
        :raises AttributeError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        """
        logging.debug("[CrowdSimPlus] In reset")

        # If we're testing, we can override the number of humans
        if testing_human_num != -1:
            self.human_num=testing_human_num
        else:
            # if training, we rotate through # humans in scene
            if phase == 'train':
                self.human_num = self.training_schema[self.training_schema_rot]
                self.training_schema_rot += 1
                self.training_schema_rot = self.training_schema_rot%self.num_choices
        self.phase = phase

        # Set a variable to hold what the current simulation environment is, such that the human agents can be given correct goals.
        if phase == 'train' or phase == 'val':
            self.sim_env = self.train_val_sim
        elif phase == 'test':
            self.sim_env = self.test_sim
        else:
            raise ValueError('Invalid phase: {}'.format(phase))


        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0.0

        self.human_times = [0] * self.human_num

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            # set robot pose (position, orientation), velocity, goal position
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase]) # kept s.t. SARL(via CADRL) seeds are repeatable
                rng = np.random.default_rng(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    # human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_static_obstacles(rule=self.train_val_sim)
                    self.generate_random_human_position(human_num=self.human_num, rule=self.train_val_sim, rng=rng)
                else:
                    self.generate_static_obstacles(rule=self.test_sim)
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim, rng=rng)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.global_times=list()
        self.states = list()
        self.robocentric_states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()


        # initialize to prevent error in self.step before reward initialization
        self.robot_goal_pos = np.array([0,0])
        self.robot_prev_dist_to_goal = 0
        self.prev_action_angular = None
        self.prev_action_linear = None
        # Move the environment forward a few timesteps
        self.global_time = 0.0
        self.global_time_step = 0

        # now run the humans for a few timesteps such that they are moving at the start of the scenario
        if self.starts_moving > 0:
            num_ts = self.starts_moving
            # set the time to negative such that when rendering, the time is correct
            self.global_time = -num_ts*self.time_step
            self.global_time_step = -num_ts
            self.robot_goal_pos = np.array(self.robot.get_full_state(original=True).goal_position)
            if self.robot.kinematics == 'holonomic':
                dummy_action = ActionXY(0.0, 0.0)
            else:
                dummy_action = ActionRot(0.0, 0.0)
            for _ in range(num_ts):
                self.step(dummy_action, dummy_start=True)

        if self.occlusion:
            self.occlusions = list()

        self.prev_action_angular = None
        self.prev_action_linear = None

        # get current observation in world state coords
        if self.robot.sensor == 'coordinates':

            if SB3_override is False and self.SB3:
                robot_world_state = self.robot.get_full_state(original=False)
                # need this to track progress towards goal
                robot_prev_pos = np.array([robot_world_state[0], robot_world_state[1]])
                self.robot_goal_pos = np.array([robot_world_state[5], robot_world_state[6]])
                self.robot_prev_dist_to_goal = np.linalg.norm(robot_prev_pos - self.robot_goal_pos)
                human_state_list = [human.get_observable_state(original=False) for human in self.humans]

                if self.SB3_model == "sarl" or self.SB3_model == "rgl":
                    # For SARL, we need to propogate into the future, so we just keep original states here
                    ob = self.SARL_input_complete(robot_world_state, human_state_list)
                elif self.SB3_model == "rgl_multistep":
                    ob = self.RGL_multistep_input_complete(robot_world_state, human_state_list)
                elif self.SB3_model == "qsarl":
                    ob = self.SARL_input_single_state(robot_world_state, human_state_list)
                elif self.SB3_model == "campc":
                    ob = [human.get_full_state() for human in self.humans]
                else:
                    raise NotImplementedError
            else:
                robot_world_state = self.robot.get_full_state(original=True)
                # need this to track progress towards goal
                robot_prev_pos = np.array(robot_world_state.position)
                self.robot_goal_pos = np.array(robot_world_state.goal_position)
                self.robot_prev_dist_to_goal = np.linalg.norm(robot_prev_pos - self.robot_goal_pos)
                ob = [human.get_observable_state() for human in self.humans]

        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        if return_stat:
            return ob, self.static_obstacles
        else:
            return ob


    def SARL_input_single_state(self, robot_state, human_states, action_index=None, reward=None):
        """_summary_

        :param robot_state: _description_
        :param human_states: _description_
        :param action_index: _description_, defaults to None
        :param reward: _description_, defaults to None
        :return: _description_
        """
        if action_index is None:
            # create current state
            ob = {f'human_state_{i}': human for i, human in enumerate(human_states)}
            ob['robot_state'] = robot_state
            ob["num_humans"] = np.array([self.human_num]) # will change for occlusions
            ob_dummy = {f'human_state_{i}': np.array(self.humans[0].get_observable_state(original=False)) for i in range(self.human_num, self.max_human_num)}
            # complete state space observation
            ob = {**ob, **ob_dummy}
        else:
            # create current state
            ob = {f'human_state_{i}_{action_index}': human for i, human in enumerate(human_states)}
            ob[f'robot_state_{action_index}'] = robot_state
            ob["num_humans"] = np.array([self.human_num]) # will change for occlusions
            ob[f"reward_{action_index}"] = np.array([reward])
            ob_dummy = {f'human_state_{i}_{action_index}': np.array(self.humans[0].get_observable_state(original=False)) for i in range(self.human_num, self.max_human_num)}
            # complete state space observation
            ob = {**ob, **ob_dummy}

        return ob


    def SARL_input_complete(self, robot_state, human_states):
        """_summary_

        :param robot_state: _description_
        :param human_states: _description_
        :return: _description_
        """
        # create current state
        ob = self.SARL_input_single_state(robot_state, human_states)

        # propogate state into future for actions
        for action_index in range(self.num_actions):
            ob_future, _, _, _ = self.step(action_index, update=False)
            ob = {**ob, **ob_future}
        return ob


    def RGL_multistep_input_single_state(self, robot_state, human_states, prev_actions, action_index, reward):
        """_summary_

        :param robot_state: _description_
        :param human_states: _description_
        :param prev_actions: _description_
        :param action_index: _description_
        :param reward: _description_
        :return: _description_
        """
        # construct prepend notation
        if prev_actions is not None:
            prev_a = f""
            for a in prev_actions:
                prev_a += f"{a}_"
        else:
            prev_a = ""

        ob = {f'human_state_{i}_{prev_a}{action_index}': human for i, human in enumerate(human_states)}
        ob[f'robot_state_{prev_a}{action_index}'] = robot_state

        ob[f"reward_{prev_a}{action_index}"] = np.array([reward])
        ob_dummy = {f'human_state_{i}_{prev_a}{action_index}': np.array(self.humans[0].get_observable_state(original=False)) for i in range(self.human_num, self.max_human_num)}
        # complete state space observation
        ob = {**ob, **ob_dummy}
        return ob


    def RGL_multistep_input_complete(self, robot_state, human_states):
        """_summary_

        :param robot_state: _description_
        :param human_states: _description_
        :return: _description_
        """
        # grab current state
        ob = self.SARL_input_single_state(robot_state, human_states)

        # propogate state into future for actions
        for action_index in range(self.num_actions):
            self.prev_actions = None
            ob_future, _, _, _ = self.step(action_index, update=False)
            ob = {**ob, **ob_future}

        # propogate state into future another time for multistep.
        for key in deepcopy(ob):
            if 'robot_state_' in key:
                first_action = key.split('_')[-1]
                self.prev_actions = [first_action]
                for action_index in range(self.num_actions):
                    ob_future, _, _, _ = self.step(action_index, update=False)
                    ob = {**ob, **ob_future}
        return ob


    def constrain_agent_action_exact(self, agent, action):
        """ Constrain agent action to not move throuh static obstacles

        :param agent: _description_
        :param action: _description_
        :return: _description_
        """
        curr_x, curr_y = agent.px, agent.py
        fut_x, fut_y = agent.compute_position(action, agent.time_step)
        cur_pos = np.array([curr_x, curr_y])
        fut_pos = np.array([fut_x, fut_y])
        movement_dir = fut_pos - cur_pos
        movement_mag = np.linalg.norm(movement_dir)

        # find lines that are in collision with agent
        collision_lines = []
        for idx, line in enumerate(self.static_obstacles):
            start_x, start_y = line[0]
            end_x, end_y = line[1]

            # Find closest distance between agent travel and line segment
            pA, pB, closest_distance = closest_distance_between_line_segments(np.array([start_x,start_y,0]), np.array([end_x,end_y,0]), np.array([curr_x, curr_y, 0]), np.array([fut_x, fut_y, 0]))
            # pA is the closest point on the line-segment obstacle and pB is the closest point on the agent's trajectory when following the action
            if closest_distance - agent.radius < 0.0:
                collision_lines.append((line, closest_distance, pA, pB, idx))

        final_action = deepcopy(action)
        # constrain against each line
        for line, closest_distance, pA, pB, idx in collision_lines:
            start_x, start_y = line[0]
            end_x, end_y = line[1]
            r = agent.radius

            # find whether the collision is with middle of line or end-point of line
            # i.e. if closest point on obstacle line is one of the endpoints, and also
            if (np.linalg.norm(pA[:2] - np.array(line[0])) < 1e-8 or np.linalg.norm(pA[:2] - np.array(line[1])) < 1e-8) and np.linalg.norm(pA - pB) > 1e-8:
                # This means that the collision is against an end-point of the line segment
                direction_vec = pB[:2] - cur_pos # vector from current position to the closest point on the direction of travel to the end-point of the line-segment
                dir_mag = np.linalg.norm(direction_vec)
                if dir_mag > 0.0 and np.linalg.norm(pA[:2] - cur_pos) - r < 1e-4 and np.dot(movement_dir, pA[:2] - cur_pos) > -1e-8:
                    # first check if the current position is already touching the closest point,
                    # and if so, then whether it is moving in a forbidden direction i.e. direction
                    # is not at least pi/2 away from the line-segment end-point.
                    # If so then do not move at all.
                    _direction_vec = direction_vec / dir_mag
                    redux = dir_mag
                elif dir_mag > 0.0:
                    # if the robot displacement is greater than 0, then we need to constrain the agent to move to the closest point on the line
                    _direction_vec = direction_vec / dir_mag

                    # Let pC be the constrained position of the agent and let p0 be the position of the agent before taking its action.
                    # We need to find the vector ||pC-pB||. We know that ||pC-pA|| = r, and we know that pB-pC is colinear with direction_vec.
                    # So we can solve for the triangle between pB, pA, and pC, using https://en.wikipedia.org/wiki/Solution_of_triangles.
                    # Let alpha be the angle at pB, beta be the angle at pA, and gamma be the angle at pC in the triangle pB-pA-pC.
                    # First, find alpha: the angle between -direction_vec (i.e. the vector from pB to pC) and the vector from pB to pA
                    arccos_value = -direction_vec.T @ (pA[:2] - pB[:2]) / (dir_mag * closest_distance)

                    # N.B. alpha can be pi when the agent is heading directly towards the line-segment end-point, meaning alpha = pi.
                    # In order to ensure numerical stability in the alpha = pi case we use clipping to avoid arccos_value > 0.
                    clipped_arccos_value = np.clip(arccos_value, -1.0, 1.0)
                    alpha = np.arccos(clipped_arccos_value)
                    if alpha == np.pi:
                        # this means that the line from p0 through pC and pB also goes through pA. Thus the angle alpha = pi, and beta=gamma=0
                        redux = r - closest_distance
                    else:
                        # The agent's heading is NOT directly towards the corner and we need to solve the triangle
                        assert alpha > 0
                        # then find gamma using the law of sines
                        gamma = np.arcsin(closest_distance * np.sin(alpha) / r)
                        assert gamma >= 0
                        beta = np.pi - alpha - gamma
                        assert beta >= 0
                        # now the reduction in the direction_Vec is the side opposite beta in the triangle
                        redux = r * np.sin(beta) / np.sin(alpha) + 1e-7
                else:
                    # in the case that dir_mag is 0 then the agent's action has no movement and we do not need to reduce anything.
                    redux = 0.0
                    _direction_vec = direction_vec
                final_position = cur_pos + _direction_vec * max(dir_mag - redux, 0)
            else:
                # If the closest point on the line is NOT one of the corners, we need to constrain the agent against the obstacle line-segment, as if it were extending to infinity
                cl_x, cl_y = closest_point_on_segment_extended(line[0][0], line[0][1], line[1][0], line[1][1], cur_pos[0], cur_pos[1])
                pA_cl = np.array([cl_x, cl_y, 0.0])
                if movement_mag > 0.0 and np.linalg.norm(pA_cl[:2] - cur_pos) - r < 1e-4 and np.dot(movement_dir, pA_cl[:2] - cur_pos) > -1e-8:
                    # first check if the current position is already touching the closest point,
                    # and if so, then whether it is moving in a forbidden direction i.e. direction
                    # is not at least pi/2 away from the line-segment end-point.
                    # If so then do not move at all.
                    final_position = cur_pos
                elif movement_mag > 0.0:
                    # if the robot displacement is greater than 0, then we need to constrain the agent to move to touch the closest point on the line
                    int_x, int_y = intersection_of_vec_line_and_2p_line(cur_pos[0], cur_pos[1], movement_dir[0], movement_dir[1], line[0][0], line[0][1], line[1][0], line[1][1])
                    d_vec = np.array([int_x - cur_pos[0], int_y - cur_pos[1]])
                    dc_0 = np.sqrt((cur_pos[0]-cl_x)**2 + (cur_pos[1]-cl_y)**2)
                    des_scaling = (dc_0 - (r+1e-7)) / dc_0
                    des_scaling = max(0.0, des_scaling)
                    final_position = cur_pos + d_vec * des_scaling
                else:
                    final_position = cur_pos

            # Make the new action
            if isinstance(action, ActionXY):
                v_x = (final_position[0] - curr_x) / agent.time_step
                v_y = (final_position[1] - curr_y) / agent.time_step
                potential_new_action = ActionXY(v_x, v_y)
                if (v_x**2+v_y**2) < (final_action.vx**2 + final_action.vy**2):
                    final_action = potential_new_action
            else:
                if action.v > 0:
                    v = np.linalg.norm(final_position - cur_pos) / agent.time_step
                    potential_new_action = ActionRot(v, action.r)
                    # self.render('human', output_file='./temp2.png')
                    if v < final_action.v:
                        final_action = potential_new_action
                else:
                    v = -np.linalg.norm(final_position - cur_pos) / agent.time_step
                    potential_new_action = ActionRot(v, action.r)
                    if v > final_action.v:
                        final_action = potential_new_action

        return final_action


    def set_imitation_learning(self, value):
        """_summary_

        :param value: _description_
        """
        self.imitation_learning = value


    def set_detailed_reward_return(self, value):
        """_summary_

        :param value: _description_
        :return: _description_
        """
        self.detailed_reward_return = value


    def set_SB3(self, value):
        """_summary_

        :param value: _description_
        """
        self.SB3 = value


    def set_model(self, value):
        """_summary_

        :param value: _description_
        """
        self.SB3_model = value


    def step(self, action, update=True, dummy_start=False):
        """Compute actions for all agents, detect collisions, update environment and return (ob, reward, done, info)
        Extend CrowdSim to (1) account for robot freezing, (2) continue towards the goal after a collision, and
        the biggest change (3) detect collisions with static obstacles modelled as 2D lines and also do not allow
        the agents to move through for static obstacles.

        :param action: action of the robot
        :param update: If true, update the robot & human states in the step otherwise, defaults to True
        :param dummy_start: If true, do not count the step in self.tot_env_steps even if update=True, defaults to False
        :return: (ob, reward, done, info) tuple containing the next observation, the reward, whether the episode is done, and additional information
        """


        if not (isinstance(action, ActionXY) or isinstance(action, ActionRot)):
            # map action index of model to actual action
            action_index = action
            action = self.action_space_map[action]

        # get human actions
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state(original=True)]
            else:
                if self.mode == 'test':
                    logging.warn('[CrowdSimPlus] Robot is not visible to humans. This is not realistic for testing. Please set robot.visible to True.]')
            human_action = human.act(ob, self.static_obstacles)
            constrained_human_action = self.constrain_agent_action_exact(human, human_action)
            human_actions.append(constrained_human_action)

       # get constrained robot action + check for static collision
        constrained_robot_action =self.constrain_agent_action_exact(self.robot, action)
        if isinstance(action, ActionXY):
            stat_collision = True if action.vx != constrained_robot_action.vx else False
        else:

            stat_collision = True if action.v != constrained_robot_action.v else False
        action = constrained_robot_action


        # collision detection between robot and humans
        dmin = float('inf')
        collision = False
        x1_rob, y1_rob = self.robot.compute_position(action, self.time_step)
        for i, human in enumerate(self.humans):
            # check if robot and human collide
            x1_hum, y1_hum = human.compute_position(human_actions[i], self.time_step)
            closest_dist = np.linalg.norm(np.array([x1_rob,y1_rob]) - np.array([x1_hum, y1_hum]))
            if closest_dist < (self.robot.radius + human.radius):
                collision = True
                # logging.debug("[CrowdSimPlus] Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if the robot has frozen
        frozen_robot = False
        if self.robot.kinematics == 'holonomic':
            frozen_robot = np.sqrt(action.vx ** 2 + action.vy ** 2) * self.time_step < 0.01
        else:
            frozen_robot = np.abs(action.v * self.time_step) < 0.01

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reached_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        # check progress to goal
        curr_dist_to_goal = np.linalg.norm(self.robot_goal_pos - end_position)

        info = {}
        reward = 0
        done = False

        set_of_rewards = ("ReachGoal", "Timeout", "Collision", "WallCollision", "Frozen", "Danger", "AngularSmoothness", "LinearSmoothness")

        if (self.detailed_reward_return or "success_reward" in self.rewards) and reached_goal:
            success_reward = self.rewards["success_reward"]
            reward += success_reward
            info["ReachGoal"] = ReachGoal(success_reward)
            done = True

        elif self.global_time >= self.time_limit:

            if (self.detailed_reward_return or "timeout" in self.rewards):
                timeout_reward = self.rewards["timeout"]
                reward += timeout_reward
                info["Timeout"] = Timeout(timeout_reward)
            done = True


        if (self.detailed_reward_return or "collision_penalty" in self.rewards) and collision:
            collision_penalty = self.rewards["collision_penalty"]
            reward += collision_penalty
            info["Collision"] = Collision(collision_penalty)

        if (self.detailed_reward_return or "wall_collision_penalty" in self.rewards) and stat_collision:
            wall_collision_penalty = self.rewards["wall_collision_penalty"]
            reward += wall_collision_penalty
            info["WallCollision"] = WallCollision(wall_collision_penalty)

        if (self.detailed_reward_return or self.rewards["discomfort"] is True) and dmin < self.rewards["discomfort_dist"]:
            discomfort_reward = (dmin - self.rewards["discomfort_dist"]) * self.rewards["discomfort_penalty_factor"] * self.time_step
            reward += discomfort_reward
            info["Danger"] = Danger(discomfort_reward, dmin)

        if (self.detailed_reward_return or "progress_factor" in self.rewards):
            progress_reward = (self.robot_prev_dist_to_goal - curr_dist_to_goal)*self.rewards["progress_factor"]
            reward += progress_reward
            info["Progress"] = Progress(progress_reward)
            if update:
                self.robot_prev_dist_to_goal = curr_dist_to_goal

        if (self.detailed_reward_return or "freezing_penalty" in self.rewards) and frozen_robot:
            freezing_penalty = self.rewards["freezing_penalty"]
            reward += freezing_penalty
            info["Frozen"] = Frozen(freezing_penalty)

        if (self.detailed_reward_return or "angular_smoothness_factor" in self.rewards):
            if self.prev_action_angular is None:
                self.prev_action_angular = np.arctan2(action.vy, action.vx) if isinstance(action, ActionXY) else action.r
            else:
                curr_angular = np.arctan2(action.vy, action.vx) if isinstance(action, ActionXY) else action.r
                # if holonomic, we need to compute diff in theta of prev and curr action. If point-turn then it's just the angular change
                angular_diff = np.abs(curr_angular - self.prev_action_angular) if isinstance(action, ActionXY) else curr_angular * self.time_step
                #angular_smoothness_reward = np.abs(self.prev_action_angular - curr_angular) * self.rewards["angular_smoothness_factor"]
                angular_smoothness_reward = np.abs(angular_diff) * self.rewards["angular_smoothness_factor"]
                self.prev_action_angular = curr_angular
                reward += angular_smoothness_reward
                info["AngularSmoothness"] = AngularSmoothness(angular_smoothness_reward)

        if (self.detailed_reward_return or "linear_smoothness_factor" in self.rewards):
            if self.prev_action_linear is None:
                self.prev_action_linear = np.sqrt(action.vx ** 2 + action.vy ** 2) if isinstance(action, ActionXY) else action.v
            else:
                curr_linear = np.sqrt(action.vx ** 2 + action.vy ** 2) if isinstance(action, ActionXY) else action.v
                linear_smoothness_reward = np.abs(self.prev_action_linear - curr_linear) * self.rewards["linear_smoothness_factor"]
                self.prev_action_linear = curr_linear
                reward += linear_smoothness_reward
                info["LinearSmoothness"] = LinearSmoothness(linear_smoothness_reward)

        # if self.detailed_reward_return or (not self.imitation_learning and self.SB3):
        info["TotalReward"] = TotalReward(reward)
        info["Done"] = Done(done)
        for reward_type in set_of_rewards:
            if reward_type not in info:
                info[reward_type] = eval(reward_type)(0)

        if update:
            self.global_times.append(self.global_time)
            # if not self.SB3:
            # store state, action value and attention weights
            human_state_list = [human.get_full_state() for human in self.humans]
            robocentric_joint_state = self.robot.get_robocentric_state(human_state_list, self.static_obstacles)
            self.states.append([self.robot.get_full_state(), human_state_list, self.static_obstacles])
            self.robocentric_states.append([robocentric_joint_state.self_state, robocentric_joint_state.human_states, robocentric_joint_state.static_obs])

            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                try:
                    if self.robot.policy.get_attention_weights().any(): # dont try to append if Nonetype attention weights
                        self.attention_weights.append(self.robot.policy.get_attention_weights())
                except:
                    self.attention_weights = None

            # update all agents. Note that is constrained above
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)

            # update the environment's time and time step counters only if not doing a dummy start of the other agents
            self.global_time += self.time_step
            self.global_time_step += 1
            if not dummy_start:
                self.tot_env_steps += 1

            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                if (not self.imitation_learning) and self.SB3:
                    # retreive robot and human states in world coords

                    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta], dtype=np.float32
                    robot_world_state = self.robot.get_full_state(original=False)
                    # self.px, self.py, self.vx, self.vy, self.radius], dtype=np.float32)
                    human_state_list = [human.get_observable_state(original=False) for human in self.humans]

                    if self.robot.policy.name == "sarl" or self.robot.policy.name == "rgl":
                        # For SARL, we need to propogate into the future, so we just keep original states here
                        ob = self.SARL_input_complete(robot_world_state, human_state_list)
                    elif self.robot.policy.name == "rgl_multistep":
                        ob = self.RGL_multistep_input_complete(robot_world_state, human_state_list)
                    elif self.robot.policy.name == "qsarl":
                         ob = self.SARL_input_single_state(robot_world_state, human_state_list)
                    elif self.SB3_model == "campc":
                        ob = [human.get_full_state() for human in self.humans]
                    else:
                        raise NotImplementedError

                else:
                    ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                if self.SB3:
                    if self.robot.policy.name == "sarl" or self.robot.policy.name == "rgl":
                        # propogate forward one step
                        robot_world_state = self.robot.get_next_full_state(action, original=False)
                        human_state_list = [human.get_next_observable_state(action, original=False) for human, action in zip(self.humans, human_actions)]
                        ob = self.SARL_input_single_state(robot_world_state, human_state_list, action_index=action_index, reward=reward)
                    elif self.robot.policy.name == "rgl_multistep":
                        robot_world_state = self.robot.get_next_full_state(action, original=False)
                        human_state_list = [human.get_next_observable_state(action, original=False) for human, action in zip(self.humans, human_actions)]
                        ob = self.RGL_multistep_input_single_state(robot_world_state, human_state_list, self.prev_actions, action_index, reward)
                    elif self.SB3_model == "qsarl":
                        robot_world_state = self.robot.get_next_full_state(action, original=False)
                        human_state_list = [human.get_next_observable_state(action, original=False) for human, action in zip(self.humans, human_actions)]
                        ob = self.SARL_input_single_state(robot_world_state, human_state_list, action_index=action_index, reward=reward)
                    else:
                        raise NotImplementedError
                else:
                    ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info


    def render(self, mode='human', output_file=None, joint_state=None, debug=False):
        """renders the environment's current state to a file or screen

        :param mode: 'human', 'joint_state', 'human', 'traj', or 'video'. Defaults to 'human'.
        :param output_file: path to output file. Defaults to None.
        :param joint_state: joint state to render if in 'joint_state' mode. Defaults to None.
        :param debug: whether to add debug labels to plot in 'video' mode, defaults to False
        :raises NotImplementedError: if mode is not 'human', 'joint_state', 'human', 'traj', or 'video'
        """
        # get ffmpeg path from the which command
        import subprocess
        ffmpeg_path = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
        if not ffmpeg_path and mode == 'video' and output_file is not None:
            logging.error("ffmpeg not found. Please install ffmpeg to render videos.")
        else:
            logging.info(f"ffmpeg found at {ffmpeg_path}. Setting matplotlib to use ffmpeg for rendering videos.")
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

        robot_color = 'yellow'
        mpc_line_color = 'dodgerblue'
        mpc_guess_line_color = "lightsteelblue"
        forecast_color = 'gray'
        hum_line_color = '#70AD47' #green from ppt
        hum_guess_line_color = 'darkseagreen'
        fail_color = 'tab:orange'
        # goal_color = 'red'
        goal_color = 'darkgreen'
        start_color = 'gray'
        human_color = 'lightgray'
        human_goal_color = 'darkgray'
        arrow_color = 'chocolate'
        real_orca_color = 'tab:olive'

        x_offset = -0.10
        y_offset = 0.10

        if mode == 'joint_state':
            logging.info('[CrowdSimPlus] Rendering in joint state mode')
            fig, ax = plt.subplots(figsize=(7, 7))
            # ax.axis('equal')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)


            for idx, line in enumerate(joint_state.static_obs):
                stat_line = mlines.Line2D([line[0][0], line[1][0]], [line[0][1], line[1][1]], linewidth=2.0, color='tab:blue')
                ax.add_line(stat_line)
                line_number = plt.text((line[0][0]+line[1][0])/2.0, (line[0][1]+line[1][1])/2.0, str(idx), color='tab:blue', fontsize=9, horizontalalignment='center', verticalalignment='center')
                ax.add_artist(line_number)

            self_state = joint_state.self_state
            ax.add_artist(plt.Circle(self_state.position, self_state.radius, fill=True, color=robot_color, label='robot', zorder=2))

            humans=[]
            for idx, human_state in enumerate(joint_state.human_states):
                if idx == 0:
                    human_circle = plt.Circle(human_state.position, human_state.radius, fill=False, color='b', label='human')
                else:
                    human_circle = plt.Circle(human_state.position, human_state.radius, fill=False, color='b')
                humans.append(human_circle)
                ax.add_artist(human_circle)

            human_numbers = [plt.text(humans[i].center[0], humans[i].center[1], str(i+1),
                                      color='black', fontsize=11, horizontalalignment='center', verticalalignment='center') for i in range(len(joint_state.human_states))]

            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                if i == 0:
                    agent_state = joint_state.self_state
                    theta = agent_state.theta
                else:
                    agent_state = joint_state.human_states[i - 1]
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                # orientation.append(((agent_state.px, agent_state.py), (agent_state.px + agent_state.vx * 5,
                                            # agent_state.py + agent_state.vy * 5)))
                orientation.append(((agent_state.px, agent_state.py), (np.cos(theta), np.sin(theta))))
                orientations.append(orientation)
            # arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
            #           for orientation in orientations]
            arrows = [patches.FancyArrow(*orientation[0], orientation[1][0]*0.15, width=0.02, head_width=0.15, head_length=0.1, overhang=0.05, color=arrow_color)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)

            ax.legend()
            if output_file:
                fig.savefig(output_file)
            else:
                plt.show()
        elif mode == 'human':
            logging.info('[CrowdSimPlus] Rendering in human mode')
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

            for idx, line in enumerate(self.static_obstacles):
                stat_line = mlines.Line2D([line[0][0], line[1][0]], [line[0][1], line[1][1]], linewidth=2.0, color='tab:blue')
                line_number = plt.text((line[0][0]+line[1][0])/2.0, (line[0][1]+line[1][1])/2.0, str(idx), color='tab:blue', fontsize=9, horizontalalignment='center', verticalalignment='center')
                ax.add_line(stat_line)
                ax.add_artist(line_number)

            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=True, color=human_color, label='humans', alpha=1, zorder=2)
                ax.add_artist(human_circle)

            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            if output_file is not None:
                fig.savefig(output_file)
                plt.close(fig)
            else:
                plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            for line in self.static_obstacles:
                stat_line = mlines.Line2D([line[0][0], line[1][0]], [line[0][1], line[1][1]], linewidth=2, color='tab:blue')
                ax.add_line(stat_line)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16, loc='upper right')
            if output_file is not None:
                fig.savefig(output_file)
                plt.close(fig)
            else:
                plt.show()
        else:
            sim_times = self.global_times
            if not hasattr(self, 'sim_display_times'):
                display_times = self.global_times
            else:
                display_times = self.sim_display_times
            if self.static_obstacles:
                if len(self.static_obstacles) == 1:
                    x_min = -self.rect_width * 0.5
                    x_max = self.rect_width * 0.5
                else:

                    top = self.static_obstacles[1]
                    x_min = min(top[0][0], top[1][0])
                    x_max = max(top[0][0], top[1][0])
                y_min = -self.rect_height * 0.5
                y_max = self.rect_height * 0.5
                fig, ax = plt.subplots(figsize=(max(6, 9*(x_max-x_min)/(y_max-y_min)), 9))
                ax.axis('equal')
                ax.set_ylim(y_min*1.1, y_max*1.1)
                time = plt.text(0.0, y_max*1.01, 'Time: {}'.format(sim_times[0]), fontsize=16, horizontalalignment='center', verticalalignment='center')
                ax.add_artist(time)
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_ylim(-6, 6)
                ax.set_xlim(-6, 6)
                time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
                ax.add_artist(time)




            # ax.set_xlim(-6, 6)
            ax.tick_params(labelsize=16)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            # states_list = self.states
            if mode == 'joint_state':
                states_list = get_vizable_robocentric(self.robocentric_states)
            else:
                states_list = self.states
            robot_positions = [state[0].position for state in states_list]
            robot_states = [state[0] for state in states_list]
            human_states = [state[1] for state in states_list]
            stat_obs_pos = [state[2] for state in states_list]
            # goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            goal = mlines.Line2D([robot_states[0].gx], [robot_states[0].gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')

            # start = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=start_color, label='Start')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color, zorder=3)
            # ax.add_artist(start)
            ax.add_artist(goal)
            ax.add_artist(robot)

            if hasattr(self.robot.policy, 'all_origin_x'):
                all_origin_x = self.robot.policy.all_origin_x
                all_origin_y = self.robot.policy.all_origin_y
                all_opt_x = self.robot.policy.all_opt_x
                all_opt_y = self.robot.policy.all_opt_y

                hum_origin_lines = []
                hum_opt_lines = []
                for a_idx in range(all_origin_x[0].shape[0]):
                    if a_idx == 0:
                        orig_line, = ax.plot(all_origin_x[0][0, :], all_origin_y[0][0, :], linewidth=2.0, linestyle='-', marker='x', color=mpc_guess_line_color, alpha=1, zorder=5)
                        orig_line.set_visible(False)
                        opt_line, = ax.plot(all_opt_x[0][0, :], all_opt_y[0][0, :], linewidth=4.0, linestyle='-', marker='o', color=mpc_line_color, alpha=1, zorder=4)
                        opt_line.set_visible(False)
                    else:
                        hum_orig_line, = ax.plot(all_origin_x[0][a_idx, :], all_origin_y[0][a_idx, :], linewidth=1.0, linestyle='-', marker='x', color=hum_guess_line_color, zorder=5)
                        hum_orig_line.set_visible(False)
                        hum_origin_lines.append(hum_orig_line)

                        hum_opt_line, = ax.plot(all_opt_x[0][a_idx, :], all_opt_y[0][a_idx, :], linewidth=2.0, linestyle='-', marker='.', color=hum_line_color, zorder=4)
                        hum_opt_line.set_visible(False)
                        hum_opt_lines.append(hum_opt_line)


            if hasattr(self.robot.policy, 'all_x_val'):
                all_x_val = self.robot.policy.all_x_val
                all_x_goals = self.robot.policy.all_x_goals
                x_val = all_x_val[0]
                x_goals = all_x_goals[0]
                ref_line, = ax.plot(x_goals[0,:], x_goals[1,: ], linewidth=1.0, linestyle='--', marker='x', markersize=3, color=goal_color, alpha=1.0)
                ref_line.set_visible(False)
                mpc_line, = ax.plot(x_val[0,:], x_val[1,: ], linewidth=4.0, linestyle='-', marker='o', color=mpc_line_color, alpha=1, zorder=4)
                mpc_line.set_visible(False)
                if debug:
                    mpc_succ_text = plt.text(-5, -6, 'MPC: {}'.format('Success'), fontsize=14, color='tab:green')
            else:
                mpc_robots = False
                mpc_line = False


                plt.legend([robot, goal], ['robot', 'rob. goal'], fontsize=16, loc='upper right')

            # add static line obstacles
            stat_lines = [mlines.Line2D([line[0][0], line[1][0]], [line[0][1], line[1][1]], linewidth=2.0, color='midnightblue') for line in stat_obs_pos[0]]
            for line2d in stat_lines:
                ax.add_line(line2d)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(human_states[0]))] for state in states_list]
            humans = [plt.Circle(human_positions[0][i], human_states[0][i].radius, fill=True, color=human_color, label='humans', alpha=1, zorder=2)
                      for i in range(len(human_states[0]))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i+1),
                                      color='black', fontsize=9) for i in range(len(human_states[0]))]

            human_gps = [(human.final_gx,  human.final_gy) for human in self.humans]
            human_intermediate_goal = [[(state[1][j].gx, state[1][j].gy) for j in range(len(human_states[0]))] for state in states_list]
            human_goal_lines = [mlines.Line2D([human_state.px, human_state.gx, human_gps[hidx][0]], [human_state.py, human_state.gy, human_gps[hidx][1]], color=human_goal_color, linewidth=1, alpha=1, linestyle='dotted', marker='*', markersize=7, zorder=1, label='hum. goal' if hidx==0 else None) for hidx, human_state in enumerate(human_states[0])]
            human_goal_numbers = [ax.text(human_gps[i][0], human_gps[i][1], str(i+1), color=human_goal_color, fontsize=10, horizontalalignment='right', verticalalignment='bottom', zorder=1, alpha=1) for i in range(len(human_states[0]))]

            for i, human in enumerate(humans):
                ax.add_artist(human_goal_lines[i])
                ax.add_artist(human_goal_numbers[i])
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            if hasattr(self.robot.policy, 'all_x_val'):
                mpc_env = self.robot.policy.mpc_env
                all_x_val_correct_orca = self.robot.policy.calc_actual_orca_for_x_val(all_x_val)
                x_val_hum_corr = all_x_val_correct_orca[0]
                human_mpc_lines = []
                human_orca_lines = []
                os = mpc_env.nx_r + mpc_env.np_g
                mt = mpc_env.nx_hum
                for j, _ in enumerate(human_positions[0]):
                    hum_line_corr, = ax.plot(x_val_hum_corr[mt*j,:], x_val_hum_corr[mt*j+1,: ], linewidth=1.0, linestyle='--', marker='.', color=real_orca_color, alpha=0.5, markersize=2)
                    hum_line_corr.set_visible(False)

                    hum_line, = ax.plot(x_val[os+mt*j,:], x_val[os+mt*j+1,: ], linewidth=3.0, linestyle='-', marker='.', color=hum_line_color, zorder=4)
                    hum_line.set_visible(False)
                    human_mpc_lines.append(hum_line)
                    human_orca_lines.append(hum_line_corr)
                if True or debug:
                    human_mpc_diff_text = []
                    for i in range(len(human_states[0])):
                        diff_txt = plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset, str(0.0), color='black', fontsize=12)
                        diff_txt.set_visible(False)
                        human_mpc_diff_text.append(diff_txt)


                plt.legend([robot, humans[0], mpc_line, human_mpc_lines[0], goal, ref_line], ['robot', 'human', 'MPC plan','predictions', 'rob. goal'], fontsize=14, loc='upper left', ncol=1)


            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in states_list:
                    if i == 0:
                        agent_state = state[0]
                        theta = agent_state.theta
                    else:
                        agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                    # orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta), agent_state.py + radius * np.sin(theta))))
                    orientation.append([(agent_state.px, agent_state.py), (0.15 * np.cos(theta), 0.15 * np.sin(theta))])
                orientations.append(orientation)
            # arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style, zorder=3) for orientation in orientations]
            arrows = [patches.FancyArrow(*orientation[0][0], *orientation[0][1], width=0.02, head_width=0.15, head_length=0.1, overhang=0.05, color=arrow_color, zorder=10) for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = -1
            offset = 0
            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                nonlocal offset
                global_step += 1
                robot.center = robot_positions[frame_num]
                if hasattr(self.robot.policy, 'all_origin_x') and sim_times[frame_num] >= 0.0:
                    offset = global_step - 1 if sim_times[frame_num] == 0 else offset
                    origin_x = all_origin_x[frame_num-offset]
                    origin_y = all_origin_y[frame_num-offset]
                    opt_x = all_opt_x[frame_num-offset]
                    opt_y = all_opt_y[frame_num-offset]

                    orig_line.set_xdata(origin_x[0,:])
                    orig_line.set_ydata(origin_y[0,:])
                    orig_line.set_visible(True)

                    opt_line.set_xdata(opt_x[0,:])
                    opt_line.set_ydata(opt_y[0,:])
                    opt_line.set_visible(True)

                    for j, hum_orig_line in enumerate(hum_origin_lines):
                        hum_orig_line.set_xdata(origin_x[j+1,:])
                        hum_orig_line.set_ydata(origin_y[j+1,:])
                        hum_orig_line.set_visible(True)

                        hum_opt_line = hum_opt_lines[j]
                        hum_opt_line.set_xdata(opt_x[j+1,:])
                        hum_opt_line.set_ydata(opt_y[j+1,:])
                        hum_opt_line.set_visible(True)

                if mpc_line and sim_times[frame_num] >= 0.0:
                    offset = global_step - 1 if sim_times[frame_num] == 0 else offset
                    # succ_text = 'Success' if self.robot.policy.mpc_sol_succ[frame_num] else 'Failure'

                    if debug:
                        succ_text = self.robot.policy.all_debug_text[frame_num-offset]
                        succ_color = 'tab:green' if self.robot.policy.mpc_sol_succ[frame_num-offset] else 'tab:red'
                        mpc_succ_text.set_text('MPC: {}, time: {:.3f}s'.format(succ_text, self.robot.policy.calc_times[frame_num-offset]))
                        mpc_succ_text.set_visible(True)
                        mpc_succ_text.set_color(succ_color)
                    else:
                        succ_text = 'NA'
                    x_val = all_x_val[frame_num-offset]
                    x_goals = all_x_goals[frame_num-offset]
                    x_val_hum_corr = all_x_val_correct_orca[frame_num-offset]
                    mpc_line.set_xdata(x_val[0,:])
                    mpc_line.set_ydata(x_val[1,:])
                    mpc_line.set_visible(True)
                    ref_line.set_xdata(x_goals[0,:])
                    ref_line.set_ydata(x_goals[1,:])
                    ref_line.set_visible(True)
                    if self.robot.policy.mpc_sol_succ[frame_num-offset]:
                        mpc_line.set_color(mpc_line_color)
                    elif 'EMERG' in succ_text:
                        mpc_line.set_color(mpc_line_color)
                    for j, hum_line in enumerate(human_mpc_lines):
                        hum_line.set_xdata(x_val[os+mt*j,:])
                        hum_line.set_ydata(x_val[os+mt*j+1,:])
                        hum_line.set_visible(True)
                        if self.robot.policy.mpc_sol_succ[frame_num-offset]:
                            hum_line.set_color(hum_line_color)
                        elif 'EMERG' in succ_text:
                            hum_line.set_color(hum_line_color)
                    for j, hum_line_corr in enumerate(human_orca_lines):
                        hum_line_corr.set_xdata(x_val_hum_corr[mt*j,:])
                        hum_line_corr.set_ydata(x_val_hum_corr[mt*j+1,:])
                        if debug:
                            hum_line_corr.set_visible(True)
                        else:
                            hum_line_corr.set_visible(False)
                    if debug:
                        for j, hum_text in enumerate(human_mpc_diff_text):
                            # diff = np.mean(np.linalg.norm(x_val[os+mt*j:os+mt*j+2,:] - x_val_hum_corr[mt*j:mt*j+2,:], axis=0)) if self.robot.policy.mpc_sol_succ[frame_num] else np.nan
                            diff = np.mean(np.linalg.norm(x_val[os+mt*j:os+mt*j+2,:] - x_val_hum_corr[mt*j:mt*j+2,:], axis=0))
                            hum_text.set_text('{:.3f}'.format(diff)) if self.robot.policy.mpc_sol_succ[frame_num-offset] else hum_text.set_text('error')
                            hum_text.set_position((x_val_hum_corr[mt*j,0] - x_offset, x_val_hum_corr[mt*j+1,0] + y_offset))
                            hum_text.set_visible(True)

                goal.set_data([robot_states[frame_num].gx], [robot_states[frame_num].gy])
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    if mpc_line and sim_times[frame_num] >= 0.0 and (self.robot.policy.mpc_env.hum_model == 'orca_casadi_kkt' and not self.robot.policy.priviledged_info):
                        # get it from x_val the goal estimates will be the fifth and sixth elements for each human
                        human_goal_lines[i].set_xdata([human.center[0], x_val[os+mt*i+4,0]])
                        human_goal_lines[i].set_ydata([human.center[1], x_val[os+mt*i+5,0]])
                    else:
                        human_goal_lines[i].set_xdata([human.center[0], human_intermediate_goal[frame_num][i][0], human_gps[i][0]])
                        human_goal_lines[i].set_ydata([human.center[1], human_intermediate_goal[frame_num][i][1], human_gps[i][1]])

                for arrow in arrows:
                    arrow.remove()
                arrows = [patches.FancyArrow(*orientation[frame_num][0], *orientation[frame_num][1], width=0.02, head_width=0.15, head_length=0.1, overhang=0.05, color=arrow_color, zorder=3) for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)
                for idx, line2d in enumerate(stat_lines):
                    line = stat_obs_pos[frame_num][idx]
                    line2d.set_data([line[0][0], line[1][0]], [line[0][1], line[1][1]])
                # time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                time.set_text('Time: {:.2f}'.format(display_times[frame_num]))
                if hasattr(self.robot.policy, 'all_x_val') and frame_num < 2:
                    # plt.legend([stat_lines[0], robot, humans[0], mpc_line, human_mpc_lines[0], human_orca_lines[0], goal, ref_line], ['stat. obs.', 'robot', 'human', 'MPC plan','predictions', 'ORCA GT', 'rob. goal'], fontsize=14, loc='best', ncol=2)
                    plt.legend([stat_lines[0], robot, humans[0], mpc_line, human_mpc_lines[0], goal, ref_line], ['stat. obs.', 'robot', 'human', 'MPC plan','predictions', 'rob. goal'], fontsize=14, loc='upper left', ncol=1)

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [states_list[global_step][0]] +states_list[global_step][1]:
                    logging.debug(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(states_list)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()
            if mode == 'video':
                logging.info('[CrowdSimPlus] Rendering in video mode with {:} frames. Debug mode: {:}'.format(len(self.states), debug))
                def on_click(event):
                    anim.running ^= True
                    if anim.running:
                        anim.event_source.stop()
                        if hasattr(self.robot.policy, 'action_values'):
                            plot_value_heatmap()
                    else:
                        anim.event_source.start()

                fig.canvas.mpl_connect('key_press_event', on_click)
                anim = animation.FuncAnimation(fig, update, frames=len(states_list), interval=int(round(self.time_step * 1000.0)))
                # anim = animation.FuncAnimation(fig, update, frames=len(states_list), fps=int(round(1.0/self.time_step)))
                anim.running = True

                if output_file is not None:
                    try:
                        logging.info('[CrowdSimPlus] Attempting to save video via ffmpeg to: {:}'.format(output_file))
                        ffmpeg_writer = animation.writers['ffmpeg']
                        if not debug:
                            ffmpeg_extra_args = []
                        else:
                            ffmpeg_extra_args = []
                        writer = ffmpeg_writer(fps=int(round(1.0/self.time_step))*2, metadata=dict(artist='Me'), bitrate=1800, extra_args=ffmpeg_extra_args)
                        anim.save(output_file, writer=writer)
                        logging.info('[CrowdSimPlus] Success saving video via ffmpeg to: {:}'.format(output_file))
                    except Exception as e:
                        logging.error('[CrowdSimPlus] Failed to save video via ffmpeg to: {:}'.format(output_file))
                        logging.error(e)
                        plt.show()
                else:
                    plt.show()
            elif mode == 'step':
                logging.info('[CrowdSimPlus] Rendering single frame. Debug mode: {:}'.format(debug))
                plt_time = self.global_time - self.time_step
                frame_num = np.argwhere(plt_time == np.array(sim_times)).item()
                offset = np.argwhere(0.0 == np.array(sim_times)).item()
                global_step = frame_num+1
                update(frame_num)
                if output_file is not None:
                    plt.savefig(output_file)
                    # plt.close(fig)
                else:
                    melon = 1
                    # plt.show()
            else:
                raise NotImplementedError
        logging.info('[CrowdSimPlus] Finished rendering')
        return fig, ax, update