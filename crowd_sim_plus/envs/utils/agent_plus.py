import torch
import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from crowd_sim_plus.envs.utils.action import ActionXY, ActionRot
from crowd_sim_plus.envs.utils.state_plus import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')

        self.SB3 = config.getboolean('env', "SB3", fallback=False)
        if not self.SB3 or section=="humans":
            self.policy = policy_factory[config.get(section, 'policy')]()
        else:
            # if this is the robot, the policy is not an attribute because SB3 overrides it
            self.policy = None
        self.sensor = config.get(section, 'sensor')
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.omega = None
        self.time_step = None


        # for SB3 specific portions that deal with numpy array
        self.position_indexes = [0, 1]
        self.velocity_indexes = [2, 3]
        self.radius_index = 4
        self.goal_indexes = [5, 6]
        self.vpref_index = 7
        self.theta_index = 8

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics
        if hasattr(policy, 'radius'):
            self.policy.radius = self.radius

    def sample_random_attributes(self, rng):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = rng.uniform(0.5, 1.5)
        self.radius = rng.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self, original=True):
        if original:
            return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

        if self.SB3:
            return np.array([self.px, self.py, self.vx, self.vy, self.radius], dtype=np.float32)

    def get_next_observable_state(self, action, original=True):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        if original:
            return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)
        else:
            return np.array([next_px, next_py, next_vx, next_vy, self.radius], dtype=np.float32)

    def get_full_state(self, original=True):
        if original:
            return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.omega)

        if self.SB3:
            return np.array([self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta], dtype=np.float32) #, self.omega])

    def get_next_full_state(self, action, original=True):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos

        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
            next_theta = np.arctan2(self.vy, self.vx)
        else:
            unwrapped_theta = (self.theta + action.r) % (2 * np.pi)
            next_theta = unwrapped_theta - 2 * np.pi if unwrapped_theta > np.pi else unwrapped_theta
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        if original:
            raise NotImplementedError
        else:
            return np.array([next_px, next_py, next_vx, next_vy, self.radius, self.gx, self.gy, self.v_pref, next_theta], dtype=np.float32)



    def get_ang_velocity(self):
        return self.omega

    def set_ang_velocity(self, angular_velocity):
        self.omega = angular_velocity

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    @abc.abstractmethod
    def act(self, ob, static_obs):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    @staticmethod
    def compute_position_by_state(state, action, delta_t):
        if isinstance(action, ActionXY):
            px = state[0] + action.vx * delta_t
            py = state[1] + action.vy * delta_t
        else:
            theta = state[2] + action.r
            px = state[0] + np.cos(theta) * action.v * delta_t
            py = state[1] + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
            self.theta = np.arctan2(self.vy, self.vx) # my addition
        else:
            unwrapped_theta = (self.theta + action.r) % (2 * np.pi)
            self.theta = unwrapped_theta - 2 * np.pi if unwrapped_theta > np.pi else unwrapped_theta # my addition to wrap to (-pi, pi
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

