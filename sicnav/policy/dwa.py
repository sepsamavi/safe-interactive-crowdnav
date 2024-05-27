import numpy as np
import logging
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionRot
import sicnav.utils.PythonRobotics.dynamic_window_approach as PRdwa
from sicnav.utils.PythonRobotics.dynamic_window_approach import RobotType

class DynamicWindowApproach(Policy):
    """
    Template to implement a teleoperation.
    This placeholder is a copy of crowd_sim_plus.envs.policy.linear
    """

    def __init__(self):
        super().__init__()
        self.name = "DWA"
        self.trainable = False
        self.kinematics = 'unicycle'
        self.multiagent_training = True
        # self.safety_space = 0
        # self.neighbor_dist = 10
        # self.max_neighbors = 10
        self.time_horizon = 2.5
        self.radius = 0.3
        self.max_speed = 0.8
        self.sim_config = None
        self.prev_theta = None

    def configure(self, config):
        pass

    def configure_dwa(self, policy_config, env_config):
        self.radius = env_config.getfloat('robot', 'radius')
        self.max_speed = env_config.getfloat('robot', 'v_pref')
        # self.max_yaw_rate = policy_config.getfloat('dwa', 'wmax')

        self.sim_config = PRdwa.Config()
        self.sim_config.max_speed = self.max_speed
        self.sim_config.min_speed = -self.max_speed
        self.sim_config.max_accel = policy_config.getfloat('dwa', 'acc_max')  # [m/ss]
        self.sim_config.max_delta_yaw_rate = policy_config.getfloat('dwa', 'ang_acc_max')  # [rad/ss]
        self.sim_config.max_d = policy_config.getfloat('dwa', 'max_d')
        self.sim_config.v_resolution = 0.1  # [m/s]
        self.sim_config.yaw_rate_resolution = policy_config.getfloat('dwa', 'ang_acc_res_deg') * np.pi / 180.0  # [rad/s] #1 * np.pi / 180.0  # [rad/s]
        self.sim_config.dt = env_config.getfloat('env', 'time_step') #self.time_step  # [s] Time tick for motion prediction
        self.sim_config.predict_time = self.time_horizon  # [s]
        self.sim_config.to_goal_cost_gain = 0.5 #0.1
        self.sim_config.speed_cost_gain = 1 #1.1
        self.sim_config.obstacle_cost_gain = 2 #1
        self.sim_config.robot_stuck_flag_cons = 0.01  # constant to prevent robot stucked
        self.sim_config.robot_type = RobotType.circle
        self.sim_config.robot_radius = self.radius  # [m] for collision check
        # self.sim_config.robot_width = 0.5  # [m] for collision check
        # self.sim_config.robot_length = 1.2  # [m] for collision check
        self.sim_config.ob = np.zeros((1, 3))
        self.sim_config.line_obs = []

    def reset_dwa(self):
        self.prev_theta = None

    def predict(self, state):
        """
        Create a PythonRobotics DWA simulation at each time step and run one step

        # Function structure based on CrowdSim ORCA

        :param state:
        :return:
        """
        self_state = state.self_state

        # reset DWA if this is a new scenario:
        if self.env.global_time < self.env.time_step:
            self.reset_dwa()

        # if there are a different number of obstacles
        if len(self.sim_config.ob) != len(state.human_states):
            self.sim_config.ob = np.zeros((len(state.human_states), 3))

        # update static obstacles if they are not in the sim_config
        if len(self.sim_config.line_obs) != len(state.static_obs):
            self.sim_config.line_obs = state.static_obs

        # update state of obstacles for the sim_config
        for idx, human_state in enumerate(state.human_states):
            self.sim_config.ob[idx, :] = [human_state.position[0], human_state.position[1], human_state.radius]


        v = np.sqrt(self_state.vx**2 + self_state.vy**2)
        w = self_state.omega
        if w is None:
            if self.prev_theta is None:
                self.prev_theta = self_state.theta
            w = (self_state.theta - self.prev_theta) / self.time_step
        PRdwa_state = [self_state.px, self_state.py, self_state.theta, v, w]


        u = PRdwa.dwa_control(PRdwa_state, self.sim_config, self_state.goal_position, self.sim_config.ob, self.sim_config.line_obs)
        action = ActionRot(u[0], u[1] * self.time_step)
        logging.debug(str(action))
        self.prev_theta = self_state.theta

        return action
