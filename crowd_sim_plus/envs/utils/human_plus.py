import numpy as np
from crowd_sim_plus.envs.utils.agent_plus import Agent
from crowd_sim_plus.envs.utils.state_plus import FullState, JointState, FullyObservableJointState

class Human(Agent):
    def __init__(self, config, section, fully_observable=False, env=None):
        super().__init__(config, section)
        self.fully_observable = fully_observable
        # human_orca_safety_space = config.getfloat(section, 'safety_space')
        # self.policy.safety_space = human_orca_safety_space

        # ORCA does not have any configurations
        try:
            self.policy.configure(config, section)
        except:
            pass
        self.env = env

    def get_g_xy(self, px, py):
        """
        Get an intermediate goal in the case of hallways with doors: if the hallway-door obstructs path to human goal,
        intermediate goal is the middle of the hallway.
        :param ob:
        :return:
        """
        if len(self.env.static_obstacles) > 0:
            # xs = [px, self.final_gx]
            ys = [py, self.final_gy]

            if (self.env.sim_env == "hallway_static" or self.env.sim_env == "hallway_static_with_back" or self.env.sim_env == "hallway_bottleneck") and np.min(ys) < self.env.door_y_mid_min and np.max(ys) > self.env.door_y_mid_max:
                # i.e. if the goal involves going through the door.
                int_gx = self.env.door_x_mid
                int_gy = 0.5 * (self.env.door_y_min + self.env.door_y_max)
                vec = np.array([int_gx - px, int_gy - py])
                vec_norm = np.linalg.norm(vec)
                # if np.linalg.norm(vec) < self.v_pref * self.policy.time_step:
                    # vec = self.v_pref * self.policy.time_step * 1.01 * vec / vec_norm
                # if np.linalg.norm(vec) < 1:
                #     vec = 1.01 * vec / vec_norm
                # gx = px + vec[0]
                # gy = py + vec[1]
                if vec_norm <= self.env.door_width / 2.0:
                    gx = self.final_gx
                    gy = self.final_gy
                else:
                    gx = int_gx
                    gy = int_gy
                return gx, gy

        gx = self.final_gx
        gy = self.final_gy
        return gx, gy

    def set_g_xy(self, px, py):
        """
        Set an intermediate goal in the case of hallways with doors: if the hallway-door obstructs path to human goal,
        intermediate goal is the middle of the hallway.
        :param ob:
        :return:
        """
        self.gx, self.gy = self.get_g_xy(px, py)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.final_gx = gx
        self.final_gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
        self.set_g_xy(px, py)


    # overwrites method from Agent to be fully observable
    def get_observable_state(self, original=True):
        if not self.fully_observable:
            return super().get_observable_state(original)
        return self.get_full_state(original)

    # overwrites method from Agent to be fully observable
    def get_next_observable_state(self, action, original=True):
        if not self.fully_observable:
            return super().get_next_observable_state(action,original=original)
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
        next_gx, next_gy = self.get_g_xy(next_px, next_py)
        return FullState(next_px, next_py, next_vx, next_vy, self.radius, next_gx, next_gy, self.v_pref, next_theta)

    def act(self, ob, static_obs=[]):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        if not self.fully_observable:
            state = JointState(self.get_full_state(original=True), ob, static_obs)
            action = self.policy.predict(state)
            return action
        state = FullyObservableJointState(self.get_full_state(original=True), ob, static_obs)
        action = self.policy.predict(state)
        return action

    # overwrite step to also update gx and gy based on whether or not still need intermediate goal or not
    def step(self, action):
        super().step(action)
        self.set_g_xy(self.px, self.py)