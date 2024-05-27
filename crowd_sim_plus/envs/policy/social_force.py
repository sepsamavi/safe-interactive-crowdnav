"""
    Reference:

    This policy has been adapted from the repository of "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph" in ICRA 2023. The repository can be found at: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph
"""
import logging
import numpy as np
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionXY
from crowd_sim_plus.envs.utils.utils_plus import closest_point_on_segment

class SFM(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'sfm'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.is_bottleneck = False

    def configure(self, config, section='sfm'):
        try:
            self.time_step = config.getfloat('env', 'time_step')
        except:
            logging.warn("[SFM POLICY] problem with policy config")

        self.radius = config.getfloat(section, 'radius')
        self.A = config.getfloat(section, 'A')
        self.B = config.getfloat(section, 'B')
        self.KI = config.getfloat(section, 'KI')
        self.A_static = config.getfloat(section, 'A_static')
        self.B_static = config.getfloat(section, 'B_static')
        self.A_bottleneck = config.getfloat(section, 'A_bottleneck')
        self.B_bottleneck = config.getfloat(section, 'B_bottleneck')

        return

    def predict(self, state):
        """
        Produce action for agent with circular specification of social force model.
        """
        # Pull force to goal
        delta_x = state.self_state.gx - state.self_state.px
        delta_y = state.self_state.gy - state.self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        dist_to_goal = 1.0 if dist_to_goal < 1e-6 else dist_to_goal
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref
        KI = self.KI # Inverse of relocation time K_i
        curr_delta_vx = KI * (desired_vx - state.self_state.vx)
        curr_delta_vy = KI * (desired_vy - state.self_state.vy)

        # Push force(s) from other agents
        A = self.A # Other observations' interaction strength: 1.5
        B = self.B # Other observations' interaction range: 1.0
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            adjustment = np.abs(self.radius - other_human_state.radius) + 0.01
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            interaction_vx += A * np.exp((state.self_state.radius + other_human_state.radius + adjustment - dist_to_human) / B) * (delta_x / dist_to_human)
            interaction_vy += A * np.exp((state.self_state.radius + other_human_state.radius + adjustment - dist_to_human) / B) * (delta_y / dist_to_human)



        for idx, stat_ob_pts in enumerate(state.static_obs):
            if self.is_bottleneck and idx >= 2:
                A_static = self.A_bottleneck # Static obstacle's interaction strength: 1.5
                B_static = self.B_bottleneck # Static obstacle's interaction range: 1.0
            else:
                A_static = self.A_static # Static obstacle's interaction strength: 1.5
                B_static = self.B_static # Static obstacle's interaction range: 1.0
            ox, oy = closest_point_on_segment(*stat_ob_pts[0], *stat_ob_pts[1], state.self_state.px, state.self_state.py)
            delta_x = state.self_state.px - ox
            delta_y = state.self_state.py - oy
            dist_to_obs = np.sqrt(delta_x**2 + delta_y**2)
            interaction_vx += A_static * np.exp((state.self_state.radius + 0.01 - dist_to_obs) / (B_static)) * (delta_x / dist_to_obs)
            interaction_vy += A_static * np.exp((state.self_state.radius + 0.01 - dist_to_obs) / (B_static)) * (delta_y / dist_to_obs)


        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        new_vy = state.self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])

        if act_norm > state.self_state.v_pref:
            return ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(new_vx, new_vy)