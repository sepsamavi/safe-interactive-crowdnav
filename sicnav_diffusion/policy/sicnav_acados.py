import logging
import time
from copy import deepcopy

import numpy as np
import casadi as cs

import os
import pickle

# Imports that are needed for the wrapper
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionRot

from crowd_sim_plus.envs.utils.human_plus import Human

from sicnav.utils.mpc_utils.sicnav_acados.campc_acados_opt import export_campc_ocp
from sicnav_diffusion.utils.mpc_utils.mpc_env_new import MPCEnv
from sicnav.utils.mpc_utils.orca_c_wrapper import ORCACWrapper # for ground truth ORCA calculations used in plotting sim runs

from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
import sys

from sicnav_diffusion.JMID.mid_sim_wrapper import HumanTrajectoryForecasterSim

import einops


DO_DEBUG = False
DO_DEBUG_LITE = False
DO_VIDS = False
DISP_TIME = True

ACADOS_STATUS = {
    0 : "success",
    1 : "failure",
    2 : "maximum number of iterations reached",
    3 : "minimum step size in QP solver reached",
    4 : "qp solver failed",
}


def residual(z1, z2):
    # z1 - z2
    # wrap each of the z's pose parts to pi
    z1[2] = z1[2] % (2 * np.pi)
    if z1[2] > np.pi:
        z1[2] -= 2 * np.pi
    z2[2] = z2[2] % (2 * np.pi)
    if z2[2] > np.pi:
        z2[2] -= 2 * np.pi
    # find the difference between the two poses
    y = z1 - z2
    # wrap the difference to pi, ensure that the residual is in the correct direction and the smallest possible
    sin_y2 = np.sin(z1[2])*np.cos(z2[2]) - np.cos(z1[2])*np.sin(z2[2])
    cos_y2 = np.cos(z1[2])*np.cos(z2[2]) + np.sin(z1[2])*np.sin(z2[2])
    y[2] = np.arctan2(sin_y2, cos_y2)
    return y

def point_to_segment_dist(segments, point):
    # Extract coordinates for readability
    p1x, p1y = segments[:, 0, 0], segments[:, 0, 1]
    p2x, p2y = segments[:, 1, 0], segments[:, 1, 1]
    px, py = point

    # Vector from p1 to point
    p1_to_point_x = px - p1x
    p1_to_point_y = py - p1y

    # Vector from p1 to p2 (the line segment)
    p1_to_p2_x = p2x - p1x
    p1_to_p2_y = p2y - p1y

    # Squared length of the segment
    l2 = p1_to_p2_x**2 + p1_to_p2_y**2

    # Projection of point on the segment, clamped from 0 to 1
    # This is the fractional point along the segment which is closest to the external point
    t = (p1_to_point_x * p1_to_p2_x + p1_to_point_y * p1_to_p2_y) / l2
    t = np.clip(t, 0, 1)

    # The coordinates of the closest point on the segment
    projection_x = p1x + t * p1_to_p2_x
    projection_y = p1y + t * p1_to_p2_y

    # Vector from closest point on segment to the point
    dx = projection_x - px
    dy = projection_y - py

    # Distance from point to segment
    distance = np.sqrt(dx**2 + dy**2)

    return distance

def intersections_in_point_frame_to_segment_dist(segments_untsfed, point, theta):
    # Extract coordinates for readability

    # transform the points to the point frame
    rot_mtx = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # transform the segments into the point frame
    segments = np.dot(segments_untsfed, rot_mtx.T)  - rot_mtx @ np.array([point[0], point[1]])
    px,py = 0., 0.

    p1x, p1y = segments[:, 0, 0], segments[:, 0, 1]
    p2x, p2y = segments[:, 1, 0], segments[:, 1, 1]

    # Vector from p1 to point
    p1_to_point_x = px - p1x
    p1_to_point_y = py - p1y

    # Vector from p1 to p2 (the line segment)
    p1_to_p2_x = p2x - p1x
    p1_to_p2_y = p2y - p1y

    # Squared length of the segment
    l2 = p1_to_p2_x**2 + p1_to_p2_y**2

    # Projection of point on the segment, clamped from 0 to 1
    # This is the fractional point along the segment which is closest to the external point
    t = (p1_to_point_x * p1_to_p2_x + p1_to_point_y * p1_to_p2_y) / l2
    t = np.clip(t, 0, 1)

    # The coordinates of the closest point on the segment
    projection_x = p1x + t * p1_to_p2_x
    projection_y = p1y + t * p1_to_p2_y

    # Vector from closest point on segment to the point
    dx = projection_x - px
    dy = projection_y - py

    # Distance from point to segment
    distance = np.sqrt(dx**2 + dy**2)

    return distance, projection_x, projection_y



class SICNavAcados(Policy):

    def __init__(self):
        self.trainable = False
        self.kinematics = 'unicycle'
        self.multiagent_training = True
        self.config = None
        self.robustness_eval = False

        # environment attributes
        self.num_hums = None
        self.hum_radii = None

        # mpc solver attributes
        self.horiz = None
        self.soft_constraints = True
        self.soft_dyn_consts = False
        self.ineq_dyn_consts = False
        self.new_ref_each_step = False
        self.use_casadi_init = False # Use my casadi optimization for ref. traj./guess traj. generation
        self.warmstart = True
        self.cvmm_pol = None # for warmstarting with a cvmm model mpc first,
        self.use_term_const = False

        # mpc env and its attributes (mpc_env and attributes)
        self.mpc_env = None
        self.callback_orca = None
        self.dynamics_func = None
        self.Q = None
        self.R = None

        # reference actions
        self.ref_poses_all = None
        self.ref_actions_all = None
        self.pos_ctrl_inv_vel = None
        self.x_prev = None
        self.u_prev = None
        self.outdoor_robot_setting = False # IMPORTANT whether the environment is the real robot or the simulator.
        # If you would like to implement a ROS node to use this code with your robot, set this to true.
        self.prev_rev = False
        self.priviledged_info = False

        self.do_callback_to_avoid_optifail = False


    def configure(self, config):
        super().configure(config)
        self.config = config
        self.horiz = config.getint('campc', 'horiz')
        self.soft_constraints = config.getboolean('campc', 'soft_constraints')
        self.soft_dyn_consts = config.getboolean('campc', 'soft_dyn_consts')
        self.ineq_dyn_consts = config.getboolean('campc', 'ineq_dyn_consts')
        self.ref_type = config.get('campc', 'ref_type')
        self.use_term_const = config.getboolean('campc', 'term_const')
        self.new_ref_each_step = config.getboolean('campc', 'new_ref_each_step')
        self.ref_type = 'new_path_eachstep'if self.new_ref_each_step and self.ref_type == 'path_foll' else self.ref_type
        self.warmstart = config.getboolean('campc', 'warmstart')
        self.human_goal_cvmm = config.getboolean('campc', 'human_goal_cvmm')
        self.human_goal_cvmm_horizon = config.getfloat('campc', 'human_goal_cvmm_horizon')
        self.human_pred_MID = config.getboolean('campc', 'human_pred_MID')
        self.human_pred_MID_joint = config.getboolean('campc', 'human_pred_MID_joint')
        assert (self.human_goal_cvmm and not self.human_pred_MID) or (not self.human_goal_cvmm and self.human_pred_MID) or (not self.human_goal_cvmm and not self.human_pred_MID)
        self.randomize_rob_goal = config.getboolean('campc', 'randomize_rob_goal')
        self.do_test_and_dock = config.getboolean('campc', 'do_test_and_dock', fallback=False)
        assert (self.human_pred_MID or self.human_goal_cvmm) and not (self.human_pred_MID and self.human_goal_cvmm), 'Configuration Error. Cannot have both MID and CVMM for intent prediction'
        print('[CAMPC] Config {:} = {:}'.format('horiz', self.horiz))
        print('[CAMPC] Config {:} = {:}'.format('soft_constraints', self.soft_constraints))
        print('[CAMPC] Config {:} = {:}'.format('soft_dyn_consts', self.soft_dyn_consts))
        print('[CAMPC] Config {:} = {:}'.format('ineq_dyn_consts', self.ineq_dyn_consts))
        print('[CAMPC] Config {:} = {:}'.format('new_ref_each_step', self.new_ref_each_step))
        print('[CAMPC] Config {:} = {:}'.format('warmstart', self.warmstart))
        print('[CAMPC] Config {:} = {:}'.format('human_goal_cvmm', self.human_goal_cvmm))
        print('[CAMPC] Config {:} = {:}'.format('human_goal_cvmm_horizon', self.human_goal_cvmm_horizon))
        print('[CAMPC] Config {:} = {:}'.format('human_pred_MID', self.human_pred_MID))
        print('[CAMPC] Config {:} = {:}'.format('randomize_rob_goal', self.randomize_rob_goal))
        self.human_pred_MID_vanil_as_joint = config.getboolean('campc', 'human_pred_MID_vanil_as_joint')
        self.human_pred_AF = config.getboolean('campc', 'human_pred_AF')
        assert (self.human_pred_MID_joint or self.human_pred_MID_vanil_as_joint) and not (self.human_pred_MID_joint and self.human_pred_MID_vanil_as_joint), 'Configuration Error. Cannot have both joint predictions and use the iMID model'

    def init_warmstart_solver(self):
        self.warmstart_horiz, self.warmstart_onestep, self.warmstart_correction, self.warmstart_debug = self.mpc_env.casadi_orca.get_rob_warmstart_fn(self.mpc_env)


    def convert_to_mpc_state_vector(self, state, nx_r=None, np_g=None, nX_hums=None, forecasts_init_weights=None, get_numpy=True):
        """Made to test the system model. from state object, return the state in the format that the orca solver would expect.

        :param state: _description_
        :param nx_r: _description_
        :param np_g: _description_
        :param nX_hums: _description_
        :return: _description_
        """
        if nx_r is None:
            nx_r = self.mpc_env.nx_r
        if np_g is None:
            np_g = self.mpc_env.np_g
        if nX_hums is None:
            nX_hums = self.mpc_env.nX_hums

        val = np.zeros(nx_r+np_g+nX_hums)
        val[0] = state.self_state.px
        val[1] = state.self_state.py
        val[2] = np.sin(state.self_state.theta)
        val[3] = np.cos(state.self_state.theta)

        if hasattr(state.self_state, 'lvel') and state.self_state.lvel is not None:
            val[4] = state.self_state.lvel
        else:
            v_coeff = -1 if self.prev_rev else 1
            val[4] = v_coeff * np.sqrt(state.self_state.velocity[0]**2 + state.self_state.velocity[1]**2)
            if not np.abs(val[4] * val[3] - state.self_state.velocity[0])<1e-5 or not np.abs(val[3] * val[2] - state.self_state.velocity[1]) < 1e-5:
                print('[MPC ENV] PROBLEM PROBLEM PROBLEM with vel and heading')
        if state.self_state.omega is not None:
            val[5] = state.self_state.omega
        else:
            val[5] = 0.0
        if hasattr(state.self_state, "v_dot") and state.self_state.v_dot is not None:
            val[6] = state.self_state.v_dot
        else:
            val[6] = 0.0
        if hasattr(state.self_state, "omega_dot") and state.self_state.omega_dot is not None:
            val[7] = state.self_state.omega_dot
        else:
            val[7] = 0.0
        val[8] = state.self_state.gx
        val[9] = state.self_state.gy
        offset = nx_r+np_g
        for i, h_state in enumerate(state.human_states):
            if i >= self.mpc_env.num_hums:
                break
            val[offset+i*self.mpc_env.nx_hum] = h_state.px
            val[offset+i*self.mpc_env.nx_hum+1] = h_state.py
            val[offset+i*self.mpc_env.nx_hum+2] = h_state.vx
            val[offset+i*self.mpc_env.nx_hum+3] = h_state.vy
            val[offset+i*self.mpc_env.nx_hum+4] = h_state.gx
            val[offset+i*self.mpc_env.nx_hum+5] = h_state.gy
            if self.human_pred_MID and not self.human_pred_MID_joint:
                if forecasts_init_weights is None:
                    val[offset+i*self.mpc_env.nx_hum+6:offset+(i+1)*self.mpc_env.nx_hum] = np.log(np.ones(self.mpc_env.num_MID_samples) / self.mpc_env.num_MID_samples)
                else:
                    val[offset+i*self.mpc_env.nx_hum+6:offset+(i+1)*self.mpc_env.nx_hum] = forecasts_init_weights[i,:]
        if self.human_pred_MID and self.human_pred_MID_joint:
            if forecasts_init_weights is None:
                val[-self.mpc_env.num_MID_samples:] = np.log(np.ones(self.mpc_env.num_MID_samples) / self.mpc_env.num_MID_samples)
            else:
                val[-self.mpc_env.num_MID_samples:] = forecasts_init_weights[:]

        if get_numpy:
            return val.reshape(val.shape[0], 1)
        return val.tolist()


    def generate_traj(self, joint_state, ref_steps, x_rob=None, u_rob=None, use_casadi_init=None, for_guess=False):
        """Generates a trajectory from the current state to the goal state. If x_rob and/pr u_rob are provided then those are taken as the trajectory
           and only the ORCA agents are simulated forward to fill the rest of the state.

        :param joint_state: starting state
        :param ref_steps: number of steps to generate trajectory for
        :param x_rob: a set of robot states, will be calculated from u_rob if None, defaults to None
        :param u_rob: a set of robot actions, will be a linear traj at maximum accelerateion if None, defaults to None
        :param use_casadi_init: simulate forward with casadi implementation of ORCA, defaults to None
        :param for_guess: whether or not the trajectory is for x_guess, u_guess. If it is for guess then humans projected forward with cvmm, defaults to False
        :return: a vector of states and a vector of actions for the whole environment
        """
        if use_casadi_init == None:
            use_casadi_init = self.use_casadi_init
        self_state = joint_state.self_state
        init_dist = np.sqrt((self_state.gx - self_state.px)**2+(self_state.gy - self_state.py)**2)
        px, py = self_state.px, self_state.py
        init_dist = np.sqrt((self_state.gx - px)**2+(self_state.gy - py)**2)

        dpg = np.array([self_state.gx - px, self_state.gy - py])
        theta_enroute = np.arctan2(dpg[1], dpg[0])
        # see the number of steps required to align (only rotate toward) with the goal
        if np.linalg.norm(dpg) > self.robot_radius:
            N_req_init_angle = 0
        else:
            init_pose_diff = residual(np.array([py, px, theta_enroute]), np.array([px, py, self_state.theta]))
            tot_init_angle_change = init_pose_diff[2]
            N_req_init_angle = int(np.ceil(np.abs(tot_init_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        theta_target = theta_enroute
        final_pose_diff = residual(np.array([self_state.gy, self_state.gx, theta_target]), np.array([py, px, theta_enroute]))
        tot_angle_change = final_pose_diff[2]

        N_req_angle_arrival = int(np.ceil(np.abs(tot_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        N_req_move = int(np.ceil(init_dist / (self.time_step * self.mpc_env.pref_speed)))
        N_req = N_req_move+N_req_init_angle

        # initialize vectors for reference states and actions of the robot
        if x_rob is None:
            ref_x = np.zeros(ref_steps+1, dtype=float) # here
            ref_y = np.zeros(ref_steps+1, dtype=float)
            ref_th = np.zeros(ref_steps+1, dtype=float)
            ref_x[0] = self_state.px
            ref_y[0] = self_state.py
            ref_th[0] = self_state.theta
            # reference actions
            ref_v = np.zeros(ref_steps, dtype=float)
            ref_om = np.zeros(ref_steps, dtype=float)
        else:
            ref_x = x_rob[0,:]
            ref_y = x_rob[1,:]
            ref_th = x_rob[2,:]
            # reference actions
            ref_v = u_rob[0,:]
            ref_om = u_rob[1,:]

        start_idx = 1
        if x_rob is None:
            for idx in range(start_idx, ref_steps+1):
                dpg_x = self_state.gx - ref_x[idx-1]
                dpg_y = self_state.gy - ref_y[idx-1]
                dist_to_goal = np.sqrt(dpg_x**2+dpg_y**2)

                target_heading = np.arctan2(dpg_y, dpg_x) if np.abs(dpg_y) > 1e-5 or np.abs(dpg_x) > 1e-5 else theta_target

                z1 = np.array([self_state.gx, self_state.gy, target_heading])
                z2 = np.array([ref_x[idx-1], ref_y[idx-1], ref_th[idx-1]])
                dpose = residual(z1, z2)
                dpg_theta = dpose[2]


                if u_rob is None:
                    if idx < N_req and idx > N_req_init_angle:
                        v_pref = self.mpc_env.pref_speed
                        ref_v[idx-1] = v_pref
                        ref_om[idx-1] = dpg_theta / self.time_step
                    elif idx == N_req:
                        v_pref = np.sqrt(dpg_x**2+dpg_y**2) / self.time_step
                        ref_v[idx-1] = v_pref
                        ref_om[idx-1] = dpg_theta / self.time_step
                    else:
                        v_pref = 0.0
                        ref_v[idx-1] = v_pref
                        if dpg_theta > 0.0:
                            corrected_dpg_theta = max(self.mpc_env.max_rot * self.time_step, dpg_theta)
                        else:
                            corrected_dpg_theta = min(-self.mpc_env.max_rot * self.time_step, dpg_theta)
                        ref_om[idx-1] = corrected_dpg_theta / self.time_step


                else:
                    ref_v[idx-1] = u_rob[0,idx-1]
                    ref_om[idx-1] = u_rob[1,idx-1]

                unwrapped_theta = (ref_th[idx-1] + self.time_step * ref_om[idx-1]) % (2 * np.pi)
                next_theta = unwrapped_theta - 2 * np.pi if unwrapped_theta >= np.pi else unwrapped_theta # my addition to wrap to (-pi, pi
                ref_x[idx] = ref_x[idx-1] + self.time_step * ref_v[idx-1] * np.cos(next_theta)
                ref_y[idx] = ref_y[idx-1] + self.time_step * ref_v[idx-1] * np.sin(next_theta)
                ref_th[idx] = next_theta

        # Simulate forward other agents via ORCA
        hum_offset = self.mpc_env.nx_r+self.mpc_env.np_g
        oc_next_val = self.convert_to_mpc_state_vector(joint_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)
        ref_oc_next_val = np.zeros((self.mpc_env.nX_hums, ref_steps+1), dtype=float)
        if use_casadi_init:
            ref_oc_next_U_val = np.zeros((self.mpc_env.nVars_hums, ref_steps), dtype=float)
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            ref_oc_lambdas = np.zeros((self.mpc_env.nLambda, ref_steps), dtype=float)
        ref_oc_next_val[:, 0] = np.array(oc_next_val[hum_offset:, 0])
        for t_step in range(ref_steps):
            col_idx = t_step+1

            if not for_guess or (self.mpc_env.hum_model == 'orca_casadi_kkt' and for_guess and self.mpc_env.orca_kkt_horiz > 0 and t_step <= self.mpc_env.orca_kkt_horiz):
                # get the next state from the ORCA callback
                if use_casadi_init:
                    oc_next_hum_val, next_U_hums, next_lambda_hums = self.mpc_env.casadi_orca.optimize_all(oc_next_val)
                    ref_oc_next_U_val[:, t_step] = next_U_hums[:,0]
                else:
                    oc_next_hum_val = self.mpc_env.callback_orca(oc_next_val).toarray()
                    next_lambda_hums = np.zeros((self.mpc_env.nLambda, 1))
            else:
                # get the next state by simulating the hums forward with CVMM
                slice_indices_posns = np.array([[idx1, idx1+1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)
                prev_X_hums = np.take(ref_oc_next_val[:, col_idx-1], slice_indices_posns, axis=0)
                slice_indices_goals = np.array([[idx1+4, idx1+5] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)
                prev_G_hums = np.take(ref_oc_next_val[:, col_idx-1], slice_indices_goals, axis=0)
                if use_casadi_init:
                    prev_U_hums_all = ref_oc_next_U_val[:, t_step-1]
                    ref_oc_next_U_val[:, t_step] = prev_U_hums_all
                else:
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        slice_indices = np.array([[idx1+2, idx1+3, -1, -1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                        prev_U_hums_all = np.take(np.hstack([ref_oc_next_val[:, col_idx-1], 0]), slice_indices, axis=0)
                        prev_U_just_vals = prev_U_hums_all.reshape(self.mpc_env.num_hums, self.mpc_env.nvars_hum).T
                        prev_U_hums = prev_U_just_vals[:-1,:].T.reshape(self.mpc_env.num_hums*2)
                    else:
                        slice_indices = np.array([[idx1+2, idx1+3] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                        prev_U_hums = np.take(ref_oc_next_val[:, col_idx-1], slice_indices, axis=0)

                next_X_hums = prev_X_hums + self.mpc_env.time_step * prev_U_hums
                oc_next_hum_val = np.array([[next_X_hums[idx1*2], next_X_hums[idx1*2+1], prev_U_hums[idx1*2], prev_U_hums[idx1*2+1], prev_G_hums[idx1*2], prev_G_hums[idx1*2+1]] for idx1 in range(self.mpc_env.num_hums)]).reshape(self.mpc_env.nX_hums, 1)
                next_lambda_hums = np.zeros((self.mpc_env.nLambda, 1))

            # add the value of the next state to the reference array
            ref_oc_next_val[:, col_idx] = oc_next_hum_val.reshape(oc_next_hum_val.shape[0],)
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                ref_oc_lambdas[:, col_idx-1] = next_lambda_hums[:,0]
            # generate next environment state to iterate next
            oc_next_rob_val = np.vstack([ref_x[col_idx], ref_y[col_idx], np.sin(ref_th[col_idx]), np.cos(ref_th[col_idx]), ref_v[t_step], ref_om[t_step], 0.0, 0.0, self_state.gx, self_state.gy])
            oc_next_val = np.vstack([oc_next_rob_val, oc_next_hum_val])

        if self.mpc_env.hum_model == 'orca_casadi_simple' or self.mpc_env.hum_model == 'orca_casadi_kkt':
            if use_casadi_init:
                ref_hum_U = ref_oc_next_U_val
            else:
                slice_indices = np.array([[idx1+2, idx1+3, -1, -1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                ref_hum_U = np.take(np.vstack([ref_oc_next_val, np.zeros(ref_oc_next_val.shape[1])]), slice_indices, axis=0)[:,1:]
            omega_0 =  self_state.omega if self_state.omega is not None else 0.0
            ref_X = np.vstack([ref_x, ref_y, np.sin(ref_th), np.cos(ref_th), np.hstack([np.array([np.linalg.norm(self_state.velocity)]), ref_v]), np.hstack([np.array([omega_0]), ref_om]), np.zeros_like(ref_x), np.zeros_like(ref_y), np.ones(ref_x.shape)*self_state.gx, np.ones(ref_x.shape)*self_state.gy]+[ref_oc_next_val[idx, :] for idx in range(ref_oc_next_val.shape[0])])
            ref_U = np.vstack([ref_v, ref_om, ref_hum_U])
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                ref_lambdas = ref_oc_lambdas
                ref_U = np.vstack([ref_U, ref_lambdas])
        else:
            omega_0 =  self_state.omega if self_state.omega is not None else 0.0
            ref_X = np.vstack([ref_x, ref_y, np.sin(ref_th), np.cos(ref_th), np.hstack([np.array([np.linalg.norm(joint_state.self_state.velocity)]), ref_v]), np.hstack([np.array([omega_0]), ref_om]), np.zeros_like(ref_x), np.zeros_like(ref_y), np.ones(ref_x.shape)*self_state.gx, np.ones(ref_x.shape)*self_state.gy, ref_oc_next_val])
            ref_U = np.vstack([ref_v, ref_om])
        assert ref_X.shape[0] == self.mpc_env.nx
        assert ref_U.shape[0] == self.mpc_env.nu
        return ref_X, ref_U


    def gen_ref_traj(self, state):
        """Generates a reference trajectory as the straight-line distance from

        :param state: the joint state
        :return: reference values for states and actions
        """
        self_state = state.self_state
        px, py = self_state.px, self_state.py
        init_dist = np.sqrt((self_state.gx - px)**2+(self_state.gy - py)**2)

        dpg = np.array([self_state.gx - px, self_state.gy - py])
        theta_enroute = np.arctan2(dpg[1], dpg[0])
        # see the number of steps required to align (only rotate toward) with the goal
        if np.linalg.norm(dpg) > self.robot_radius:
            N_req_init_angle = 0
        else:
            init_pose_diff = residual(np.array([py, px, theta_enroute]), np.array([px, py, self_state.theta]))
            tot_init_angle_change = init_pose_diff[2]
            N_req_init_angle = int(np.ceil(np.abs(tot_init_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        theta_target = theta_enroute
        final_pose_diff = residual(np.array([self_state.gy, self_state.gx, theta_target]), np.array([py, px, theta_enroute]))
        tot_angle_change = final_pose_diff[2]

        N_req_angle_arrival = int(np.ceil(np.abs(tot_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        N_req_angle = N_req_init_angle + N_req_angle_arrival
        N_req_move = int(np.ceil(init_dist / (self.time_step * self.mpc_env.pref_speed)))
        ref_steps = N_req_move+N_req_angle+N_req_angle_arrival+2
        # reference states
        ref_poses, ref_actions = self.generate_traj(state, ref_steps)

        self.ref_poses_all = ref_poses
        self.ref_actions_all = ref_actions
        self.pos_ctrl_inv_vel = np.zeros((self.mpc_env.nu,1))

    def get_ref_traj(self, state):
        """Generate a reference trajectory for the controller to follow based on the type of MPC controller configured

        :param state: starting state object
        :raises NotImplementedError: if reference type is not one of point stabalization, path following, or trajectory tracking
        :return: reference states and reference actions
        """
        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        # Adapted from https://github.com/utiasDSL/safe-control-gym/
        # (file safe-control-gym/safe_control_gym/controllers/mpc/mpc.py)
        p_init = np.array(state.self_state.position).reshape(2,1)
        p_goal = np.array(state.self_state.goal_position).reshape(2,1)
        # check if the p_goal is behind the robot or in front, given the current position and heading of the robot
        cur_to_goal_vec = p_goal - p_init
        cur_head_dir = np.array([np.cos(state.self_state.theta), np.sin(state.self_state.theta)]).reshape(2,1)
        cur_to_goal_dir = cur_to_goal_vec / np.linalg.norm(cur_to_goal_vec) if np.linalg.norm(cur_to_goal_vec) > 1e-5 else np.zeros_like(cur_to_goal_vec)
        dot_prod = np.dot(cur_head_dir.T, cur_to_goal_dir).item()
        if dot_prod < 0.5:
            ref_type = 'point_stab'
        else:
            ref_type = self.ref_type

        if ref_type == 'traj_track':
        # Just follow reference with time, no matter where the robot is at time step i
            start = min(self.traj_step, self.ref_actions_all.shape[-1])
            end = min(self.traj_step + self.horiz, self.ref_actions_all.shape[-1])
            remain = max(0, self.horiz - (end - start))
            x_ref_regular = np.concatenate([self.ref_poses_all[:, start:end+1],
                                        np.tile(self.ref_poses_all[:, -1:], (1, remain))
                                    ], -1)
            u_ref_regular = np.concatenate([self.ref_actions_all[:, start:end],
                                    np.tile(self.pos_ctrl_inv_vel, (1, remain))
                                    ], -1)
        elif ref_type == 'path_foll' or ref_type == 'new_path_eachstep':
        # Take closest reference point at current timestep for reference traj
            p_init = np.array(state.self_state.position).reshape(2,1)
            p_goal = np.array(state.self_state.goal_position).reshape(2,1)
            if self.outdoor_robot_setting and self.curr_global_plan_path is not None and np.linalg.norm(p_init-p_goal) > 0.75:
                self.g_plan_lock.acquire()
                curr_global_plan_path = deepcopy(self.curr_global_plan_path)
                self.g_plan_lock.release()

                start_min = min(3, len(curr_global_plan_path)-1)
                if len(curr_global_plan_path) < 2:
                    rob_gplan_poses = np.atleast_2d(np.array(curr_global_plan_path))
                    rob_gplan = np.atleast_1d(rob_gplan_poses[:2])
                else:
                    rob_gplan_poses = np.array(curr_global_plan_path).T[:, start_min:]
                    rob_gplan = rob_gplan_poses[:2, :]

                curpos_diff = np.linalg.norm(p_init - p_goal, axis=0).item()
                target_start_diff = curpos_diff - self.pref_vel_rob * self.time_step
                plan_dists_to_end = np.linalg.norm(rob_gplan - p_goal, axis=0)
                start_idx = 0
                while start_idx < len(plan_dists_to_end)-2 and plan_dists_to_end[start_idx] > target_start_diff:
                    start_idx += 1
                rob_gplan_poses = np.atleast_2d(rob_gplan_poses[:, start_idx:])
                rob_gplan = rob_gplan[:, start_idx:]

                # new way to get dists:
                # start with the first point in the plan, then for each point in the plan, get the distance to the next point
                dist_to_first_point = np.linalg.norm(rob_gplan[:,0] - p_init[:,0])
                diffs_to_next = np.linalg.norm(rob_gplan[:,1:] - rob_gplan[:,:-1], axis=0)
                while not diffs_to_next.shape[0] == 0 and np.mean(diffs_to_next) < self.pref_vel_rob * self.time_step and curpos_diff > 0.25 :
                    rob_gplan = rob_gplan[:, ::2]
                    rob_gplan_poses = rob_gplan_poses[:, ::2]
                    diffs_to_next = np.linalg.norm(rob_gplan[:,1:] - rob_gplan[:,:-1], axis=0)

                dists = np.concatenate([[dist_to_first_point], dist_to_first_point+diffs_to_next])
                resampled_plan = np.zeros((2, self.horiz+1))
                # resampled_plan_thetas = np.zeros(self.horiz+1)
                if self.latest_static_obs is None or not hasattr(self, 'latest_static_obs_dists'):
                    pref_vel_rob = 0.05
                elif len(self.latest_static_obs_dists) == 0:
                    pref_vel_rob = self.pref_vel_rob
                else:
                    dists_mask = np.ones_like(self.latest_static_obs_dists, dtype=bool)
                    # look at the closest static obstacle that the robot might collide with actually, rather than things on the
                    if self.latest_state is not None:
                        cur_vel_rob = self.latest_state.self_state.lvel
                        dists_mask = np.abs(self.latest_static_obs_int_y) < self.mpc_env.width / 2.0
                        dists_mask &= self.latest_static_obs_int_y > -self.mpc_env.width / 2.0
                        if cur_vel_rob < -0.1:
                            dists_mask &= self.latest_static_obs_int_x < 2.0
                        elif cur_vel_rob > 0.1:
                            dists_mask &= self.latest_static_obs_int_x > -2.0

                    closest_dist = np.array(self.latest_static_obs_dists[dists_mask].tolist()+[np.inf])
                    closest_static_obs = np.min(closest_dist)

                    closest_dist_side = np.min(np.array(self.latest_static_obs_dists.tolist()+[np.inf]))
                    # mask any point where the self.latest_static_obs_py absolute
                    # preferred velocity is TTC of 2 seconds with closest static obstacle

                    pref_vel_rob = max(0.05, min(self.pref_vel_rob, closest_static_obs/4))

                    pref_vel_rob_sides = max(0.05, min(self.pref_vel_rob, closest_dist_side/1.0))
                    pref_vel_rob = min(pref_vel_rob, pref_vel_rob_sides)

                # properly resample the plan so that the points are equidistant based on target dist
                p_current = p_init[:,0]
                for idx in range(self.horiz+1):
                    target_dist = pref_vel_rob * (idx+1) * self.time_step
                    gplan_idx = min(np.argmin(np.abs(dists - target_dist)), rob_gplan.shape[1]-1)
                    dist_diff = dists[gplan_idx] - target_dist
                    norm_val = np.linalg.norm(rob_gplan[:, gplan_idx] - p_current)
                    if norm_val < 1e-6:
                        resampled_plan[:, idx] = rob_gplan[:, gplan_idx]
                    else:
                        vec = (rob_gplan[:, gplan_idx] - p_current) / norm_val
                        resampled_plan[:, idx] = p_current + pref_vel_rob*self.time_step * vec
                    p_current = resampled_plan[:, idx]

                resampled_plan_thetas = np.arctan2(resampled_plan[1, 1:] - resampled_plan[1, :-1], resampled_plan[0, 1:] - resampled_plan[0, :-1])
                resampled_plan_thetas = np.concatenate([resampled_plan_thetas, [rob_gplan_poses[2, gplan_idx]]])

                self.gen_ref_traj(state)
                x_ref_regular = np.zeros((self.mpc_env.nx, self.horiz+1))
                x_ref_regular[:2, :] = resampled_plan
                x_ref_regular[2, :] = resampled_plan_thetas
                remain = max(0, self.horiz+1-self.ref_poses_all.shape[1])
                x_ref_regular[3:,:] = np.concatenate([self.ref_poses_all[3:, :min(self.ref_poses_all.shape[1], self.horiz+1)],
                                            np.tile(self.ref_poses_all[3:, -1:], (1, remain))
                                        ], -1)

                u_ref_regular = np.zeros((self.mpc_env.nu, self.horiz))
                # global plan is 2xN, where N is the number of points in the global plan we need insted to have self.horiz instead of N. So resample via linear interpolation in a way that if it is
            else:
                p_goal = np.array(state.self_state.goal_position).reshape(2,1)
                if self.ref_type == 'new_path_eachstep' and np.linalg.norm(p_init-p_goal) > self.robot_radius*2:
                    self.gen_ref_traj(state)
                p_ref_diff = np.linalg.norm((self.ref_poses_all[:2, :] - p_init), axis=0)
                curgoal_diff = np.linalg.norm(p_init-p_goal)
                refgoal_diff = np.linalg.norm((self.ref_poses_all[:2, :] - p_goal), axis=0)
                p_ref_diff[refgoal_diff>curgoal_diff] = np.inf
                start = np.argmin(p_ref_diff)
                end = min(start + self.horiz, self.ref_actions_all.shape[-1])
                if state.self_state.g_theta is None:
                    g_theta = state.self_state.theta
                else:
                    g_theta = state.self_state.g_theta
                if curgoal_diff < 0.02 and np.abs(state.self_state.theta - g_theta) < 2 * np.pi/180.:
                    start = self.ref_actions_all.shape[-1]-1
                    end = self.ref_actions_all.shape[-1]

                remain = max(0, self.horiz - (end - start))
                x_ref_regular = np.concatenate([self.ref_poses_all[:, start:end+1],
                                            np.tile(self.ref_poses_all[:, -1:], (1, remain))
                                        ], -1)
                u_ref_regular = np.concatenate([self.ref_actions_all[:, start:end],
                                        np.tile(self.pos_ctrl_inv_vel, (1, remain))
                                        ], -1)
        elif ref_type == 'point_stab':
            p_init = np.array(state.self_state.position).reshape(2,1)
            p_goal = np.array(state.self_state.goal_position).reshape(2,1)
            curgoal_diff = np.linalg.norm(p_init-p_goal)
            if np.linalg.norm(p_init-p_goal) > self.robot_radius:
                self.gen_ref_traj(state)
                refgoal_diff = np.linalg.norm((self.ref_poses_all[:2, :] - p_goal), axis=0)
                p_ref_diff = np.linalg.norm((self.ref_poses_all[:2, :] - p_init), axis=0)
                p_ref_diff[refgoal_diff>curgoal_diff] = np.inf
                start = 0
                end = min(self.horiz, self.ref_actions_all.shape[-1])
            else:
                start = self.ref_actions_all.shape[-1]-1
                end = self.ref_actions_all.shape[-1]

            if curgoal_diff < 0.02 and np.abs(state.self_state.theta - np.pi/2) < 2 * np.pi/180.:
                start = self.ref_actions_all.shape[-1]-1
                end = self.ref_actions_all.shape[-1]

            remain = max(0, self.horiz - (end - start))
            x_ref_regular = np.concatenate([self.ref_poses_all[:, start:end+1],
                                        np.tile(self.ref_poses_all[:, -1:], (1, remain))
                                        ], -1)
            u_ref_regular = np.concatenate([self.ref_actions_all[:, start:end],
                                    np.tile(self.pos_ctrl_inv_vel, (1, remain))
                                    ], -1)

            if np.linalg.norm(p_init-p_goal) > self.robot_radius:
                if p_ref_diff[-1] < 1:
                    idx = self.ref_actions_all.shape[-1]-1
                else:
                    idx = np.argmin(np.abs(1 - p_ref_diff))

                x_ref_regular[:2, :] = np.tile(self.ref_poses_all[:2, idx:idx+1], (1, x_ref_regular.shape[-1]))



        else:
            raise NotImplementedError
        # take ref poses for robot, but take hallucenation of how humans would move if the robot were just stationary the whole mpc horizon
        return x_ref_regular, np.zeros(u_ref_regular.shape)


    def generate_traj(self, joint_state, ref_steps, x_rob=None, u_rob=None, use_casadi_init=None, for_guess=False):
        """Generates a trajectory from the current state to the goal state. If x_rob and/pr u_rob are provided then those are taken as the trajectory
           and only the ORCA agents are simulated forward to fill the rest of the state.

        :param joint_state: starting state
        :param ref_steps: number of steps to generate trajectory for
        :param x_rob: a set of robot states, will be calculated from u_rob if None, defaults to None
        :param u_rob: a set of robot actions, will be a linear traj at maximum accelerateion if None, defaults to None
        :param use_casadi_init: simulate forward with casadi implementation of ORCA, defaults to None
        :param for_guess: whether or not the trajectory is for x_guess, u_guess. If it is for guess then humans projected forward with cvmm, defaults to False
        :return: a vector of states and a vector of actions for the whole environment
        """
        if use_casadi_init == None:
            use_casadi_init = self.use_casadi_init
        self_state = joint_state.self_state
        init_dist = np.sqrt((self_state.gx - self_state.px)**2+(self_state.gy - self_state.py)**2)
        px, py = self_state.px, self_state.py
        init_dist = np.sqrt((self_state.gx - px)**2+(self_state.gy - py)**2)

        dpg = np.array([self_state.gx - px, self_state.gy - py])
        theta_enroute = np.arctan2(dpg[1], dpg[0])
        # see the number of steps required to align (only rotate toward) with the goal
        if np.linalg.norm(dpg) > self.robot_radius:
            N_req_init_angle = 0
        else:
            init_pose_diff = residual(np.array([py, px, theta_enroute]), np.array([px, py, self_state.theta]))
            tot_init_angle_change = init_pose_diff[2]
            N_req_init_angle = int(np.ceil(np.abs(tot_init_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        theta_target = theta_enroute # 0.999*np.pi if self_state.gx > 0.0 else 0.0
        final_pose_diff = residual(np.array([self_state.gy, self_state.gx, theta_target]), np.array([py, px, theta_enroute]))
        tot_angle_change = final_pose_diff[2]

        N_req_angle_arrival = int(np.ceil(np.abs(tot_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        N_req_move = int(np.ceil(init_dist / (self.time_step * self.mpc_env.pref_speed)))
        N_req = N_req_move+N_req_init_angle

        # initialize vectors for reference states and actions of the robot
        if x_rob is None:
            ref_x = np.zeros(ref_steps+1, dtype=float) # here
            ref_y = np.zeros(ref_steps+1, dtype=float)
            ref_th = np.zeros(ref_steps+1, dtype=float)
            ref_x[0] = self_state.px
            ref_y[0] = self_state.py
            ref_th[0] = self_state.theta
            # reference actions
            ref_v = np.zeros(ref_steps, dtype=float)
            ref_om = np.zeros(ref_steps, dtype=float)
        else:
            ref_x = x_rob[0,:]
            ref_y = x_rob[1,:]
            ref_th = x_rob[2,:]
            # reference actions
            ref_v = u_rob[0,:]
            ref_om = u_rob[1,:]

        start_idx = 1
        if x_rob is None:
            for idx in range(start_idx, ref_steps+1):
                dpg_x = self_state.gx - ref_x[idx-1]
                dpg_y = self_state.gy - ref_y[idx-1]
                dist_to_goal = np.sqrt(dpg_x**2+dpg_y**2)

                target_heading = np.arctan2(dpg_y, dpg_x) if np.abs(dpg_y) > 1e-5 or np.abs(dpg_x) > 1e-5 else theta_target

                z1 = np.array([self_state.gx, self_state.gy, target_heading])
                z2 = np.array([ref_x[idx-1], ref_y[idx-1], ref_th[idx-1]])
                dpose = residual(z1, z2)
                dpg_theta = dpose[2]


                if u_rob is None:
                    if idx < N_req and idx > N_req_init_angle:
                        v_pref = self.mpc_env.pref_speed
                        ref_v[idx-1] = v_pref
                        ref_om[idx-1] = dpg_theta / self.time_step
                    elif idx == N_req:
                        v_pref = np.sqrt(dpg_x**2+dpg_y**2) / self.time_step
                        ref_v[idx-1] = v_pref
                        ref_om[idx-1] = dpg_theta / self.time_step
                    else:
                        v_pref = 0.0
                        ref_v[idx-1] = v_pref
                        if dpg_theta > 0.0:
                            corrected_dpg_theta = max(self.mpc_env.max_rot * self.time_step, dpg_theta)
                        else:
                            corrected_dpg_theta = min(-self.mpc_env.max_rot * self.time_step, dpg_theta)
                        ref_om[idx-1] = corrected_dpg_theta / self.time_step


                else:
                    ref_v[idx-1] = u_rob[0,idx-1]
                    ref_om[idx-1] = u_rob[1,idx-1]

                unwrapped_theta = (ref_th[idx-1] + self.time_step * ref_om[idx-1]) % (2 * np.pi)
                next_theta = unwrapped_theta - 2 * np.pi if unwrapped_theta >= np.pi else unwrapped_theta # my addition to wrap to (-pi, pi
                ref_x[idx] = ref_x[idx-1] + self.time_step * ref_v[idx-1] * np.cos(next_theta)
                ref_y[idx] = ref_y[idx-1] + self.time_step * ref_v[idx-1] * np.sin(next_theta)
                ref_th[idx] = next_theta

        # Simulate forward other agents via ORCA
        hum_offset = self.mpc_env.nx_r+self.mpc_env.np_g
        oc_next_val = self.convert_to_mpc_state_vector(joint_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)
        ref_oc_next_val = np.zeros((self.mpc_env.nX_hums, ref_steps+1), dtype=float)
        if use_casadi_init:
            ref_oc_next_U_val = np.zeros((self.mpc_env.nVars_hums, ref_steps), dtype=float)
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            ref_oc_lambdas = np.zeros((self.mpc_env.nLambda, ref_steps), dtype=float)
        ref_oc_next_val[:, 0] = np.array(oc_next_val[hum_offset:, 0])
        for t_step in range(ref_steps):
            col_idx = t_step+1

            if not for_guess or (self.mpc_env.hum_model == 'orca_casadi_kkt' and for_guess and self.mpc_env.orca_kkt_horiz > 0 and t_step <= self.mpc_env.orca_kkt_horiz):
                # get the next state from the ORCA callback
                if use_casadi_init:
                    oc_next_hum_val, next_U_hums, next_lambda_hums = self.mpc_env.casadi_orca.optimize_all(oc_next_val)
                    ref_oc_next_U_val[:, t_step] = next_U_hums[:,0]
                else:
                    oc_next_hum_val = self.mpc_env.callback_orca(oc_next_val).toarray()
                    next_lambda_hums = np.zeros((self.mpc_env.nLambda, 1))
            else:
                # get the next state by simulating the hums forward with CVMM
                slice_indices_posns = np.array([[idx1, idx1+1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)
                prev_X_hums = np.take(ref_oc_next_val[:, col_idx-1], slice_indices_posns, axis=0)
                slice_indices_goals = np.array([[idx1+4, idx1+5] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)
                prev_G_hums = np.take(ref_oc_next_val[:, col_idx-1], slice_indices_goals, axis=0)
                if use_casadi_init:
                    prev_U_hums_all = ref_oc_next_U_val[:, t_step-1]
                    ref_oc_next_U_val[:, t_step] = prev_U_hums_all
                else:
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        slice_indices = np.array([[idx1+2, idx1+3, -1, -1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                        prev_U_hums_all = np.take(np.hstack([ref_oc_next_val[:, col_idx-1], 0]), slice_indices, axis=0)
                        prev_U_just_vals = prev_U_hums_all.reshape(self.mpc_env.num_hums, self.mpc_env.nvars_hum).T
                        prev_U_hums = prev_U_just_vals[:-1,:].T.reshape(self.mpc_env.num_hums*2)
                    else:
                        slice_indices = np.array([[idx1+2, idx1+3] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                        prev_U_hums = np.take(ref_oc_next_val[:, col_idx-1], slice_indices, axis=0)

                next_X_hums = prev_X_hums + self.mpc_env.time_step * prev_U_hums
                oc_next_hum_val = np.array([[next_X_hums[idx1*2], next_X_hums[idx1*2+1], prev_U_hums[idx1*2], prev_U_hums[idx1*2+1], prev_G_hums[idx1*2], prev_G_hums[idx1*2+1]] for idx1 in range(self.mpc_env.num_hums)]).reshape(self.mpc_env.nX_hums, 1)
                next_lambda_hums = np.zeros((self.mpc_env.nLambda, 1))

            # add the value of the next state to the reference array
            ref_oc_next_val[:, col_idx] = oc_next_hum_val.reshape(oc_next_hum_val.shape[0],)
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                ref_oc_lambdas[:, col_idx-1] = next_lambda_hums[:,0]
            # generate next environment state to iterate next
            oc_next_rob_val = np.vstack([ref_x[col_idx], ref_y[col_idx], np.sin(ref_th[col_idx]), np.cos(ref_th[col_idx]), ref_v[t_step], ref_om[t_step], 0.0, 0.0, self_state.gx, self_state.gy])
            oc_next_val = np.vstack([oc_next_rob_val, oc_next_hum_val])

        if self.mpc_env.hum_model == 'orca_casadi_simple' or self.mpc_env.hum_model == 'orca_casadi_kkt':
            if use_casadi_init:
                ref_hum_U = ref_oc_next_U_val
            else:
                slice_indices = np.array([[idx1+2, idx1+3, -1, -1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                ref_hum_U = np.take(np.vstack([ref_oc_next_val, np.zeros(ref_oc_next_val.shape[1])]), slice_indices, axis=0)[:,1:]
            omega_0 =  self_state.omega if self_state.omega is not None else 0.0
            ref_X = np.vstack([ref_x, ref_y, np.sin(ref_th), np.cos(ref_th), np.hstack([np.array([np.linalg.norm(self_state.velocity)]), ref_v]), np.hstack([np.array([omega_0]), ref_om]), np.zeros_like(ref_x), np.zeros_like(ref_y), np.ones(ref_x.shape)*self_state.gx, np.ones(ref_x.shape)*self_state.gy]+[ref_oc_next_val[idx, :] for idx in range(ref_oc_next_val.shape[0])])
            ref_U = np.vstack([ref_v, ref_om, ref_hum_U])
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                ref_lambdas = ref_oc_lambdas
                ref_U = np.vstack([ref_U, ref_lambdas])
        else:
            omega_0 =  self_state.omega if self_state.omega is not None else 0.0
            ref_X = np.vstack([ref_x, ref_y, np.sin(ref_th), np.cos(ref_th), np.hstack([np.array([np.linalg.norm(joint_state.self_state.velocity)]), ref_v]), np.hstack([np.array([omega_0]), ref_om]), np.zeros_like(ref_x), np.zeros_like(ref_y), np.ones(ref_x.shape)*self_state.gx, np.ones(ref_x.shape)*self_state.gy, ref_oc_next_val])
            ref_U = np.vstack([ref_v, ref_om])
        assert ref_X.shape[0] == self.mpc_env.nx
        assert ref_U.shape[0] == self.mpc_env.nu
        return ref_X, ref_U


    def gen_ref_traj(self, state):
        """Generates a reference trajectory as the straight-line distance from

        :param state: the joint state
        :return: reference values for states and actions
        """
        self_state = state.self_state
        px, py = self_state.px, self_state.py
        init_dist = np.sqrt((self_state.gx - px)**2+(self_state.gy - py)**2)

        dpg = np.array([self_state.gx - px, self_state.gy - py])
        theta_enroute = np.arctan2(dpg[1], dpg[0])
        # see the number of steps required to align (only rotate toward) with the goal
        if np.linalg.norm(dpg) > self.robot_radius:
            N_req_init_angle = 0
        else:
            init_pose_diff = residual(np.array([py, px, theta_enroute]), np.array([px, py, self_state.theta]))
            tot_init_angle_change = init_pose_diff[2]
            N_req_init_angle = int(np.ceil(np.abs(tot_init_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        theta_target = theta_enroute
        final_pose_diff = residual(np.array([self_state.gy, self_state.gx, theta_target]), np.array([py, px, theta_enroute]))
        tot_angle_change = final_pose_diff[2]

        N_req_angle_arrival = int(np.ceil(np.abs(tot_angle_change) / (self.time_step * self.mpc_env.max_rot)))
        N_req_angle = N_req_init_angle + N_req_angle_arrival
        N_req_move = int(np.ceil(init_dist / (self.time_step * self.mpc_env.pref_speed)))
        ref_steps = N_req_move+N_req_angle+N_req_angle_arrival+2
        # reference states
        ref_poses, ref_actions = self.generate_traj(state, ref_steps)

        self.ref_poses_all = ref_poses
        self.ref_actions_all = ref_actions
        self.pos_ctrl_inv_vel = np.zeros((self.mpc_env.nu,1))


    def get_int_goal(self, state):
        """_summary_

        :param state: _description_
        """
        self_state = state.self_state
        ys = [self_state.py, self_state.gy]
        if (self.env.sim_env == "hallway_static" or self.env.sim_env == "hallway_static_with_back" or self.env.sim_env == "hallway_bottleneck") and np.min(ys) < self.env.door_y_mid_min and np.max(ys) > self.env.door_y_mid_max:
            int_gx = self.env.door_x_mid
            int_gy = 0.5 * (self.env.door_y_min + self.env.door_y_max)
            vec = np.array([int_gx - self_state.px, int_gy - self_state.py])
            vec_norm = np.linalg.norm(vec)
            if np.linalg.norm(vec) < self.mpc_env.max_speed * self.time_step * self.horiz:
                vec = self.horiz * self.mpc_env.max_speed * self.time_step * 1.01 * vec / vec_norm
            gx = self_state.px + vec[0]
            gy = self_state.py + vec[1]
        else:
            gx, gy = self_state.gx, self_state.gy
        return gx, gy



    def get_ref_traj(self, state):
        """Generate a reference trajectory for the controller to follow based on the type of MPC controller configured

        :param state: starting state object
        :raises NotImplementedError: if reference type is not one of point stabalization, path following, or trajectory tracking
        :return: reference states and reference actions
        """
        if self.new_ref_each_step or (self.ref_type == 'point_stab' and (self.env.sim_env == "hallway_static" or self.env.sim_env == "hallway_static_with_back" or self.env.sim_env == "hallway_bottleneck")):
            gx, gy = self.get_int_goal(state)
            state.self_state.gx = gx
            state.self_state.gy = gy
            state.self_state.goal_position = (gx, gy)
            self.gen_ref_traj(state)

        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        # Adapted from https://github.com/utiasDSL/safe-control-gym/
        # (file safe-control-gym/safe_control_gym/controllers/mpc/mpc.py)
        if self.ref_type == 'traj_track':
        # Just follow reference with time, no matter where the robot is at time step i
            start = min(self.traj_step, self.ref_actions_all.shape[-1])
            end = min(self.traj_step + self.horiz, self.ref_actions_all.shape[-1])
        elif self.ref_type == 'path_foll':
        # Take closest reference point at current timestep for reference traj
            p_init = np.array(state.self_state.position).reshape(2,1)
            p_goal = np.array(state.self_state.goal_position).reshape(2,1)
            p_ref_diff = np.linalg.norm((self.ref_poses_all[:2, :] - p_init), axis=0)
            curgoal_diff = np.linalg.norm(p_init-p_goal)
            refgoal_diff = np.linalg.norm((self.ref_poses_all[:2, :] - p_goal), axis=0)
            p_ref_diff[refgoal_diff>curgoal_diff] = np.inf
            start = np.argmin(p_ref_diff)
            end = min(start + self.horiz, self.ref_actions_all.shape[-1])
        elif self.ref_type == 'point_stab':
            start = self.ref_actions_all.shape[-1]-1
            end = self.ref_actions_all.shape[-1]
        else:
            raise NotImplementedError

        remain = max(0, self.horiz - (end - start))
        x_ref_regular = np.concatenate([self.ref_poses_all[:, start:end+1],
                                    np.tile(self.ref_poses_all[:, -1:], (1, remain))
                                   ], -1)
        u_ref_regular = np.concatenate([self.ref_actions_all[:, start:end],
                                np.tile(self.pos_ctrl_inv_vel, (1, remain))
                                ], -1)

        # take ref poses for robot, but take hallucenation of how humans would move if the robot were just stationary the whole mpc horizon
        return x_ref_regular, np.zeros(u_ref_regular.shape)



    def init_fwding_function(self):
        X_r_fwded = cs.MX.zeros((self.mpc_env.nx_r, self.mpc_env.horiz))
        X_r_vec = cs.MX.sym('X_r_vec', self.mpc_env.nx_r, self.mpc_env.horiz)
        U_vec = cs.MX.sym('U_vec', self.mpc_env.nu_r, self.mpc_env.horiz)
        for idx in range(self.mpc_env.horiz):
            X_r_fwded[:, idx] = self.mpc_env.next_X_r_fn(X_r_vec[:, idx], U_vec[:, idx], 1./self.rate)

        self.X_r_fwd_atrate = cs.Function('X_r_fwd_atrate', [X_r_vec, U_vec], [X_r_fwded], ['X_r_vec', 'U_vec'], ['X_r_fwded']).expand()

    def init_solver(self, file_run_id=None):
        self.acados_solver, self.acados_ocp = export_campc_ocp(self.mpc_env, file_run_id=file_run_id)
        if self.mpc_env.human_pred_MID:
            if self.outdoor_robot_setting:
                self.stage_con_fn = cs.Function('stage_con_fn', [cs.vertcat(self.mpc_env.X, self.mpc_env.U, self.mpc_env.stat_obs_params_vecced), self.mpc_env.MID_samples_t_all_hums_stacked], [self.acados_solver.acados_ocp.model.con_h_expr])
                self.term_con_fn = cs.Function('term_con_fn', [cs.vertcat(self.mpc_env.X, self.mpc_env.stat_obs_params_vecced), self.mpc_env.MID_samples_t_all_hums_stacked], [self.acados_solver.acados_ocp.model.con_h_expr_e])
            else:
                self.stage_con_fn = cs.Function('stage_con_fn', [cs.vertcat(self.mpc_env.X, self.mpc_env.U), self.mpc_env.MID_samples_t_all_hums_stacked], [self.acados_solver.acados_ocp.model.con_h_expr])
                self.term_con_fn = cs.Function('term_con_fn', [cs.vertcat(self.mpc_env.X), self.mpc_env.MID_samples_t_all_hums_stacked], [self.acados_solver.acados_ocp.model.con_h_expr_e])
        else:
            if self.outdoor_robot_setting:
                self.stage_con_fn = cs.Function('stage_con_fn', [cs.vertcat(self.mpc_env.X, self.mpc_env.U, self.mpc_env.stat_obs_params_vecced)], [self.acados_solver.acados_ocp.model.con_h_expr])
                self.term_con_fn = cs.Function('term_con_fn', [cs.vertcat(self.mpc_env.X, self.mpc_env.stat_obs_params_vecced)], [self.acados_solver.acados_ocp.model.con_h_expr_e])
            else:
                self.stage_con_fn = cs.Function('stage_con_fn', [cs.vertcat(self.mpc_env.X, self.mpc_env.U)], [self.acados_solver.acados_ocp.model.con_h_expr])
                self.term_con_fn = cs.Function('term_con_fn', [cs.vertcat(self.mpc_env.X)], [self.acados_solver.acados_ocp.model.con_h_expr_e])


    def init_human_prediction(self):
        if not self.human_pred_MID_vanil_as_joint and self.human_pred_MID_joint:
            self.hum_traj_forecaster = HumanTrajectoryForecasterSim(env_config=self.env.config, mid_config_file="sicnav_diffusion/JMID/test_time_configs/mid_jp.yaml")
        else:
            self.hum_traj_forecaster = HumanTrajectoryForecasterSim(env_config=self.env.config, mid_config_file="sicnav_diffusion/JMID/test_time_configs/mid.yaml")

    def init_gt_callback_orca(self, env, test_case):
        ob, static_obs = env.reset('test', test_case, return_stat=True)
        joint_state = env.robot.get_joint_state(ob, static_obs)
        self.gt_orca = ORCACWrapper('callback_orca', self.time_step, joint_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, env.config, {'enable_fd':True}, nx_hum=self.mpc_env.nx_hum, num_humans=self.num_hums)

    # Overwrite set_env to also configure the policy's env-related values
    def set_env(self, env, file_run_id=None):
        super().set_env(env)
        env.set_human_observability(False)
        tot_time = env.time_limit
        self.time_step = env.time_step
        self.rate = 1.0 / self.time_step # rate of the mpc solver, i.e. how many times it runs per second this can be different from the env discretization time step when running on a real robot

        _ = env.reset() # to ensure that static obstacles are created as an object.

        init_start_time = time.time()
        env_config = env.config
        policy_config = self.config
        sim_env = env
        self.time_step = env_config.getfloat('env', 'time_step')

        self.robot_radius = env_config.getfloat('robot', 'radius')
        # nb the v_pref in the state objects is actually used for max speed in the orca constraints, both for the warmstart and for the humans
        self.pref_vel_rob = policy_config.getfloat('mpc_env', 'max_speed')

        self.human_radius = policy_config.getfloat('humans', 'radius')
        self.human_max_vel = policy_config.getfloat('humans', 'v_pref')


        self.num_hums = env_config.getint('sim', 'human_num')

        self.forecaster_publish_ts = env_config.getfloat('human_trajectory_forecaster', 'publish_freq')

        self.configure(policy_config)

        self.static_obs = dummy_static_obs = deepcopy(sim_env.static_obstacles)

        # compile solver with dummy state and observations
        dummy_robot_state = FullState(px=0.0, py=0.0, vx=0.0, vy=0.1, radius=self.robot_radius, gx=0.0, gy=1.0, v_pref=self.pref_vel_rob, theta=np.pi/2.0)

        human_xs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        human_ys = [-3.0, -2.0, 1.0, 2.0, 4.4, 5.0, 6.0, 7.0]

        self.dummy_obs = dummy_obs = []
        for h_idx in range(self.num_hums):
            dummy_vx = 0.0
            dummy_vy = 0.5
            dummy_theta = np.pi/2
            if self.human_goal_cvmm:
                dummy_gx = human_xs[h_idx] + self.human_goal_cvmm_horizon * dummy_vx
                dummy_gy = human_ys[h_idx] + self.human_goal_cvmm_horizon * dummy_vy
                dummy_obs.append(FullState(px=human_xs[h_idx], py=human_ys[h_idx], vx=dummy_vx, vy=dummy_vy, radius=self.human_radius, gx=dummy_gx, gy=dummy_gy, v_pref=self.human_max_vel, theta=dummy_theta))
            else:
                dummy_obs.append(FullState(px=human_xs[h_idx], py=human_ys[h_idx], vx=0.0, vy=0.001, radius=self.human_radius, gx=human_xs[h_idx], gy=human_ys[h_idx], v_pref=self.human_max_vel, theta=np.pi/2))


        dummy_state = FullyObservableJointState(dummy_robot_state, dummy_obs, dummy_static_obs)

        self.configure(policy_config)



        self.mpc_env_creation_args = mpc_env_creation_args = {
            'time_step' : self.time_step,
            'joint_state' : dummy_state,
            'num_hums' : self.num_hums,
            'K' : self.horiz,
            'dummmy_human_args' : {'env_config':env_config, 'section':'humans', 'fully_observable':True},
            'config' : policy_config,
            'env_config' : env_config,
        }
        self.mpc_env = MPCEnv(self.time_step, dummy_state, self.num_hums, self.horiz, policy_config, env_config, isSim=True)




        self.warmstart = True
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            self.init_warmstart_solver()
        self.reuse_K = self.horiz

        self.init_fwding_function()

        print('[CAMPC] done basic setup of mpc, now initialize solver and tracked values')
        self.init_solver(file_run_id=file_run_id)
        self.reset_scenario_values()
        start_time = time.time()

        if self.human_pred_MID:
            # ie if using simulator
            print('[CAMPC] done initialize solver, now first ref traj generation')
            self.gen_ref_traj(dummy_state)
            print('[CAMPC] done first ref traj generation, now running once to potentially compile')
            goal_states, goal_actions = self.get_ref_traj(dummy_state)
            if not self.human_pred_MID_joint:
                forecasts_init_weights = np.log(np.ones((self.num_hums, self.mpc_env.num_MID_samples))/self.mpc_env.num_MID_samples)
            else:
                forecasts_init_weights = np.log(np.ones((self.mpc_env.num_MID_samples))/self.mpc_env.num_MID_samples)
            mpc_state = self.convert_to_mpc_state_vector(dummy_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, forecasts_init_weights, get_numpy=True)
            # make the forecasts just be the goals for every sample
            dummy_forecasts_list = []
            for h_idx in range(self.num_hums):
                # make a self.horiz+1 x self.mpc_env.num_MID_samples x dim array where the dim dimension is 2 for x and y and contains repetitions of the mpc_state's current goal position for each human
                dummy_forecasts_list.append(np.tile(np.expand_dims(np.array([dummy_state.human_states[h_idx].gx, dummy_state.human_states[h_idx].gy]), axis=0), (self.horiz+1, self.mpc_env.num_MID_samples, 1)))
            # now stack dummy_forecasts list on the axis1
            forecasts_reshaped = np.concatenate(dummy_forecasts_list, axis=1)
            # forecasts_reshaped = np.zeros((self.horiz+1, self.mpc_env.num_hums*self.mpc_env.num_MID_samples, 2))
            _ = self.select_action(mpc_state, dummy_state, goal_states, goal_actions, MID_samples=forecasts_reshaped)
            self.iterate_save_file = 'temp_files/iterate_save_file_{:}.json'.format(file_run_id) if file_run_id is not None else 'temp_files/iterate_save_file.json'
        else:
            print('[CAMPC] done initialize solver, now first ref traj generation')
            self.gen_ref_traj(dummy_state)
            print('[CAMPC] done first ref traj generation, now running once to potentially compile')
            goal_states, goal_actions = self.get_ref_traj(dummy_state)
            mpc_state = self.convert_to_mpc_state_vector(dummy_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)
            _ = self.select_action(mpc_state, dummy_state, goal_states, goal_actions)
            self.iterate_save_file = 'temp_files/iterate_save_file_{:}.json'.format(file_run_id) if file_run_id is not None else 'temp_files/iterate_save_file.json'
            if not os.path.exists('temp_files'):
                os.makedirs('temp_files')
            self.acados_solver.store_iterate(filename=self.iterate_save_file, overwrite=True)

        end_time = time.time()
        print('[CAMPC] done running optimization once, it took {:.3f} seconds.'.format(end_time-start_time))
        print('[CAMPC] done initiaizing mpc')
        init_end_time = time.time()
        #format end_time - start time as mm:ss.SSS
        print('[CAMPC] done initializing mpc, it took {:.3f} minutes.'.format((init_end_time-init_start_time)/60.0))


    def reset_scenario_values(self):
            self.traj_step = 0
            self.prev_lvel = 0.0
            self.prev_avel = 0.0
            self.x_prev = None
            self.u_prev = None
            self.num_prev_used = 0
            self.mpc_progress = None # Track if we're moving forward or not.
            self.all_goal_diff = []
            self.all_x_val = []
            self.all_u_val = []
            self.all_x_guess = []
            self.all_x_goals = []
            self.all_u_guess = []
            self.all_u_goals = []
            self.all_debug_text = []
            self.mpc_sol_succ = []
            self.calc_times = []
            self.all_prev_status = []
            self.all_prev_num_iter = []
            self.solver_summary = {
                'traj_step' : [],
                'sol_success' : [],
                'optim_status' : [],
                'iter_count' : [],
                'prep_time' : [],
                'sol_time' : [],
                'final_nopenal_cost' : [],
                'final_term_cost' : [],
                'debug_text' : [],
                'ipopt_iterations' : [],
            }
            if self.human_pred_MID:
                self.all_forecasts = []
                self.all_weights = []
                self.all_cum_weights = []
                self.all_forecasts_shaped = []
                self.all_log_weights = []
                self.all_log_weights_unnormed = []
                # also redefine the human trajectory forecaster object in case there is some statefulness.
                self.init_human_prediction()

                start_time_step = self.env.global_time_step - self.hum_traj_forecaster.num_hist_frames-1
                end_time_step = self.env.global_time_step-1
                time_steps = np.arange(start_time_step+1, end_time_step+1, 1) * self.time_step
                try:
                    states_history = self.env.states[-self.hum_traj_forecaster.num_hist_frames-1:-1]
                except IndexError as e:
                    logging.error('[CAMPC] ERROR: Could not retrieve enough states from the environment to predict human trajectories. Has enough "starts_moving" steps to generate sufficient history for predictions \n {}'.format(e))
                    raise e
                for idx in range(len(time_steps)):
                    self.hum_traj_forecaster.update_state_hists(states_history[idx][0], states_history[idx][1], time_steps[idx])

    def init_robustness_eval(self, seed, std):
        if std == 0.0:
            self.robustness_eval = False
        else:
            self.robustness_eval = True
        self.rng = np.random.default_rng(seed)
        self.std = std


    def randomize_state_robustness_eval(self, state):
        # randomize robot state:
        rob_px = state.self_state.px + self.rng.normal(0, self.std)
        rob_py = state.self_state.py + self.rng.normal(0, self.std)
        rob_vx = state.self_state.vx + self.rng.normal(0, self.std)
        rob_vy = state.self_state.vy + self.rng.normal(0, self.std)
        rob_theta = np.arctan2(rob_vy, rob_vx)
        new_robot_state = FullState(
            px=rob_px,
            py=rob_py,
            vx=rob_vx,
            vy=rob_vy,
            gx=state.self_state.gx,
            gy=state.self_state.gy,
            radius=state.self_state.radius,
            v_pref=state.self_state.v_pref,
            theta=rob_theta,
        )
        new_human_states = []
        for human_state in state.human_states:
            hum_px = human_state.px + self.rng.normal(0, self.std)
            hum_py = human_state.py + self.rng.normal(0, self.std)
            hum_vx = human_state.vx + self.rng.normal(0, self.std)
            hum_vy = human_state.vy + self.rng.normal(0, self.std)
            hum_theta = np.arctan2(hum_vy, hum_vx)
            new_human_states.append(FullState(
                px=hum_px,
                py=hum_py,
                vx=hum_vx,
                vy=hum_vy,
                gx=human_state.gx,
                gy=human_state.gy,
                radius=human_state.radius,
                v_pref=human_state.v_pref,
                theta=hum_theta,
            ))
        new_stat_obs = []
        for stat_ob in state.static_obs:
            new_stat_ob = [(stat_ob[0][0]+self.rng.normal(0, self.std), stat_ob[0][1]+self.rng.normal(0, self.std)),
                            (stat_ob[1][0]+self.rng.normal(0, self.std), stat_ob[1][1]+self.rng.normal(0, self.std))]
            new_stat_obs.append(new_stat_ob)
        new_state = FullyObservableJointState(new_robot_state, new_human_states, new_stat_obs)
        return new_state


    def bring_fwd(self, joint_state, obs, x_prev, u_prev, for_guess=True, MID_samples=None):
        u_prev_fwded = deepcopy(u_prev)
        x_prev_fwded = deepcopy(x_prev)
        if self.time_step - 1./self.rate > 1e-5:
            print('[CAMPC] [BRING FWD] rate and time_step are not consistent, using prev soln as is.')
            x_r_fwded = self.X_r_fwd_atrate(x_prev_fwded[:self.mpc_env.nx_r, :-1], u_prev_fwded[:self.mpc_env.nu_r, :]).toarray()
            x_prev_fwded[:self.mpc_env.nx_r, :-1] = x_r_fwded
        else:
            u_prev_fwded[:, :-1] = u_prev_fwded[:, 1:]
            x_prev_fwded[:, :-1] = x_prev_fwded[:, 1:]

        if self.mpc_env.hum_model == 'orca_casadi_kkt' and self.warmstart:
            slice_indices_posns = np.array([[self.mpc_env.nx_r+self.mpc_env.np_g+idx1, self.mpc_env.nx_r+self.mpc_env.np_g+idx1+1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)

            if self.time_step - 1./self.rate > 1e-5 or np.any(np.abs(x_prev_fwded[:3, 0] - obs[:3, 0]) > 1e-2) or np.any(np.abs(np.take(x_prev_fwded[:, 0], slice_indices_posns, axis=0) - np.take(obs[:, 0], slice_indices_posns, axis=0)) > self.mpc_env.rob_len_buffer):

                ws_horiz_start = time.time()
                if self.mpc_env.human_pred_MID:
                    print('[CAMPC] [BRING FWD] discrepancy between obs and first timestep of previous solution. Will correct.')
                    stacked_forecasts = einops.rearrange(MID_samples, 't hs d -> (t hs) d')
                    if not self.outdoor_robot_setting:
                        ans = self.warmstart_correction(X_0=obs, X_rob_vec=x_prev_fwded[:self.mpc_env.nx_r,:], MID_samples_stacked=stacked_forecasts)
                    else:
                        ans = self.warmstart_correction(X_0=np.vstack([obs, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,1)]), X_rob_vec=x_prev_fwded[:self.mpc_env.nx_r,:], MID_samples_stacked=stacked_forecasts)
                    x_prev_fwded = ans['X_vec'].toarray()
                    u_prev_fwded = ans['U_vec'].toarray()
                else:
                    print('[CAMPC] [BRING FWD] discrepancy between obs and first timestep of previous solution. Will correct.')
                    # map all print output from stderr and stdout to null for this bit

                    if not self.outdoor_robot_setting:
                        ans = self.warmstart_correction(X_0=obs, X_rob_vec=x_prev_fwded[:self.mpc_env.nx_r,:])
                    else:
                        ans = self.warmstart_correction(params_0=np.vstack([obs, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,1)]), X_rob_vec=x_prev_fwded[:self.mpc_env.nx_r,:])
                    x_prev_fwded = ans['X_vec'].toarray()
                    u_prev_fwded = ans['U_vec'].toarray()
                ws_horiz_end = time.time()
                ws_horiz_time = ws_horiz_end-ws_horiz_start
                print('[CAMPC] [BRING FWD] time: {:.3f}s'.format(ws_horiz_time))
                if ws_horiz_time > 0.02 :
                    logging.warn('[CAMPC] warmstart was {:.3f}s'.format(ws_horiz_time))

            else:
                # run the feasible warmstart for one step
                ws_onestep_start = time.time()
                if self.mpc_env.human_pred_MID:
                    if not self.outdoor_robot_setting:
                        ans = self.warmstart_onestep(X_0=np.atleast_2d(x_prev_fwded[:,-2]).T, MID_samples_0_stacked=MID_samples[-2,:,:], MID_samples_1_stacked=MID_samples[-1,:,:])
                    else:
                        ans = self.warmstart_onestep(X_0=np.vstack([np.atleast_2d(x_prev_fwded[:,-2]).T, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,1)]), MID_samples_0_stacked=MID_samples[-2,:,:], MID_samples_1_stacked=MID_samples[-1,:,:])
                else:
                    if not self.outdoor_robot_setting:
                        ans = self.warmstart_onestep(X_0=np.atleast_2d(x_prev_fwded[:,-2]).T)
                    else:
                        ans = self.warmstart_onestep(params_0=np.vstack([np.atleast_2d(x_prev_fwded[:,-2]).T, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,1)]))
                ws_onestep_end = time.time()
                print('[CAMPC] [BRING FWD] [WARMSTART HORIZ] time: {:.3f}s'.format(ws_onestep_end-ws_onestep_start))
                ws_onsestep_time = ws_onestep_end-ws_onestep_start
                if ws_onsestep_time > 0.02:
                    logging.warn('[CAMPC] warmstart onestep was {:.3f}s'.format(ws_onsestep_time))
                x_next = ans['X_1'].toarray()
                u_next = ans['U_0'].toarray()
                u_prev_fwded[:, self.mpc_env.orca_kkt_horiz-1:] = u_next[:, :]
                x_prev_fwded[:, self.mpc_env.orca_kkt_horiz:] = x_next[:, :]
        else:
            # just simulate forward keeping the same final robot action, except with no rotation
            x_prev_fwded, u_prev_fwded = self.generate_traj(joint_state, self.horiz, x_rob=None, u_rob=u_prev_fwded, for_guess=for_guess)

        return x_prev_fwded, u_prev_fwded


    def select_action(self, obs, joint_state, goal_states, goal_actions, return_all=False, MID_samples=None):
        """Solves nonlinear mpc problem to get next action.
        Args:
            obs (np.array): current state/observation.

        Returns:
            np.array: input/action to the task/env.
        """



        solver = self.acados_solver

        # Select x_guess, u_guess
        start_ref = time.time()
        if self.x_prev is not None and self.u_prev is not None:
            for_guess = True if self.mpc_env.hum_model != 'orca_casadi_kkt' or self.mpc_env.orca_kkt_horiz > 0 else False
            x_prev = self.x_prev
            u_prev = self.u_prev
            bfwd_start = time.time()
            if self.mpc_env.human_pred_MID:
                x_prev_fwded, u_prev_fwded = self.bring_fwd(joint_state, obs, x_prev, u_prev, for_guess=for_guess, MID_samples=MID_samples)
            else:
                x_prev_fwded, u_prev_fwded = self.bring_fwd(joint_state, obs, x_prev, u_prev, for_guess=for_guess)
            bfwd_end = time.time()
            print('[CAMPC] [BRING FWD] Wall Time: {:.3f}s'.format(bfwd_end-bfwd_start))


        if not self.warmstart or (self.mpc_env.hum_model != 'orca_casadi_kkt' and (self.x_prev is None or self.u_prev is None or self.mpc_sol_succ is None or (not self.mpc_sol_succ and self.num_prev_used >= self.reuse_K) or self.new_ref_each_step)):
            for_guess = True if self.mpc_env.hum_model != 'orca_casadi_kkt' or self.mpc_env.orca_kkt_horiz > 0 else False
            x_guess, u_guess = self.generate_traj(joint_state, self.horiz, for_guess=for_guess)
            can_use_guess = False # whether or not the guess can be used for the next step in case optimization fails
        elif self.mpc_env.hum_model == 'orca_casadi_kkt' and (self.x_prev is None or self.u_prev is None or self.mpc_sol_succ is None or not self.mpc_sol_succ or self.new_ref_each_step):
            print('[CAMPC] generating warmstart for entire horizon')
            if self.mpc_env.human_pred_MID:
                stacked_forecasts = einops.rearrange(MID_samples, 't hs d -> (t hs) d')
                if not self.outdoor_robot_setting:
                    if not 'path' in self.ref_type or 'traj' in self.ref_type:
                        ans = self.warmstart_horiz(X_0=obs, MID_samples_stacked=stacked_forecasts)
                    else:
                        ans = self.warmstart_correction(X_0=obs, X_rob_vec=goal_states[:self.mpc_env.nx_r, :], MID_samples_stacked=stacked_forecasts)
                else:
                    if not 'path' in self.ref_type or 'traj' in self.ref_type:
                        ans = self.warmstart_horiz(X_0=np.vstack([obs, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs, 4).reshape(self.mpc_env.num_stat_obs*4, 1)]), MID_samples_stacked=stacked_forecasts)
                    else:
                        ans = self.warmstart_correction(X_0=np.vstack([obs, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,1)]), X_rob_vec=goal_states[:self.mpc_env.nx_r, :], MID_samples_stacked=stacked_forecasts)
            else:
                if not self.outdoor_robot_setting:
                    # the warmstart fn uses a single goal for every step, if doing path or traj tracking, need to give it the goal for each step, then it is correct to use the ws_correction_fn
                    if not 'path' in self.ref_type or 'traj' in self.ref_type:
                        ans = self.warmstart_horiz(X_0=obs)
                    else:
                        ans = self.warmstart_correction(X_0=obs, X_rob_vec=goal_states[:self.mpc_env.nx_r, :])
                else:
                    # the warmstart fn uses a single goal for every step, if doing path or traj tracking, need to give it the goal for each step, then it is correct to use the ws_correction_fn
                    if not 'path' in self.ref_type or 'traj' in self.ref_type:
                        ans = self.warmstart_horiz(params_0=np.vstack([obs, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs, 4).reshape(self.mpc_env.num_stat_obs*4, 1)]))
                    else:
                        ans = self.warmstart_correction(params_0=np.vstack([obs, np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,1)]), X_rob_vec=goal_states[:self.mpc_env.nx_r, :])

                # ans = self.warmstart_horiz(X_0=obs)
            x_guess = ans['X_vec'].toarray()
            u_guess = ans['U_vec'].toarray()

            can_use_guess = True # whether or not the guess can be used for the next step in case optimization fails

        else:
            x_guess = x_prev_fwded
            u_guess = u_prev_fwded
            can_use_guess = True # whether or not the guess can be used for the next step in case optimization fails

        x_guess[:, 0] = obs[:, 0]

        if self.outdoor_robot_setting:
            properly_shaped_static_obs = np.array(joint_state.static_obs).reshape(self.mpc_env.num_stat_obs,4).reshape(self.mpc_env.num_stat_obs*4,)
        # set the values for the constraint
        solver.constraints_set(0, "lbx", x_guess[:, 0])
        solver.constraints_set(0, "ubx", x_guess[:, 0])
        # set the initial guess
        for idx in range(self.horiz):
            if self.mpc_env.human_pred_MID:
                MID_sample_t = MID_samples[idx,:,:]
                MID_sample_tp1 = MID_samples[idx+1,:,:]
                if not self.outdoor_robot_setting:
                    solver.set(idx, "p", np.hstack([goal_states[:,idx], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag, MID_sample_t[:,0], MID_sample_t[:,1], MID_sample_tp1[:,0], MID_sample_tp1[:,1]]))
                else:
                    solver.set(idx, "p", np.hstack([goal_states[:,idx], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag, MID_sample_t[:,0], MID_sample_t[:,1], MID_sample_tp1[:,0], MID_sample_tp1[:,1], properly_shaped_static_obs]))
            else:
                if not self.outdoor_robot_setting:
                    solver.set(idx, "p", np.hstack([goal_states[:,idx], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag]))
                else:
                    solver.set(idx, "p", np.hstack([goal_states[:,idx], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag, properly_shaped_static_obs]))


            solver.set(idx, "x", x_guess[:, idx])
            solver.set(idx, "u", u_guess[:, idx])
        solver.set(self.horiz, "x", x_guess[:, -1])
        if self.mpc_env.human_pred_MID:
            idx = self.horiz-1
            MID_sample_t = MID_samples[idx,:,:]
            MID_sample_tp1 = MID_samples[idx+1,:,:]
            if not self.outdoor_robot_setting:
                solver.set(self.horiz, "p", np.hstack([goal_states[:,idx+1], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag, MID_sample_t[:,0], MID_sample_t[:,1], MID_sample_tp1[:,0], MID_sample_tp1[:,1]]))
            else:
                solver.set(self.horiz, "p", np.hstack([goal_states[:,idx+1], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag, MID_sample_t[:,0], MID_sample_t[:,1], MID_sample_tp1[:,0], MID_sample_tp1[:,1], properly_shaped_static_obs]))
        else:
            # solver.set(self.horiz, "p", np.hstack([goal_states[:,idx+1], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag]))
            if not self.outdoor_robot_setting:
                solver.set(self.horiz, "p", np.hstack([goal_states[:,idx], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag]))
            else:
                solver.set(self.horiz, "p", np.hstack([goal_states[:,idx], goal_actions[:,idx], self.mpc_env.Q_diag, self.mpc_env.R_diag, self.mpc_env.term_Q_diag, properly_shaped_static_obs]))

        # obtain the values
        init_obj_val = solver.get_cost()
        end_ref = time.time()
        refset_time = end_ref - start_ref
        time_text = '. Total prep. time {:.3f}s'.format(refset_time)
        print('[CAMPC] Start solve step {:}{:}'.format(self.traj_step, time_text))
        sol_start = time.time()
        status = solver.solve()
        sol_end = time.time()
        solver.print_statistics()
        final_obj_val = solver.get_cost()
        residuals = solver.get_stats("residuals")
        print('[CAMPC] init_val: {:.3f}'.format(init_obj_val))
        print('[CAMPC] final_val: {:.3f}'.format(final_obj_val))
        debug_text = "[CAMPC] optim. status: {:}. init_fval: {:.4f}, final_fval: {:.4f}. residuals: {:}.\n Prep. Time: {:.3f}. Solve time: {:.3f}.".format(ACADOS_STATUS[status], init_obj_val, final_obj_val, residuals, refset_time, sol_end-sol_start)
        print(debug_text)

        if self.mpc_env.hum_model == 'orca_casadi_kkt' and not self.mpc_env.human_pred_MID:
            succ_cond = (
            (status == 0 and final_obj_val <= init_obj_val and np.all(residuals[1:3] < 5e-2)) or # if we have total success
            (status == 2 and final_obj_val <= init_obj_val and np.all(residuals[1:3] < 5e-2)) or
            (status == 4 and final_obj_val < init_obj_val and np.all(residuals[1:3] < 5e-2))
         )
        else:
            succ_cond = (
            (status == 0 and np.all(residuals[1:3] < 5e-2)) or # if we have total success
            (status == 2 and np.all(residuals[1:3] < 5e-2)) or
            (status == 4 and np.all(residuals[1:3] < 5e-2))
         )

        # Solve the optimization problem.
        if succ_cond:
            self.mpc_sol_succ = True
            # get solution
            x_val = np.zeros_like(x_guess)
            u_val = np.zeros_like(u_guess)
            for i in range(self.horiz):
                x_val[:, i] = solver.get(i, "x")
                u_val[:, i] = solver.get(i, "u")
            x_val[:, self.horiz] = solver.get(self.horiz, "x")

            if u_val.ndim > 1:
                action = u_val[:, 0]
            else:
                action = np.array([u_val[0]])
            self.prev_action = action
            post_sol_end = time.time()
            time_text = ', Wall Time: {:.4f}'.format(post_sol_end-sol_start)
            print('[CAMPC] Optim. success step {:}: {:}.  Num Iter: {:}{:}'.format(self.traj_step, status, solver.get_stats("sqp_iter"), time_text))
        else:
            self.mpc_sol_succ = False
            K_orca = self.mpc_env.orca_kkt_horiz
            if can_use_guess and self.warmstart and self.mpc_env.hum_model == 'orca_casadi_kkt':
                debug_text2 = " USING WARMSTART GUESS"
                if len(u_guess.shape) > 1:
                    action = u_guess[:, 0]
                else:
                    action = np.array([u_guess[0]])
                x_val = x_guess
                u_val = u_guess
            elif can_use_guess and self.num_prev_used < self.reuse_K:
                debug_text2 = " USING PREV. SOLN. FWDED"
                if len(u_prev_fwded.shape) > 1:
                    action = u_prev_fwded[:, 0]
                else:
                    action = np.array([u_prev_fwded[0]])
                x_val = x_prev_fwded
                u_val = u_prev_fwded
                self.num_prev_used += 1
            else:
                debug_text2 = " EMER. BRAKE"
                cur_vel = x_guess[3, 0]
                actions = []
                rob_states = [np.atleast_2d(x_guess[:, 0]).T]
                f_still_moving = True
                count = 1
                while f_still_moving:
                    if count > self.horiz:
                        break
                    if self.mpc_env.dyn_type == 'dynamic': raise NotImplementedError
                    lowest_feasible_vel = np.max((cur_vel + self.mpc_env.max_l_dcc*self.time_step, 0.0))

                    prev_state = rob_states[count-1]
                    next_action = np.array([lowest_feasible_vel, 0.0]+[0.0 for _ in range(self.mpc_env.nVars_hums)]+[0.0 for _ in range(self.mpc_env.nLambda)])
                    actions.append(next_action)
                    next_state = self.mpc_env.system_model.f_func_nonlin(prev_state, next_action).toarray()
                    rob_states.append(next_state)
                    cur_vel = lowest_feasible_vel
                    f_still_moving = lowest_feasible_vel - 0.0 > 1e-10
                    count += 1
                action = actions[0]

                u_val = np.concatenate([np.atleast_2d(action).T for action in actions]+
                                    [np.tile(np.atleast_2d(next_action).T, (1, self.horiz-len(actions)))], -1)
                if self.mpc_env.hum_model == 'orca_casadi_kkt':
                    u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                    u_val = np.vstack([u_val[:, :], u_val_hums])
                x_val_noorca = np.concatenate([rob_state for rob_state in rob_states]+
                                    [np.tile(next_state, (1, self.horiz-len(actions)))], -1)
                x_val_hums = x_guess[self.mpc_env.nx_r+self.mpc_env.np_g:,:]
                x_val = np.vstack([x_val_noorca[:self.mpc_env.nx_r+self.mpc_env.np_g,:], x_val_hums])
                self.num_prev_used = self.horiz + 1
            debug_text += debug_text2
            post_sol_end = time.time()
            time_text = debug_text2+', Wall Time: {:.4f}'.format(post_sol_end-sol_end)
            print('[CAMPC] Optim. error step {:}: {:}.  Num Iter: {:}{:}'.format(self.traj_step, status, solver.get_stats("sqp_iter"), time_text))
        if status == 1 or status == 4:
            print("GUESS")
            for idx in range(self.horiz):
                if self.mpc_env.hum_model == 'orca_casadi_kkt' and self.human_pred_MID:
                    if not self.outdoor_robot_setting:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_guess[:,idx:idx+1], u_guess[:, idx:idx+1]]), np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                    else:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_guess[:,idx:idx+1], u_guess[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)]), np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                else:
                    if not self.outdoor_robot_setting:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_guess[:,idx:idx+1], u_guess[:, idx:idx+1]])))
                    else:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_guess[:,idx:idx+1], u_guess[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)])))
                con_vals = con_vals.squeeze().tolist()
                for cidx, con_val in enumerate(con_vals):
                    if '_eq_' in self.mpc_env.all_state_names[cidx]:
                        con_vals[cidx] = np.abs(con_val)
                arg_max_con = np.argmax(con_vals)
                print("idx {:},\t con: {:} {:},\t val: {:.2f}".format(idx, arg_max_con, self.mpc_env.all_state_names[arg_max_con], con_vals[arg_max_con]))
            # same thing for terminal consts
            idx+=1
            if not self.outdoor_robot_setting:
                if self.mpc_env.human_pred_MID:
                    con_vals = np.array(self.term_con_fn(x_guess[:, idx:idx+1], np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                else:
                    con_vals = np.array(self.term_con_fn(x_guess[:, idx:idx+1]))
            else:
                if self.mpc_env.human_pred_MID:
                    con_vals = np.array(self.term_con_fn(np.vstack([x_guess[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)]), np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                else:
                    con_vals = np.array(self.term_con_fn(np.vstack([x_guess[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)])))
            print("idx {:},\t con: {:},\t val: {:.2f}".format(idx, np.argmax(con_vals), con_vals[np.argmax(con_vals)].item()))
            print("idx {:},\t con: {:},\t val: {:.2f}".format(idx, np.argmin(con_vals), con_vals[np.argmin(con_vals)].item()))
            print(); print(); print()

        if self.mpc_env.isSim or status == 1 or status == 4:
            print("FINAL")
            for idx in range(self.horiz):
                if self.mpc_env.human_pred_MID:
                    if self.outdoor_robot_setting:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_val[:,idx:idx+1], u_val[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)]), np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                    else:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_val[:,idx:idx+1], u_val[:, idx:idx+1]]), np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                else:
                    if self.outdoor_robot_setting:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_val[:,idx:idx+1], u_val[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)])))
                    else:
                        con_vals = np.array(self.stage_con_fn(np.vstack([x_val[:,idx:idx+1], u_val[:, idx:idx+1]])))
                con_vals = con_vals.squeeze().tolist()
                for cidx, con_val in enumerate(con_vals):
                    if '_eq_' in self.mpc_env.all_state_names[cidx]:
                        con_vals[cidx] = np.abs(con_val)
                arg_max_con = np.argmax(con_vals)
                print("idx {:},\t con: {:} {:},\t val: {:.2f}".format(idx, arg_max_con, self.mpc_env.all_state_names[arg_max_con], con_vals[arg_max_con]))
            # same thing for terminal consts
            idx+=1
            if self.outdoor_robot_setting:
                if self.mpc_env.human_pred_MID:
                    con_vals = np.array(self.term_con_fn(np.vstack([x_val[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)]), np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                else:
                    con_vals = np.array(self.term_con_fn(np.vstack([x_val[:, idx:idx+1], properly_shaped_static_obs.reshape(self.mpc_env.num_stat_obs*4,1)])))
            else:
                if self.mpc_env.human_pred_MID:
                    con_vals = np.array(self.term_con_fn(x_val[:, idx:idx+1], np.vstack([MID_samples[idx,:,0], MID_samples[idx,:,1]]).T))
                else:
                    con_vals = np.array(self.term_con_fn(x_val[:, idx:idx+1]))

            # print(con_vals)
            print("idx {:},\t con: {:},\t val: {:.2f}".format(idx, np.argmax(con_vals), con_vals[np.argmax(con_vals)].item()))
            print("idx {:},\t con: {:},\t val: {:.2f}".format(idx, np.argmin(con_vals), con_vals[np.argmin(con_vals)].item()))

        self.x_prev = x_val
        self.u_prev = u_val
        self.x_val = x_val
        self.u_val = u_val
        self.x_guess = x_guess
        self.u_guess = u_guess
        self.x_goals = goal_states
        self.u_goals = goal_actions
        self.debug_text = debug_text
        self.prev_status = status
        self.prev_num_iter = solver.get_stats("sqp_iter")
        if self.mpc_env.hum_model == 'orca_casadi_kkt' and self.mpc_env.human_pred_MID:
            # all weights is what
            self.forecasts_shaped = MID_samples
            mid_samples_map_arrange = einops.rearrange(MID_samples[:-1], 't hs d -> hs (t d)')
            ofs = self.mpc_env.nx_r+self.mpc_env.np_g
            nxh = self.mpc_env.nx_hum
            nsamp = self.mpc_env.num_MID_samples
            log_weights_t = np.stack([x_val[ofs+h_idx*nxh+6:ofs+h_idx*nxh+6+nsamp,:] for h_idx in range(self.mpc_env.num_hums)], axis=0)
            weights_t = np.exp(log_weights_t)
            self.log_weights = log_weights_t
            self.weights = weights_t


        if return_all:
            return action, x_val, u_val
        return action




    def predict(self, state):
        if self.robustness_eval:
            state = self.randomize_state_robustness_eval(state)

        # initialize casadi options etc. for new scenario case.
        if not self.mpc_env or self.env.global_time == 0.0:
            self.reset_scenario_values()
            # set the goal etc. for the humans s.t. this gt value ends up being correct

        # update the state provided by the sim environment here because there is no ros callback with simulator to do this stuff in.
        if self.human_goal_cvmm or self.human_pred_MID:
            if self.human_pred_MID:
                self.hum_traj_forecaster.update_state_hists(state.self_state, state.human_states, self.env.global_time)
                # shape of forecasts is (num_humans, num_samples, horiz, xy)
                # forecasts = self.hum_traj_forecaster.predict()[:, :, 1:, :]
                top_k_forecasts, top_k_weights = self.hum_traj_forecaster.predict_ret_best()
                forecasts = top_k_forecasts[:, :, 1:, :]
                if self.human_pred_MID_joint:
                    forecasts_init_weights = top_k_weights[0,:]
                else:
                    forecasts_init_weights = top_k_weights

                forecasts_reshaped = einops.rearrange(forecasts, 'h s t d -> t (h s) d')[:self.horiz+1, :, :]
                self.all_forecasts.append(forecasts)
            new_human_states = []
            for h_idx in range(len(state.human_states)):
                if self.human_goal_cvmm:
                    gx = state.human_states[h_idx].px + self.human_goal_cvmm_horizon * state.human_states[h_idx].vx
                    gy = state.human_states[h_idx].py + self.human_goal_cvmm_horizon * state.human_states[h_idx].vy
                    v_pref = self.human_max_vel
                else:
                    # estimate the gx for the first time step of the prediction horizon by
                    gx = np.mean(forecasts[h_idx, :, 0, 0])
                    gy = np.mean(forecasts[h_idx, :, 0, 1])
                    # estimate v_pref i.e. the max velocity of the human
                    # find difference between every two consecutive points (i.e. finite diff dim 2)
                    # then find the norm of the difference and divide by the time step to get the velocity
                    # then take the mean of all the velocities to get the average velocity
                    v_pref = np.max(np.linalg.norm(np.diff(forecasts[h_idx, :, :, :], axis=1), axis=2) / self.time_step)

                new_human_states.append(FullState(
                    px=state.human_states[h_idx].px,
                    py=state.human_states[h_idx].py,
                    vx=state.human_states[h_idx].vx,
                    vy=state.human_states[h_idx].vy,
                    gx=gx,
                    gy=gy,
                    radius=state.human_states[h_idx].radius,
                    v_pref=v_pref,
                    theta=np.arctan2(state.human_states[h_idx].vy, state.human_states[h_idx].vx) if state.human_states[h_idx].vx != 0 or state.human_states[h_idx].vy != 0 else 0.0
                ))
            joint_state = FullyObservableJointState(state.self_state, new_human_states, state.static_obs)
        else:
            joint_state = state

        robot_state = joint_state.self_state
        goal_states, goal_actions = self.get_ref_traj(joint_state)

        if self.human_pred_MID:
            mpc_state = self.convert_to_mpc_state_vector(joint_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, forecasts_init_weights, get_numpy=True)
        else:
            mpc_state = self.convert_to_mpc_state_vector(joint_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)


        start_time = time.time()
        if self.human_pred_MID:
            mpc_action = self.select_action(mpc_state, joint_state, goal_states, goal_actions, MID_samples=forecasts_reshaped)
        else:
            mpc_action = self.select_action(mpc_state, joint_state, goal_states, goal_actions)
        end_just_action = time.time()
        action = ActionRot(mpc_action[0], mpc_action[1]*self.time_step)
        self.prev_lvel = mpc_action[0]

        x_val = self.x_val
        u_val = self.u_val
        x_guess = self.x_guess
        u_guess = self.u_guess
        x_goals = self.x_goals
        u_goals = self.u_goals
        debug_text = self.debug_text
        traj_step = self.traj_step
        prev_status = self.prev_status

        self.all_x_val.append(deepcopy(x_val))
        self.all_u_val.append(deepcopy(u_val))
        self.all_x_guess.append(deepcopy(x_guess))
        self.all_u_guess.append(deepcopy(u_guess))
        self.all_x_goals.append(deepcopy(goal_states))
        self.all_u_goals.append(deepcopy(goal_actions))
        self.all_debug_text.append(deepcopy(debug_text))
        self.all_prev_status.append(deepcopy(prev_status))
        self.all_prev_num_iter.append(deepcopy(self.prev_num_iter))

        if self.mpc_env.human_pred_MID:
            self.all_forecasts_shaped.append(deepcopy(self.forecasts_shaped))
            self.all_weights.append(deepcopy(self.weights))
            self.all_log_weights.append(deepcopy(self.log_weights))

        end_time = time.time()
        # step along trajectory
        self.calc_times.append(end_time-start_time)
        if DISP_TIME:
            logging.info('[CAMPC] Total wall time to solve MPC for step {:} was {:.3f}s'.format(self.traj_step, end_time-start_time))

        self.traj_step += 1
        return action



