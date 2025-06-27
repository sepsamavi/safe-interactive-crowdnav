from copy import deepcopy
from os import stat
import os, contextlib
import numpy as np
import casadi as cs

from sicnav.utils.mpc_utils.orca_c_wrapper import get_human_radii, get_human_goals

numstab_epsilon = 1e-6

def det(vector1, vector2):
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]

def abs_sq(vector1):
    return vector1.T @ vector1

def safe_divide(numer, denom):
    return numer * denom / (denom*denom + numstab_epsilon)

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


@contextlib.contextmanager
def nostdout():
    blockPrint()
    yield
    enablePrint()



class casadiORCA(object):


    def __init__(self, mpc_env, joint_state, X):
        self.orca_pol = mpc_env.callback_orca # Temp for testing
        self.mpc_env = mpc_env
        self.time_step = mpc_env.time_step
        self.num_hums = mpc_env.num_hums
        self.nx_hum = mpc_env.nx_hum
        self.nX_hums = mpc_env.nX_hums
        self.num_hums = mpc_env.num_hums
        self.nx_r = mpc_env.nx_r
        self.np_g = mpc_env.np_g
        self.nvars_hum = mpc_env.nvars_hum
        self.v_max_unobservable = self.orca_pol.max_speed
        self.X = X
        self.v_max_prefs = None # to match sim
        self.human_gxs = None
        self.human_gys = None
        self.safety_space = self.orca_pol.safety_space
        self.time_coll_hor = self.orca_pol.time_horizon * 1.0
        self.time_coll_hor_obst = self.orca_pol.time_horizon_obst * 1.0
        self.curcoll_hor = 1e-1
        self.humB_idcs_list = self.get_humB_idcs_list()
        self.reset_humans(joint_state)
        self.reset_stat_obs(joint_state)
        self.return_list = np.zeros(self.nX_hums, dtype=float)
        self.init_one_hum_eqns()

        self.get_ORCA_pairwise = self.init_get_ORCA_pairwise_casadi_fns()
        self.get_ORCA_humstatic_noadj = self.init_get_ORCA_humstatic_fns()
        self.get_ORCA_humstatic_withadj = self.init_get_ORCA_humstatic_fns(with_adj_pt=True)
        self.get_v_pref_fromstate_csfunc = self.init_get_v_pref_fromstate_csfunc()
        self.get_rob_ws_v_pref_fromstate_csfunc = self.init_get_rob_ws_v_pref_fromstate_csfunc()

        self.outdoor_robot_setting = mpc_env.outdoor_robot_setting
        if self.outdoor_robot_setting:
            self.stat_obs_params = mpc_env.stat_obs_params
            self.hum_extra_params = mpc_env.hum_extra_params


    def get_humB_idcs_list(self):
        """Initialize a list (one entry for each humA) of np arrays with the indices of humB for humA
        """
        humB_idcs_list = []
        for humA_idx in range(self.num_hums):
            humB_for_humA = np.concatenate([np.arange(0, humA_idx, 1, dtype=int),
                                            np.arange(humA_idx+1, self.num_hums, 1, dtype=int),
                                            np.array([-1])])
            humB_idcs_list.append(humB_for_humA)

        humB_idcs_list.append(np.arange(0, self.num_hums, 1, dtype=int)) # for the robot's orca-based warm start

        return humB_idcs_list


    def reset_stat_obs(self, state):
        self.num_stat_obs = len(state.static_obs)

        # static obs as 2 cs.DM points
        static_obs = []
        static_obs_cvx_adj = []
        for stat_ob in state.static_obs:
            p_1 = cs.DM(stat_ob[0])
            p_2 = cs.DM(stat_ob[1])
            static_obs.append((p_1, p_2))
            # if this line-segment obstacle and the previous line-segment obstacle share a point
            eps = 1e-4
            if len(static_obs) > 1 and cs.norm_2(static_obs[-1][0] - static_obs[-2][1]) < eps:
                static_obs_cvx_adj.append(len(static_obs) - 2)
            else:
                static_obs_cvx_adj.append(None)

        self.static_obs = static_obs
        self.static_obs_cvx_adj = static_obs_cvx_adj


    def reset_humans(self, state, new_h_gxs=None, new_h_gys=None):
        self.num_humans = len(state.human_states)
        # goals
        if new_h_gxs is None or new_h_gys is None:
            self.human_gxs, self.human_gys = get_human_goals(state)
        else:
            self.human_gxs, self.human_gys = new_h_gxs, new_h_gys

        # radii
        human_radii = get_human_radii(state) + self.safety_space
        agent_radii = np.append(human_radii, state.self_state.radius)
        self.agent_radii = agent_radii + 0.01
        self.agent_radii_orig = agent_radii

        # max speed settings
        v_max_prefs = np.zeros(len(self.human_gxs)+1)
        for hum_idx, human_state in enumerate(state.human_states):
            v_max_prefs[hum_idx] = human_state.v_pref
        v_max_prefs[-1] = state.self_state.v_pref # for the robot (used in robot's warm start)
        self.v_max_prefs = v_max_prefs


    def get_ORCA_set_list(self, X, humA_idx, casadi_dicts=None):
        """ Generate ORCA_{humA_idx} set i.e. list of ORCA_{humA_idx|humB_idx} lines

        :param X: state of entire environment (including robot)
        :param humA_idx: index of human whose orca sets we want
        """
        line_norms = []
        line_pts = []
        line_norms_checked = []
        line_scalars_checked = []

        humB_idcs = self.humB_idcs_list[humA_idx]
        for humB_idx in humB_idcs:
            if casadi_dicts is not None:
                casadi_dict = {}
                casadi_dict['X'] = X
                casadi_dict['humA_idx'] = humA_idx
                txt = 'hum{:}'.format(humB_idx) if humB_idx != -1 else 'rob'
                casadi_dicts['hum{:}_{:}'.format(humA_idx, txt)] = casadi_dict
            else:
                casadi_dict = None
            if humA_idx == -1:
                line_norm, line_pt, line_norm_checked, line_scalar_checked = self.get_ORCA_pairwise_robhum(X, humB_idx=humB_idx, casadi_dict=casadi_dict)
            else:
                if humB_idx == -1:
                    # i.e. if we are dealing with the robot
                    line_norm, line_pt, line_norm_checked, line_scalar_checked = self.get_ORCA_pairwise_humrob(X, humA_idx=humA_idx, casadi_dict=casadi_dict)
                else:
                    line_norm, line_pt, line_norm_checked, line_scalar_checked = self.get_ORCA_pairwise_humhum(X, humA_idx=humA_idx, humB_idx=humB_idx, casadi_dict=casadi_dict)
            line_norms.append(line_norm)
            line_pts.append(line_pt)
            line_norms_checked.append(line_norm_checked)
            line_scalars_checked.append(line_scalar_checked)

        return line_norms, line_pts, line_norms_checked, line_scalars_checked


    def get_ORCA_stat_set_list(self, X, humA_idx, get_line_pts=False, casadi_dict=None, stat_obs_params=None):
        """ Generate ORCA_{humA_idx} set i.e. list of ORCA_{humA_idx|humB_idx} lines

        :param X: state of entire environment (including robot)
        :param humA_idx: index of human whose orca sets we want
        """
        X_hums = X[self.nx_r+self.np_g:]
        if humA_idx == -1:
            if self.mpc_env.human_pred_MID and not self.mpc_env.human_pred_MID_joint:
                X_humA = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[8], X[9], cs.log(cs.DM.ones(self.mpc_env.num_MID_samples)/self.mpc_env.num_MID_samples))
            else:
                X_humA = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[8], X[9])
        else:
            X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]

        radA = self.agent_radii[humA_idx]

        line_norms = []
        line_scalars = []
        line_pts=[]
        for stat_idx in range(self.num_stat_obs):
            if self.outdoor_robot_setting:
                if stat_obs_params is None:
                    stat_obs_params = self.stat_obs_params

                p_1 = stat_obs_params[stat_idx, :2].reshape((2,1))
                p_2 = stat_obs_params[stat_idx, 2:].reshape((2,1))

            else:
                p_1 = self.static_obs[stat_idx][0]
                p_2 = self.static_obs[stat_idx][1]
            cvx_adj_idx = self.static_obs_cvx_adj[stat_idx]
            if cvx_adj_idx is not None and not self.outdoor_robot_setting:
                adj_line_pt = line_pts[cvx_adj_idx]
                adj_line_norm = line_norms[cvx_adj_idx]
                line_norm, line_scalar, line_pt = self.get_ORCA_humstatic_withadj(X_humA, p_1, p_2, radA, adj_line_pt, adj_line_norm)
            else:
                adj_line_pt = None
                adj_line_norm = None
                line_norm, line_scalar, line_pt = self.get_ORCA_humstatic_noadj(X_humA, p_1, p_2, radA)

            line_norms.append(line_norm)
            line_scalars.append(line_scalar)
            line_pts.append(line_pt)

        if get_line_pts:
            return line_norms, line_scalars, line_pts
        return line_norms, line_scalars


    def init_get_ORCA_pairwise_casadi_fns(self, casadi_dict=None, time_coll_hor=None):
        time_coll_hor = self.time_coll_hor if time_coll_hor is None else time_coll_hor
        X_humA = cs.SX.sym('X_humA', self.nx_hum)
        X_humB = cs.SX.sym('X_humB', self.nx_hum)
        radA = cs.SX.sym('radA', 1)
        radB = cs.SX.sym('radB', 1)
        fn_inputs = (X_humA, X_humB, radA, radB)

        rel_pos = X_humB[0:2] - X_humA[0:2]
        rel_vel = X_humA[2:4] - X_humB[2:4]
        dist_sq = cs.dot(rel_pos, rel_pos)
        comb_rad = radA + radB
        comb_rad_sq = comb_rad ** 2

        cond_nocoll = dist_sq > comb_rad_sq

        def get_w_vec(inv_time):
            return rel_vel - inv_time * rel_pos

        def get_nocoll():
            inv_time_horiz = 1.0 / time_coll_hor
            w = get_w_vec(inv_time_horiz)
            w_len_sq = cs.dot(w, w)
            dotprod_1 = cs.dot(w, rel_pos)
            cond_projcutoff = (dotprod_1 < 0.0) * (dotprod_1**2 > comb_rad_sq * w_len_sq)

            def get_proj_cutoffcirc():
                w_len = cs.sqrt(w_len_sq)
                unit_w = safe_divide(w, w_len)
                line_dir = cs.vertcat(unit_w[1], -unit_w[0])
                u = (comb_rad * inv_time_horiz - w_len) * unit_w
                return cs.vertcat(line_dir, u)

            def get_proj_leg():
                # py-rvo-2 way
                leg = cs.sqrt(cs.fabs(dist_sq - comb_rad_sq) + numstab_epsilon) # is this num.stab. Apparently not, only get here if dist_sq > comb_rad_sq, but when we dont the nan still effects result of if_else(.)
                def proj_left():
                    return safe_divide(cs.vertcat(rel_pos[0] * leg - rel_pos[1] * comb_rad,
                                                    rel_pos[0] * comb_rad + rel_pos[1] * leg), dist_sq)
                def proj_right():
                    return safe_divide(-1 * cs.vertcat(rel_pos[0] * leg + rel_pos[1] * comb_rad,
                                                        -rel_pos[0] * comb_rad + rel_pos[1] * leg), dist_sq)


                cond_leftleg = det(rel_pos, w) > 0.0
                proj_left_cs = cs.Function('proj_left', [*fn_inputs], [proj_left()])
                proj_right_cs = cs.Function('proj_right', [*fn_inputs], [proj_right()])


                line_dir = cs.if_else(cond_leftleg,
                                    proj_left_cs(*fn_inputs),
                                    proj_right_cs(*fn_inputs),
                                    True)

                dotprod_2 = cs.dot(rel_vel, line_dir)

                u = dotprod_2 * line_dir - rel_vel

                return cs.vertcat(line_dir, u)

            get_proj_cutoffcirc_cs = cs.Function('get_proj_cutoffcirc', [*fn_inputs], [get_proj_cutoffcirc()])
            get_proj_leg_cs = cs.Function('get_proj_leg', [*fn_inputs], [get_proj_leg()])


            ans = cs.if_else(cond_projcutoff,
                             get_proj_cutoffcirc_cs(*fn_inputs),
                             get_proj_leg_cs(*fn_inputs),
                             True)

            return ans

        def get_curcoll():
            inv_ts = 1.0 / self.time_step
            # rel_pos_dist = cs.norm_2(rel_pos)
            rel_pos_dist = cs.sqrt(rel_pos.T @ rel_pos)

            # unit_rel_pos = rel_pos / rel_pos_dist
            unit_rel_pos = safe_divide(rel_pos, rel_pos_dist)

            protrusion = comb_rad - rel_pos_dist

            line_norm = - unit_rel_pos

            line_dir = cs.vertcat(line_norm[1], -line_norm[0])


            cutoff_line_pt = inv_ts * protrusion ** 2 * line_norm

            proj_pt = cutoff_line_pt + cs.dot((rel_vel - cutoff_line_pt), line_dir) * line_dir

            u = proj_pt - rel_vel

            return cs.vertcat(line_dir, u)


        get_nocoll_cs = cs.Function('get_nocoll', [*fn_inputs], [get_nocoll()])
        get_curcoll_cs = cs.Function('get_curcoll', [*fn_inputs], [get_curcoll()])


        ans = cs.if_else(cond_nocoll,
                         get_nocoll_cs(*fn_inputs),
                         get_curcoll_cs(*fn_inputs),
                         True)

        line_dir = ans[0:2]
        u = ans[2:]
        line_pt = X_humA[2:4] + 0.5 * u
        line_norm = cs.vertcat(-line_dir[1], line_dir[0])

        line_norm_checked = line_norm
        # line_scalar_checked = line_norm.T @ line_pt

        # Invalidate lines that do not matter:
        v_max = 20.0 # some upper bound on the human speeds
        dotprod_check = line_dir.T @ line_pt
        discriminant = dotprod_check ** 2 + v_max ** 2 - cs.dot(line_pt, line_pt)
        # If the ORCA constraint is beyond the upper bound maximum velocity, then we put its constraint line just outside the max_vel circle
        dummy_line_pt = -1.15*v_max*line_norm
        line_scalar_checked = cs.if_else(discriminant < 0.0, line_norm.T @ dummy_line_pt, line_norm.T @ line_pt, True)

        pairwise_orca_fn = cs.Function('get_ORCA_pairwise', [X_humA, X_humB, radA, radB], [line_norm, line_pt, line_norm_checked, line_scalar_checked])
        return pairwise_orca_fn


    def init_get_ORCA_humstatic_fns(self, with_adj_pt=False):
        inv_time_horiz_obst = 1.0 / self.time_coll_hor_obst
        X_humA = cs.SX.sym('X_humA', self.nx_hum)
        p_1_raw = cs.SX.sym('p_1_raw', 2)
        p_2_raw = cs.SX.sym('p_2_raw', 2)
        radA = cs.SX.sym('radA')

        if with_adj_pt:
            adj_line_pt = cs.SX.sym('adj_line_pt', 2)
            adj_line_norm = cs.SX.sym('adj_line_norm', 2)
            fn_inputs = (X_humA, p_1_raw, p_2_raw, radA, adj_line_pt, adj_line_norm)
        else:
            fn_inputs = (X_humA, p_1_raw, p_2_raw, radA)

        # my sketchy check:
        rel_pos_1_raw = p_1_raw - X_humA[0:2]
        p_1 = cs.if_else(det((p_1_raw - p_2_raw), rel_pos_1_raw)<0, p_1_raw, p_2_raw, True)
        p_2 = cs.if_else(det((p_1_raw - p_2_raw), rel_pos_1_raw)<0, p_2_raw, p_1_raw, True)

        rel_pos_1 = p_1 - X_humA[0:2]
        rel_pos_2 = p_2 - X_humA[0:2]
        vel = X_humA[2:4]

        dist_sq_1 =  cs.dot(rel_pos_1, rel_pos_1)
        dist_sq_2 =  cs.dot(rel_pos_2, rel_pos_2)

        rad_sq = radA ** 2

        obst_vec = p_2 - p_1

        s = safe_divide(-rel_pos_1.T @ obst_vec, abs_sq(obst_vec))

        sq_line0 = - rel_pos_1 - s * obst_vec
        dist_sq_line = cs.dot(sq_line0, sq_line0)

        cond_left_vtx_coll = (s < 0.0) * (dist_sq_1 <= rad_sq)
        cond_right_vtx_coll = (s > 1.0) * (dist_sq_2 <= rad_sq)
        cond_segment_coll = (s >= 0.0) * (s < 1.0) * (dist_sq_line <= rad_sq)
        # check to see if there is a collision
        cond_curcoll = cond_left_vtx_coll + cond_right_vtx_coll + cond_segment_coll

        def get_curcoll():
            line_pt = 0.0 * rel_pos_1

            def coll_left_vtx():
                line_dir = safe_divide(cs.vertcat(-rel_pos_1[1], rel_pos_1[0]), cs.sqrt(rel_pos_1.T @ rel_pos_1))
                return line_dir

            def not_coll_left_vtx():
                def coll_right_vtx():
                    line_dir = safe_divide(cs.vertcat(-rel_pos_2[1], rel_pos_2[0]), cs.sqrt(rel_pos_2.T @ rel_pos_2))
                    return line_dir

                def coll_segment():
                    # must be coll segment
                    line_dir = -  safe_divide(obst_vec, cs.sqrt(obst_vec.T @ obst_vec))
                    return line_dir

                coll_right_vtx_cs = cs.Function('coll_right_vtx', [*fn_inputs], [coll_right_vtx()])
                coll_segment_cs = cs.Function('coll_segment', [*fn_inputs], [coll_segment()])

                return cs.if_else(cond_right_vtx_coll, coll_right_vtx_cs(*fn_inputs), coll_segment_cs(*fn_inputs), True)

            coll_left_vtx_cs = cs.Function('coll_left_vtx', [*fn_inputs], [coll_left_vtx()])
            not_coll_left_vtx_cs = cs.Function('not_coll_left_vtx', [*fn_inputs], [not_coll_left_vtx()])
            line_dir = cs.if_else(cond_left_vtx_coll, coll_left_vtx_cs(*fn_inputs), not_coll_left_vtx_cs(*fn_inputs), True)
            return cs.vertcat(line_dir, line_pt)


        def get_nocoll():

            # Construct the legs of the velocity obstacle
            cond_left_vtx_nocoll = (s < 0.0) * (dist_sq_line <= rad_sq)
            cond_right_vtx_nocoll = (s > 1.0) * (dist_sq_line <= rad_sq)
            cond_vtxs = cond_left_vtx_nocoll + cond_right_vtx_nocoll

            def nocoll_vtxs():
                # oblique view of obstacle => obstacle defined by left vertex
                def nocoll_left_vtx():
                    # NB OBS2 = OBS1
                    leg1 = cs.sqrt(cs.fabs(dist_sq_1 - rad_sq) + numstab_epsilon)
                    left_leg_dir = safe_divide(cs.vertcat(rel_pos_1[0] * leg1 - rel_pos_1[1] * radA, rel_pos_1[0] * radA + rel_pos_1[1] * leg1), dist_sq_1)
                    right_leg_dir = safe_divide(cs.vertcat(rel_pos_1[0] * leg1 + rel_pos_1[1] * radA, -rel_pos_1[0] * radA + rel_pos_1[1] * leg1), dist_sq_1)
                    left_cutoff = inv_time_horiz_obst * rel_pos_1 #(obstacle1->point_ - position_)
                    right_cutoff = inv_time_horiz_obst * rel_pos_1 #(obstacle2->point_ - position_)
                    return cs.vertcat(left_leg_dir, right_leg_dir, left_cutoff, right_cutoff)

                # oblique view of obstacle => obstacle defined by right vertex
                def nocoll_right_vtx():
                    # NB OBS1 = OBS2
                    leg2 = cs.sqrt(cs.fabs(dist_sq_2 - rad_sq) + numstab_epsilon)
                    left_leg_dir = safe_divide(cs.vertcat(rel_pos_2[0] * leg2 - rel_pos_2[1] * radA, rel_pos_2[0] * radA + rel_pos_2[1] * leg2), dist_sq_2)
                    right_leg_dir = safe_divide(cs.vertcat(rel_pos_2[0] * leg2 + rel_pos_2[1] * radA, -rel_pos_2[0] * radA + rel_pos_2[1] * leg2), dist_sq_2)
                    left_cutoff = inv_time_horiz_obst * rel_pos_2 #(obstacle1->point_ - position_)
                    right_cutoff = inv_time_horiz_obst * rel_pos_2 #(obstacle2->point_ - position_)
                    return cs.vertcat(left_leg_dir, right_leg_dir, left_cutoff, right_cutoff)

                nocoll_left_vtx_cs = cs.Function('nocoll_left_vtx', [*fn_inputs], [nocoll_left_vtx()])
                nocoll_right_vtx_cs = cs.Function('nocoll_right_vtx', [*fn_inputs], [nocoll_right_vtx()])
                return cs.if_else(cond_left_vtx_nocoll, nocoll_left_vtx_cs(*fn_inputs), nocoll_right_vtx_cs(*fn_inputs), True)


            def nocoll_else():
                # assuming obs 1 is convex:
                leg1 = cs.sqrt(cs.fabs(dist_sq_1 - rad_sq) + numstab_epsilon)
                left_leg_dir = safe_divide(cs.vertcat(rel_pos_1[0] * leg1 - rel_pos_1[1] * radA, rel_pos_1[0] * radA + rel_pos_1[1] * leg1), dist_sq_1)

                # assuming obs 2 is convex:
                leg2 = cs.sqrt(cs.fabs(dist_sq_2 - rad_sq) + numstab_epsilon)
                right_leg_dir = safe_divide(cs.vertcat(rel_pos_2[0] * leg2 + rel_pos_2[1] * radA, -rel_pos_2[0] * radA + rel_pos_2[1] * leg2), dist_sq_2)


                left_cutoff = inv_time_horiz_obst * rel_pos_1 #(obstacle1->point_ - position_)
                right_cutoff = inv_time_horiz_obst * rel_pos_2 #(obstacle2->point_ - position_)
                return cs.vertcat(left_leg_dir, right_leg_dir, left_cutoff, right_cutoff)

            nocoll_vtxs_cs = cs.Function('nocoll_vtxs', [*fn_inputs], [nocoll_vtxs()])
            nocoll_else_cs = cs.Function('nocoll_else', [*fn_inputs], [nocoll_else()])

            legs_and_cutoff = cs.if_else(cond_vtxs, nocoll_vtxs_cs(*fn_inputs), nocoll_else_cs(*fn_inputs), True)

            left_leg_dir = cs.vertcat(legs_and_cutoff[0], legs_and_cutoff[1])
            right_leg_dir = cs.vertcat(legs_and_cutoff[2], legs_and_cutoff[3])

            left_cutoff = cs.vertcat(legs_and_cutoff[4], legs_and_cutoff[5])
            right_cutoff = cs.vertcat(legs_and_cutoff[6], legs_and_cutoff[7])
            cutoff_vec = right_cutoff - left_cutoff

            # Projecting onto velocity obstacle
            option2 = cs.Function('option2', [*fn_inputs], [safe_divide((vel - left_cutoff).T @ cutoff_vec, abs_sq(cutoff_vec))])
            t = cs.if_else(cond_vtxs, 0.5, option2(*fn_inputs), True)
            t_left = ((vel - left_cutoff).T @ left_leg_dir)
            t_right = ((vel - right_cutoff).T @ right_leg_dir)

            cond_proj_left_cutoff_circ = ((t < 0.0) * (t_left < 0.0)) + (cond_vtxs * (t_left < 0.0) * (t_right < 0.0))
            cond_proj_right_cutoff_circ = ((t > 1.0) * (t_right < 0.0))
            cond_proj_cutoff_circs = cond_proj_left_cutoff_circ + cond_proj_right_cutoff_circ

            def proj_cutoff_circ():
                cutoff = cs.if_else(cond_proj_left_cutoff_circ, left_cutoff, right_cutoff, True)
                v_diff = vel - cutoff
                unitW = safe_divide(v_diff, cs.sqrt(v_diff.T @ v_diff))
                line_dir = cs.vertcat(unitW[1], -unitW[0])
                line_pt = cutoff + radA * inv_time_horiz_obst * unitW
                return cs.vertcat(line_dir, line_pt)

            def proj_legs_or_cutoff_line():
                dist_sq_cutoff = cs.if_else(((t < 0.0) + (t > 1.0) + cond_vtxs), cs.inf, abs_sq(vel - (left_cutoff + t * cutoff_vec)), True)
                dist_sq_left = cs.if_else((t_left < 0.0), cs.inf, abs_sq(vel - (left_cutoff + t_left * left_leg_dir)), True)
                dist_sq_right = cs.if_else((t_right < 0.0), cs.inf, abs_sq(vel - (right_cutoff + t_right * right_leg_dir)), True)
                cond_proj_cutoff_line = (dist_sq_cutoff <= dist_sq_left) * (dist_sq_cutoff <= dist_sq_right)
                cond_proj_left_leg = (dist_sq_left <= dist_sq_right)
                cond_firstwo = cond_proj_cutoff_line + cond_proj_left_leg

                def firstwo():
                    def proj_cutoff_line():
                        unit_dir = safe_divide(p_1 - p_2, cs.sqrt((p_1 - p_2).T @ (p_1 - p_2)))
                        line_dir = unit_dir
                        line_pt = left_cutoff + radA * inv_time_horiz_obst * cs.vertcat(-line_dir[1], line_dir[0])
                        return cs.vertcat(line_dir, line_pt)

                    def proj_left_leg():
                        line_dir = left_leg_dir
                        line_pt = left_cutoff + radA * inv_time_horiz_obst * cs.vertcat(-line_dir[1], line_dir[0])
                        return cs.vertcat(line_dir, line_pt)

                    proj_cutoff_line_cs = cs.Function('proj_cutoff_line', [*fn_inputs], [proj_cutoff_line()])
                    proj_left_leg_cs = cs.Function('proj_left_leg', [*fn_inputs], [proj_left_leg()])
                    return cs.if_else(cond_proj_cutoff_line, proj_cutoff_line_cs(*fn_inputs), proj_left_leg_cs(*fn_inputs), True)

                def proj_right_leg():
                    line_dir = -right_leg_dir
                    line_pt = right_cutoff + radA * inv_time_horiz_obst * cs.vertcat(-line_dir[1], line_dir[0])
                    return cs.vertcat(line_dir, line_pt)

                firstwo_cs = cs.Function('firstwo', [*fn_inputs], [firstwo()])
                proj_right_leg_cs = cs.Function('proj_right_leg', [*fn_inputs], [proj_right_leg()])
                return cs.if_else(cond_firstwo, firstwo_cs(*fn_inputs), proj_right_leg_cs(*fn_inputs), True)

            proj_cutoff_circ_cs = cs.Function('proj_cutoff_circ', [*fn_inputs], [proj_cutoff_circ()])
            proj_legs_or_cutoff_line_cs = cs.Function('proj_legs_or_cutoff_line', [*fn_inputs], [proj_legs_or_cutoff_line()])
            return cs.if_else(cond_proj_cutoff_circs, proj_cutoff_circ_cs(*fn_inputs), proj_legs_or_cutoff_line_cs(*fn_inputs), True)

        get_curcoll_cs = cs.Function('get_curcoll', [*fn_inputs], [get_curcoll()])
        get_nocoll_cs = cs.Function('get_nocoll', [*fn_inputs], [get_nocoll()])


        ans = cs.if_else(cond_curcoll,
                        get_curcoll_cs(*fn_inputs),
                        get_nocoll_cs(*fn_inputs),
                        True)


        line_dir = ans[0:2]
        line_pt_calc = ans[2:]
        line_norm = cs.vertcat(-line_dir[1], line_dir[0])


        if with_adj_pt:
            eps = 1e-2 # some epsilon value
            v_max = 10.0 # some upper bound for the max velocity value # some upper bound for the max velocity value

            case1 = cs.Function('case1', [*fn_inputs], [-1.15*v_max*line_norm])
            case2 = cs.Function('case2', [*fn_inputs], [line_pt_calc])

            line_pt = cs.if_else((cs.fabs(line_norm.T @ line_pt_calc - adj_line_norm.T @ adj_line_pt) < eps)*(cs.sqrt((line_norm - adj_line_norm).T @ (line_norm - adj_line_norm)) < eps),
                                case1(*fn_inputs),
                                case2(*fn_inputs),
                                True)

            line_scalar = line_norm.T @ line_pt
            cs_fn = cs.Function('get_ORCA_humstatic_withadj', [X_humA, p_1_raw, p_2_raw, radA, adj_line_pt, adj_line_norm], [line_norm, line_scalar, line_pt])
        else:
            line_pt = line_pt_calc
            line_scalar = line_norm.T @ line_pt
            cs_fn = cs.Function('get_ORCA_humstatic_noadj', [X_humA, p_1_raw, p_2_raw, radA], [line_norm, line_scalar, line_pt])

        return cs_fn


    def get_ORCA_rob_simulatedconsts(self, X, rot_max, get_pts=False):
        """Simulated ORCA constraints to underestimate the robot's kinodynamic constraints

        :param X: state of the environment at current timestep
        :param rot_max: maximum rotational velocity of the robot
        :param deltavel_max: maximum change in velocity of the robot
        :param get_pts: _description_
        :returns: _description_
        """

        sin_theta_k = X[2]
        cos_theta_k = X[3]
        speed_k = cs.if_else(cs.fabs(X[4]) < 1e-5, 1e-5, X[4], True)

        sin_theta_left = sin_theta_k * cs.cos(rot_max) + cos_theta_k * cs.sin(rot_max)
        sin_theta_right = sin_theta_k * cs.cos(-rot_max) + cos_theta_k * cs.sin(-rot_max)

        cos_theta_left = cos_theta_k * cs.cos(rot_max) - sin_theta_k * cs.sin(rot_max)
        cos_theta_right = cos_theta_k * cs.cos(-rot_max) - sin_theta_k * cs.sin(-rot_max)


        # Let's do this a better way
        max_inc = self.mpc_env.max_l_acc * self.mpc_env.time_step
        max_dec = -self.mpc_env.max_l_dcc * self.mpc_env.time_step

        dir_vec = cs.sign(speed_k) * cs.vertcat(cos_theta_k, sin_theta_k)

        max_speed = cs.sign(speed_k) * (cs.fabs(speed_k) + max_inc)
        min_speed = cs.sign(speed_k) * (cs.fabs(speed_k) - cs.fmax(max_inc, cs.fmin(cs.fabs(speed_k), max_dec)))

        # For the delta_theta constraints
        line_dir_left = cs.vertcat(cos_theta_left, sin_theta_left)*cs.sign(speed_k)
        line_norm_left = cs.vertcat(line_dir_left[1], -line_dir_left[0])
        line_pt_left = 1e-3 * dir_vec
        line_scalar_left = line_norm_left.T @ line_pt_left

        line_dir_right = -cs.vertcat(cos_theta_right, sin_theta_right)*cs.sign(speed_k)
        line_norm_right = cs.vertcat(line_dir_right[1], -line_dir_right[0])
        line_pt_right = 1e-3 * dir_vec
        line_scalar_right = line_norm_right.T @ line_pt_right
        # the pt is the origin

        # for the delta_v constraints
        line_dir_min = cs.vertcat(dir_vec[1], -dir_vec[0])
        line_pt_min = min_speed*cs.vertcat(cos_theta_k, sin_theta_k)
        line_norm_min = cs.vertcat(-line_dir_min[1], line_dir_min[0])
        line_scalar_min = line_norm_min.T @ line_pt_min

        line_dir_max = -cs.vertcat(dir_vec[1], -dir_vec[0])
        line_pt_max = max_speed*cs.vertcat(cos_theta_k, sin_theta_k)
        line_norm_max = cs.vertcat(-line_dir_max[1], line_dir_max[0])
        line_scalar_max = line_norm_max.T @ line_pt_max

        line_norms = [line_norm_left, line_norm_right, line_norm_min, line_norm_max]
        if get_pts:
            line_pts = [cs.vertcat(0.0,0.0), cs.vertcat(0.0,0.0), line_pt_min, line_pt_max]
            return line_norms, line_pts
        line_scalars = [line_scalar_left, line_scalar_right, line_scalar_min, line_scalar_max]
        return line_norms, line_scalars


    def get_ORCA_pairwise_robhum(self, X, humB_idx, casadi_dict=None):
        """Returns the norm vector and rho-value of pair-wise ORCA_{humA|humB} set

        :param X: state of the environment at current timestep
        :param humA_idx: _description_
        """
        # NB must use: cs.if_else(cond, iftrue_val, iffalse_val)
        X_hums = X[self.nx_r+self.np_g:]
        if self.mpc_env.human_pred_MID and not self.mpc_env.human_pred_MID_joint:
            X_humA = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g], cs.log(cs.DM.ones(self.mpc_env.num_MID_samples)/self.mpc_env.num_MID_samples))
        else:
            X_humA = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])
        X_humB = X_hums[self.nx_hum*humB_idx:(self.nx_hum*humB_idx+self.nx_hum)]

        radA = self.agent_radii[-1] + max(self.mpc_env.rob_len_buffer, self.mpc_env.rob_wid_buffer)
        radB = self.agent_radii[humB_idx]
        return self.get_ORCA_pairwise_rob_ws(X_humA, X_humB, radA, radB)


    def get_ORCA_pairwise_humrob(self, X, humA_idx, casadi_dict=None):
        """Returns the norm vector and rho-value of pair-wise ORCA_{humA|humB} set

        :param X: state of the environment at current timestep
        :param humA_idx: _description_
        """
        # NB must use: cs.if_else(cond, iftrue_val, iffalse_val)
        X_hums = X[self.nx_r+self.np_g:]
        X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
        if self.mpc_env.human_pred_MID and not self.mpc_env.human_pred_MID_joint:
            X_humB = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g], cs.log(cs.DM.ones(self.mpc_env.num_MID_samples)/self.mpc_env.num_MID_samples))
        else:
            X_humB = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])

        radA = self.agent_radii[humA_idx]
        radB = self.agent_radii[-1]
        return self.get_ORCA_pairwise(X_humA, X_humB, radA, radB)


    def get_ORCA_pairwise_humhum(self, X, humA_idx, humB_idx, casadi_dict=None):
        """Returns the norm vector and rho-value of pair-wise ORCA_{humA|humB} set

        :param X: state of the environment at current timestep
        :param humA_idx: _description_
        :param humB_idx: _description_
        """

        # NB must use: cs.if_else(cond, iftrue_val, iffalse_val)
        X_hums = X[self.nx_r+self.np_g:]
        X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
        X_humB = X_hums[self.nx_hum*humB_idx:(self.nx_hum*humB_idx+self.nx_hum)]

        radA = self.agent_radii[humA_idx]
        radB = self.agent_radii[humB_idx]
        return self.get_ORCA_pairwise(X_humA, X_humB, radA, radB)



    def init_get_v_pref_fromstate_csfunc(self):
        X_humA = cs.SX.sym('X_humA', self.nx_hum)
        v_max = cs.SX.sym('v_max')

        gx = X_humA[4]
        gy = X_humA[5]
        v_pref = cs.vertcat(gx - X_humA[0], gy - X_humA[1]) / self.time_step
        v_pref_mag = cs.sqrt(v_pref[0]**2 + v_pref[1]**2 + numstab_epsilon)

        epsilon = 1e-2

        v_pref_normed = cs.if_else(v_pref_mag > 1e-4,
                                   v_pref / (v_pref_mag) * (v_max - epsilon),
                                   v_pref / 1.0,
                                   True)

        if self.mpc_env.human_pred_MID:
            case1 = cs.Function('case1', [X_humA, v_max], [v_pref_normed])
            case2 = cs.Function('case2', [X_humA, v_max], [v_pref])
        else:
            case1 = cs.Function('case1', [X_humA, v_max], [v_pref_normed])
            case2 = cs.Function('case2', [X_humA, v_max], [v_pref])

        v_pref_ret =  cs.if_else(v_pref_mag >= v_max,
                        case1(X_humA, v_max),
                        case2(X_humA, v_max),
                        True)

        # get the v_pref for the humans
        cs_fn = cs.Function('get_v_pref_fromstate', [X_humA, v_max], [v_pref_ret])

        return cs_fn

    def init_get_rob_ws_v_pref_fromstate_csfunc(self):

        X_var = cs.MX.sym('X', self.mpc_env.nx)
        X_humA = self.get_X_humA(X_var, -1)
        v_max = self.v_max_prefs[-1]
        v_pref_raw = self.get_v_pref_fromstate_csfunc(X_humA, v_max)



        ## Let's try to get this robot to turn around. see note from 2024-04-18 16:51 EDT.
        # Make a rotation matrix to get v_pref with respect to the orientation of the robot
        # v_pref_raw_vf_raw = cs.vertcat(cs.cos(X_var[2])*v_pref_raw[0] + cs.sin(X_var[2])*v_pref_raw[1],
        #                               -cs.sin(X_var[2])*v_pref_raw[0] + cs.cos(X_var[2])*v_pref_raw[1])

        sin_head = X_var[2]
        cos_head = X_var[3]
        v_pref_raw_vf_raw = cs.vertcat(cos_head*v_pref_raw[0] + sin_head*v_pref_raw[1],
                                      -sin_head*v_pref_raw[0] + cos_head*v_pref_raw[1])

        get_v_pref_vf_func = cs.Function('get_rob_ws_v_pref_vf', [X_var], [v_pref_raw_vf_raw]).expand()
        v_pref_raw_vf = get_v_pref_vf_func(X_var)
        v_pref_flipped_y_vf = cs.vertcat(v_pref_raw_vf[0]*0.01, -v_pref_raw_vf[1])

        v_pref_flipped_y_i_raw = cs.vertcat(cos_head*v_pref_flipped_y_vf[0] - sin_head*v_pref_flipped_y_vf[1],
                                            sin_head*v_pref_flipped_y_vf[0] + cos_head*v_pref_flipped_y_vf[1])

        flipped_v_func = cs.Function('flipped_v', [X_var], [v_pref_flipped_y_i_raw]).expand()
        v_pref_flipped_y_i = flipped_v_func(X_var)

        v_pref_sym0 = cs.if_else((v_pref_raw_vf[0] > 0.0)*(cs.fabs(cs.atan2(v_pref_raw_vf[1], v_pref_raw_vf[0])) < 80*np.pi/180),
                                v_pref_raw,
                                v_pref_flipped_y_i,
                                True)

        cur_ang_vel = X_var[5]
        sin_next_head_cur_rot = sin_head*cs.cos(cur_ang_vel*self.time_step) + cos_head*cs.sin(cur_ang_vel*self.time_step)
        cos_next_head_cur_rot = cos_head*cs.cos(cur_ang_vel*self.time_step) - sin_head*cs.sin(cur_ang_vel*self.time_step)
        next_v_cur_v = cs.vertcat(X_var[4]*cos_next_head_cur_rot, X_var[4]*sin_next_head_cur_rot)

        v_pref_sym = 0.05*v_pref_sym0+0.95*next_v_cur_v

        get_v_pref_csfunc = cs.Function('get_rob_ws_v_pref', [X_var], [v_pref_sym]).expand()

        v_pref_final = get_v_pref_csfunc(X_var)

        cs_fn = cs.Function('get_rob_ws_v_pref_fromstate', [X_var], [v_pref_final]).expand()
        return cs_fn

    def get_X_humA(self, X, humA_idx):
        if humA_idx == -1:
            if self.mpc_env.human_pred_MID and not self.mpc_env.human_pred_MID_joint:
                # X_humA = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g], cs.log(cs.DM.ones(self.mpc_env.num_MID_samples)/self.mpc_env.num_MID_samples))
                X_humA = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g], cs.log(cs.DM.ones(self.mpc_env.num_MID_samples)/self.mpc_env.num_MID_samples))
            else:
                # X_humA = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])
                X_humA = cs.vertcat(X[0], X[1], X[4]*X[3], X[4]*X[2], X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])
        else:
            X_humA = X[self.nx_r+self.np_g+self.nx_hum*humA_idx:(self.nx_r+self.np_g+self.nx_hum*humA_idx+self.nx_hum)]
        return X_humA

    def get_v_pref_fromstate(self, humA_idx, X, MID_samples_t=None, MID_samples_tp1=None):
        """Obtain the preferred velocity for human ORCA agent given the state X, and the value of the agent's goal position.

        :param humA_idx: index for human agent
        :param X: symbolic variable for the state of the environment
        :return: symbolic equation for the preferred velocity
        """
        X_humA = self.get_X_humA(X, humA_idx)
        v_max = self.v_max_prefs[humA_idx]
        return self.get_v_pref_fromstate_csfunc(X_humA, v_max)


    def init_one_hum_eqns(self):
        U_humA = cs.SX.sym('U_humA', 2, 1) # new_vx, new_vy
        ksi_humA = cs.SX.sym('ksi_humA', 1, 1) # ksi
        ksi_2_humA = cs.SX.sym('ksi_2_humA', 1, 1) # ksi_2
        vars_humA = cs.vertcat(U_humA, ksi_humA, ksi_2_humA)

        self.U_humA = U_humA

        U_humA_pref = cs.SX.sym('U_humA_pref', 2,1) # new_vx, new_vy

        # Make the optimizer cost function
        cost_eqn = (U_humA - U_humA_pref).T @ (U_humA - U_humA_pref)
        self.cost_dict = {"cost_eqn": cost_eqn, "vars": {"U_humA": U_humA, "U_humA_pref": U_humA_pref}}
        self.cost_func = cs.Function('loss_ORCA_humA', [U_humA, U_humA_pref], [cost_eqn], ['U_humA', 'U_humA_pref'], ['l'])

        ksi_penal_eqn = 100 * 1.0 * vars_humA[2] ** 2
        self.ksi_penal_func = cs.Function('loss_ORCA_humA_ksi_penal', [ksi_humA], [ksi_penal_eqn], ['ksi_humA'], ['l'])

        ksi_2_penal_eqn = 50 * 1.0 * vars_humA[3] ** 2 #+ 40 * 1.0 * vars_humA[3]
        self.ksi_2_penal_func = cs.Function('loss_ORCA_humA_ksi_penal', [ksi_2_humA], [ksi_2_penal_eqn], ['ksi_2_humA'], ['l'])


    def get_hums_next_U_ws(self, X_var, params_var=None):
        mpc_env = self.mpc_env
        hums_orca_consts_list = mpc_env.hums_orca_consts
        assert len(hums_orca_consts_list) > 0
        hums_max_vel_consts = mpc_env.hums_max_vel_consts
        assert len(hums_max_vel_consts) > 0
        hums_max_acc_consts = mpc_env.hums_max_acc_consts
        assert len(hums_max_acc_consts) > 0
        hums_ksi_consts = mpc_env.hums_ksi_consts
        assert len(hums_ksi_consts) > 0
        hums_ksi_2_consts = mpc_env.hums_ksi_2_consts
        assert len(hums_ksi_2_consts) > 0
        orca_ksi_scaling = mpc_env.orca_ksi_scaling
        orca_vxy_scaling = mpc_env.orca_vxy_scaling

        nvars = 4
        var_humAs_list = [cs.MX.sym('vars_k_hum{:}'.format(idx), nvars, 1) for idx in range(mpc_env.num_hums)]

        humA_ws_solvers = []
        humA_ws_solver_funcs = []
        next_U_hums = cs.MX.zeros((self.nvars_hum*self.num_hums, 1))
        next_lambda_hums = cs.MX.zeros((self.mpc_env.nLambda, 1))

        next_U_hums_cvmm = cs.MX.zeros((self.nvars_hum*self.num_hums, 1))

        for humA_idx in range(mpc_env.num_hums):
            vars_humA = var_humAs_list[humA_idx]
            # the cost based on v_pref
            v_pref = self.get_v_pref_fromstate(humA_idx, X_var)
            cost_l = self.cost_func(U_humA=orca_vxy_scaling*vars_humA[:2], U_humA_pref=v_pref)['l']  + self.ksi_penal_func(ksi_humA=orca_ksi_scaling*vars_humA[2])['l'] + self.ksi_2_penal_func(ksi_2_humA=orca_ksi_scaling*vars_humA[3])['l']
            # + 100 * 1.0 * vars_humA[2] ** 2 + 20 * vars_humA[2]

            # orca constraints (dynamic and static)
            orca_con_list = []
            for idx, orca_con_obj in enumerate(hums_orca_consts_list[humA_idx]):
                const_fn_of_humA = orca_con_obj.casadi_dict['const_fn_of_humA']
                orca_con_list.append(const_fn_of_humA(U_ksi_humA=vars_humA, X=X_var)['const'])

            # v_max constraint
            const_fn_of_humA = hums_max_vel_consts[humA_idx].casadi_dict['maxvel_const_fn_of_humA']
            v_max_con = const_fn_of_humA(U_ksi_humA=vars_humA, X=X_var)['const']

            # ksi geq 0 constraint
            const_fn_of_humA = hums_ksi_consts[humA_idx].casadi_dict['ksi_const_fn_of_humA']
            ksi_con = const_fn_of_humA(U_ksi_humA=vars_humA)['const']

            # acc_max constraint
            const_fn_of_humA = hums_max_acc_consts[humA_idx].casadi_dict['maxacc_const_fn_of_humA']
            acc_max_con = const_fn_of_humA(U_ksi_humA=vars_humA, X=X_var)['const']

            # ksi_2 geq 0 constraint
            const_fn_of_humA = hums_ksi_2_consts[humA_idx].casadi_dict['ksi_2_const_fn_of_humA']
            ksi_con_2 = const_fn_of_humA(U_ksi_humA=vars_humA)['const']

            all_cons = orca_con_list+[v_max_con]+[acc_max_con]+[ksi_con, ksi_con_2]


            const_g = cs.vertcat(*tuple(all_cons))

            if not self.outdoor_robot_setting:
                prob = {'f': cost_l, 'x': var_humAs_list[humA_idx], 'g': const_g, 'p': X_var}
            else:
                prob = {'f': cost_l, 'x': var_humAs_list[humA_idx], 'g': const_g, 'p': params_var}


            opts = {"print_time": 0, "calc_lam_p": False}
            opts["expand"] = True
            opts["jit"] = True
            opts["compiler"] = "shell"
            # opts["jit_options"] = {"verbose":True, "flags":["-O3", "-ffast-math", "-mfma", "-mavx"]}
            opts["jit_options"] = {"verbose":True, "flags":["-O3", "-ffast-math"]}

            opts["max_iter"] = 10
            opts["regularity_check"] = True
            opts["qpsol_options"] = {"printLevel":"none","enableRegularisation":True}
            opts["print_iteration"] = False
            opts["print_header"] = False

            solver = cs.nlpsol('ws_solver_hum{:}'.format(humA_idx), 'blocksqp', prob, opts)




            X_hums_init = X_var[self.nx_r+self.np_g:]
            V_humA_init = orca_vxy_scaling*X_hums_init[self.nx_hum*humA_idx+2:(self.nx_hum*humA_idx+4)]
            ksi_init = 0.0
            ksi_2_init = 0.0
            x_init = cs.vertcat(V_humA_init, ksi_init, ksi_2_init)
            if not self.outdoor_robot_setting:
                sol_return = solver(x0=x_init, ubg=0, p=X_var)
            else:
                sol_return = solver(x0=x_init, ubg=0, p=params_var)
            sol_ans = sol_return['x']
            v_next_opt = sol_ans[:2]
            ksi_opt = sol_ans[2]
            ksi_2_opt = sol_ans[3]
            fin_cost = sol_return['f']
            fin_consts = sol_return['g']
            fin_dual = sol_return['lam_g']
            if not self.outdoor_robot_setting:
                humA_solver_func = cs.Function('ws_hum{:}_vnext'.format(humA_idx), [X_var], [v_next_opt, ksi_opt, ksi_2_opt, fin_cost, fin_consts, sol_ans, fin_dual], ['X'], ['v_next', 'ksi_opt', 'ksi_2_opt', 'fin_cost', 'fin_consts', 'fin_prim', 'fin_dual'])
            else:
                humA_solver_func = cs.Function('ws_hum{:}_vnext'.format(humA_idx), [params_var], [v_next_opt, ksi_opt, ksi_2_opt, fin_cost, fin_consts, sol_ans, fin_dual], ['params'], ['v_next', 'ksi_opt', 'ksi_2_opt', 'fin_cost', 'fin_consts', 'fin_prim', 'fin_dual'])
            humA_ws_solver_funcs.append(humA_solver_func)
            humA_ws_solvers.append(solver)

            # Now use the solver and get the next_U_hums
            if not self.outdoor_robot_setting:
                sol_return_humA = humA_solver_func(X=X_var)
            else:
                sol_return_humA = humA_solver_func(params=params_var)
            v_next = sol_return_humA['v_next']
            ksi = sol_return_humA['ksi_opt']
            ksi_2 = sol_return_humA['ksi_2_opt']
            lambdas = sol_return_humA['fin_dual']
            next_U_hums[humA_idx*self.nvars_hum] = v_next[0]
            next_U_hums[humA_idx*self.nvars_hum+1] = v_next[1]
            next_U_hums[humA_idx*self.nvars_hum+2] = ksi
            next_U_hums[humA_idx*self.nvars_hum+3] = ksi_2
            next_lambda_hums[humA_idx*self.mpc_env.nlambda_hum:((humA_idx+1)*self.mpc_env.nlambda_hum), 0] = lambdas

            next_U_hums_cvmm[humA_idx*self.nvars_hum] = V_humA_init[0]
            next_U_hums_cvmm[humA_idx*self.nvars_hum+1] = V_humA_init[1]
            next_U_hums_cvmm[humA_idx*self.nvars_hum+2] = ksi_init
            next_U_hums_cvmm[humA_idx*self.nvars_hum+3] = ksi_2_init

        self.humA_ws_solvers = humA_ws_solvers
        self.humA_ws_solver_funcs = humA_ws_solver_funcs

        if not self.outdoor_robot_setting:
            next_U_hums_fn = cs.Function('next_U_hums_fn', [X_var], [cs.vertcat(next_U_hums, next_lambda_hums)], ['X'], ['next_U_hums'])
            next_U_hums_cvmm_fn = cs.Function('next_U_hums_cvmm_fn', [X_var], [cs.vertcat(next_U_hums, next_lambda_hums)], ['X'], ['next_U_hums'])
        else:
            next_U_hums_fn = cs.Function('next_U_hums_fn', [params_var], [cs.vertcat(next_U_hums, next_lambda_hums)], ['params'], ['next_U_hums'])
            next_U_hums_cvmm_fn = cs.Function('next_U_hums_cvmm_fn', [params_var], [cs.vertcat(next_U_hums, next_lambda_hums)], ['params'], ['next_U_hums'])



        return next_U_hums_fn, next_U_hums, next_lambda_hums, next_U_hums_cvmm, next_U_hums_cvmm_fn


    def get_rob_warmstart_fn(self, mpc_env):
        humA_idx = -1
        # Make the optimizer and variables
        v_var = cs.MX.sym('v_k_robws', 2, 1)

        ksi_var = cs.MX.sym('ksi_k_robws', 1, 1)
        ksi_var_2 = cs.MX.sym('ksi_k_2_robws', 1, 1)
        if not self.outdoor_robot_setting:
            X_var = cs.MX.sym('X_var', mpc_env.nx, 1)
            stat_obs_params_vecced = None
            stat_obs_params = None
        else:
            num_params = mpc_env.nx + mpc_env.num_stat_obs*4
            params_var = cs.MX.sym('params', num_params, 1)
            X_var = params_var[:mpc_env.nx]
            stat_obs_params_vecced = params_var[mpc_env.nx:]
            stat_obs_params = cs.vertcat(*tuple([stat_obs_params_vecced[idx*4:idx*4+4].T for idx in range(self.num_stat_obs)]))

        MID_samples_all_t_all_hums = []


        if self.mpc_env.human_pred_MID:
            # make MX variables for just the one (current) timestep and the next timestep (i.e. NOT for the whole horizon) to be used for generating single-step functions
            MID_samples_dummy_t_all_hums_stacked = cs.MX.sym('MID_samples_t_all_hums_stacked', mpc_env.num_MID_samples*mpc_env.num_hums, 2)
            MID_samples_dummy_tp1_all_hums_stacked = cs.MX.sym('MID_samples_tp1_all_hums_stacked', mpc_env.num_MID_samples*mpc_env.num_hums, 2)
            # make MX variables for ALL the MID samples (i.e. for the whole horizon)
            MID_samples_stacked = cs.MX.sym('MID_samples_all_t_all_hums_stacked', (mpc_env.horiz+1)*mpc_env.num_MID_samples*mpc_env.num_hums, 2)
            MID_samples = cs.vertsplit(MID_samples_stacked, [mpc_env.num_MID_samples*mpc_env.num_hums*t_idx for t_idx in range(mpc_env.horiz+2)])


            # spoof a single MID sample in the case of the robot.
            # pretend there is one sample and the current point is the X_var
            # repeat X_var[:2].T mpc_env.num_MID_samples times
            MID_samples_t_rob_dummy = cs.repmat(X_var[:2].T, mpc_env.num_MID_samples, 1)
            # the next point is the goal itself
            MID_samples_tp1_rob_dummy =cs.repmat(X_var[mpc_env.nx_r:mpc_env.nx_r+1].T, mpc_env.num_MID_samples, 1)
            # v_pref = self.get_v_pref_fromstate(humA_idx, X_var, MID_samples_t_rob_dummy, MID_samples_tp1_rob_dummy)

        v_pref = self.get_rob_ws_v_pref_fromstate_csfunc(X_var)

        # set the cost and constraints NB for the robot
        # cost
        cost = self.cost_func(U_humA=v_var, U_humA_pref=v_pref)['l'] + 100 * 1.0 * ksi_var ** 2 - 20 * ksi_var + 10000 * 1.0 * ksi_var_2 ** 2 - 2000 * ksi_var_2

        # constraints
        # pairwise ORCA constraints
        time_coll_hor_rob_ws = 1.0 #self.time_coll_hor - 1.0
        self.get_ORCA_pairwise_rob_ws = self.init_get_ORCA_pairwise_casadi_fns(time_coll_hor=time_coll_hor_rob_ws)
        _, _, line_norms_checked, line_scalars_checked = self.get_ORCA_set_list(X_var, humA_idx)
        orca_con_list = []
        for idx in range(len(line_norms_checked)):
            orca_con_list.append( -line_norms_checked[idx].T @ v_var + line_scalars_checked[idx] - ksi_var)

        # static obs constraints
        line_norms_stat, line_scalars_stat = self.get_ORCA_stat_set_list(X_var, humA_idx, stat_obs_params=stat_obs_params)
        for idx in range(len(line_norms_stat)):
            orca_con_list.append( -line_norms_stat[idx].T @ v_var + line_scalars_stat[idx] - ksi_var_2)

        # robot dynamics-limiting constraints
        deltavel_max = np.min([np.abs(mpc_env.max_l_acc), np.abs(mpc_env.max_l_dcc)]) * self.time_step
        self.deltavel_max = deltavel_max
        rot_max = mpc_env.max_rot * mpc_env.time_step * 0.99
        self.rot_max = rot_max
        line_norms_rob, line_scalars_rob = self.get_ORCA_rob_simulatedconsts(X_var, rot_max)
        for idx in range(len(line_scalars_rob)):
            # we will make these norms switch directions for values of v_var below zero
            if idx < 2:
                orca_con_list.append( -line_norms_rob[idx].T @ v_var + line_scalars_rob[idx])
            else:
                orca_con_list.append( -line_norms_rob[idx].T @ v_var + line_scalars_rob[idx])

        # v_max constraint
        v_max_con = v_var.T @ v_var - self.mpc_env.max_speed ** 2 - ksi_var_2

        # ksi geq 0 constraint
        ksi_con = -ksi_var
        ksi_con_2 = -ksi_var_2
        all_cons = orca_con_list+[v_max_con]+[ksi_con, ksi_con_2]

        cons_leq0_vec = cs.vertcat(*all_cons)

        # Create an NLP
        if not self.outdoor_robot_setting:
            prob = {'f': cost, 'x': cs.vertcat(v_var, ksi_var, ksi_var_2), 'g': cons_leq0_vec, 'p': X_var}
        else:
            prob = {'f': cost, 'x': cs.vertcat(v_var, ksi_var, ksi_var_2), 'g': cons_leq0_vec, 'p': params_var}

        opts = {}
        opts["print_time"] = 0
        opts["calc_lam_p"] = False
        opts["expand"] = True
        opts["jit"] = True
        opts["compiler"] = "shell"
        # opts["jit_options"] = {"verbose":True, "flags":["-O3", "-ffast-math", "-mfma", "-mavx"]}
        opts["jit_options"] = {"verbose":True, "flags":["-O3", "-ffast-math"]}
        opts["qpsol"] = "qpoases"
        opts["regularity_check"] = True
        opts["qpsol_options"] = {"printLevel":"none", "enableRegularisation":True}
        opts["max_iter"] = 10
        opts["print_iteration"] = False
        opts["print_header"] = False

        solver = cs.nlpsol('solver_robws', 'blocksqp', prob, opts)

        V_robws_init = cs.vertcat(X_var[4]*X_var[3], X_var[4]*X_var[2])

        ksi_init = 0.0
        x_init = cs.vertcat(V_robws_init, ksi_init, ksi_init)

        if not self.outdoor_robot_setting:
            sol_return_fwddir = solver(x0=x_init, ubg=0, p=X_var)
        else:
            sol_return_fwddir = solver(x0=x_init, ubg=0, p=params_var)
        sol_ans_fwddir = sol_return_fwddir['x']
        v_next_opt_fwddir = sol_ans_fwddir[:2]
        ksi_opt_fwddir = sol_ans_fwddir[2]
        fin_cost_fwddir = sol_return_fwddir['f']
        fin_consts_fwddir = sol_return_fwddir['g']
        fin_dual_fwddir = sol_return_fwddir['lam_g']

        # Now do it as if the robot is facing the opposite direction
        X_var_opdir = cs.vertcat(X_var[:2], -X_var[2], -X_var[3], X_var[4], -X_var[5], X_var[6:])

        if not self.outdoor_robot_setting:
            sol_return_opdir = solver(x0=x_init, ubg=0, p=X_var_opdir)
        else:
            sol_return_opdir = solver(x0=x_init, ubg=0, p=cs.vertcat(X_var_opdir, params_var[mpc_env.nx:]))
        sol_ans_opdir = sol_return_opdir['x']
        v_next_opt_opdir = sol_ans_opdir[:2]
        ksi_opt_opdir = sol_ans_opdir[2]
        fin_cost_opdir = sol_return_opdir['f']
        fin_consts_opdir = sol_return_opdir['g']
        fin_dual_opdir = sol_return_opdir['lam_g']

        speed_k = X_var[4]

        sol_ans = cs.if_else(speed_k**2 < 0.02**2,
                                cs.if_else(fin_cost_opdir < fin_cost_fwddir,
                                           sol_ans_opdir,
                                           sol_ans_fwddir,
                                           True),
                                sol_ans_fwddir,
                                True)
        v_next_opt = cs.if_else(speed_k**2 < 0.02**2,
                                cs.if_else(fin_cost_opdir < fin_cost_fwddir,
                                           v_next_opt_opdir,
                                           v_next_opt_fwddir,
                                           True),
                                v_next_opt_fwddir,
                                True)
        ksi_opt = cs.if_else(speed_k**2 < 0.02**2,
                                cs.if_else(fin_cost_opdir < fin_cost_fwddir,
                                           ksi_opt_opdir,
                                           ksi_opt_fwddir,
                                           True),
                                ksi_opt_fwddir,
                                True)
        fin_cost = cs.if_else(speed_k**2 < 0.02**2,
                                cs.if_else(fin_cost_opdir < fin_cost_fwddir,
                                           fin_cost_opdir,
                                           fin_cost_fwddir,
                                           True),
                                fin_cost_fwddir,
                                True)
        fin_consts = cs.if_else(speed_k**2 < 0.02**2,
                                cs.if_else(fin_cost_opdir < fin_cost_fwddir,
                                           fin_consts_opdir,
                                           fin_consts_fwddir,
                                           True),
                                fin_consts_fwddir,
                                True)
        fin_dual = cs.if_else(speed_k**2 < 0.02**2,
                                cs.if_else(fin_cost_opdir < fin_cost_fwddir,
                                           fin_dual_opdir,
                                           fin_dual_fwddir,
                                           True),
                                fin_dual_fwddir,
                                True)

        # find the v_next_opt in the frame of the current position fo the robot.
        v_next_opt_robot_frame = cs.vertcat(v_next_opt[0]*X_var[3]           + v_next_opt[1]*X_var[2],         -v_next_opt[0]*X_var[2]         + v_next_opt[1]*X_var[3])
        # if the v_next_opt_robot_frame is in the negative x direction, then l_vel_next is negative
        l_vel_next_sign = cs.sign(v_next_opt_robot_frame[0])
        l_vel_next = l_vel_next_sign*cs.sqrt(v_next_opt_robot_frame.T @ v_next_opt_robot_frame)
        # to find the rotation, we can use the atan2 function. But if the l_vel_next is negative then we need to flip the v_next_opt_robot_frame vector to do atan2
        v_next_opt_robot_frame_foratan = l_vel_next_sign*v_next_opt_robot_frame
        delta_theta = cs.atan2(v_next_opt_robot_frame_foratan[1], v_next_opt_robot_frame_foratan[0])
        omega_next = delta_theta / mpc_env.time_step

        if not self.outdoor_robot_setting:
            robws_solver_func = cs.Function('robws_vnext'.format(humA_idx), [X_var], [l_vel_next, omega_next, V_robws_init, v_next_opt, ksi_opt, fin_cost, fin_consts, fin_dual], ['X'], ['l_vel_next', 'omega_next', 'v_prev', 'v_next', 'ksi_opt', 'fin_cost', 'fin_consts', 'fin_dual'])
        else:
            robws_solver_func = cs.Function('robws_vnext'.format(humA_idx), [params_var], [l_vel_next, omega_next, V_robws_init, v_next_opt, ksi_opt, fin_cost, fin_consts, fin_dual], ['params'], ['l_vel_next', 'omega_next', 'v_prev', 'v_next', 'ksi_opt', 'fin_cost', 'fin_consts', 'fin_dual'])

        if not self.outdoor_robot_setting:
            next_U_hums_fn, next_U_hums, next_lambda_hums, next_U_hums_cvmm, next_U_hums_cvmm_fn = self.get_hums_next_U_ws(X_var)
        else:
            next_U_hums_fn, next_U_hums, next_lambda_hums, next_U_hums_cvmm, next_U_hums_cvmm_fn = self.get_hums_next_U_ws(X_var, params_var)


        next_U = cs.vertcat(l_vel_next, omega_next, next_U_hums, next_lambda_hums)
        if not self.outdoor_robot_setting:
            next_U_fn = cs.Function('next_U_ws_fn', [X_var], [next_U, V_robws_init, sol_ans[:3], fin_dual], ['X'], ['next_U', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])
        else:
            next_U_fn = cs.Function('next_U_ws_fn', [params_var], [next_U, V_robws_init, sol_ans[:3], fin_dual], ['params'], ['next_U', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])

        next_X = mpc_env.system_model.f_func_nonlin(x=X_var, u=next_U)['f']




        X_ws = cs.MX.zeros((mpc_env.nx, mpc_env.horiz+1))
        X_ws[:, 0] = X_var
        U_ws = cs.MX.zeros((mpc_env.nu, mpc_env.horiz))
        U_rob_fakews = cs.MX.zeros((3, mpc_env.orca_kkt_horiz))
        U_prev_rob_fakews = cs.MX.zeros((2, mpc_env.orca_kkt_horiz))


        for k in range(mpc_env.orca_kkt_horiz):
            if not self.outdoor_robot_setting:
                next_u_ans = next_U_fn(X=X_ws[:, k])
            else:
                next_u_ans = next_U_fn(params=cs.vertcat(X_ws[:, k], stat_obs_params_vecced))
            U_ws[:,k] = next_u_ans['next_U']
            if mpc_env.human_pred_MID:
                MID_samples_t_all_hums_stacked = MID_samples[k]
                MID_samples_tp1_all_hums_stacked = MID_samples[k+1]
                stacked_preds = self.mpc_env.stack_MID_preds(MID_samples_t_all_hums_stacked, MID_samples_tp1_all_hums_stacked)
                X_ws[:,k+1] = mpc_env.system_model.f_func(x=X_ws[:, k], u=U_ws[:,k], stacked_preds=stacked_preds)['f']
            else:
                X_ws[:,k+1] = mpc_env.system_model.f_func(x=X_ws[:, k], u=U_ws[:,k])['f']
            U_rob_fakews[:,k] = next_u_ans['next_Vars_rob_fake']
            U_prev_rob_fakews[:,k] = next_u_ans['v_prev_rob_fake']


        # in case the orca_kkt horizon is less than the mpc horizon, we need to repeat the last action

        if mpc_env.orca_kkt_horiz < mpc_env.horiz: # TODO These whole block of stuff needs to be removed?
            next_U_withhumscvmm = cs.vertcat(l_vel_next, omega_next, next_U_hums_cvmm, cs.MX.zeros(next_lambda_hums.shape))
            next_U_withhumscvmm_fn = cs.Function('next_U_ws_withumscvmm_fn', [X_var], [next_U_withhumscvmm, V_robws_init, sol_ans, fin_dual], ['X'], ['next_U', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])

            for k in range(mpc_env.orca_kkt_horiz, mpc_env.horiz):
                next_u_ans = next_U_withhumscvmm_fn(X=X_ws[:, k])
                U_ws[:,k] = next_u_ans['next_U']
                X_ws[:,k+1] = mpc_env.system_model.f_func(x=X_ws[:, k], u=U_ws[:,k])['f']

            ws_fn = cs.Function('warmstart_horiz', [X_var], [X_ws, U_ws, U_rob_fakews, U_prev_rob_fakews], ['X_0'], ['X_vec', 'U_vec', 'U_rob_fakews', 'U_prev_rob_fakews'])

            X_ws_one_step = cs.MX.zeros((mpc_env.nx, mpc_env.horiz-mpc_env.orca_kkt_horiz+2))
            X_ws_one_step[:, 0] = X_var
            U_ws_one_step = cs.MX.zeros((mpc_env.nu, mpc_env.horiz-mpc_env.orca_kkt_horiz+1))
            # ws_onestep_fn = cs.Function('warmstart_one', [X_var], [next_X, next_U, V_robws_init, sol_ans, fin_dual], ['X_0'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])
            # do the final step with the hums modelled to run orca
            U_ws_one_step[:,0] = next_U
            X_ws_one_step[:,1] = mpc_env.system_model.f_func(x=X_var, u=next_U)['f']
            # fill the rest of the warm start with the hums modelled to run orca
            for k in range(1, mpc_env.horiz-mpc_env.orca_kkt_horiz+1):
                next_u_ans = next_U_withhumscvmm_fn(X=X_ws_one_step[:, k])
                U_ws_one_step[:,k] = next_u_ans['next_U']
                X_ws_one_step[:,k+1] = mpc_env.system_model.f_func(x=X_ws_one_step[:, k], u=U_ws_one_step[:,k])['f']

            ws_onestep_fn = cs.Function('warmstart_one', [X_var], [X_ws_one_step[:,1:], U_ws_one_step, V_robws_init, sol_ans, fin_dual], ['X_0'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])

        else:
            if mpc_env.human_pred_MID:
                if not self.outdoor_robot_setting:
                    ws_fn = cs.Function('warmstart_horiz', [X_var, MID_samples_stacked], [X_ws, U_ws, U_rob_fakews, U_prev_rob_fakews], ['X_0', 'MID_samples_stacked'], ['X_vec', 'U_vec', 'U_rob_fakews', 'U_prev_rob_fakews'])
                    ws_onestep_fn = cs.Function('warmstart_one', [X_var, MID_samples_dummy_t_all_hums_stacked, MID_samples_dummy_tp1_all_hums_stacked], [next_X, next_U, V_robws_init, sol_ans, fin_dual], ['X_0', 'MID_samples_0_stacked', 'MID_samples_1_stacked'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])
                else:
                    ws_fn = cs.Function('warmstart_horiz', [params_var, MID_samples_stacked], [X_ws, U_ws, U_rob_fakews, U_prev_rob_fakews], ['X_0', 'MID_samples_stacked'], ['X_vec', 'U_vec', 'U_rob_fakews', 'U_prev_rob_fakews'])
                    ws_onestep_fn = cs.Function('warmstart_one', [params_var, MID_samples_dummy_t_all_hums_stacked, MID_samples_dummy_tp1_all_hums_stacked], [next_X, next_U, V_robws_init, sol_ans, fin_dual], ['X_0', 'MID_samples_0_stacked', 'MID_samples_1_stacked'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])

            else:
                if not self.outdoor_robot_setting:
                    ws_fn = cs.Function('warmstart_horiz', [X_var], [X_ws, U_ws, U_rob_fakews, U_prev_rob_fakews], ['X_0'], ['X_vec', 'U_vec', 'U_rob_fakews', 'U_prev_rob_fakews'])
                    ws_onestep_fn = cs.Function('warmstart_one', [X_var], [next_X, next_U, V_robws_init, sol_ans, fin_dual], ['X_0'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])
                else:
                    ws_fn = cs.Function('warmstart_horiz', [params_var], [X_ws, U_ws, U_rob_fakews, U_prev_rob_fakews], ['params_0'], ['X_vec', 'U_vec', 'U_rob_fakews', 'U_prev_rob_fakews'])
                    ws_onestep_fn = cs.Function('warmstart_one', [params_var], [next_X, next_U, V_robws_init, sol_ans, fin_dual], ['params_0'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])

        # create a function that can fix the human models, given a series of robot actions. We use this in case the goal position of the human changes from one step to the next.
        X_corrected = cs.MX.zeros((mpc_env.nx, mpc_env.horiz+1))
        X_rob = cs.MX.sym('X_rob_dummy', (mpc_env.nx_r, mpc_env.horiz+1))
        U_rob = cs.MX.sym('U_rob_dummy', (mpc_env.nu_r, mpc_env.horiz))
        U_corrected = cs.MX.zeros((mpc_env.nu, mpc_env.horiz))
        X_corrected[:, 0] = X_var
        for k in range(mpc_env.horiz):
            # spoof the expected position of X_rob at next step as the goal instead of the actual goal.
            X_dummy = cs.vertcat(X_corrected[:mpc_env.nx_r, k], X_rob[:2, k+1], X_corrected[mpc_env.nx_r+mpc_env.np_g:, k])
            if mpc_env.human_pred_MID:
                MID_samples_t_all_hums_stacked = MID_samples[k]
                MID_samples_tp1_all_hums_stacked = MID_samples[k+1]
                stacked_preds = self.mpc_env.stack_MID_preds(MID_samples_t_all_hums_stacked, MID_samples_tp1_all_hums_stacked)
                # next_u_ans = next_U_fn(X=X_dummy, MID_samples_t_stacked=MID_samples_t_all_hums_stacked, MID_samples_tp1_stacked=MID_samples_tp1_all_hums_stacked)
            # else:
            if not self.outdoor_robot_setting:
                next_u_ans = next_U_fn(X=X_dummy)
            else:
                next_u_ans = next_U_fn(params=cs.vertcat(X_dummy, stat_obs_params_vecced))
            U_corrected[:,k] = next_u_ans['next_U']
            if mpc_env.human_pred_MID:
                X_corrected[:,k+1] = mpc_env.system_model.f_func(x=X_corrected[:, k], u=U_corrected[:, k], stacked_preds=stacked_preds)['f']
            else:
                X_corrected[:,k+1] = mpc_env.system_model.f_func(x=X_corrected[:, k], u=U_corrected[:, k])['f']
        if mpc_env.human_pred_MID:
            if not self.outdoor_robot_setting:
                ws_correction_fn = cs.Function('warmstart_correction', [X_var, X_rob, MID_samples_stacked], [X_corrected, U_corrected], ['X_0', 'X_rob_vec', 'MID_samples_stacked'], ['X_vec', 'U_vec'])
            else:
                ws_correction_fn = cs.Function('warmstart_correction', [params_var, X_rob, MID_samples_stacked], [X_corrected, U_corrected], ['X_0', 'X_rob_vec', 'MID_samples_stacked'], ['X_vec', 'U_vec'])
        else:
            if not self.outdoor_robot_setting:
                ws_correction_fn = cs.Function('warmstart_correction', [X_var, X_rob], [X_corrected, U_corrected], ['X_0', 'X_rob_vec'], ['X_vec', 'U_vec'])
            else:
                ws_correction_fn = cs.Function('warmstart_correction', [params_var, X_rob], [X_corrected, U_corrected], ['params_0', 'X_rob_vec'], ['X_vec', 'U_vec'])

        if mpc_env.orca_kkt_horiz < mpc_env.horiz:
            raise UserWarning('The horizon of the ORCA solver is less than the horizon of the MPC solver. This is not implemented for the correction. The correction will assume the same horizon.')

        debug_ws = {
            "orca_con_list" : orca_con_list,
            "v_max_con" : v_max_con,
            "ksi_con" : ksi_con,
            "robws_solver_func" : robws_solver_func,
        }

        return ws_fn, ws_onestep_fn, ws_correction_fn, debug_ws
