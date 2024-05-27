from copy import deepcopy
import logging
from os import stat
import numpy as np
import casadi as cs
from crowd_sim_plus.envs.utils.state_plus import FullyObservableJointState
from .orca_callback import get_human_radii, get_human_goals


def det(vector1, vector2):
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]

def abs_sq(vector1):
    return vector1.T @ vector1

def safe_divide(numer, denom):
    return numer * denom / (denom*denom + 1e-100)

class casadiORCA(object):


    def __init__(self, mpc_env, joint_state, X):
        self.orca_pol = mpc_env.dummy_human.policy # Temp for testing
        self.mpc_env = mpc_env
        self.time_step = mpc_env.time_step
        self.num_hums = mpc_env.num_hums
        self.nx_hum = mpc_env.nx_hum
        self.nX_hums = mpc_env.nX_hums
        self.num_hums = mpc_env.num_hums
        self.nx_r = mpc_env.nx_r
        self.np_g = mpc_env.np_g
        self.nvars_hum = mpc_env.nvars_hum
        self.v_max_unobservable = mpc_env.human_max_speed #self.orca_pol.max_speed
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

        self.opti_dicts = self.init_optimizers()
        humA_solvers, humA_solver_objs = self.init_solvers()
        self.humA_solvers = humA_solvers
        self.humA_solver_objs = humA_solver_objs


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
            # if len(static_obs) > 1 and len(stat_ob) > 2 and stat_ob[2] < 0:
            eps = 1e-4
            if len(static_obs) > 1 and cs.norm_2(static_obs[-1][0] - static_obs[-2][1]) < eps:
                static_obs_cvx_adj.append(len(static_obs) - 2)
            else:
                static_obs_cvx_adj.append(None)

        self.static_obs = static_obs
        self.static_obs_cvx_adj = static_obs_cvx_adj


    def reset_humans(self, state, new_h_gxs=None, new_h_gys=None):
        self.num_humans = len(state.human_states)
        # radii
        human_radii = get_human_radii(state)
        agent_radii = np.append(human_radii, state.self_state.radius)
        self.agent_radii = agent_radii + 0.01 + self.safety_space
        self.agent_radii_orig = agent_radii

        # goals
        if isinstance(state, FullyObservableJointState):
            if new_h_gxs is None or new_h_gys is None:
                self.human_gxs, self.human_gys = get_human_goals(state)
            else:
                self.human_gxs, self.human_gys = new_h_gxs, new_h_gys

                    # max speed settings
            v_max_prefs = np.zeros(len(human_radii)+1)
            for hum_idx, human_state in enumerate(state.human_states):
                v_max_prefs[hum_idx] = human_state.v_pref
            v_max_prefs[-1] = state.self_state.v_pref # for the robot (used in robot's warm start)
            self.v_max_prefs = v_max_prefs
        else:
            self.v_max_prefs = np.zeros(len(human_radii)+1)
            for hum_idx in range(len(human_radii)):
                self.v_max_prefs[hum_idx] = self.v_max_unobservable
            self.v_max_prefs[-1] = state.self_state.v_pref


    def get_ORCA_set_list(self, X, humA_idx, debug_dicts=None):
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
            if debug_dicts is not None:
                debug_dict = {}
                debug_dict['X'] = X
                debug_dict['humA_idx'] = humA_idx
                txt = 'hum{:}'.format(humB_idx) if humB_idx != -1 else 'rob'
                debug_dicts['hum{:}_{:}'.format(humA_idx, txt)] = debug_dict
            else:
                debug_dict = None
            if humA_idx == -1:
                line_norm, line_pt, line_norm_checked, line_scalar_checked = self.get_ORCA_pairwise_robhum(X, humB_idx=humB_idx, debug_dict=debug_dict)
            else:
                if humB_idx == -1:
                    # i.e. if we are dealing with the robot
                    line_norm, line_pt, line_norm_checked, line_scalar_checked = self.get_ORCA_pairwise_humrob(X, humA_idx=humA_idx, debug_dict=debug_dict)
                else:
                    line_norm, line_pt, line_norm_checked, line_scalar_checked = self.get_ORCA_pairwise_humhum(X, humA_idx=humA_idx, humB_idx=humB_idx, debug_dict=debug_dict)
            line_norms.append(line_norm)
            line_pts.append(line_pt)
            line_norms_checked.append(line_norm_checked)
            line_scalars_checked.append(line_scalar_checked)

        return line_norms, line_pts, line_norms_checked, line_scalars_checked


    def get_ORCA_stat_set_list(self, X, humA_idx, get_line_pts=False, debug_dict=None):
        """ Generate ORCA_{humA_idx} set i.e. list of ORCA_{humA_idx|humB_idx} lines

        :param X: state of entire environment (including robot)
        :param humA_idx: index of human whose orca sets we want
        """
        X_hums = X[self.nx_r+self.np_g:]
        if humA_idx == -1:
            X_humA = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[5], X[6])
        else:
            X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]

        radA = self.agent_radii[humA_idx]

        line_norms = []
        line_scalars = []
        line_pts=[]
        for stat_idx in range(self.num_stat_obs):
            p_1 = self.static_obs[stat_idx][0]
            p_2 = self.static_obs[stat_idx][1]
            cvx_adj_idx = self.static_obs_cvx_adj[stat_idx]
            if cvx_adj_idx is not None:
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


    def init_get_ORCA_pairwise_casadi_fns(self, debug_dict=None):
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
            inv_time_horiz = 1.0 / self.time_coll_hor
            w = get_w_vec(inv_time_horiz)
            w_len_sq = cs.dot(w, w)
            dotprod_1 = cs.dot(w, rel_pos)
            cond_projcutoff = (dotprod_1 < 0.0) * (dotprod_1**2 > comb_rad_sq * w_len_sq)

            def get_proj_cutoffcirc():
                w_len = cs.norm_2(w)
                unit_w = safe_divide(w, w_len)
                line_dir = cs.vertcat(unit_w[1], -unit_w[0])
                u = (comb_rad * inv_time_horiz - w_len) * unit_w
                return cs.vertcat(line_dir, u)

            def get_proj_leg():
                # py-rvo-2 way
                leg = cs.sqrt(cs.fabs(dist_sq - comb_rad_sq))

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
            rel_pos_dist = cs.norm_2(rel_pos)
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

        # Invalidate lines that do not matter:
        v_max = 2.0 # some upper bound on the human speeds
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
        cond_curcoll = cond_left_vtx_coll + cond_right_vtx_coll + cond_segment_coll

        def get_curcoll():
            line_pt = 0.0 * rel_pos_1

            def coll_left_vtx():
                line_dir = safe_divide(cs.vertcat(-rel_pos_1[1], rel_pos_1[0]), cs.norm_2(rel_pos_1))
                return line_dir

            def not_coll_left_vtx():
                def coll_right_vtx():
                    line_dir = safe_divide(cs.vertcat(-rel_pos_2[1], rel_pos_2[0]), cs.norm_2(rel_pos_2))
                    return line_dir

                def coll_segment():
                    # must be coll segment
                    line_dir = -  safe_divide(obst_vec, cs.norm_2(obst_vec))
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
                    leg1 = cs.sqrt(cs.fabs(dist_sq_1 - rad_sq))
                    left_leg_dir = safe_divide(cs.vertcat(rel_pos_1[0] * leg1 - rel_pos_1[1] * radA, rel_pos_1[0] * radA + rel_pos_1[1] * leg1), dist_sq_1)
                    right_leg_dir = safe_divide(cs.vertcat(rel_pos_1[0] * leg1 + rel_pos_1[1] * radA, -rel_pos_1[0] * radA + rel_pos_1[1] * leg1), dist_sq_1)
                    left_cutoff = inv_time_horiz_obst * rel_pos_1 #(obstacle1->point_ - position_)
                    right_cutoff = inv_time_horiz_obst * rel_pos_1 #(obstacle2->point_ - position_)
                    return cs.vertcat(left_leg_dir, right_leg_dir, left_cutoff, right_cutoff)

                # oblique view of obstacle => obstacle defined by right vertex
                def nocoll_right_vtx():
                    # NB OBS1 = OBS2
                    leg2 = cs.sqrt(cs.fabs(dist_sq_2 - rad_sq))
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
                leg1 = cs.sqrt(cs.fabs(dist_sq_1 - rad_sq))
                left_leg_dir = safe_divide(cs.vertcat(rel_pos_1[0] * leg1 - rel_pos_1[1] * radA, rel_pos_1[0] * radA + rel_pos_1[1] * leg1), dist_sq_1)

                # assuming obs 2 is convex:
                leg2 = cs.sqrt(cs.fabs(dist_sq_2 - rad_sq))
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
                unitW = safe_divide(v_diff, cs.norm_2(v_diff))
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
                        unit_dir = safe_divide(p_1 - p_2, cs.norm_2(p_1 - p_2))
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
            v_max = 1.5 # some upper bound for the max velocity value

            case1 = cs.Function('case1', [*fn_inputs], [-1.15*v_max*line_norm])
            case2 = cs.Function('case2', [*fn_inputs], [line_pt_calc])

            line_pt = cs.if_else((cs.fabs(line_norm.T @ line_pt_calc - adj_line_norm.T @ adj_line_pt) < eps)*(cs.norm_2(line_norm - adj_line_norm) < eps),
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


    def get_ORCA_rob_simulatedconsts(self, X, rot_max, deltavel_max, get_pts=False):
        theta_k = X[2]
        speed_k = X[3]

        theta_left = theta_k + rot_max
        theta_right = theta_k - rot_max

        min_speed = speed_k - deltavel_max # this is required to make sure that the warmstart is feasible: the total value function of the warmstart must be leq the terminal cost of the initial condition
        max_speed = speed_k + deltavel_max

        # For the delta_theta constraints
        line_dir_left = cs.vertcat(cs.cos(theta_left), cs.sin(theta_left))
        line_norm_left = cs.vertcat(line_dir_left[1], -line_dir_left[0])
        line_scalar_left = 0.0

        line_dir_right = -cs.vertcat(cs.cos(theta_right), cs.sin(theta_right))
        line_norm_right = cs.vertcat(line_dir_right[1], -line_dir_right[0])
        line_scalar_right = 0.0
        # the pt is the origin

        # for the delta_v constraints
        line_dir_min = cs.vertcat(cs.sin(theta_k), -cs.cos(theta_k))
        line_pt_min = min_speed*cs.vertcat(cs.cos(theta_k), cs.sin(theta_k))
        line_norm_min = cs.vertcat(-line_dir_min[1], line_dir_min[0])
        line_scalar_min = line_norm_min.T @ line_pt_min

        line_dir_max = -cs.vertcat(cs.sin(theta_k), -cs.cos(theta_k))
        line_pt_max = max_speed*cs.vertcat(cs.cos(theta_k), cs.sin(theta_k))
        line_norm_max = cs.vertcat(-line_dir_max[1], line_dir_max[0])
        line_scalar_max = line_norm_max.T @ line_pt_max

        line_norms = [line_norm_left, line_norm_right, line_norm_min, line_norm_max]
        if get_pts:
            line_pts = [cs.vertcat(0.0,0.0), cs.vertcat(0.0,0.0), line_pt_min, line_pt_max]
            return line_norms, line_pts
        line_scalars = [line_scalar_left, line_scalar_right, line_scalar_min, line_scalar_max]
        return line_norms, line_scalars


    def get_ORCA_pairwise_robhum(self, X, humB_idx, debug_dict=None):
        """Returns the norm vector and rho-value of pair-wise ORCA_{humA|humB} set

        :param X: state of the environment at current timestep
        :param humA_idx: _description_
        """
        X_hums = X[self.nx_r+self.np_g:]
        X_humA = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])
        X_humB = X_hums[self.nx_hum*humB_idx:(self.nx_hum*humB_idx+self.nx_hum)]

        radA = self.agent_radii[-1]
        radB = self.agent_radii[humB_idx]
        return self.get_ORCA_pairwise(X_humA, X_humB, radA, radB)


    def get_ORCA_pairwise_humrob(self, X, humA_idx, debug_dict=None):
        """Returns the norm vector and rho-value of pair-wise ORCA_{humA|humB} set

        :param X: state of the environment at current timestep
        :param humA_idx: _description_
        """
        X_hums = X[self.nx_r+self.np_g:]
        X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
        X_humB = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])

        radA = self.agent_radii[humA_idx]
        radB = self.agent_radii[-1]
        return self.get_ORCA_pairwise(X_humA, X_humB, radA, radB)


    def get_ORCA_pairwise_humhum(self, X, humA_idx, humB_idx, debug_dict=None):
        """Returns the norm vector and rho-value of pair-wise ORCA_{humA|humB} set

        :param X: state of the environment at current timestep
        :param humA_idx: _description_
        :param humB_idx: _description_
        """
        X_hums = X[self.nx_r+self.np_g:]
        X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
        X_humB = X_hums[self.nx_hum*humB_idx:(self.nx_hum*humB_idx+self.nx_hum)]

        radA = self.agent_radii[humA_idx]
        radB = self.agent_radii[humB_idx]
        return self.get_ORCA_pairwise(X_humA, X_humB, radA, radB)


    def init_get_v_pref_fromstate_csfunc(self):
        X_humA = cs.SX.sym('X_humA', self.nx_hum)
        v_max = cs.SX.sym('v_max')

        gx = X_humA[-2]
        gy = X_humA[-1]
        v_pref = cs.vertcat(gx - X_humA[0], gy - X_humA[1])
        v_pref_mag = cs.sqrt(v_pref[0]**2 + v_pref[1]**2) + 0.001

        epsilon = 1e-3
        v_pref_normed = v_pref / v_pref_mag * (v_max - epsilon)
        # v_pref_ret =  cs.if_else(v_pref_mag >= v_max,
        #                  v_pref_normed,
        #                  v_pref,
        #                  True)


        case1 = cs.Function('case1', [X_humA, v_max], [v_pref_normed])
        case2 = cs.Function('case2', [X_humA, v_max], [v_pref])
        v_pref_ret =  cs.if_else(v_pref_mag >= v_max,
                         case1(X_humA, v_max),
                         case2(X_humA, v_max),
                         True)


        cs_fn = cs.Function('get_v_pref_fromstate', [X_humA, v_max], [v_pref_ret])
        return cs_fn


    def get_v_pref_fromstate(self, humA_idx, X):
        """Obtain the preferred velocity for human ORCA agent given the state X, and the value of the agent's goal position.

        :param humA_idx: index for human agent
        :param X: symbolic variable for the state of the environment
        :return: symbolic equation for the preferred velocity
        """
        if humA_idx == -1:
            X_humA = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])
        else:
            X_humA = X[self.nx_r+self.np_g+self.nx_hum*humA_idx:(self.nx_r+self.np_g+self.nx_hum*humA_idx+self.nx_hum)]
        v_max = self.v_max_prefs[humA_idx]
        return self.get_v_pref_fromstate_csfunc(X_humA, v_max)


    def get_v_pref_fromstate_old(self, humA_idx, X):
        """Obtain the preferred velocity for human ORCA agent given the state X, and the value of the agent's goal position.

        :param humA_idx: index for human agent
        :param X: symbolic variable for the state of the environment
        :return: symbolic equation for the preferred velocity
        """
        if humA_idx == -1:
            X_humA = cs.vertcat(X[0], X[1], X[3]*cs.cos(X[2]), X[3]*cs.sin(X[2]), X[self.mpc_env.nx_r:self.mpc_env.nx_r+self.mpc_env.np_g])
        else:
            X_humA = X[self.nx_r+self.np_g+self.nx_hum*humA_idx:(self.nx_r+self.np_g+self.nx_hum*humA_idx+self.nx_hum)]
        v_max = self.v_max_prefs[humA_idx]

        gx = X_humA[-2]
        gy = X_humA[-1]
        v_pref = cs.vertcat(gx - X_humA[0], gy - X_humA[1])
        v_pref_mag = cs.sqrt(v_pref[0]**2 + v_pref[1]**2) + 0.001
        v_max = self.v_max_prefs[humA_idx]
        epsilon = 1e-3
        v_pref_normed = v_pref / v_pref_mag * (v_max - epsilon)
        v_pref_ret =  cs.if_else(v_pref_mag >= v_max,
                         v_pref_normed,
                         v_pref,
                         True)
        return v_pref_ret


    def init_one_hum_eqns(self):
        U_humA = cs.SX.sym('U_humA', 2, 1) # new_vx, new_vy
        ksi_humA = cs.SX.sym('ksi_humA', 1, 1) # ksi
        vars_humA = cs.vertcat(U_humA, ksi_humA)

        self.U_humA = U_humA

        U_humA_pref = cs.SX.sym('U_humA_pref', 2,1) # new_vx, new_vy

        # Make the optimizer cost function
        cost_eqn = (U_humA - U_humA_pref).T @ (U_humA - U_humA_pref)
        self.cost_dict = {"cost_eqn": cost_eqn, "vars": {"U_humA": U_humA, "U_humA_pref": U_humA_pref}}
        self.cost_func = cs.Function('loss_ORCA_humA', [U_humA, U_humA_pref], [cost_eqn], ['U_humA', 'U_humA_pref'], ['l'])

        ksi_penal_eqn = 100 * 1.0 * vars_humA[-1] ** 2
        self.ksi_penal_func = cs.Function('loss_ORCA_humA_ksi_penal', [ksi_humA], [ksi_penal_eqn], ['ksi_humA'], ['l'])


    def get_nlpsol(self, humA_idx):
        # Make the optimizer and variables
        v_var = cs.MX.sym('v_k_hum{:}'.format(humA_idx), 2, 1)
        X_var = cs.MX.sym('X_var', self.mpc_env.nx, 1)
        ksi_var = cs.MX.sym('ksi_k_hum{:}'.format(humA_idx), 1, 1)
        v_pref = self.get_v_pref_fromstate(humA_idx, X_var)

        # set the cost and constraints
        # cost
        cost = self.cost_func(U_humA=v_var, U_humA_pref=v_pref)['l'] + self.ksi_penal_func(ksi_humA=ksi_var)['l']
        # constraints
        # pairwise ORCA constraints
        _, _, line_norms_checked, line_scalars_checked = self.get_ORCA_set_list(X_var, humA_idx)
        orca_con_list = []
        for idx in range(len(line_norms_checked)):
            orca_con_list.append( -line_norms_checked[idx].T @ v_var + line_scalars_checked[idx] - ksi_var)

        # static obs constraints
        line_norms_stat, line_scalars_stat = self.get_ORCA_stat_set_list(X_var, humA_idx)
        for idx in range(len(line_norms_stat)):
            orca_con_list.append( -line_norms_stat[idx].T @ v_var + line_scalars_stat[idx])
            # opti.subject_to(orca_con_list[len(line_norms_checked)+idx])

        # v_max constraint
        v_max_con = v_var.T @ v_var - self.v_max_prefs[humA_idx] ** 2

        # ksi geq 0 constraint
        ksi_con = -ksi_var
        all_cons = orca_con_list+[v_max_con]+[ksi_con]

        cons_leq0_vec = cs.vertcat(*all_cons)

        # Create an NLP ag1_
        prob = {'f': cost, 'x': cs.vertcat(v_var, ksi_var), 'g': cons_leq0_vec, 'p': X_var}


        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb":"yes","error_on_fail":False, "calc_lam_p": False}

        solver = cs.nlpsol('solver_hum{:}'.format(humA_idx), 'ipopt', prob, opts)

        X_hums_init = X_var[self.nx_r+self.np_g:]
        V_humA_init = X_hums_init[self.nx_hum*humA_idx+2:(self.nx_hum*humA_idx+self.nx_hum-2)]
        ksi_init = 0
        x_init = cs.vertcat(V_humA_init, ksi_init)

        sol_return = solver(x0=x_init, ubg=0, p=X_var)
        sol_ans = sol_return['x']
        v_next_opt = sol_ans[:-1]
        ksi_opt = sol_ans[-1]
        fin_cost = sol_return['f']
        fin_consts = sol_return['g']
        fin_dual = sol_return['lam_g']
        humA_solver_func = cs.Function('hum{:}_vnext'.format(humA_idx), [X_var], [v_next_opt, ksi_opt, fin_cost, fin_consts, sol_ans, fin_dual], ['X'], ['v_next', 'ksi_opt', 'fin_cost', 'fin_consts', 'fin_prim', 'fin_dual'])

        return solver, humA_solver_func


    def init_solvers(self):
        humA_solvers = []
        humA_solver_objs = []
        for humA_idx in range(self.num_hums):
            solver_obj, solvers_funcs = self.get_nlpsol(humA_idx)
            humA_solvers.append(solvers_funcs)
            humA_solver_objs.append(solver_obj)

        return humA_solvers, humA_solver_objs


    def get_cs_opti(self, humA_idx):
        # Make the optimizer and variables
        opti = cs.Opti()
        v_var = opti.variable(2, 1)
        # ksi_var = opti.variable(1, 1)

        X_var = opti.parameter(self.mpc_env.nx, 1)
        v_pref = self.get_v_pref_fromstate(humA_idx, X_var)

        # set the cost and constraints
        # cost
        cost = self.cost_func(U_humA=v_var, U_humA_pref=v_pref)['l']
        # ksi penalty
        # cost += self.ksi_penal_fn(cs.vertcat(v_var,ksi_var))['l']

        # constraints
        # pairwise ORCA constraints
        _, _, line_norms_checked, line_scalars_checked = self.get_ORCA_set_list(X_var, humA_idx)
        orca_con_list = []
        for idx in range(len(line_norms_checked)):
            orca_con_list.append( -line_norms_checked[idx].T @ v_var + line_scalars_checked[idx] <= 0)
            opti.subject_to(orca_con_list[idx])

        # static obs constraints
        line_norms_stat, line_scalars_stat = self.get_ORCA_stat_set_list(X_var, humA_idx)
        for idx in range(len(line_norms_stat)):
            orca_con_list.append( -line_norms_stat[idx].T @ v_var + line_scalars_stat[idx] <= 0)
            opti.subject_to(orca_con_list[len(line_norms_checked)+idx])

        # v_max constraint
        v_max_con = v_var.T @ v_var - self.v_max_prefs[humA_idx] ** 2 <= 0
        opti.subject_to(v_max_con)

        # ksi constraint
        # ksi_con = -ksi_var <= 0
        # opti.subject_to(ksi_con)

        opti.minimize(cost)

        # Create solver (IPOPT solver in this version)
        opts = {"expand": False,"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}
        # opts = {"ipopt.print_level": 3, "ipopt.max_iter" : 1000, "ipopt.tol":1e-15, "ipopt.constr_viol_tol":1e-20, "ipopt.dual_inf_tol":1e-15}
        # opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.max_iter" : 1000}
        # opts = {"expand": False}
        opti.solver('ipopt', opts)
        opti_dict = {
            "opti": opti,
            "v_var": v_var,
            # "ksi_var": ksi_var,
            "X_var": X_var,
            "cost": cost,
            "orca_con_list": orca_con_list,
            "v_max_con": v_max_con,
            # "ksi_con": ksi_con
        }


        return opti_dict


    def init_optimizers(self):
        opti_dicts = []
        for humA_idx in range(self.num_hums):
            opti_dicts.append(self.get_cs_opti(humA_idx))

        return opti_dicts



    def optimize_all(self, X_init, get_dual=True):
        X_hums = np.array(X_init[self.nx_r+self.np_g:])
        X_hums = X_hums.reshape((len(X_hums), 1))
        next_X_hums = np.zeros(X_hums.shape)
        next_U_hums = np.zeros((self.nvars_hum*self.num_hums, 1))
        next_lambda_hums = np.zeros((self.mpc_env.nLambda, 1))
        for humA_idx in range(self.num_hums):
            px = X_hums[humA_idx*self.nx_hum]
            py = X_hums[humA_idx*self.nx_hum+1]
            v_next, ksi, orca_lambda_arr, v_max_lambda, ksi_con_lambda  = self.optimize_one(humA_idx, X_init, True)
            next_px = px + self.mpc_env.time_step * v_next[0]
            next_py = py + self.mpc_env.time_step * v_next[1]
            next_X_hums[humA_idx*self.nx_hum] = next_px
            next_X_hums[humA_idx*self.nx_hum+1] = next_py
            next_X_hums[humA_idx*self.nx_hum+2] = v_next[0]
            next_X_hums[humA_idx*self.nx_hum+3] = v_next[1]
            next_X_hums[humA_idx*self.nx_hum+4] = X_hums[humA_idx*self.nx_hum+4]
            next_X_hums[humA_idx*self.nx_hum+5] = X_hums[humA_idx*self.nx_hum+5]
            next_U_hums[humA_idx*self.nvars_hum] = v_next[0]
            next_U_hums[humA_idx*self.nvars_hum+1] = v_next[1]
            next_U_hums[humA_idx*self.nvars_hum+2] = ksi
            next_lambda_hums[humA_idx*self.mpc_env.nlambda_hum:((humA_idx+1)*self.mpc_env.nlambda_hum-2), 0] = orca_lambda_arr
            next_lambda_hums[(humA_idx+1)*self.mpc_env.nlambda_hum-2] = v_max_lambda
            next_lambda_hums[(humA_idx+1)*self.mpc_env.nlambda_hum-1] = ksi_con_lambda

        return next_X_hums, next_U_hums, next_lambda_hums


    def optimize_one(self, humA_idx, X_init, get_dual=False):
        opti_dict = self.opti_dicts[humA_idx]
        opti = opti_dict["opti"]
        v_var = opti_dict["v_var"]
        # ksi_var = opti_dict["ksi_var"]
        X_var = opti_dict["X_var"]
        cost = opti_dict["cost"]
        orca_con_list = opti_dict["orca_con_list"]
        v_max_con = opti_dict["v_max_con"]
        # ksi_con = opti_dict["ksi_con"]

        # set initial value to the previous starting velocity
        X_hums_init = X_init[self.nx_r+self.np_g:]
        V_humA_init = X_hums_init[self.nx_hum*humA_idx+2:(self.nx_hum*humA_idx+self.nx_hum-2)]

        opti.set_initial(v_var, deepcopy(V_humA_init))
        # opti.set_initial(ksi_var, 0.0)

        opti.set_value(X_var, X_init)

        try:
            sol = opti.solve()
            v_next = sol.value(v_var)
            # ksi = sol.value(ksi_var)
            orca_lambda_array = np.array([sol.value(opti.dual(orca_con)) for orca_con in orca_con_list])
            v_max_lambda = sol.value(opti.dual(v_max_con))
            cost_fin = sol.value(cost)
            # ksi_con_lambda = sol.value(opti.dual(ksi_con))
            # logging.debug('[CasADi ORCA OPT] humA_idx: {:}, v_next* {:}, ksi* {:}, orca_lambda_list* {:}, v_max_lambda* {:}, ksi_const_lambda* {:}'.format(humA_idx, v_next, ksi, orca_lambda_array, v_max_lambda, ksi_con_lambda))
            logging.info('[CasADi ORCA opti] humA_idx: {:}, v_next* {:}, orca_lambda_list* {:}, v_max_lambda* {:}'.format(humA_idx, v_next, orca_lambda_array, v_max_lambda))
        except RuntimeError:
            logging.warn('[CasADi ORCA OPT] Casadi ORCA opt failed, humA_idx: {:}'.format(humA_idx))
            v_next = np.array([np.nan, np.nan])
            orca_lambda_array = np.array([np.nan for _ in orca_con_list])
            v_max_lambda = np.nan
            cost_fin = np.nan
            ksi = np.nan

        if get_dual:
            # return v_next, ksi, orca_lambda_array, v_max_lambda, ksi_con_lambda
            return v_next, orca_lambda_array, v_max_lambda, cost_fin
        return v_next, ksi


    def get_hums_next_U_ws(self, X_var):
        mpc_env = self.mpc_env
        hums_orca_consts_list = mpc_env.hums_orca_consts
        assert len(hums_orca_consts_list) > 0
        hums_max_vel_consts = mpc_env.hums_max_vel_consts
        assert len(hums_max_vel_consts) > 0
        hums_ksi_consts = mpc_env.hums_ksi_consts
        assert len(hums_ksi_consts) > 0
        orca_ksi_scaling = mpc_env.orca_ksi_scaling
        orca_vxy_scaling = mpc_env.orca_vxy_scaling

        var_humAs_list = [cs.MX.sym('vars_k_hum{:}'.format(idx), 3, 1) for idx in range(mpc_env.num_hums)]

        humA_ws_solvers = []
        humA_ws_solver_funcs = []
        next_U_hums = cs.MX.zeros((self.nvars_hum*self.num_hums, 1))
        next_lambda_hums = cs.MX.zeros((self.mpc_env.nLambda, 1))

        next_U_hums_cvmm = cs.MX.zeros((self.nvars_hum*self.num_hums, 1))

        for humA_idx in range(mpc_env.num_hums):
            vars_humA = var_humAs_list[humA_idx]

            # the cost based on v_pref
            v_pref = self.get_v_pref_fromstate(humA_idx, X_var)
            cost_l = self.cost_func(U_humA=orca_vxy_scaling*vars_humA[:-1], U_humA_pref=v_pref)['l'] + self.ksi_penal_func(ksi_humA=orca_ksi_scaling*vars_humA[-1])['l']

            # orca constraints (dynamic and static)
            orca_con_list = []
            for idx, orca_con_obj in enumerate(hums_orca_consts_list[humA_idx]):
                const_fn_of_humA = orca_con_obj.debug_dict['const_fn_of_humA']
                orca_con_list.append(const_fn_of_humA(U_ksi_humA=vars_humA, X=X_var)['const'])

            # v_max constraint
            const_fn_of_humA = hums_max_vel_consts[humA_idx].debug_dict['maxvel_const_fn_of_humA']
            v_max_con = const_fn_of_humA(U_ksi_humA=vars_humA)['const']

            # ksi geq 0 constraint
            const_fn_of_humA = hums_max_vel_consts[humA_idx].debug_dict['ksi_const_fn_of_humA']
            ksi_con = const_fn_of_humA(U_ksi_humA=vars_humA)['const']

            all_cons = orca_con_list+[v_max_con]+[ksi_con]
            const_g = cs.vertcat(*tuple(all_cons))

            prob = {'f': cost_l, 'x': vars_humA, 'g': const_g, 'p': X_var}
            opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb":"yes","error_on_fail":False, "calc_lam_p": False}
            opts["ipopt.check_derivatives_for_naninf"] = "no"
            opts["ipopt.jacobian_regularization_value"] = 1e-4
            opts["ipopt.min_hessian_perturbation"] = 1e-4

            solver = cs.nlpsol('ws_solver_hum{:}'.format(humA_idx), 'ipopt', prob, opts)

            X_hums_init = X_var[self.nx_r+self.np_g:]
            V_humA_init = orca_vxy_scaling**(-1)*X_hums_init[self.nx_hum*humA_idx+2:(self.nx_hum*humA_idx+self.nx_hum-2)]
            ksi_init = 0.0
            x_init = cs.vertcat(V_humA_init, ksi_init)

            sol_return = solver(x0=x_init, ubg=0, p=X_var)
            sol_ans = sol_return['x']
            v_next_opt = sol_ans[:-1]
            ksi_opt = sol_ans[-1]
            fin_cost = sol_return['f']
            fin_consts = sol_return['g']
            fin_dual = sol_return['lam_g']
            humA_solver_func = cs.Function('ws_hum{:}_vnext'.format(humA_idx), [X_var], [v_next_opt, ksi_opt, fin_cost, fin_consts, sol_ans, fin_dual], ['X'], ['v_next', 'ksi_opt', 'fin_cost', 'fin_consts', 'fin_prim', 'fin_dual'])
            humA_ws_solver_funcs.append(humA_solver_func)
            humA_ws_solvers.append(solver)

            # Now use the solver and get the next_U_hums
            sol_return_humA = humA_solver_func(X=X_var)
            v_next = sol_return_humA['v_next']
            ksi = sol_return_humA['ksi_opt']
            lambdas = sol_return_humA['fin_dual']
            next_U_hums[humA_idx*self.nvars_hum] = v_next[0]
            next_U_hums[humA_idx*self.nvars_hum+1] = v_next[1]
            next_U_hums[humA_idx*self.nvars_hum+2] = ksi
            next_lambda_hums[humA_idx*self.mpc_env.nlambda_hum:((humA_idx+1)*self.mpc_env.nlambda_hum), 0] = lambdas

            next_U_hums_cvmm[humA_idx*self.nvars_hum] = V_humA_init[0]
            next_U_hums_cvmm[humA_idx*self.nvars_hum+1] = V_humA_init[1]
            next_U_hums_cvmm[humA_idx*self.nvars_hum+2] = ksi_init

        self.humA_ws_solvers = humA_ws_solvers
        self.humA_ws_solver_funcs = humA_ws_solver_funcs
        next_U_hums_fn = cs.Function('next_U_hums_fn', [X_var], [cs.vertcat(next_U_hums, next_lambda_hums)], ['X'], ['next_U_hums'])
        next_U_hums_cvmm_fn = cs.Function('next_U_hums_cvmm_fn', [X_var], [cs.vertcat(next_U_hums, next_lambda_hums)], ['X'], ['next_U_hums'])
        return next_U_hums_fn, next_U_hums, next_lambda_hums, next_U_hums_cvmm, next_U_hums_cvmm_fn


    def get_hums_next_U_vanilla_nlpsol(self, X_var):
        next_U_hums = cs.MX.zeros((self.nvars_hum*self.num_hums, 1))
        next_lambda_hums = cs.MX.zeros((self.mpc_env.nLambda, 1))
        for humA_idx in range(self.num_hums):
            sol_return_humA = self.humA_solvers[humA_idx](X=X_var)
            v_next = sol_return_humA['v_next']
            ksi = sol_return_humA['ksi_opt']
            lambdas = sol_return_humA['fin_dual']
            next_U_hums[humA_idx*self.nvars_hum] = v_next[0]
            next_U_hums[humA_idx*self.nvars_hum+1] = v_next[1]
            next_U_hums[humA_idx*self.nvars_hum+2] = ksi
            next_lambda_hums[humA_idx*self.mpc_env.nlambda_hum:((humA_idx+1)*self.mpc_env.nlambda_hum), 0] = lambdas

        return next_U_hums, next_lambda_hums


    def get_rob_warmstart_fn(self, mpc_env):
        humA_idx = -1
        # Make the optimizer and variables
        v_var = cs.MX.sym('v_k_robws', 2, 1)
        X_var = cs.MX.sym('X_var', mpc_env.nx, 1)
        ksi_var = cs.MX.sym('ksi_k_robws', 1, 1)
        v_pref = self.get_v_pref_fromstate(humA_idx, X_var)

        # set the cost and constraints
        # cost
        cost = self.cost_func(U_humA=v_var, U_humA_pref=v_pref)['l'] + self.ksi_penal_func(ksi_humA=ksi_var)['l']

        # constraints
        # pairwise ORCA constraints
        _, _, line_norms_checked, line_scalars_checked = self.get_ORCA_set_list(X_var, humA_idx)
        orca_con_list = []
        for idx in range(len(line_norms_checked)):
            orca_con_list.append( -line_norms_checked[idx].T @ v_var + line_scalars_checked[idx] - ksi_var)

        # static obs constraints
        line_norms_stat, line_scalars_stat = self.get_ORCA_stat_set_list(X_var, humA_idx)
        for idx in range(len(line_norms_stat)):
            orca_con_list.append( -line_norms_stat[idx].T @ v_var + line_scalars_stat[idx])

        # robot dynamics-limiting constraints
        deltavel_max = np.min([np.abs(mpc_env.max_l_acc), np.abs(mpc_env.max_l_dcc)])
        self.deltavel_max = deltavel_max
        rot_max = mpc_env.max_rot * mpc_env.time_step
        self.rot_max = rot_max
        line_norms_rob, line_scalars_rob = self.get_ORCA_rob_simulatedconsts(X_var, rot_max, deltavel_max)
        for idx in range(len(line_scalars_rob)):
            # we will make these norms switch directions for values of v_var below zero
            orca_con_list.append( -cs.sign(v_var.T @ cs.vertcat(cs.cos(X_var[2]), cs.sin(X_var[2])))*line_norms_rob[idx].T @ v_var + line_scalars_rob[idx])

        # v_max constraint
        v_max_con = v_var.T @ v_var - self.v_max_prefs[humA_idx] ** 2

        # ksi geq 0 constraint
        ksi_con = -ksi_var
        all_cons = orca_con_list+[v_max_con]+[ksi_con]

        cons_leq0_vec = cs.vertcat(*all_cons)

        # Create an NLP
        prob = {'f': cost, 'x': cs.vertcat(v_var, ksi_var), 'g': cons_leq0_vec, 'p': X_var}

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb":"yes", "error_on_fail":False, "calc_lam_p": False, "expand":True}

        solver = cs.nlpsol('solver_robws', 'ipopt', prob, opts)

        V_robws_init = cs.vertcat(X_var[3]*cs.cos(X_var[2]), X_var[3]*cs.sin(X_var[2]))

        ksi_init = 0.0
        x_init = cs.vertcat(V_robws_init, ksi_init)

        sol_return = solver(x0=x_init, ubg=0, p=X_var)
        sol_ans = sol_return['x']
        v_next_opt = sol_ans[:-1]
        ksi_opt = sol_ans[-1]
        fin_cost = sol_return['f']
        fin_consts = sol_return['g']
        fin_dual = sol_return['lam_g']

        l_vel_next = cs.sign(v_next_opt.T @ cs.vertcat(cs.cos(X_var[2]), cs.sin(X_var[2])))*cs.norm_2(v_next_opt)

        vnext_foratan = cs.sign(v_next_opt.T @ cs.vertcat(cs.cos(X_var[2]), cs.sin(X_var[2])))*v_next_opt
        new_theta = cs.if_else(cs.fabs(l_vel_next) < 1e-5, X_var[2], cs.atan2(vnext_foratan[1], vnext_foratan[0]), True)
        omega_next = (new_theta - X_var[2]) / mpc_env.time_step

        robws_solver_func = cs.Function('robws_vnext'.format(humA_idx), [X_var], [l_vel_next, omega_next, V_robws_init, v_next_opt, ksi_opt, fin_cost, fin_consts, fin_dual], ['X'], ['l_vel_next', 'omega_next', 'v_prev', 'v_next', 'ksi_opt', 'fin_cost', 'fin_consts', 'fin_dual'])

        next_U_hums_fn, next_U_hums, next_lambda_hums, next_U_hums_cvmm, next_U_hums_cvmm_fn = self.get_hums_next_U_ws(X_var)

        next_U = cs.vertcat(l_vel_next, omega_next, next_U_hums, next_lambda_hums)
        next_U_fn = cs.Function('next_U_ws_fn', [X_var], [next_U, V_robws_init, sol_ans, fin_dual], ['X'], ['next_U', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])
        next_X = mpc_env.system_model.f_func(x=X_var, u=next_U)['f']

        X_ws = cs.MX.zeros((mpc_env.nx, mpc_env.horiz+1))
        X_ws[:, 0] = X_var
        U_ws = cs.MX.zeros((mpc_env.nu, mpc_env.horiz))
        U_rob_fakews = cs.MX.zeros((3, mpc_env.orca_kkt_horiz))
        U_prev_rob_fakews = cs.MX.zeros((2, mpc_env.orca_kkt_horiz))

        for k in range(mpc_env.orca_kkt_horiz):
            next_u_ans = next_U_fn(X=X_ws[:, k])
            U_ws[:,k] = next_u_ans['next_U']
            X_ws[:,k+1] = mpc_env.system_model.f_func(x=X_ws[:, k], u=U_ws[:,k])['f']
            U_rob_fakews[:,k] = next_u_ans['next_Vars_rob_fake']
            U_prev_rob_fakews[:,k] = next_u_ans['v_prev_rob_fake']

        # in case the orca_kkt horizon is less than the mpc horizon, we need to repeat the last action
        if mpc_env.orca_kkt_horiz < mpc_env.horiz:
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
            ws_fn = cs.Function('warmstart_horiz', [X_var], [X_ws, U_ws, U_rob_fakews, U_prev_rob_fakews], ['X_0'], ['X_vec', 'U_vec', 'U_rob_fakews', 'U_prev_rob_fakews'])
            ws_onestep_fn = cs.Function('warmstart_one', [X_var], [next_X, next_U, V_robws_init, sol_ans, fin_dual], ['X_0'], ['X_1', 'U_0', 'v_prev_rob_fake', 'next_Vars_rob_fake', 'dual_rob_fake'])

        # create a function that can fix the human models, given a series of robot actions. We use this in case the goal position of the human changes from one step to the next.
        X_corrected = cs.MX.zeros((mpc_env.nx, mpc_env.orca_kkt_horiz))
        U_rob = cs.MX.sym('U_rob_dummy', (mpc_env.nu_r, mpc_env.orca_kkt_horiz-1))
        U_corrected = cs.MX.zeros((mpc_env.nu, mpc_env.orca_kkt_horiz-1))
        X_corrected[:, 0] = X_var
        for k in range(mpc_env.orca_kkt_horiz-1):
            next_u_hums = next_U_hums_fn(X=X_corrected[:, k])['next_U_hums']
            next_u_rob = U_rob[:, k]
            U_corrected[:,k] = cs.vertcat(next_u_rob, next_u_hums)
            X_corrected[:,k+1] = mpc_env.system_model.f_func(x=X_corrected[:, k], u=U_corrected[:, k])['f']
        ws_correction_fn = cs.Function('warmstart_correction', [X_var, U_rob], [X_corrected, U_corrected], ['X_0', 'U_rob_vec'], ['X_corrected', 'U_corrected'])

        debug_ws = {
            "orca_con_list" : orca_con_list,
            "v_max_con" : v_max_con,
            "ksi_con" : ksi_con,
            "robws_solver_func" : robws_solver_func,
        }
        return ws_fn, ws_onestep_fn, ws_correction_fn, debug_ws

