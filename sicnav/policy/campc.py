from copy import deepcopy
import os
import logging
import time
import numpy as np
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionRot
import pandas as pd
import casadi as cs
from crowd_sim_plus.envs.utils.human_plus import Human
from crowd_sim_plus.envs.utils.state_plus import FullState
from crowd_sim_plus.envs.utils.state_plus import FullyObservableJointState
from sicnav.utils.mpc_utils.constraints import ConstrainedVariableType
from sicnav.utils.mpc_utils.mpc_env import MPCEnv


from matplotlib import pyplot as plt
import pickle


DO_DEBUG = False
os.environ['MKL_VERBOSE'] = '2'
DO_DEBUG_LITE = False
DO_VIDS = False
DISP_TIME = False
return_stat_dict = {'Solved_To_Acceptable_Level' : 1, 'Solve_Succeeded' : 2, 'Infeasible_Problem_Detected' : -1, 'Maximum_Iterations_Exceeded' : -2}

class CollisionAvoidMPC(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'unicycle'
        self.multiagent_training = True
        self.config = None

        # environment attributes
        self.num_hums = None
        self.hum_radii = None

        # mpc solver attributes
        self.horiz = 12
        self.soft_constraints = True
        self.ineq_dyn_consts = False
        self.new_ref_each_step = False
        self.warmstart = True
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

        self.do_callback_to_avoid_optifail = True

    # Overwrite set_env to also configure the policy's env-related values
    def set_env(self, env):
        super().set_env(env)
        env.set_human_observability(self.priviledged_info)
        tot_time = env.time_limit
        self.time_step = env.time_step
        env_config_for_dummy_human = deepcopy(env.config)
        env_config_for_dummy_human.set('humans', 'policy', 'orca_plus')
        self.dummy_human = Human(env_config_for_dummy_human, section='humans', fully_observable=self.priviledged_info)


    def configure(self, config):
        self.config = config
        self.horiz = config.getint('campc', 'horiz')
        self.soft_constraints = config.getboolean('campc', 'soft_constraints')
        self.ref_type = config.get('campc', 'ref_type')
        self.use_term_const = config.getboolean('campc', 'term_const')
        self.new_ref_each_step = config.getboolean('campc', 'new_ref_each_step')
        self.ref_type = 'new_path_eachstep'if self.new_ref_each_step else self.ref_type
        self.warmstart = config.getboolean('campc', 'warmstart')
        hum_model = config.get('mpc_env', 'hum_model')
        if hum_model == 'orca_casadi_kkt' and not self.warmstart:
            logging.warn('[CAMPC] WARNING! without using the feasible warmstart strategy, SICNav will likely fail!')
        self.priviledged_info = config.getboolean('mpc_env', 'priviledged_info')
        self.human_max_speed = config.getfloat('mpc_env', 'human_v_max_assumption')
        logging.info('[CAMPC] Config {:} = {:}'.format('horiz', self.horiz))
        logging.info('[CAMPC] Config {:} = {:}'.format('soft_constraints', self.soft_constraints))
        logging.info('[CAMPC] Config {:} = {:}'.format('ineq_dyn_consts', self.ineq_dyn_consts))
        logging.info('[CAMPC] Config {:} = {:}'.format('new_ref_each_step', self.new_ref_each_step))
        logging.info('[CAMPC] Config {:} = {:}'.format('warmstart', self.warmstart))
        logging.info('[CAMPC] Config {:} = {:}'.format('priviledged_info', self.priviledged_info))
        logging.info('[CAMPC] Config {:} = {:}'.format('human_v_max_assumption', self.human_max_speed))



    def init_mpc(self, state, ipopt_print_level=0):
        if self.num_hums is None:
            self.num_hums = len(state.human_states)

        # generating the MPCEnv object
        self.mpc_env = MPCEnv(self.time_step, state, self.num_hums, self.horiz, self.dummy_human, self.config)
        self.mpc_env.casadi_orca.reset_humans(state)
        self.Q = self.mpc_env.Q
        self.term_Q = self.mpc_env.term_Q
        self.R = self.mpc_env.R
        self.callback_orca = self.mpc_env.callback_orca
        self.callback_orca.reset_humans(state)
        self.dynamics_func = self.mpc_env.system_model.f_func
        self.cost_func = self.mpc_env.system_model.cost_func

        # self.system_model = self.mpc_env.system_model
        if self.warmstart and self.mpc_env.hum_model == 'orca_casadi_kkt':
            logging.info('[CAMPC] initialize solver for warmstarting')
            self.init_warmstart_solver()

        # the number of steps allowed to use previous solution controller, if not possible then conduct emergency braking
        if self.mpc_env.orca_kkt_horiz > 0 and self.mpc_env.hum_model == 'orca_casadi_kkt':
            self.reuse_K = np.inf if self.warmstart else self.mpc_env.orca_kkt_horiz
        else:
            self.reuse_K = self.horiz

        logging.info('[CAMPC] done basic setup of mpc, now initialize solver and tracked values')
        self.init_solver(ipopt_print_level=ipopt_print_level)
        self.reset_scenario_values()
        logging.info('[CAMPC] done initialize solver, now first ref traj generation')
        self.gen_ref_traj(state)
        logging.info('[CAMPC] done first ref traj generation, now running once to potentially compile')
        goal_states, goal_actions = self.get_ref_traj(state)
        mpc_state = self.mpc_env.convert_to_mpc_state_vector(state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)
        start_time = time.time()
        _ = self.select_action(mpc_state, state, goal_states, goal_actions, dummy=True)
        end_time = time.time()
        logging.info('[CAMPC] done running optimization once, it took {:.3f} seconds.'.format(end_time-start_time))
        logging.info('[CAMPC] done initiaizing mpc')

    def init_warmstart_solver(self):
        self.warmstart_horiz, self.warmstart_onestep, self.warmstart_correction, self.warmstart_debug = self.mpc_env.casadi_orca.get_rob_warmstart_fn(self.mpc_env)

    def reset_f_orca(self, state, new_h_gxs, new_h_gys):
        if self.num_hums != len(state.human_states):
            self.init_mpc(state)
            return
        self.mpc_env.callback_orca.reset_humans(state, new_h_gxs, new_h_gys)
        self.mpc_env.casadi_orca.reset_humans(state, new_h_gxs, new_h_gys)
        logging.info('rest f_orca in dynamics model')


    def get_const_names(self, name_stub,  k, start_idx, end_idx, mapped_row_names=None):
        indices = list(range(start_idx, end_idx))
        if k>-1:
            names = [{'name': name_stub + '_k{:}_{:}'.format(k, idx), 'name_stub': name_stub, 'k' : k, 'idx' : idx}  for idx in range(len(indices))]
        else:
            if mapped_row_names:
                assert len(indices) == len(mapped_row_names)
                names = [{'name': name_stub + '_{:}'.format(mapped_row_names[idx]), 'name_stub': name_stub, 'k' : k, 'idx' : idx}  for idx in range(len(indices))]
            else:
                names = [{'name': name_stub + '_{:}'.format(idx), 'name_stub': name_stub, 'k' : k, 'idx' : idx}  for idx in range(len(indices))]
        return indices, names


    def subject_to_const(self, opti, eqn, k, name_stub, debug_obj=None, mapped_row_names=None):
        start_idx = opti.ng
        opti.subject_to(eqn)
        end_idx = opti.ng
        const_idcs, const_names = self.get_const_names(name_stub,  k=k, start_idx=start_idx, end_idx=end_idx, mapped_row_names=mapped_row_names)
        self.all_const_idcs += const_idcs
        self.all_const_names += const_names
        self.all_const_objs += [debug_obj for _ in const_idcs]
        logging.debug('[CAMPC] Done adding const {:} for step k = {:}. New rows added {:}'.format(name_stub, k, end_idx-start_idx))


    def get_slackvar_initializer(self, subject_eqn, slackvar):
        return {'critical_val_eqn' : self.slack_scaling**(-1)*cs.fmax(0.0, cs.mmax(subject_eqn)), 'slackvar' : slackvar}

    def get_slackvar_initializer_eq(self, subject_eqn, slackvar):
        return {'critical_val_eqn' : subject_eqn, 'slackvar' : slackvar}

    def init_solver(self, ipopt_print_level=6, do_spec=False):
        K = self.horiz
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            K_orca = self.mpc_env.orca_kkt_horiz
        else:
            K_orca = None

        self.opti_dict, self.debug_cs_functions = self.get_opti_dict(K, K_orca, ipopt_print_level, do_spec)


    def get_opti_dict(self, K, K_orca=None, ipopt_print_level=0, do_spec=False):
        """Sets up nonlinear optimization problem.
        Adapted from https://github.com/utiasDSL/safe-control-gym/
        (safe-control-gym/safe_control_gym/controllers/mpc/mpc.py)
        """
        nx, nu = self.mpc_env.nx, self.mpc_env.nu
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, K + 1)
        # Inputs.
        u_var = opti.variable(self.mpc_env.nu_r, K)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        if K_orca is not None:
            u_var_hums = opti.variable(self.mpc_env.nVars_hums+self.mpc_env.nLambda, K_orca)

        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, K + 1)

        # Reference (equilibrium point or trajectory, last step for terminal cost).
        u_ref = opti.parameter(nu, K)

        x_eval = None
        u_eval = None

        # Add slack variables if using a penalty method for soft constraints.
        if self.soft_constraints:
            self.slack_scaling = 1e-3
            slack_penal_coeff = 1e9
            state_slack_pureSYM = [opti.variable(1,1) for _ in range(len(self.mpc_env.state_constraints_sym))]
            state_slack = [self.slack_scaling*state_slack_pureSYM[idx] for idx in range(len(self.mpc_env.state_constraints_sym))]

            stat_coll_slack_pureSYM = opti.variable(1,1)
            stat_coll_slack = self.slack_scaling*stat_coll_slack_pureSYM

            input_slack_pureSYM = [opti.variable(1,1) for _ in range(len(self.mpc_env.input_constraints_sym))]
            input_slack = [self.slack_scaling*input_slack_pureSYM[idx] for idx in range(len(self.mpc_env.input_constraints_sym))]

            input_state_slack_pureSYM = [opti.variable(1,1) for _ in range(len(self.mpc_env.input_state_constraints_sym))]
            input_state_slack = [self.slack_scaling*input_state_slack_pureSYM[idx] for idx in range(len(self.mpc_env.input_state_constraints_sym))]


            if self.mpc_env.hum_model == 'orca_casadi_kkt' and self.mpc_env.orca_kkt_horiz >= 0:
                hum_orca_kkt_ineq_slack_pureSYM = opti.variable(len(self.mpc_env.hum_orca_kkt_ineq_consts))
                hum_orca_kkt_ineq_slack = self.slack_scaling*hum_orca_kkt_ineq_slack_pureSYM
                hum_orca_kkt_eq_slack_pureSYM = opti.variable(len(self.mpc_env.hum_orca_kkt_eq_consts))
                hum_orca_kkt_eq_slack = self.slack_scaling*hum_orca_kkt_eq_slack_pureSYM

            if self.use_term_const:
                logging.warn('[CAMPC] Using terminal constraint. This has not been tested.')
                term_state_slack_pureSYM = opti.variable(1)
                term_state_slack = self.slack_scaling*term_state_slack_pureSYM

        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            u_var_vec = cs.horzcat(cs.vertcat(u_var[:, :K_orca], u_var_hums), cs.vertcat(u_var[:, K_orca:], cs.repmat(u_var_hums[:, K_orca-1], 1, K-K_orca)))
        else:
            u_var_vec = u_var

        # cost (cumulative)
        cost = 0
        cost_func = self.cost_func

        cost_func_map = cost_func.map(K+1, "thread", os.cpu_count())
        cost_cumsum = cs.cumsum(cost_func_map(x=x_var,
                                   u=cs.horzcat(u_var_vec, cs.sparsify(np.zeros((nu, 1)))),
                                   Xr=x_ref,
                                   Ur=cs.sparsify(np.zeros((nu, K+1))),
                                   Q=cs.sparsify(cs.horzcat(cs.repmat(self.Q, 1, K), self.term_Q)),
                                   R=cs.sparsify(cs.repmat(self.R, 1, K+1)))["l"])
        cost_mapac = cost_cumsum[-1]
        cost_term = cost_cumsum[-1] - cost_cumsum[-2]

        cost += cost_mapac

        # Constraints
        self.all_const_idcs = []
        self.all_const_names = []
        self.all_const_objs = []

        self.all_used_slack_vars = []
        self.other_slackvars = []

        for sc_i, state_constraint in enumerate(self.mpc_env.state_constraints):
            logging.info('[CAMPC] Adding map of state constraint for entire horizon: ' + state_constraint.name)
            cs_fn_map, mapped_row_names = state_constraint.get_cs_fn_map(K+1, "thread", os.cpu_count())
            if self.soft_constraints:
                self.subject_to_const(opti, cs.vec(cs_fn_map(x_var)) <= state_slack[sc_i], k=-1, name_stub=state_constraint.name+'_soft', debug_obj=state_constraint, mapped_row_names=mapped_row_names)
                cost += (K+1) * slack_penal_coeff * state_slack[sc_i] **2 + ((K+1)*100.0*slack_penal_coeff) * 1 * state_slack[sc_i] ** 2
                self.subject_to_const(opti, state_slack[sc_i] >= 0, -1, 'slackvar_'+self.mpc_env.state_constraints[sc_i].name)
                self.all_used_slack_vars.append(self.get_slackvar_initializer(cs.vec(cs_fn_map(x_var)), state_slack_pureSYM[sc_i]))
            else:
                self.subject_to_const(opti, cs.vec(cs_fn_map(x_var)) <= cs.zeros(1,K+1), -1, state_constraint.name, debug_obj=state_constraint, mapped_row_names=mapped_row_names)

        stat_coll_eqns_list = []
        for sc_i, stat_coll_con in enumerate(self.mpc_env.stat_coll_consts):
            logging.info('[CAMPC] Adding map of static collision constraint for entire horizon: ' + stat_coll_con.name)
            cs_fn_map, mapped_row_names = stat_coll_con.get_cs_fn_map(K+1, "thread", os.cpu_count())
            if self.soft_constraints:
                self.subject_to_const(opti, cs.vec(cs_fn_map(x_var)) <= stat_coll_slack, k=-1, name_stub=stat_coll_con.name+'_soft', debug_obj=stat_coll_con, mapped_row_names=mapped_row_names)
                stat_coll_eqns_list.append(cs_fn_map(x_var))
            else:
                self.subject_to_const(opti, cs.vec(cs_fn_map(x_var)) <= 0, -1, stat_coll_con.name, debug_obj=stat_coll_con, mapped_row_names=mapped_row_names)

        if self.soft_constraints:
            cost += (sc_i+1)*(K+1) * slack_penal_coeff * stat_coll_slack **2 + (sc_i+1) * (K+1) * 100.0 * slack_penal_coeff * stat_coll_slack ** 2
            self.subject_to_const(opti, stat_coll_slack >= 0, -1, 'slackvar_all_stat_obs_consts')
            self.all_used_slack_vars.append(self.get_slackvar_initializer(cs.vertcat(*tuple(stat_coll_eqns_list)), stat_coll_slack_pureSYM))

        for ic_i, input_constraint in enumerate(self.mpc_env.input_constraints):
            if self.soft_constraints:
                logging.info('[CAMPC] Adding map of input constraint for entire horizon: ' + input_constraint.name)
                cs_fn_map, mapped_row_names = input_constraint.get_cs_fn_map(K, "thread", os.cpu_count())
                self.subject_to_const(opti,  cs.vec(cs_fn_map(u_var_vec)) <= input_slack[ic_i], -1, input_constraint.name+'_soft', debug_obj=input_constraint, mapped_row_names=mapped_row_names)
                cost += K * slack_penal_coeff * input_slack[ic_i] ** 2 + K * 100.0 * slack_penal_coeff * input_slack[ic_i] ** 2
                self.subject_to_const(opti, input_slack[ic_i] >= 0, -1, 'slackvar_'+input_constraint.name)
                self.all_used_slack_vars.append(self.get_slackvar_initializer(cs.vec(cs_fn_map(u_var_vec)), input_slack_pureSYM[ic_i]))
            else:
                self.subject_to_const(opti,  cs.vec(cs_fn_map(u_var_vec)) <= 0, -1, input_constraint.name, debug_obj=input_constraint, mapped_row_names=mapped_row_names)

        for sc_i, input_state_constraint in enumerate(self.mpc_env.input_state_constraints):
            logging.info('[CAMPC] Adding map of input+state constraint for entire horizon: ' + input_state_constraint.name)
            if self.soft_constraints:
                cs_fn_map, mapped_row_names = input_state_constraint.get_cs_fn_map(K, "thread", os.cpu_count())
                self.subject_to_const(opti, cs.vec(cs_fn_map(cs.vertcat(x_var[:, :-1], u_var_vec))) <= input_state_slack[sc_i], -1, input_state_constraint.name+'_soft', debug_obj=input_state_constraint, mapped_row_names=mapped_row_names)
                cost += K * slack_penal_coeff * input_state_slack[sc_i] **2 #+ K * 100.0 * slack_penal_coeff * input_state_slack[sc_i] ** 2
                self.subject_to_const(opti, input_state_slack[sc_i] >= 0, -1, 'slackvar_'+self.mpc_env.input_state_constraints[sc_i].name)
                self.all_used_slack_vars.append(self.get_slackvar_initializer(cs.vec(cs_fn_map(cs.vertcat(x_var[:, :-1], u_var_vec))), input_state_slack_pureSYM[sc_i]))
            else:
                self.subject_to_const(opti, cs.vec(cs_fn_map(cs.vertcat(x_var[:, :-1], u_var_vec))) <= 0, -1, input_state_constraint.name, debug_obj=input_state_constraint)

        # if we use ORCA constraints for subset of horizon only, i.e. ORCA KKT horizon < MPC horizon.
        orca_kkt_penal_cost = 0
        if self.mpc_env.hum_model == 'orca_casadi_kkt' and self.mpc_env.orca_kkt_horiz > 0:
            # hum_orca_kkt_consts, input and state, hum_orca_kkt_slack
            for sc_i, con in enumerate(self.mpc_env.hum_orca_kkt_ineq_consts):
                logging.info('[CAMPC] Adding map of hum_orca_kkt_ineq_consts constraint for entire horizon: ' + con.name)
                cs_fn_map, mapped_row_names = con.get_cs_fn_map(K_orca, "thread", os.cpu_count())
                if self.soft_constraints:
                    self.subject_to_const(opti, cs.vec(cs_fn_map(cs.vertcat(x_var[:, :K_orca], u_var_vec[:, :K_orca]))) <= hum_orca_kkt_ineq_slack[sc_i], -1, con.name+'_soft', debug_obj=con, mapped_row_names=mapped_row_names)
                    orca_kkt_penal_cost += K_orca * slack_penal_coeff * hum_orca_kkt_ineq_slack[sc_i] ** 2 + K_orca * 100.0 * slack_penal_coeff * 1 * hum_orca_kkt_ineq_slack[sc_i] ** 2
                    self.subject_to_const(opti, hum_orca_kkt_ineq_slack[sc_i] >= 0, -1, 'slackvar_'+con.name)
                    self.other_slackvars.append(hum_orca_kkt_ineq_slack_pureSYM[sc_i])
                else:
                    self.subject_to_const(opti, cs.vec(cs_fn_map(cs.vertcat(x_var[:, :K_orca], u_var_vec[:, :K_orca]))) <= 0, -1, con.name, debug_obj=con, mapped_row_names=mapped_row_names)

            # hum_orca_kkt_consts, input and state, hum_orca_kkt_slack
            for sc_i, con in enumerate(self.mpc_env.hum_orca_kkt_eq_consts):
                logging.info('[CAMPC] Adding map of hum_orca_kkt_eq_consts constraint for entire horizon: ' + con.name)
                cs_fn_map, mapped_row_names = con.get_cs_fn_map(K_orca, "thread", os.cpu_count())
                if self.soft_constraints:
                    self.subject_to_const(opti, cs.vec(cs_fn_map(cs.vertcat(x_var[:, :K_orca], u_var_vec[:, :K_orca]))) == hum_orca_kkt_eq_slack[sc_i], -1, con.name+'_soft', debug_obj=con, mapped_row_names=mapped_row_names)
                    orca_kkt_penal_cost += K_orca * 1e-1 * slack_penal_coeff * hum_orca_kkt_eq_slack[sc_i]**2
                    self.other_slackvars.append(hum_orca_kkt_eq_slack_pureSYM[sc_i])
                else:
                    self.subject_to_const(opti, cs.vec(cs_fn_map(cs.vertcat(x_var[:, :K_orca], u_var_vec[:, :K_orca]))) == 0, -1, con.name, debug_obj=con, mapped_row_names=mapped_row_names)

            # state constraints for numerical stability:
            for sc_i, unspec_constraint in enumerate(self.mpc_env.hum_numstab_consts):

                if self.mpc_env.hum_numstab_consts[sc_i].constrained_variable == ConstrainedVariableType.INPUT:
                    const_var = u_var_vec[:, :K_orca]
                    reps = K_orca
                elif self.mpc_env.hum_numstab_consts[sc_i].constrained_variable == ConstrainedVariableType.STATE:
                    const_var = x_var[:, :K_orca]
                    reps = K_orca
                elif self.mpc_env.hum_numstab_consts[sc_i].constrained_variable == ConstrainedVariableType.INPUT_AND_STATE:
                    const_var = cs.vertcat(x_var[:, :K_orca], u_var_vec[:, :K_orca])
                    reps = K_orca
                cs_fn_map = unspec_constraint.get_cs_fn().map(reps, "thread", os.cpu_count())
                self.subject_to_const(opti, cs.vec(cs_fn_map(const_var)) <= 0, -1, unspec_constraint.name, debug_obj=unspec_constraint)


        for i in range(K):
            u_ts = u_var_vec[:, i]

            # Dynamics constraints.
            next_state = self.dynamics_func(x=x_var[:, i], u=u_ts)['f']

            self.subject_to_const(opti, x_var[:, i + 1] == next_state, k=i, name_stub='dynamics') # always have hard dynamics constraints

        # initial condition constraints
        self.subject_to_const(opti, x_var[:, 0] - x_init[:, 0] == 0, -1, 'init_cond_const')

        # terminal constraint
        if self.use_term_const:
            cost_imp_req = 0.99
            p_g = x_ref[:2, -1]
            p_rob_end  = x_var[:2, -1]
            p_rob_init = x_init[:2, 0]
            term_con_eq = (p_g - p_rob_end).T @ (p_g - p_rob_end) - cost_imp_req * (p_g - p_rob_init).T @ (p_g - p_rob_init)
            if self.soft_constraints:
                self.subject_to_const(opti, term_con_eq <= term_state_slack, -1, 'term_const')
                cost += slack_penal_coeff * term_state_slack ** 2
                self.subject_to_const(opti, term_state_slack >= 0, -1, 'slackvar_term_const')
                self.all_used_slack_vars.append(self.get_slackvar_initializer(term_con_eq, term_state_slack_pureSYM))
            else:
                self.subject_to_const(opti, term_con_eq <= 0, -1, 'term_const')

        opti.minimize(cost+orca_kkt_penal_cost)
        logging.info('[CAMPC] Create initializers for slack variables, to be used for calculating cost of initialization')
        combined_slackvars_list = []
        critical_vals_list = []
        for slack_var_dict in self.all_used_slack_vars:
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                slack_var_dict['min_val_fn'] = cs.Function('init_fn', [x_var, u_var, u_var_hums, x_ref, u_ref], [slack_var_dict['critical_val_eqn']], ['x_var', 'u_var', 'u_var_hums', 'x_ref', 'u_ref'], ['min_val']).expand()
            else:
                slack_var_dict['min_val_fn'] = cs.Function('init_fn', [x_var, u_var, x_ref, u_ref], [slack_var_dict['critical_val_eqn']], ['x_var', 'u_var', 'x_ref', 'u_ref'], ['min_val']).expand()
            combined_slackvars_list.append(slack_var_dict['slackvar'])
            critical_vals_list.append(slack_var_dict['critical_val_eqn'])


        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            penalized_cost_fn_temp = cs.Function('penal_cost_fn', [x_var, u_var, u_var_hums, x_ref, u_ref]+combined_slackvars_list, [cost])
            penalized_cost_val = penalized_cost_fn_temp(x_var, u_var, u_var_hums, x_ref, u_ref, *critical_vals_list)
            penalized_cost_fn = cs.Function('penal_cost_fn', [x_var, u_var, u_var_hums, x_ref, u_ref], [penalized_cost_val], ['x_var', 'u_var', 'u_var_hums', 'x_ref', 'u_ref'], ['cost']).expand()
        else:
            penalized_cost_fn_temp = cs.Function('penal_cost_fn', [x_var, u_var, x_ref, u_ref]+combined_slackvars_list, [cost])
            penalized_cost_val = penalized_cost_fn_temp(x_var, u_var, x_ref, u_ref, *critical_vals_list)
            penalized_cost_fn = cs.Function('penal_cost_fn', [x_var, u_var, x_ref, u_ref], [penalized_cost_val], ['x_var', 'u_var', 'x_ref', 'u_ref'], ['cost']).expand()

        # Sometimes Opti will throw this warning and fail: WARNING("solver:nlp_hess_l failed: NaN detected for output hess_gamma_x_x,...
        # It may be necessary to calculate the hessian of the Lagrangian manually, instead of letting Casadi do it as per https://github.com/casadi/casadi/blob/402fe583f0d3cf1fc77d1e1ac933f75d86083124/docs/examples/python/sysid.py#L95

        # logging.info('[CAMPC] Calculating Hessian of Lagrangian manually to avoid bug in CasADi')
        # lag = opti.f+cs.dot(opti.lam_g,opti.g)
        # temp_lam_f = cs.MX.sym("temp_lam_f")
        hessLag=None

        # logging.debug('[CAMPC] Defined Lagrangian, now calculating Hessian using CasADi')
        # H_lag = cs.triu(cs.hessian(lag, opti.x)[0], True)
        # logging.debug('[CAMPC] Calculated Hessian of Lagrangian as a CasADi symbolic expression, now creating CasADi function')
        # hessLag = cs.Function('nlp_hess_l',{'x':opti.x, 'p': opti.p, 'lam_f':temp_lam_f, 'lam_g': opti.lam_g, 'hess_gamma_x_x':temp_lam_f*H_lag},
        #              ['x','p','lam_f','lam_g'], ['hess_gamma_x_x']).expand()

        logging.debug('[CAMPC] Done creating CasADi function, now setting solver options')


        # Check to see if HSL is available
        logging.info('[CAMPC] Checking if MA57 linear solver is available for IPOPT by solving a dummy problem.')
        dummy_var = cs.MX.sym('dummy_var', 1)
        dummy_cost = dummy_var**2
        dummy_const = -dummy_var+1
        dummy_prob = {'x': dummy_var, 'f': dummy_cost, 'g': dummy_const}
        dummy_nlpsol = cs.nlpsol('dummy', 'ipopt', dummy_prob, {'ipopt.linear_solver':'ma57', 'ipopt.print_level':0, 'print_time':0})
        ans = dummy_nlpsol(x0=100)
        if dummy_nlpsol.stats()['return_status'] == 'Solve_Succeeded':
            logging.info('[CAMPC] MA57 linear solver available, using it')
            hsl_available = True
        else:
            logging.warn('[CAMPC]\t=== [CAMPC] WARNING =====================================================================================\n\n\n \
                                \t HSL solvers are not loading properly. MA57 linear solver not available for IPOPT, using default instead.\n \
                                \t This is untested and may be slower and less stable. Please install HSL libraries.\n\n\n \
                                \t=== [CAMPC] WARNING =====================================================================================')
            hsl_available = False

        if self.mpc_env.hum_model == 'cvmm':
            opts   = {"expand" : 1, "ipopt.print_level": ipopt_print_level, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter":500, "error_on_fail":False,
                    #   "ipopt.linear_solver":"ma57", "ipopt.ma57_automatic_scaling" : "yes", "ipopt.ma57_pre_alloc": 2
                      } #
        else:
            opts   = {"expand" : 1, "ipopt.print_level": ipopt_print_level, "ipopt.sb": "yes", "print_time": 0, "ipopt.max_iter":250, "error_on_fail":False,
                    #   "ipopt.linear_solver":"ma57", "ipopt.ma57_automatic_scaling" : "yes", "ipopt.ma57_pre_alloc": 2,
                "ipopt.acceptable_constr_viol_tol" : 1e-3,
                "ipopt.acceptable_compl_inf_tol" : 1e-3,
                "ipopt.acceptable_obj_change_tol" : 1e-3,
                } #

        if hsl_available:
            opts["ipopt.linear_solver"] = "ma57"
            opts["ipopt.ma57_automatic_scaling"] = "yes"
            opts["ipopt.ma57_pre_alloc"] = 2

        # making my hessLag calculation the hesslag for the ipopt solver as per https://github.com/casadi/casadi/blob/402fe583f0d3cf1fc77d1e1ac933f75d86083124/docs/examples/python/sysid.py#L95
        logging.debug('[CAMPC] Done setting solver options, now setting my Hessian of Lagrangian for the solver (ipopt)')
        # opts["hess_lag"] = hessLag
        opti.solver('ipopt', opts)

        nopenal_cost_fn = cs.Function('nopenal_cost_fn', [opti.x, opti.p], [cost_mapac], ['opti_x', 'opti_p'], ['cost']).expand()

        # make a function that finds the minimum values of the slack variables, then calculates the cost


        opti_dict = {
            "opti": opti,
            # "hess_lag": hessLag,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "x_ref": x_ref,
            "u_ref": u_ref,
            "x_eval": x_eval,
            "u_eval": u_eval,
            "cost": cost+orca_kkt_penal_cost,
            "cost_norcakkt" : cost,
            "cost_mapac" : cost_mapac,
            "cost_term" : cost_term,
            "nopenal_cost_fn" : nopenal_cost_fn,
            "penalized_cost_fn" : penalized_cost_fn,
            "slackvar_dicts" : self.all_used_slack_vars,
            "other_slackvars" : self.other_slackvars,
            "opts" : opts
        }
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            opti_dict["u_var_hums"] = u_var_hums


        # in order to be able to get the optimization variables in a format that I am familiar with easily during debugging.
        debug_cs_functions = {}
        debug_cs_functions['x_var_fn'] = cs.Function('get_x_var', [opti.x], [x_var], ['opti_x'], ['x_var'])
        debug_cs_functions['u_var_fn'] = cs.Function('get_u_var', [opti.x], [u_var], ['opti_x'], ['u_var'])
        debug_cs_functions['u_var_vec_fn'] = cs.Function('get_u_var_vec', [opti.x], [u_var_vec], ['opti_x'], ['u_var_vec'])
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            debug_cs_functions['u_var_hums_fn'] = cs.Function('get_u_var_hums', [opti.x], [u_var_hums], ['opti_x'], ['u_var_hums'])
            debug_cs_functions['hum_orca_kkt_ineq_slack_fn'] = cs.Function('get_hum_orca_kkt_ineq_slack', [opti.x], [hum_orca_kkt_ineq_slack], ['opti_x'], ['hum_orca_kkt_ineq_slack'])
        logging.info('[CAMPC] Done building MPC solver using CasADi Opti Stack')



        return opti_dict, debug_cs_functions


    def get_debug_orca_const_all(self, debug_x_var, debug_u_var, debug_u_var_hums):
        debug_orca_const_dict = {}
        debug_orca_const_names = []
        debug_orca_const_list = []
        # for idx, const in enumerate(self.mpc_env.hum_orca_kkt_consts+self.mpc_env.hums_orca_consts):
        for idx, const in enumerate(self.mpc_env.hum_orca_kkt_consts):
            cs_fun = const.get_cs_fn()
            name = const.name
            debug_orca_const_dict[name] = list()
            for k in range(self.horiz):
                x = debug_x_var[:,k].reshape(debug_x_var.shape[0], 1)
                u = np.vstack([np.atleast_2d(debug_u_var[:,k]).T, np.atleast_2d(debug_u_var_hums[:,k]).T])
                xu = np.vstack([x,u])
                # const.
                val = cs_fun(xu).toarray()
                debug_orca_const_dict[name].append(val)
                debug_orca_const_names.extend([name+'_k_{:}_row_{:}'.format(k, const.row_names[i]) for i in range(len(val))])
                debug_orca_const_list.append(val)

        debug_orca_const_all = np.vstack(debug_orca_const_list)
        return debug_orca_const_names, debug_orca_const_all


    def get_debug_const_all(self, nlp_x_debug, nlp_p_debug, opti_g):
        val = opti_g(x=nlp_x_debug, p=nlp_p_debug)['g'].toarray()
        return [const_name['name'] for const_name in self.all_const_names], val


    def bring_fwd(self, joint_state, obs, x_prev, u_prev, for_guess=True):
        u_prev_fwded = deepcopy(u_prev)
        u_prev_fwded[:, :-1] = u_prev_fwded[:, 1:]
        x_prev_fwded = deepcopy(x_prev)
        x_prev_fwded[:, :-1] = x_prev_fwded[:, 1:]
        if self.mpc_env.hum_model == 'orca_casadi_kkt' and self.warmstart:
            slice_indices_posns = np.array([[self.mpc_env.nx_r+self.mpc_env.np_g+idx1, self.mpc_env.nx_r+self.mpc_env.np_g+idx1+1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)
            if np.any(np.abs(x_prev_fwded[:3, 0] - obs[:3, 0]) > 1e-2) or np.any(np.abs(np.take(x_prev_fwded[:, 0], slice_indices_posns, axis=0) - np.take(obs[:, 0], slice_indices_posns, axis=0)) > self.mpc_env.rob_rad_buffer):
                logging.info('[CAMPC] [BRING FWD] discrepancy between obs and first timestep of previous solution. Will correct.')
                ans = self.warmstart_horiz(X_0=obs)
                x_prev_fwded = ans['X_vec'].toarray()
                u_prev_fwded = ans['U_vec'].toarray()


            # run the feasible warmstart for one step
            ans = self.warmstart_onestep(X_0=np.atleast_2d(x_prev_fwded[:,-2]).T)
            x_next = ans['X_1'].toarray()
            u_next = ans['U_0'].toarray()
            u_prev_fwded[:, self.mpc_env.orca_kkt_horiz-1:] = u_next[:, :]
            x_prev_fwded[:, self.mpc_env.orca_kkt_horiz:] = x_next[:, :]
        else:
            # just simulate forward keeping the same final robot action, except with no rotation
            u_prev_fwded[1, -1] = 0.0
            x_prev_fwded, u_prev_fwded = self.generate_traj(joint_state, self.horiz, x_rob=None, u_rob=u_prev_fwded, for_guess=for_guess)

        return x_prev_fwded, u_prev_fwded


    def select_action(self, obs, joint_state, goal_states, goal_actions, return_all=False, dummy=False):
        """Solves nonlinear mpc problem to get next action.
        Args:
            obs (np.array): current state/observation.

        Returns:
            np.array: input/action to the task/env.
        """

        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        x_var = opti_dict["x_var"]
        u_var = opti_dict["u_var"]
        x_init = opti_dict["x_init"]
        x_ref = opti_dict["x_ref"]
        u_ref = opti_dict["u_ref"]

        # Assign the initial state.
        opti.set_value(x_init, obs)

        # Assign reference trajectory within horizon.
        opti.set_value(x_ref, goal_states)
        opti.set_value(u_ref, goal_actions)

        # Select x_guess, u_guess
        start_ref = time.time()
        if self.x_prev is not None and self.u_prev is not None:
            for_guess = True if self.mpc_env.hum_model != 'orca_casadi_kkt' or self.mpc_env.orca_kkt_horiz > 0 else False
            x_prev = deepcopy(self.x_prev)
            u_prev = deepcopy(self.u_prev)
            x_prev_fwded, u_prev_fwded = self.bring_fwd(joint_state, obs, x_prev, u_prev, for_guess=for_guess)

        if not self.warmstart or (self.mpc_env.hum_model != 'orca_casadi_kkt' and (self.x_prev is None or self.u_prev is None or self.mpc_sol_succ is None or (not self.mpc_sol_succ[-1] and self.num_prev_used >= self.reuse_K) or self.new_ref_each_step)):
            for_guess = True if self.mpc_env.hum_model != 'orca_casadi_kkt' or self.mpc_env.orca_kkt_horiz > 0 else False
            x_guess, u_guess = self.generate_traj(joint_state, self.horiz, for_guess=for_guess)
            can_use_guess = False # whether or not the guess can be used for the next step in case optimization fails
        elif self.mpc_env.hum_model == 'orca_casadi_kkt' and (self.x_prev is None or self.u_prev is None or self.mpc_sol_succ is None or not np.any(np.array(self.mpc_sol_succ)) or self.new_ref_each_step):
            logging.info('[CAMPC] generating warmstart for entire horizon')
            ans = self.warmstart_horiz(X_0=obs)
            x_guess = ans['X_vec'].toarray()
            u_guess = ans['U_vec'].toarray()
            can_use_guess = True # whether or not the guess can be used for the next step in case optimization fails

        else:
            x_guess = deepcopy(x_prev_fwded)
            u_guess = deepcopy(u_prev_fwded)
            can_use_guess = True # whether or not the guess can be used for the next step in case optimization fails

        x_guess[:, 0] = obs[:, 0]
        opti.set_initial(x_var, x_guess)
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            u_var_hums = opti_dict["u_var_hums"]
            K_orca = self.mpc_env.orca_kkt_horiz if self.mpc_env.orca_kkt_horiz > 0 else self.horiz
            u_var_guess = u_guess[:self.mpc_env.nu_r, :]
            u_hums_guess = u_guess[self.mpc_env.nu_r:, :K_orca]
            opti.set_initial(u_var, u_var_guess)
            opti.set_initial(u_var_hums, u_hums_guess)
        else:
            opti.set_initial(u_var, u_guess)

        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            init_obj_val = np.array([opti_dict['penalized_cost_fn'](
                        x_var=x_guess,
                        u_var=u_var_guess,
                        u_var_hums=u_hums_guess,
                        x_ref=goal_states,
                        u_ref=goal_actions
                    )['cost']]).item()
        else:
            init_obj_val = np.array([opti_dict['penalized_cost_fn'](
                        x_var=x_guess,
                        u_var=u_guess,
                        x_ref=goal_states,
                        u_ref=goal_actions
                    )['cost']]).item()
        end_ref = time.time()
        refset_time = end_ref - start_ref

        # Logging the intermediate optimization values via a callback function for the optimizer
        if DO_DEBUG:
            nlp_x_debug_vals = []
            nlp_p_debug_vals = []
            x_var_debug_vals = []
            u_var_debug_vals = []
            iter_debug_vals = []
            f_debug_vals = []
            def opti_callback(i, nlp_x_debug_vals=nlp_x_debug_vals, nlp_p_debug_vals=nlp_p_debug_vals, x_var_debug_vals=x_var_debug_vals, u_var_debug_vals=u_var_debug_vals, iter_debug_vals=iter_debug_vals, f_debug_vals=f_debug_vals):
                logging.debug('[CAMPC OPTI CALLBACK] opt idx {:}'.format(i))
                iter_debug_vals.append(i)
                nlp_x_debug_vals.append(deepcopy(opti.debug.value(opti.x)))
                nlp_p_debug_vals.append(deepcopy(opti.debug.value(opti.p)))
                f_debug_vals.append(deepcopy(opti.debug.value(opti.f)))
                x_var_debug_vals.append(deepcopy(self.debug_cs_functions['x_var_fn'](opti_x=opti.debug.value(opti.x))['x_var']))
                u_var_debug_vals.append(deepcopy(self.debug_cs_functions['u_var_vec_fn'](opti_x=opti.debug.value(opti.x))['u_var_vec']))
            opti.callback(opti_callback)
        else:
            x_var_debug_vals = []
            u_var_debug_vals = []
            iter_debug_vals = []
            f_debug_vals = []
            def opti_callback(i, x_var_debug_vals=x_var_debug_vals, u_var_debug_vals=u_var_debug_vals, iter_debug_vals=iter_debug_vals, f_debug_vals=f_debug_vals):
                logging.debug('[CAMPC OPTI CALLBACK] opt idx {:}'.format(i))
                infeas = opti.advanced.stats()['iterations']['inf_pr'][-1]
                if infeas < 1e-3:
                    iter_debug_vals.append(i)
                    f_debug_vals.append(deepcopy(opti.debug.value(opti.f)))
                    x_var_debug_vals.append(deepcopy(self.debug_cs_functions['x_var_fn'](opti_x=opti.debug.value(opti.x))['x_var']))
                    u_var_debug_vals.append(deepcopy(self.debug_cs_functions['u_var_vec_fn'](opti_x=opti.debug.value(opti.x))['u_var_vec']))
            # For some reason Casadi's opti sometimes fails when no callback is specifided. We also use this callback to get intermediate values
            # through the iterations of the optimization, store feasible solutions and use the last feasible solution if the optimization exhausts
            # the maximum number of iterations.
            if self.do_callback_to_avoid_optifail:
                opti.callback(opti_callback)


        # Solve the optimization problem.
        try:
            if not dummy and DISP_TIME:
                time_text = '. Prep time {:.3f}s'.format(refset_time)
                logging.info('[CAMPC] Start solve step {:}{:}'.format(self.traj_step, time_text))
            sol_start = time.time()
            sol = opti.solve()
            sol_end = time.time()
            final_obj_val = sol.value(opti.f) # final objective value
            opti_stats = opti.debug.stats()

            if self.warmstart and final_obj_val > init_obj_val:
                # take the initial values as the solution
                x_val = x_guess
                u_val = u_guess
                if self.mpc_env.hum_model == 'orca_casadi_kkt':
                    u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                    u_val = u_val[:self.mpc_env.nu_r, :]
                debug_text = 'Solution worse than warmstart'

                self.num_prev_used = 0
                self.mpc_sol_succ.append(True)
                time_text = ', Wall Time: {:.4f}'.format(sol_end-sol_start) if DISP_TIME else ''
                logging.info('[CAMPC] Optim. success {:} but warmstart better step {:}, Initial: {:.3g} -> Final: {:.3g}{:}'.format(str(opti.debug.return_status()), self.traj_step, init_obj_val, final_obj_val, time_text))
            else:
                x_val, u_val = sol.value(x_var).reshape(x_var.shape), sol.value(u_var).reshape(u_var.shape)
                if self.mpc_env.hum_model == 'orca_casadi_kkt':
                    u_val_hums = sol.value(u_var_hums).reshape(u_var_hums.shape)
                debug_text = opti.debug.return_status()
                self.num_prev_used = 0
                self.mpc_sol_succ.append(True)
                time_text = ', Wall Time: {:.4f}'.format(sol_end-sol_start) if DISP_TIME else ''
                logging.info('[CAMPC] Optim. success {:} step {:}. Initial: {:.3g} -> Final: {:.3g}. Num Iter: {:}{:}'.format(str(opti.debug.return_status()), self.traj_step, init_obj_val, final_obj_val, opti_stats['iter_count'], time_text))
            if u_val.ndim > 1:
                action = u_val[:, 0]
            else:
                action = np.array([u_val[0]])
            self.prev_action = action
        except Exception as e:
            sol_end = time.time()
            def deal_with_fail():
                self.mpc_sol_succ.append(False)
                time_text = ', Wall Time: {:.4f}'.format(sol_end-sol_start) if DISP_TIME else ''
                logging.info('[CAMPC] Optim. error step {:}: {:}.  Num Iter: {:}{:}'.format(self.traj_step, str(opti.debug.return_status()), opti_stats['iter_count'], time_text))
                logging.debug('[CAMPC] Error message {:}'.format(e))
                debug_x_var = opti.debug.value(x_var)

                if can_use_guess and self.warmstart and self.mpc_env.hum_model == 'orca_casadi_kkt':
                    debug_text = deepcopy(str(opti.debug.return_status()) + ' USING WARMSTART GUESS')
                    if len(u_guess.shape) > 1:
                        action = u_guess[:, 0]
                    else:
                        action = np.array([u_guess[0]])
                    x_val = x_guess
                    u_val = u_guess
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                        u_val = u_val[:self.mpc_env.nu_r, :]
                elif can_use_guess and self.num_prev_used < self.reuse_K:
                    debug_text = deepcopy(str(opti.debug.return_status()) + ' PREV. SOLN. FWDED')
                    if len(u_prev_fwded.shape) > 1:
                        action = u_prev_fwded[:, 0]
                    else:
                        action = np.array([u_prev_fwded[0]])
                    x_val = x_prev_fwded
                    u_val = u_prev_fwded
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                        u_val = u_val[:self.mpc_env.nu_r, :]
                    self.num_prev_used += 1
                else:
                    # Emergency braking controller
                    debug_text = deepcopy(str(opti.debug.value) + ' EMER. BRAKE')
                    cur_vel = x_guess[3, 0]
                    actions = []
                    rob_states = [np.atleast_2d(x_guess[:, 0]).T]
                    f_still_moving = True
                    count = 1
                    while f_still_moving:
                        if count > self.horiz:
                            break
                        lowest_feasible_vel = np.max((cur_vel + self.mpc_env.max_l_dcc*self.time_step, 0.0))

                        prev_state = rob_states[count-1]
                        next_action = np.array([lowest_feasible_vel, 0.0]+[0.0 for _ in range(self.mpc_env.nVars_hums)]+[0.0 for _ in range(self.mpc_env.nLambda)])
                        actions.append(next_action)
                        next_state = self.mpc_env.system_model.f_func(prev_state, next_action).toarray()
                        rob_states.append(next_state)
                        cur_vel = lowest_feasible_vel
                        f_still_moving = lowest_feasible_vel - 0.0 > 1e-10
                        count += 1
                    action = actions[0]

                    u_val = np.concatenate([np.atleast_2d(action).T for action in actions]+
                                        [np.tile(np.atleast_2d(next_action).T, (1, self.horiz-len(actions)))], -1)
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                        u_val = u_val[:self.mpc_env.nu_r, :]
                    x_val_noorca = np.concatenate([rob_state for rob_state in rob_states]+
                                        [np.tile(next_state, (1, self.horiz-len(actions)))], -1)
                    x_val_hums = debug_x_var[self.mpc_env.nx_r+self.mpc_env.np_g:,:]
                    x_val = np.vstack([x_val_noorca[:self.mpc_env.nx_r+self.mpc_env.np_g,:], x_val_hums])
                    self.num_prev_used = self.horiz + 1
                if self.mpc_env.hum_model == 'orca_casadi_kkt':
                    return action, x_val, u_val, u_val_hums, debug_text
                else:
                    return action, x_val, u_val, debug_text

            opti_stats = opti.debug.stats()
            # If return status is Maximum_Iterations_Exceeded, check initial and final cost values, and check feasibility of constraints
            if opti.debug.return_status() == 'Maximum_Iterations_Exceeded' and len(iter_debug_vals) > 0:
                last_iter = iter_debug_vals[-1]
                final_obj_val = f_debug_vals[-1]

                if self.warmstart and final_obj_val > init_obj_val:
                    # take the initial values as the solution
                    x_val = x_guess
                    u_val = u_guess
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                        u_val = u_val[:self.mpc_env.nu_r, :]
                    debug_text = 'Solution worse than warmstart'

                    self.num_prev_used = 0
                    self.mpc_sol_succ.append(True)
                    time_text = ', Wall Time: {:.4f}'.format(sol_end-sol_start) if DISP_TIME else ''
                    logging.info('[CAMPC] Optim. success {:} but warmstart better step {:}, Initial: {:.3g} -> Final: {:.3g}{:}'.format(str(opti.debug.return_status()), self.traj_step, init_obj_val, final_obj_val, time_text))
                else:
                    x_val = np.array(x_var_debug_vals[-1])
                    u_val = np.array(u_var_debug_vals[-1])
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        u_val_hums = u_val[self.mpc_env.nu_r:, :K_orca]
                        u_val = u_val[:self.mpc_env.nu_r, :]
                    debug_text = opti.debug.return_status()
                    self.num_prev_used = 0
                    self.mpc_sol_succ.append(True)
                    final_const_viol = opti.debug.value(opti.g)
                    time_text = ', Wall Time: {:.4f}'.format(sol_end-sol_start) if DISP_TIME else ''
                    logging.info('[CAMPC] Optim. success {:} step {:}. Initial: {:.3g} -> Final: {:.3g}. Num Iter: {:}{:}'.format(str(opti.debug.return_status()), self.traj_step, init_obj_val, final_obj_val, last_iter, time_text))
                if u_val.ndim > 1:
                    action = u_val[:, 0]
                else:
                    action = np.array([u_val[0]])
                self.prev_action = action
            else:
                if self.mpc_env.hum_model == 'orca_casadi_kkt':
                    action, x_val, u_val, u_val_hums, debug_text = deal_with_fail()
                else:
                    action, x_val, u_val, debug_text = deal_with_fail()




        sol_suc = True if self.mpc_sol_succ[-1] else False

        if not dummy:
            if opti_stats['return_status'] in return_stat_dict.keys():
                log_status = return_stat_dict[opti_stats['return_status']]
            else:
                log_status = -5

            self.solver_summary['traj_step'].append(self.traj_step)
            self.solver_summary['sol_success'].append(int(sol_suc))
            self.solver_summary['optim_status'].append(log_status)
            self.solver_summary['iter_count'].append(opti_stats['iter_count'])
            self.solver_summary['prep_time'].append(refset_time)
            self.solver_summary['sol_time'].append(sol_end - sol_start)
            self.solver_summary['final_nopenal_cost'].append(float(opti.debug.value(opti_dict['cost_mapac'])))
            self.solver_summary['final_term_cost'].append(float(opti.debug.value(opti_dict['cost_term'])))
            self.solver_summary['debug_text'].append(debug_text)
            try:
                self.solver_summary['ipopt_iterations'].append(deepcopy(opti_stats['iterations']))
            except:
                self.solver_summary['ipopt_iterations'].append([])


        text = 'succ' if self.mpc_sol_succ[-1] else 'fail'
        if not dummy and DO_DEBUG:
            debug_orca_const_og_names, _ = self.get_debug_const_all(opti.debug.value(opti.x), opti.debug.value(opti.p), opti.debug.casadi_solver.get_function('nlp_g'))
            debug_g_list = []
            debug_x_vals_list = []
            debug_p_vals_list = []
            debug_f_vals = []
            debug_nopenal_cost_vals = []
            debug_opt_idx = []
            for idx in range(len(x_var_debug_vals)):
                iter = iter_debug_vals[idx]
                debug_nlp_x = nlp_x_debug_vals[idx]
                debug_nlp_p = nlp_p_debug_vals[idx]
                debug_orca_const_names, debug_orca_const_all = self.get_debug_const_all(debug_nlp_x, debug_nlp_p, opti.debug.casadi_solver.get_function('nlp_g'))
                debug_g_list.append(pd.Series(debug_orca_const_all[:,0].tolist(), index=debug_orca_const_og_names))
                debug_x_vals_list.append(pd.Series(debug_nlp_x))
                debug_p_vals_list.append(pd.Series(debug_nlp_p))
                debug_f_vals.append(opti.debug.casadi_solver.get_function('nlp_f')(x=debug_nlp_x, p=debug_nlp_p)['f'].toarray().item())
                debug_nopenal_cost_vals.append(opti_dict['nopenal_cost_fn'](opti_x=debug_nlp_x, opti_p=debug_nlp_p)['cost'].toarray().item())
                debug_opt_idx.append(idx)
            debug_g_df = pd.concat(debug_g_list, axis=1)
            debug_x_vals_df = pd.concat(debug_x_vals_list, axis=1)
            debug_p_vals_df = pd.concat(debug_p_vals_list, axis=1)
            debug_g_df_tsp = debug_g_df.transpose()
            debug_f_df = pd.DataFrame({
                'opt_idx' : pd.Series(debug_opt_idx),
                'f' : pd.Series(debug_f_vals),
                'nopenal_cost' : pd.Series(debug_nopenal_cost_vals)
            })
            # Get the index values
            index_values = debug_g_df_tsp.index.values
            # Insert the index values as a new column at the first position
            debug_g_df_tsp.insert(0, 'optidx', list(range(len(index_values))))
            debug_ipopt_iter_df = pd.DataFrame(opti_stats['iterations'])
            debug_ipopt_iter_df.insert(0, 'optidx', list(range(len(debug_ipopt_iter_df))))
            debug_material = {
                'campc/traj_step' : self.traj_step,
                'campc/sol_success' : int(sol_suc),
                'campc/optim_status' : log_status,
                'campc/iter_count' : opti_stats['iter_count'],
                'campc/prep_time' : refset_time,
                'campc/sol_time' : sol_end - sol_start,
                'campc/final_nopenal_cost' : float(opti.debug.value(opti_dict['cost_mapac'])),
                'campc/final_term_cost' : float(opti.debug.value(opti_dict['cost_term'])),
                'campc/debug_text' : debug_text,
                'campc/debug/debug_x_df' : deepcopy(debug_x_vals_df),
                'campc/debug/debug_f_df' : deepcopy(debug_f_df),
                'campc/debug/debug_p_df' : deepcopy(debug_p_vals_df),
                'campc/debug/debug_g_df' : deepcopy(debug_g_df_tsp),
                'campc/debug/debug_ipopt_df' : deepcopy(debug_ipopt_iter_df),
                'campc/debug/t_proc_callback_fun' : deepcopy(opti_stats['t_proc_callback_fun']),
                'campc/debug/t_proc_nlp_f' : deepcopy(opti_stats['t_proc_nlp_f']),
                'campc/debug/t_proc_nlp_g' : deepcopy(opti_stats['t_proc_nlp_g']),
                'campc/debug/t_proc_nlp_grad' : deepcopy(opti_stats['t_proc_nlp_grad']),
                'campc/debug/t_proc_nlp_grad_f' : deepcopy(opti_stats['t_proc_nlp_grad_f']),
                'campc/debug/t_proc_nlp_hess_l' : deepcopy(opti_stats['t_proc_nlp_hess_l']),
                'campc/debug/t_proc_nlp_jac_g' : deepcopy(opti_stats['t_proc_nlp_jac_g']),
                'campc/debug/t_wall_callback_fun' : deepcopy(opti_stats['t_wall_callback_fun']),
                'campc/debug/t_wall_nlp_f' : deepcopy(opti_stats['t_wall_nlp_f']),
                'campc/debug/t_wall_nlp_g' : deepcopy(opti_stats['t_wall_nlp_g']),
                'campc/debug/t_wall_nlp_grad' : deepcopy(opti_stats['t_wall_nlp_grad']),
                'campc/debug/t_wall_nlp_grad_f' : deepcopy(opti_stats['t_wall_nlp_grad_f']),
                'campc/debug/t_wall_nlp_hess_l' : deepcopy(opti_stats['t_wall_nlp_hess_l']),
                'campc/debug/t_wall_nlp_jac_g' : deepcopy(opti_stats['t_wall_nlp_jac_g']),
                'campc/debug/x_goals' : pd.DataFrame(goal_states),
                'campc/debug/u_goals' : pd.DataFrame(goal_actions),
                'campc/debug/x_guess' : pd.DataFrame(x_guess),
                'campc/debug/u_guess' : pd.DataFrame(u_guess),
                'campc/debug/x_val' : pd.DataFrame(x_val),
                'campc/debug/u_val' : pd.DataFrame(u_val),
                }
            # dump debug material to a pickle file called debug_material.pkl
            with open('debug_material.pkl', 'wb') as f:
                logging.info('[CAMPC] Dumping debug material for solve step {:}'.format(self.traj_step))
                pickle.dump(debug_material, f)

            del nlp_x_debug_vals
            del nlp_p_debug_vals
            del x_var_debug_vals
            del u_var_debug_vals
            del iter_debug_vals
        elif not dummy and DO_DEBUG_LITE:
            debug_ipopt_iter_df = pd.DataFrame(opti_stats['iterations'])
            debug_ipopt_iter_df.insert(0, 'optidx', list(range(len(debug_ipopt_iter_df))))
            debug_material = {
                'campc/traj_step' : self.traj_step,
                'campc/sol_success' : int(sol_suc),
                'campc/optim_status' : log_status,
                'campc/iter_count' : opti_stats['iter_count'],
                'campc/prep_time' : refset_time,
                'campc/sol_time' : sol_end - sol_start,
                'campc/final_nopenal_cost' : float(opti.debug.value(opti_dict['cost_mapac'])),
                'campc/final_term_cost' : float(opti.debug.value(opti_dict['cost_term'])),
                'campc/debug_text' : debug_text,
                'campc/debug/debug_ipopt_df' : deepcopy(debug_ipopt_iter_df),
                'campc/debug/t_proc_callback_fun' : deepcopy(opti_stats['t_proc_callback_fun']),
                'campc/debug/t_proc_nlp_f' : deepcopy(opti_stats['t_proc_nlp_f']),
                'campc/debug/t_proc_nlp_g' : deepcopy(opti_stats['t_proc_nlp_g']),
                'campc/debug/t_proc_nlp_grad' : deepcopy(opti_stats['t_proc_nlp_grad']),
                'campc/debug/t_proc_nlp_grad_f' : deepcopy(opti_stats['t_proc_nlp_grad_f']),
                'campc/debug/t_proc_nlp_hess_l' : deepcopy(opti_stats['t_proc_nlp_hess_l']),
                'campc/debug/t_proc_nlp_jac_g' : deepcopy(opti_stats['t_proc_nlp_jac_g']),
                'campc/debug/t_wall_callback_fun' : deepcopy(opti_stats['t_wall_callback_fun']),
                'campc/debug/t_wall_nlp_f' : deepcopy(opti_stats['t_wall_nlp_f']),
                'campc/debug/t_wall_nlp_g' : deepcopy(opti_stats['t_wall_nlp_g']),
                'campc/debug/t_wall_nlp_grad' : deepcopy(opti_stats['t_wall_nlp_grad']),
                'campc/debug/t_wall_nlp_grad_f' : deepcopy(opti_stats['t_wall_nlp_grad_f']),
                'campc/debug/t_wall_nlp_hess_l' : deepcopy(opti_stats['t_wall_nlp_hess_l']),
                'campc/debug/t_wall_nlp_jac_g' : deepcopy(opti_stats['t_wall_nlp_jac_g']),
                'campc/debug/x_goals' : pd.DataFrame(goal_states),
                'campc/debug/u_goals' : pd.DataFrame(goal_actions),
                'campc/debug/x_guess' : pd.DataFrame(x_guess),
                'campc/debug/u_guess' : pd.DataFrame(u_guess),
                'campc/debug/x_val' : pd.DataFrame(x_val),
                'campc/debug/u_val' : pd.DataFrame(u_val),
                }
            # dump debug material to a pickle file called debug_material.pkl
            with open('debug_material.pkl', 'wb') as f:
                logging.info('[CAMPC] Dumping debug material for solve step {:}'.format(self.traj_step))
                pickle.dump(debug_material, f)

        self.x_prev = deepcopy(x_val)
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            if self.mpc_env.orca_kkt_horiz > 0:
                self.u_prev = deepcopy(np.vstack([u_val, np.hstack([u_val_hums, np.tile(u_val_hums[:, -1].reshape(u_val_hums.shape[0],1), (1,self.horiz-self.mpc_env.orca_kkt_horiz))])]))
            else:
                self.u_prev = deepcopy(np.vstack([u_val, u_val_hums]))
        else:
            self.u_prev = deepcopy(u_val)

        self.all_x_val.append(deepcopy(x_val))
        self.all_u_val.append(deepcopy(u_val))
        self.all_x_guess.append(deepcopy(x_guess))
        self.all_u_guess.append(deepcopy(u_guess))
        self.all_x_goals.append(deepcopy(goal_states))
        self.all_u_goals.append(deepcopy(goal_actions))
        self.all_debug_text.append(deepcopy(debug_text))

        if return_all:
            return action, x_val, u_val
        return action


    def generate_traj(self, joint_state, ref_steps, x_rob=None, u_rob=None, use_casadi_init=False, for_guess=False):
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
        self_state = joint_state.self_state
        init_dist = np.sqrt((self_state.gx - self_state.px)**2+(self_state.gy - self_state.py)**2)
        N_req = int(np.ceil(init_dist / (self.time_step * self.mpc_env.pref_speed)))

        # initialize vectors for reference states and actions of the robot
        if x_rob is None:
            ref_x = np.zeros(ref_steps+1, dtype=np.float64)
            ref_y = np.zeros(ref_steps+1, dtype=np.float64)
            ref_th = np.zeros(ref_steps+1, dtype=np.float64)
            ref_x[0] = self_state.px
            ref_y[0] = self_state.py
            ref_th[0] = self_state.theta
            # reference actions
            ref_v = np.zeros(ref_steps, dtype=np.float64)
            ref_om = np.zeros(ref_steps, dtype=np.float64)
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
                dpg_theta = np.arctan2(dpg_y, dpg_x) - ref_th[idx-1] if np.abs(dpg_y) > 1e-5 or np.abs(dpg_x) > 1e-5 else 0.0

                if u_rob is None:
                    if idx < N_req:
                        v_pref = self.mpc_env.pref_speed
                    elif idx == N_req:
                        v_pref = np.sqrt(dpg_x**2+dpg_y**2) / self.time_step
                    else:
                        v_pref = 0.0

                    ref_v[idx-1] = v_pref
                    ref_om[idx-1] = dpg_theta / self.time_step
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
        oc_next_val = self.mpc_env.convert_to_mpc_state_vector(joint_state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)
        ref_oc_next_val = np.zeros((self.mpc_env.nX_hums, ref_steps+1), dtype=np.float64)
        if use_casadi_init:
            ref_oc_next_U_val = np.zeros((self.mpc_env.nVars_hums, ref_steps), dtype=np.float64)
        if self.mpc_env.hum_model == 'orca_casadi_kkt':
            ref_oc_lambdas = np.zeros((self.mpc_env.nLambda, ref_steps), dtype=np.float64)
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
                prev_X_hums = deepcopy(np.take(ref_oc_next_val[:, col_idx-1], slice_indices_posns, axis=0))
                slice_indices_goals = np.array([[idx1+4, idx1+5] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape((2)*self.mpc_env.num_hums)
                prev_G_hums = deepcopy(np.take(ref_oc_next_val[:, col_idx-1], slice_indices_goals, axis=0))
                if use_casadi_init:
                    prev_U_hums_all = deepcopy(ref_oc_next_U_val[:, t_step-1])
                    ref_oc_next_U_val[:, t_step] = prev_U_hums_all
                else:
                    if self.mpc_env.hum_model == 'orca_casadi_kkt':
                        slice_indices = np.array([[idx1+2, idx1+3, -1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                        prev_U_hums_all = deepcopy(np.take(np.hstack([ref_oc_next_val[:, col_idx-1], 0]), slice_indices, axis=0))
                        prev_U_just_vals = prev_U_hums_all.reshape(self.mpc_env.num_hums, self.mpc_env.nvars_hum).T
                        prev_U_hums = prev_U_just_vals[:-1,:].T.reshape(self.mpc_env.num_hums*2)
                    else:
                        slice_indices = np.array([[idx1+2, idx1+3] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                        prev_U_hums = deepcopy(np.take(ref_oc_next_val[:, col_idx-1], slice_indices, axis=0))

                next_X_hums = prev_X_hums + self.mpc_env.time_step * prev_U_hums
                oc_next_hum_val = np.array([[next_X_hums[idx1*2], next_X_hums[idx1*2+1], prev_U_hums[idx1*2], prev_U_hums[idx1*2+1], prev_G_hums[idx1*2], prev_G_hums[idx1*2+1]] for idx1 in range(self.mpc_env.num_hums)]).reshape(self.mpc_env.nX_hums, 1)
                next_lambda_hums = np.zeros((self.mpc_env.nLambda, 1))

            # add the value of the next state to the reference array
            ref_oc_next_val[:, col_idx] = oc_next_hum_val.reshape(oc_next_hum_val.shape[0],)
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                ref_oc_lambdas[:, col_idx-1] = next_lambda_hums[:,0]
            # generate next environment state to iterate next
            oc_next_rob_val = np.vstack([ref_x[col_idx], ref_y[col_idx], ref_th[col_idx], ref_v[t_step], self_state.gx, self_state.gy])
            oc_next_val = np.vstack([oc_next_rob_val, oc_next_hum_val])

        if self.mpc_env.hum_model == 'orca_casadi_simple' or self.mpc_env.hum_model == 'orca_casadi_kkt':
            if use_casadi_init:
                ref_hum_U = ref_oc_next_U_val
            else:
                slice_indices = np.array([[idx1+2, idx1+3, -1] for idx1 in np.arange(self.mpc_env.num_hums)*self.mpc_env.nx_hum]).reshape(self.mpc_env.nvars_hum*self.mpc_env.num_hums)
                ref_hum_U = np.take(np.vstack([ref_oc_next_val, np.zeros(ref_oc_next_val.shape[1])]), slice_indices, axis=0)[:,1:]

            ref_X = np.vstack([ref_x, ref_y, ref_th, np.hstack([np.array([np.linalg.norm(self_state.velocity)]), ref_v]), np.ones(ref_x.shape)*self_state.gx, np.ones(ref_x.shape)*self_state.gy]+[ref_oc_next_val[idx, :] for idx in range(ref_oc_next_val.shape[0])])
            ref_U = np.vstack([ref_v, ref_om, ref_hum_U])
            if self.mpc_env.hum_model == 'orca_casadi_kkt':
                ref_lambdas = ref_oc_lambdas
                ref_U = np.vstack([ref_U, ref_lambdas])
        else:
            ref_X = np.vstack([ref_x, ref_y, ref_th, np.hstack([np.array([np.linalg.norm(joint_state.self_state.velocity)]), ref_v]), np.ones(ref_x.shape)*self_state.gx, np.ones(ref_x.shape)*self_state.gy, ref_oc_next_val])
            ref_U = np.vstack([ref_v, ref_om])

        return ref_X, ref_U


    def calc_actual_orca_for_x_val(self, all_x_vals):
        actual_all_x_vals_hums = []
        for x_val in all_x_vals:
            # For plotting
            nx_r, np_g = self.mpc_env.nx_r, self.mpc_env.np_g
            hum_offset = self.mpc_env.nx_r+self.mpc_env.np_g
            ref_steps = x_val.shape[1]-1
            oc_next_val = np.atleast_2d(x_val[:,0]).T
            actual_oc_next_val = np.zeros((self.mpc_env.nX_hums, ref_steps+1), dtype=np.float64)
            actual_oc_next_val[:, 0] = np.array(oc_next_val[hum_offset:]).reshape(oc_next_val.shape[0]-hum_offset,)
            for t_step in range(ref_steps):
                col_idx = t_step+1
                # get the next state from the ORCA callback
                oc_next_hum_val = self.mpc_env.callback_orca(oc_next_val).toarray()
                # add the value of the next state to the reference array
                actual_oc_next_val[:, col_idx] = oc_next_hum_val.reshape(oc_next_hum_val.shape[0],)
                # read next environment state to iterate next
                oc_next_rob_val = np.atleast_2d(x_val[:nx_r+np_g, col_idx]).T
                oc_next_val = np.vstack([oc_next_rob_val, oc_next_hum_val])
            actual_all_x_vals_hums.append(actual_oc_next_val)

        return actual_all_x_vals_hums

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
            # if np.linalg.norm(vec) < 1:
            #     vec = 1.01 * vec / vec_norm
            gx = self_state.px + vec[0]
            gy = self_state.py + vec[1]
        else:
            gx, gy = self_state.gx, self_state.gy
        return gx, gy


    def gen_ref_traj(self, state):
        """Generates a reference trajectory as the straight-line distance from

        :param state: the joint state
        :return: reference values for states and actions
        """
        self_state = state.self_state
        px, py = self_state.px, self_state.py
        init_dist = np.sqrt((self_state.gx - px)**2+(self_state.gy - py)**2)
        N_req = int(np.ceil(init_dist / (self.time_step * self.mpc_env.pref_speed)))+2
        # ref_steps = self.horiz # if we re-generate reference at each time-step
        ref_steps = N_req
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
        if self.new_ref_each_step or (self.ref_type == 'point_stab' and (self.env.sim_env == "hallway_static" or self.env.sim_env == "hallway_static_with_back" or self.env.sim_env == "hallway_bottleneck")):
            # state = deepcopy(state_actual)
            gx, gy = self.get_int_goal(state)
            state.self_state.gx = gx
            state.self_state.gy = gy
            state.self_state.goal_position = (gx, gy)
            self.gen_ref_traj(state)
            # x_ref, u_ref = self.generate_traj(state, self.horiz)
            # return x_ref, np.zeros(u_ref.shape)

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


    def predict(self, env_state):
        if self.priviledged_info:
            state = env_state
        else:
            # make a full state for each agent, but with:
            # gx, gy based on CV projection of current speed for each agent
            # v_pref based on callback max_speed
            # p, g, v, om based on current state
            human_states = []
            for hum in env_state.human_states:
                # calculate gx, gy based on p + v * 2 seconds
                gx = hum.px + hum.vx * 2
                gy = hum.py + hum.vy * 2
                hum_state = FullState(
                    px=hum.px,
                    py=hum.py,
                    vx=hum.vx,
                    vy=hum.vy,
                    gx=gx,
                    gy=gy,
                    v_pref=self.human_max_speed,
                    theta=np.arctan2(hum.vy, hum.vx),
                    radius=hum.radius,
                )
                human_states.append(hum_state)
            state = FullyObservableJointState(
                self_state=env_state.self_state,
                human_states=human_states,
                static_obs=env_state.static_obs,
            )

        # initialize casadi options etc. for new scenario case.
        if not self.mpc_env or self.env.global_time == 0.0:
            self.init_mpc(state)
            self.reset_scenario_values()

        # calculating MPC action
        goal_states, goal_actions = self.get_ref_traj(state)
        mpc_state = self.mpc_env.convert_to_mpc_state_vector(state, self.mpc_env.nx_r, self.mpc_env.np_g, self.mpc_env.nX_hums, get_numpy=True)

        start_time = time.time()
        mpc_action = self.select_action(mpc_state, state, goal_states, goal_actions)
        end_time = time.time()

        action = ActionRot(mpc_action[0], mpc_action[1]*self.time_step)
        self.prev_lvel = action.v


        # step along trajectory
        self.calc_times.append(end_time-start_time)
        if DISP_TIME:
            logging.info('[CAMPC] Total wall time to solve MPC for step {:} was {:.3f}s'.format(self.traj_step, end_time-start_time))

        self.traj_step += 1
        return action

