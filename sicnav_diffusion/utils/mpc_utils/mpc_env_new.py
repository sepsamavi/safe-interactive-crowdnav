
from copy import deepcopy
import os
import logging
from math import ceil
from re import M
from unicodedata import name
import numpy as np
import casadi as cs
import io
import hashlib


from sicnav.utils.mpc_utils.orca_c_wrapper import ORCACWrapper
from .orca_casadi_new import casadiORCA, numstab_epsilon, safe_divide
from .constraints_new import BoundedConstraint, LinearConstraint, NonlinearConstraint, QuadraticConstraint, ConstrainedVariableType, ConstraintType
from .system_model_new import SystemModel


class MPCEnv(object):
    callback_orca = None
    casadi_orca = None

    def __init__(self, time_step, joint_state, num_hums, K, config=None, env_config=None, isSim=False, rate=None):
        self.isSim = isSim
        self.time_step = time_step
        if rate is None:
            self.rate = 1./self.time_step
        else:
            self.rate = rate

        # constrain the robot to not get within this buffer distance to the other agents
        self.rob_len_buffer = 0.0
        self.rob_wid_buffer = 0.0
        self.rob_stat_buffer = 0.0

        self.dyn_type = 'kinematic'
        self.hum_model = 'orca_casadi_kkt'
        self.human_goal_cvmm = False

        # horizon that we use the kkt reformulation, after this, we would use cvmm
        self.horiz = K
        self.orca_kkt_horiz = 0

        # pref speed used for generatig refernce and warmstart
        self.pref_speed = 0.95

        # robot attributes self.nx_hum
        self.max_speed = 1.0
        self.max_rot = np.pi/4
        self.max_l_acc = 1.0
        self.max_l_dcc = -1.0

        if config is not None:
            self.config = config
            self.configure(config, env_config)
        else:
            logging.warn('[MPCEnv] No config. file, using default values')

        if self.max_l_dcc > -self.max_speed / (time_step*max(1,ceil(K*0.5))):
            logging.warn('[MPCEnv] Stopping from max speed will take more than 0.5K, max decc = {:}'.format(self.max_l_dcc))
        if self.max_l_dcc > -self.max_speed / (time_step*max(1,ceil(K))):
            logging.warn('[MPCEnv] POTENTIAL PROBLEM Stopping from max speed will take more than K, max decc = {:}'.format(self.max_l_dcc))
        # other agent state attributes
        self.num_hums = num_hums
        self.num_hums_total = len(joint_state.human_states)
        self.num_stat_obs = len(joint_state.static_obs)
        self.env_static_obs = deepcopy(joint_state.static_obs)

        # if the hums are inputs to the MPC optimization, this will get overwritten
        if self.hum_model == 'cvmm':
            self.nx_hum = 6 # pos, vel, goal_pos
            self.nvars_hum = 2
            self.nVars_hums = 0
            self.nlambda_hum = 0
            self.nLambda = 0
            nX_hums = self.nx_hum * num_hums
            self.nX_hums = nX_hums
        elif self.hum_model == 'orca_casadi_kkt':
            if self.human_pred_MID and self.MID_stateful_weights and not self.human_pred_MID_joint:
                self.nx_hum = 6 + self.num_MID_samples # pos, vel, goal_pos, MID_sample weights
                nX_hums = self.nx_hum * num_hums
                self.nX_hums = nX_hums
            elif self.human_pred_MID and self.MID_stateful_weights and self.human_pred_MID_joint:
                self.nx_hum = 6
                nX_hums = self.nx_hum * num_hums + self.num_MID_samples
                self.nX_hums = nX_hums
            else:
                self.nx_hum = 6 # pos, vel, goal_pos
                nX_hums = self.nx_hum * num_hums
                self.nX_hums = nX_hums
            self.nvars_hum = 4 # vx, vy, ksi <- for relaxed orca, ksi_2 <- for relaxed acc constraint
            self.nVars_hums = self.nvars_hum * self.num_hums
            self.nlambda_hum = self.num_hums + self.num_stat_obs + 4 # also counts the lambdas for velocity + the relaxation slack variable for the relaxed orca + acceleration constraint + the extra relaxation variable for the acceleration constraint
            self.nLambda = self.num_hums * self.nlambda_hum # one for each human
            self.rng = np.random.default_rng(5)
        else:
            raise NotImplementedError



        nx_r, np_g = 8, 2
        nu_r = 2
        self.nx_r, self.np_g = nx_r, np_g
        self.nu_r = nu_r
        # final dimensions
        nx = self.nx_r + self.np_g + nX_hums
        self.state_dim = self.nx = nx
        self.action_dim = self.nu = self.nu_r + self.nVars_hums + self.nLambda


        # Initializing system variables
        self.time_step = time_step

        # generate objects for internal ORCA model (summarized in ORCA callback function)
        # callback which calls o.g. optimization
        self.callback_orca = ORCACWrapper('callback_orca', self.time_step, joint_state, self.nx_r, self.np_g, self.nX_hums, self.config, {'enable_fd':True}, nx_hum=self.nx_hum, num_humans = self.num_hums)
        # my casadiORCA object that re-implements symbolicalle


        # dynamics model of the environment
        if self.dyn_type == 'kinematic':
            X, U, next_X = self.gen_kin_model(joint_state)
        else:
            raise NotImplementedError



        # update final dimensions given dynamics type
        nx = self.nx_r + self.np_g + nX_hums
        self.state_dim = self.nx = nx
        self.action_dim = self.nu = self.nu_r + self.nVars_hums + self.nLambda


        # MPC cost parameters:
        q_x = self.cost_params['q_x']
        q_y = self.cost_params['q_y']
        q_theta = self.cost_params['q_theta']
        q_v_prev = self.cost_params['q_v_prev']
        r_om = self.cost_params['r_om']
        q_v_prev_dot = self.cost_params['q_v_prev_dot']
        q_om_prev_dot = self.cost_params['q_om_prev_dot']
        term_q_coeff = self.term_q_coeff = self.cost_params['term_q_coeff']
        term_q_theta = self.cost_params['term_q_theta']

        if self.hum_model == 'orca_casadi_kkt':
            self.Q_diag = np.hstack([np.array([q_x, q_y, q_theta, q_theta, q_v_prev, 0.0, q_v_prev_dot, q_om_prev_dot]), np.zeros(self.nx_r-8+self.np_g+self.nX_hums)])
            self.Q = np.diag(self.Q_diag)
            self.term_Q_diag = np.hstack([term_q_coeff*np.array([q_x, q_y, term_q_theta, term_q_theta, q_v_prev, 0.0, q_v_prev_dot/term_q_coeff, q_om_prev_dot/term_q_coeff]), np.zeros(self.nx_r-8+self.np_g+self.nX_hums)])
            self.term_Q = np.diag(self.term_Q_diag)
            self.R_diag = np.hstack([np.array([0.0, r_om]), np.zeros(self.nu-self.nu_r)])
            self.R = np.diag(self.R_diag)
        else:
            self.Q_diag = np.hstack([np.array([q_x, q_y, q_theta, q_theta, q_v_prev, 0.0, q_v_prev_dot, q_om_prev_dot]), np.zeros(self.nx_r-8+self.np_g+self.nX_hums)])
            self.Q = np.diag(self.Q_diag)
            self.term_Q_diag = np.hstack([term_q_coeff*np.array([q_x, q_y, q_theta, q_theta, q_v_prev, 0.0, q_v_prev_dot/term_q_coeff, q_om_prev_dot/term_q_coeff]), np.zeros(self.nx_r-8+self.np_g+self.nX_hums)])
            self.term_Q = np.diag(self.term_Q_diag)
            self.R_diag = np.hstack([np.array([0.0, r_om]), np.zeros(self.nu-self.nu_r)])
            self.R = np.diag(self.R_diag)

        # From intialization:
        self.init_joint_state = joint_state

        # robot attributes
        self.radius = joint_state.self_state.radius

        # Defining the elements of the MPC cost function
        cost = self.get_cost_eqn(X, U)
        if self.hum_model == 'orca_casadi_kkt' and self.human_pred_MID:
            stacked_preds = self.stack_MID_preds(self.MID_samples_t_all_hums_stacked, self.MID_samples_tp1_all_hums_stacked)
            dynamics = {"dyn_eqn": next_X, "obs_eqn": None, "vars": {"X": X, "U": U}, "stacked_preds" : stacked_preds}
        else:
            dynamics = {"dyn_eqn": next_X, "obs_eqn": None, "vars": {"X": X, "U": U}}
        self.system_model = SystemModel(dynamics=dynamics, cost=cost, dt=self.time_step, linearize=False)

        # State constraints
        coll_consts = self.get_mpc_coll_constraints()
        stat_coll_consts = self.get_mpc_stat_coll_constraints(joint_state)

        # Terminal constraints
        term_consts = [] # self.get_term_constraints()

        # Action constraints
        if self.dyn_type == 'kinematic':
            bound_con = self.get_kin_bound_constraint()
            # Input and action constraintt
            acc_cons = self.get_kin_acc_constraint()
            bound_cons_state = []
            bound_cons_state_sym = []
        else:
            bound_cons_state = [self.get_kin_bound_constraint()]
            bound_cons_state_sym = [bound_cons_state[0].get_symbolic_model()]
            bound_con = self.get_dyn_bound_constraint()
            acc_cons=[]

        # Constraints on human (if to be included in MPC model)
        if self.hum_model == 'orca_casadi_kkt':
            if self.orca_kkt_horiz == 0:
                self.orca_kkt_horiz = self.horiz
            self.hums_orca_consts, self.hums_max_vel_consts, self.hums_max_acc_consts, self.hums_ksi_consts, self.hums_ksi_2_consts, self.hum_orca_kkt_ineq_consts, self.hum_orca_kkt_eq_consts, self.hum_numstab_consts = self.get_human_constraints()
            hums_orca_consts, hums_max_vel_consts, hums_max_acc_consts, hums_ksi_consts, hums_ksi_consts, hum_numstab_consts = [], [], [], [], [], [] # NB, requires that the constraints be included in the kkt constraints in this case.
        else:
            hums_orca_consts, hums_max_vel_consts, hums_max_acc_consts, hums_ksi_consts, hums_ksi_consts, hum_numstab_consts = self.get_human_constraints()

        if self.hum_model == 'orca_casadi_kkt' and self.human_pred_MID:
            self.hums_close_to_preds_consts = self.get_hums_close_to_preds_constraints(self.X, self.U)

        self.state_constraints  = [con for con in coll_consts] + bound_cons_state
        self.state_constraints_sym = [con.get_symbolic_model() for con in self.state_constraints]
        self.stat_coll_consts = stat_coll_consts
        self.stat_coll_consts_sym = [con.get_symbolic_model() for con in self.stat_coll_consts]

        self.term_constraints  = [con for con in term_consts]
        self.term_constraints_sym = [con.get_symbolic_model() for con in self.term_constraints]

        self.input_constraints  = [bound_con] + [con for con in hums_ksi_consts]
        self.input_constraints_sym = [con.get_symbolic_model() for con in self.input_constraints]

        self.input_state_constraints  = [acc_con for acc_con in acc_cons] + [con for con in hums_max_vel_consts+hums_max_acc_consts+hums_orca_consts]
        self.input_state_constraints_sym = [con.get_symbolic_model() for con in self.input_state_constraints]

        self.numstab_state_consts = [con for con in hum_numstab_consts]
        self.numstab_state_consts_sym = [con.get_symbolic_model() for con in self.numstab_state_consts]


    def configure(self, config, env_config=None):
        self.hum_model = config.get('mpc_env', 'hum_model')
        self.orca_kkt_horiz = config.getint('mpc_env', 'orca_kkt_horiz')
        self.rob_len_buffer = config.getfloat('mpc_env', 'rob_len_buffer')
        self.rob_wid_buffer = config.getfloat('mpc_env', 'rob_wid_buffer')
        self.rob_stat_buffer = config.getfloat('mpc_env', 'rob_stat_buffer')
        self.pref_speed =  config.getfloat('mpc_env', 'pref_speed')
        self.max_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rev_speed = config.getfloat('mpc_env', 'max_rev_speed')
        self.max_rot = config.getfloat('mpc_env', 'max_rot_degrees') * np.pi / 180.0
        self.max_l_acc = config.getfloat('mpc_env', 'max_l_acc')
        self.max_l_dcc = config.getfloat('mpc_env', 'max_l_dcc')


        self.human_goal_cvmm = config.getboolean('campc', 'human_goal_cvmm')
        self.human_goal_cvmm_horizon = config.getfloat('campc', 'human_goal_cvmm_horizon')

        self.human_pred_MID = config.getboolean('campc', 'human_pred_MID')
        self.human_pred_MID_joint = config.getboolean('campc', 'human_pred_MID_joint')
        assert (self.human_goal_cvmm and not self.human_pred_MID) or (not self.human_goal_cvmm and self.human_pred_MID) or (not self.human_goal_cvmm and not self.human_pred_MID)
        self.num_MID_samples = env_config.getint('human_trajectory_forecaster', 'num_samples')
        if self.human_pred_MID:
            self.MID_stateful_weights = config.getboolean('campc', 'MID_stateful_weights')
        if self.hum_model=='orca_casadi_kkt' and self.human_pred_MID :
            self.orca_ksi_scaling = 1.0
            self.orca_vxy_scaling = 1.0
        else:
            self.orca_ksi_scaling = config.getfloat('mpc_env', 'orca_ksi_scaling')
            self.orca_vxy_scaling =  config.getfloat('mpc_env', 'orca_vxy_scaling')


        self.max_hum_acc = config.getfloat('humans', 'max_acc')

        self.min_hum_max_speed = config.getfloat('campc', 'min_hum_max_speed')

        # Read in the cost params
        self.cost_params = {}
        self.cost_params['q_x'] = config.getfloat('mpc_env', 'q_x')
        self.cost_params['q_y'] = config.getfloat('mpc_env', 'q_y')
        self.cost_params['q_theta'] = config.getfloat('mpc_env', 'q_theta')
        self.cost_params['q_v_prev'] = config.getfloat('mpc_env', 'q_v_prev')
        self.cost_params['r_om'] = config.getfloat('mpc_env', 'r_om')
        self.cost_params['q_v_prev_dot'] = config.getfloat('mpc_env', 'q_v_prev_dot')
        self.cost_params['q_om_prev_dot'] = config.getfloat('mpc_env', 'q_om_prev_dot')
        self.cost_params['term_q_coeff'] = config.getfloat('mpc_env', 'term_q_coeff')
        self.cost_params['term_q_theta'] = config.getfloat('mpc_env', 'term_q_theta')


        print('[MPCEnv] Config {:} = {:}'.format('dyn_type', self.dyn_type))
        print('[MPCEnv] Config {:} = {:}'.format('hum_model', self.hum_model))
        print('[MPCEnv] Config {:} = {:}'.format('orca_kkt_horiz', self.orca_kkt_horiz))
        print('[MPCEnv] Config {:} = {:}'.format('rob_len_buffer', self.rob_len_buffer))
        print('[MPCEnv] Config {:} = {:}'.format('rob_wid_buffer', self.rob_wid_buffer))
        print('[MPCEnv] Config {:} = {:}'.format('pref_speed', self.pref_speed))
        print('[MPCEnv] Config {:} = {:}'.format('max_speed', self.max_speed))
        print('[MPCEnv] Config {:} = {:}'.format('max_rev_speed', self.max_rev_speed))
        print('[MPCEnv] Config {:} = {:}'.format('max_rot', self.max_rot))
        print('[MPCEnv] Config {:} = {:}'.format('max_l_acc', self.max_l_acc))
        print('[MPCEnv] Config {:} = {:}'.format('max_l_dcc', self.max_l_dcc))
        print('[MPCEnv] Config {:} = {:}'.format('orca_ksi_scaling', self.orca_ksi_scaling))
        print('[MPCEnv] Config {:} = {:}'.format('orca_vxy_scaling', self.orca_vxy_scaling))
        print('[MPCEnv] Config {:} = {:}'.format('human_goal_cvmm', self.orca_vxy_scaling))


        print('[MPCEnv] Config {:} = {:}'.format('human_goal_cvmm_horizon', self.human_goal_cvmm_horizon))
        print('[MPCEnv] Config {:} = {:}'.format('human_pred_MID', self.human_pred_MID))
        print('[MPCEnv] Config {:} = {:}'.format('min_hum_max_speed', self.min_hum_max_speed))
        print('[MPCEnv] Config {:} = {:}'.format('cost_params', self.cost_params))

        buffer_env = io.StringIO()
        buffer_pol = io.StringIO()
        env_config.write(buffer_env)
        config.write(buffer_pol)
        # read buffer_env and read buffer_pol and strip of all whitespace and strip \n and strip = and then append to eachother
        self.config_str = buffer_env.getvalue().replace(' ', '').replace('\n', '').replace('=', '') + buffer_pol.getvalue().replace(' ', '').replace('\n', '').replace('=', '')
        self.config_hash = hashlib.md5(self.config_str.encode()).hexdigest()

        if env_config is not None:
            self.env_config = env_config
            self.length = env_config.getfloat('robot', 'length')
            self.width = env_config.getfloat('robot', 'width')
            self.radius = env_config.getfloat('robot', 'radius')
            assert self.width >= 2*self.radius, 'Need length {:2f} >= 2*radius {:.2f}. Current parameters can lead to collisions.'.format(self.width, self.radius)
            assert self.length >= 2*self.radius, 'Need length {:.2f} >= 2*radius {:.2f}. Current parameters can lead to collisions.'.format(self.length, self.radius)
            self.outdoor_robot_setting = env_config.getboolean('env', 'wild', fallback=False)
            print('[MPCEnv] Config {:} = {:}'.format('robot length', self.length))
            print('[MPCEnv] Config {:} = {:}'.format('robot width', self.width))
            print('[MPCEnv] Config {:} = {:}'.format('robot radius', self.radius))
        else:
            raise RuntimeError('No env_config provided to MPCEnv.')

    def stack_MID_preds(self, MID_samples_t_all_hums_stacked, MID_samples_tp1_all_hums_stacked):
        return cs.vertcat(MID_samples_t_all_hums_stacked[:,0], MID_samples_t_all_hums_stacked[:,1], MID_samples_tp1_all_hums_stacked[:,0], MID_samples_tp1_all_hums_stacked[:,1])

    def get_cost_eqn(self, X, U):
        """Obtain a cost equation for the MPC system.

        :param X: symbolic variable for current state of the environment
        :param U: symbolic variable for the input to the system
        :return: symbolic equation for the cost function
        """
        # cost equation for robot's MPC objective
        # reference
        Xr = cs.MX.sym('Xr', self.nx, 1)
        Ur = cs.MX.sym('Ur', self.nu, 1)
        # quadratic cost matrices
        Q_diag = cs.MX.sym('Q_diag', self.nx, 1)
        Q = cs.diag(Q_diag)
        R_diag = cs.MX.sym('R_diag', self.nu, 1)
        R = cs.diag(R_diag)
        term_Q_diag = cs.MX.sym('term_Q_diag', self.nx, 1)
        term_Q = cs.diag(term_Q_diag)

        X_residual_p1 = X[:2] - Xr[:2]
        # X_residual_p2 = cs.sin(X[2])*cs.cos(Xr[2]) - cs.cos(X[2])*cs.sin(Xr[2])
        X_residual_p2 = X[2]*Xr[3] - X[3]*Xr[2]
        # X_residual_p3 = cs.cos(X[3])*cs.cos(Xr[3]) + cs.sin(X[3])*cs.sin(Xr[3])
        X_residual_p3 = X[3]*Xr[3] + X[2]*Xr[2]
        X_residual_p4 = X[4:] - Xr[4:]
        X_residual = cs.vertcat(X_residual_p1, X_residual_p2, X_residual_p3, X_residual_p4)

        cost_eqn = 0.5 * X_residual.T @ Q @ X_residual + 0.5 * (U - Ur).T @ R @ (U - Ur)
        term_cost_eqn = 0.5 * X_residual.T @ term_Q @ X_residual

        cost = {"cost_eqn": cost_eqn, "term_cost_eqn":term_cost_eqn, "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q_diag": Q_diag, "R_diag": R_diag, "term_Q_diag" : term_Q_diag}}
        return cost


    def get_human_dyn_eqn(self, X, X_hums, U_rob, MID_samples_t_all_hums=None, MID_samples_tp1_all_hums=None):
        """Dynamics equation for the human agents

        :param X: symbolic variable for state of environment
        :param X_hums: symbolic variable for state of human agents (included in X)
        :param U_rob: symbolic variable for input to the robot
        :return: next_X_hums (equation to dynamics of robot), U (input to entire system)
        """
        if self.hum_model == 'cvmm':
            next_X_hums = self.get_CVMM_human_eqn(X_hums)
            U = U_rob
        elif self.hum_model == 'orca_casadi_kkt':
            next_X_hums, U_temp = self.get_ORCA_human_dynamics_eqn(X_hums, U_rob, MID_samples_t_all_hums, MID_samples_tp1_all_hums)
            lambdas_list = []
            for humA_idx in range(self.num_hums):
                lambda_humA = cs.MX.sym('lambda_hum{:}'.format(humA_idx), self.nlambda_hum, 1)
                lambdas_list.append(lambda_humA)
            lambdas = cs.vertcat(*tuple(lambdas_list))
            self.lambdas_list = lambdas_list
            U = cs.vertcat(U_temp, lambdas)
        else:
            raise NotImplementedError

        return next_X_hums, U


    def get_ORCA_human_dynamics_eqn(self, X_hums, U_rob, MID_samples_t_all_hums=None, MID_samples_tp1_all_hums=None):
        """Get equation of next human dynamics when human input is solved in the KKT refomumation of the bilevel set up

        :param X_hums: symbolic variable for current state of the humans
        :param U_rob: symbolic variable for the input to the robot
        :return: next_X_hums (equation to the dynamics of the next humans), U (inputs to the entire system)
        """
        nu_hum = 2
        nksi_hum = 2
        self.nvars_hum = nu_hum + nksi_hum
        self.nVars_hums = self.nvars_hum * self.num_hums
        U_hums_list = []
        for humA_idx in range(self.num_hums):
            U_ksi_humA = cs.MX.sym('U_ksi_hum{:}'.format(humA_idx), self.nvars_hum, 1)
            U_hums_list.append(U_ksi_humA)

        X_humA_sym = cs.MX.sym('X_humA_sym', self.nx_hum, 1)
        U_humA_sym = cs.MX.sym('U_humA_sym', self.nvars_hum, 1)
        # function for getting next human position
        next_P_humj_sym = cs.vertcat(X_humA_sym[0] + self.orca_vxy_scaling * U_humA_sym[0] * self.time_step,
                                        X_humA_sym[1] + self.orca_vxy_scaling * U_humA_sym[1] * self.time_step)
        self.get_next_P_humj_cs_func = cs.Function('get_next_P_humj_cs_func', [X_humA_sym, U_humA_sym], [next_P_humj_sym]).expand()

        self.U_hums_list = U_hums_list
        U_hums = cs.vertcat(*tuple(U_hums_list))
        humA_idx = 0
        idx_x = 0
        idx_u = 0
        idx_x_next = (0+1)*self.nx_hum
        if self.human_pred_MID:

            # function for getting log weights
            next_P_humj_sym2 = self.get_next_P_humj_cs_func(X_humA_sym, U_humA_sym)
            if not self.human_pred_MID_joint:
                MID_samples_t_sym = cs.MX.sym('MID_samples_t_sym', self.num_MID_samples, 2)
                MID_samples_tp1_sym = cs.MX.sym('MID_samples_tp1_sym', self.num_MID_samples, 2)
                sq_dists_t = cs.sum1((MID_samples_t_sym.T - next_P_humj_sym2)**2).T
                log_weights_t_unnormed = -2**10*sq_dists_t
                # clip log_weights_t_unnormed to be at or above -10
                log_weights_lower_bound = cs.DM.ones(log_weights_t_unnormed.shape)*-20

                log_weights_t_unnormed_sans_prev = cs.fmax(log_weights_t_unnormed, log_weights_lower_bound)
                normalization_factor = cs.logsumexp(log_weights_t_unnormed_sans_prev)
                log_weights_t_sym = log_weights_t_unnormed_sans_prev - normalization_factor
                if self.MID_stateful_weights:
                    # log_weights_t_unnormed_sans_prev = log_weights_t_unnormed
                    log_weights_t_inc_prev_unnormed = X_humA_sym[6:] + log_weights_t_sym
                    normalization_factor_inc_prev = cs.logsumexp(log_weights_t_inc_prev_unnormed)
                    log_weights_t_inc_prev_unchecked = log_weights_t_inc_prev_unnormed - normalization_factor_inc_prev

                else:
                    log_weights_t_inc_prev_unchecked = log_weights_t_sym

                log_weights_t_inc_prev = log_weights_t_inc_prev_unchecked


                self.get_v_pref_weights_cs_func_one_hum = cs.Function('get_v_pref_weights_cs_func_one_hum', [X_humA_sym, U_humA_sym, MID_samples_t_sym], [log_weights_t_inc_prev])

                # function for getting the goals
                log_weights_t_sym2 = self.get_v_pref_weights_cs_func_one_hum(X_humA_sym, U_humA_sym, MID_samples_t_sym)
                weights_t = cs.exp(log_weights_t_sym2)
                MID_samples_tp1_weighted = cs.sum1((MID_samples_tp1_sym * cs.fmax(weights_t, numstab_epsilon))).T
                gx = MID_samples_tp1_weighted[0]
                gy = MID_samples_tp1_weighted[1]
                self.get_gx_gy_one_hum_cs_func = cs.Function('get_gx_gy_one_hum_cs_func', [X_humA_sym, U_humA_sym, MID_samples_t_sym, MID_samples_tp1_sym], [gx, gy])
            else:
                # in the case the predictions are joint, we need to first calculate the log weights based on all next steps.
                X_hums_sym = cs.MX.sym('X_hums_sym', self.nX_hums, 1)
                U_hums_sym = cs.MX.sym('U_hums_sym', self.nVars_hums, 1)
                MID_samples_allhums_t_sym = cs.MX.sym('MID_samples_allhums_t_sym', self.num_MID_samples, 2*self.num_hums)
                MID_samples_allhums_tp1_sym = cs.MX.sym('MID_samples_allhums_tp1_sym', self.num_MID_samples, 2*self.num_hums)
                next_P_hums_list = []
                for humA_idx_sym in range(self.num_hums):
                    idx_x = humA_idx_sym*self.nx_hum
                    idx_u = humA_idx_sym*self.nvars_hum
                    idx_x_next = (humA_idx_sym+1)*self.nx_hum
                    X_humA_sym2 = X_hums_sym[self.nx_hum*humA_idx_sym:(self.nx_hum*humA_idx_sym+self.nx_hum)]
                    U_humA_sym2 = U_hums_sym[idx_u:idx_u+self.nvars_hum]
                    next_P_humj_sym3 = self.get_next_P_humj_cs_func(X_humA_sym2, U_humA_sym2)

                    next_P_hums_list.append(next_P_humj_sym3)
                next_P_hums_sym = cs.vertcat(*tuple(next_P_hums_list))
                avg_sq_dists_t = cs.sum1((MID_samples_allhums_t_sym.T - next_P_hums_sym)**2).T / self.num_hums
                log_weights_t_unnormed = -2**10*avg_sq_dists_t
                log_weights_lower_bound = cs.DM.ones(log_weights_t_unnormed.shape)*-20
                log_weights_t_unnormed_sans_prev = cs.fmax(log_weights_t_unnormed, log_weights_lower_bound)
                normalization_factor = cs.logsumexp(log_weights_t_unnormed_sans_prev)
                log_weights_t_sym = log_weights_t_unnormed_sans_prev - normalization_factor
                if self.MID_stateful_weights:
                    # log_weights_t_unnormed_sans_prev = log_weights_t_unnormed
                    log_weights_t_inc_prev_unnormed = X_hums_sym[-self.num_MID_samples:] + log_weights_t_sym
                    normalization_factor_inc_prev = cs.logsumexp(log_weights_t_inc_prev_unnormed)
                    log_weights_t_inc_prev_unchecked = log_weights_t_inc_prev_unnormed - normalization_factor_inc_prev
                else:
                    log_weights_t_inc_prev_unchecked = log_weights_t_sym

                log_weights_t_inc_prev = log_weights_t_inc_prev_unchecked
                self.get_v_pref_weights_cs_func_all_hums = cs.Function('get_v_pref_weights_cs_func_all_hums', [X_hums_sym, U_hums_sym, MID_samples_allhums_t_sym], [log_weights_t_inc_prev])

                # function for getting the goals
                log_weights_t_sym2 = self.get_v_pref_weights_cs_func_all_hums(X_hums_sym, U_hums_sym, MID_samples_allhums_t_sym)
                weights_t = cs.exp(log_weights_t_sym2)
                MID_samples_allhums_tp1_weighted = cs.sum1((MID_samples_allhums_tp1_sym * cs.fmax(weights_t, numstab_epsilon))).T
                self.get_gxgy_all_hums_cs_func = cs.Function('get_gxgy_all_hums_cs_func', [X_hums_sym, U_hums_sym, MID_samples_allhums_t_sym, MID_samples_allhums_tp1_sym], [MID_samples_allhums_tp1_weighted])




            ##### end create functions
            humA_idx = 0
            idx_u = 0
            X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
            U_humA = U_hums[idx_u:idx_u+self.nvars_hum]
            next_P_humj = self.get_next_P_humj_cs_func(X_humA, U_humA)
            if not self.human_pred_MID_joint:
                MID_samples_t = MID_samples_t_all_hums[humA_idx]
                MID_samples_tp1 = MID_samples_tp1_all_hums[humA_idx]
                gx, gy = self.get_gx_gy_one_hum_cs_func(X_humA, U_humA, MID_samples_t, MID_samples_tp1)
                log_weights_t = self.get_v_pref_weights_cs_func_one_hum(X_humA, U_humA, MID_samples_t)


                next_X_hums = cs.vertcat(next_P_humj[0],
                                        next_P_humj[1],
                                        self.orca_vxy_scaling*U_hums[idx_u],
                                        self.orca_vxy_scaling*U_hums[idx_u+1],
                                        gx, # human goal posns
                                        gy,
                                        log_weights_t) # human goal posns
            else:
                # find the actual next log weights
                MID_samples_allhums_t = cs.horzcat(*tuple(MID_samples_t_all_hums))
                MID_samples_allhums_tp1 = cs.horzcat(*tuple(MID_samples_tp1_all_hums))
                all_goals_t = self.get_gxgy_all_hums_cs_func(X_hums, U_hums, MID_samples_allhums_t, MID_samples_allhums_tp1)

                # now do the first human
                next_X_hums = cs.vertcat(next_P_humj[0],
                                        next_P_humj[1],
                                        self.orca_vxy_scaling*U_hums[idx_u],
                                        self.orca_vxy_scaling*U_hums[idx_u+1],
                                        all_goals_t[0],
                                        all_goals_t[1])

        else:

            gx_sym = X_humA_sym[4]
            gy_sym = X_humA_sym[5]
            self.get_gx_gy_one_hum_cs_func = cs.Function('get_gx_gy_one_hum_cs_func', [X_humA_sym], [gx_sym, gy_sym]).expand()

            X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
            U_humA = U_hums[idx_u:idx_u+self.nvars_hum]
            next_P_humj = self.get_next_P_humj_cs_func(X_humA, U_humA)
            gx, gy = self.get_gx_gy_one_hum_cs_func(X_humA)
            next_X_hums = cs.vertcat(next_P_humj[0], #X_hums[idx_x] + self.orca_vxy_scaling*U_hums[idx_u] * self.time_step,
                                     next_P_humj[1], #X_hums[idx_x+1] + self.orca_vxy_scaling*U_hums[idx_u+1] * self.time_step,
                                     self.orca_vxy_scaling*U_hums[idx_u],
                                     self.orca_vxy_scaling*U_hums[idx_u+1],
                                     gx, # human goal posns
                                     gy) # human goal posns
        for j in range(1, self.num_hums):
            humA_idx = j
            idx_x = j*self.nx_hum
            idx_u = j*self.nvars_hum
            idx_x_next = (j+1)*self.nx_hum
            if self.human_pred_MID:

                X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
                # U_humA = U_hums_list[0]
                U_humA = U_hums[idx_u:idx_u+self.nvars_hum]
                MID_samples_t = MID_samples_t_all_hums[humA_idx]
                MID_samples_tp1 = MID_samples_tp1_all_hums[humA_idx]
                next_P_humj = self.get_next_P_humj_cs_func(X_humA, U_humA)
                if not self.human_pred_MID_joint:
                    gx, gy = self.get_gx_gy_one_hum_cs_func(X_humA, U_humA, MID_samples_t, MID_samples_tp1)
                    log_weights_t = self.get_v_pref_weights_cs_func_one_hum(X_humA, U_humA, MID_samples_t)


                    next_X_hums = cs.vertcat(next_X_hums,
                                            next_P_humj[0],
                                            next_P_humj[1],
                                            self.orca_vxy_scaling*U_hums[idx_u],
                                            self.orca_vxy_scaling*U_hums[idx_u+1],
                                            gx, # human goal posns
                                            gy,
                                            log_weights_t) # human goal posns
                else:
                    gx = all_goals_t[2*j]
                    gy = all_goals_t[2*j+1]
                    next_X_hums = cs.vertcat(next_X_hums,
                                            next_P_humj[0],
                                            next_P_humj[1],
                                            self.orca_vxy_scaling*U_hums[idx_u],
                                            self.orca_vxy_scaling*U_hums[idx_u+1],
                                            gx, # human goal posns
                                            gy)
            else:
                X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]
                U_humA = U_hums[idx_u:idx_u+self.nvars_hum]
                next_P_humj = self.get_next_P_humj_cs_func(X_humA, U_humA)
                gx, gy = self.get_gx_gy_one_hum_cs_func(X_humA)
                next_X_hums = cs.vertcat(next_X_hums,
                                        next_P_humj[0],
                                        next_P_humj[1],
                                        self.orca_vxy_scaling*U_hums[idx_u],
                                        self.orca_vxy_scaling*U_hums[idx_u+1],
                                        gx, # human goal posns
                                        gy) # human goal posns

        if self.human_pred_MID and self.human_pred_MID_joint:
            all_log_weights_t = self.get_v_pref_weights_cs_func_all_hums(X_hums, U_hums, MID_samples_allhums_t)
            next_X_hums = cs.vertcat(next_X_hums, all_log_weights_t)

        U = cs.vertcat(U_rob, U_hums)
        return next_X_hums, U


    def get_CVMM_human_eqn(self, X_hums):
        idx_x = 0
        idx_x_next = (0+1)*self.nx_hum
        next_X_hums = cs.vertcat(X_hums[idx_x] + X_hums[idx_x+2] * self.time_step,
                                 X_hums[idx_x+1] + X_hums[idx_x+3] * self.time_step,
                                 X_hums[idx_x+2],
                                 X_hums[idx_x+3],
                                 X_hums[idx_x_next-2], # human goal posns
                                 X_hums[idx_x_next-1]) # human goal posns
        for j in range(1, self.num_hums):
            idx_x = j*self.nx_hum
            idx_x_next = (j+1)*self.nx_hum
            next_X_hums = cs.vertcat(next_X_hums,
                                     X_hums[idx_x] + X_hums[idx_x+2] * self.time_step,
                                     X_hums[idx_x+1] + X_hums[idx_x+3] * self.time_step,
                                     X_hums[idx_x+2],
                                     X_hums[idx_x+3],
                                     X_hums[idx_x_next-2], # human goal posns
                                     X_hums[idx_x_next-1]) # human goal posns

        return next_X_hums


    def gen_kin_model(self, joint_state):

        # robot's state
        x_r = cs.MX.sym('x_r')
        y_r = cs.MX.sym('y_r')
        sin_th_r = cs.MX.sym('sin_theta_r')
        cos_th_r = cs.MX.sym('cos_theta_r')
        v_r_prev = cs.MX.sym('v_r_prev')
        om_r_prev = cs.MX.sym('om_r_prev')
        v_r_dot = cs.MX.sym('v_r_dot')
        om_r_dot = cs.MX.sym('om_r_dot')

        X_r = cs.vertcat(x_r, y_r, sin_th_r, cos_th_r, v_r_prev, om_r_prev, v_r_dot, om_r_dot)
        # p_r = cs.vertcat(x_r, y_r, th_r)

        # The goal position
        P_g = cs.MX.sym('P_goal', self.np_g)

        # the human agent positions and velocities
        X_hums = cs.MX.sym('X_hums', self.nX_hums)

        # the extra human agent positions and velocities
        X = cs.vertcat(X_r, P_g, X_hums)

        # the actions
        v_r = cs.MX.sym('v_r')
        om_r = cs.MX.sym('om_r')
        U = cs.vertcat(v_r, om_r)

        # Defining discrete-time dynamics equations
        # the robot's dynamics
        delta_t = cs.MX.sym('delta_t')

        next_sin_th_r = sin_th_r*cs.cos(om_r*delta_t) + cos_th_r*cs.sin(om_r*delta_t)
        next_cos_th_r = cos_th_r*cs.cos(om_r*delta_t) - sin_th_r*cs.sin(om_r*delta_t)
        next_x_r = x_r + delta_t * v_r * next_cos_th_r
        next_y_r = y_r + delta_t * v_r * next_sin_th_r

        next_v_r_prev = v_r
        next_om_r_prev = om_r
        next_v_r_dot = (v_r - v_r_prev) / delta_t
        next_om_r_dot = (om_r - om_r_prev) / delta_t
        next_X_r_sym = cs.vertcat(next_x_r, next_y_r, next_sin_th_r, next_cos_th_r, next_v_r_prev, next_om_r_prev, next_v_r_dot, next_om_r_dot)

        next_X_r_fn = cs.Function('next_X_r_dt', [X_r, U, delta_t], [next_X_r_sym], ['X_r', 'U', 'delta_t'], ['next_X_r'])
        self.next_X_r_fn = next_X_r_fn
        next_X_r = next_X_r_fn(X_r, U, self.time_step)

        # next_p_r = cs.vertcat(next_x_r, next_y_r, next_th_r)
        # self.next_p_r_fn = cs.Function('next_p_r_fn', [p_r, cs.vertcat(v_r, om_r)], [next_p_r], ['p_r', 'U'], ['next_p_r']).expand()

        # the goal position next
        next_P_g = P_g


        if self.outdoor_robot_setting:
            # make the changeable static obstacle params
            self.stat_obs_params_vecced = cs.MX.sym('static_obs_params_vecced', 4*self.num_stat_obs, 1)
            self.stat_obs_params = cs.vertcat(*tuple([self.stat_obs_params_vecced[idx*4:idx*4+4].T for idx in range(self.num_stat_obs)]))
            if self.num_hums_total - self.num_hums > 0:
                self.num_extra_hums = self.num_hums_total - self.num_hums
                self.hum_extra_params = cs.MX.sym('hum_extra_params', 4, self.num_hums_total - self.num_hums)
            else:
                self.hum_extra_params = None
                self.num_extra_hums = 0
        else:
            self.stat_obs_params_vecced = None
            self.stat_obs_params = None

        # the next step dynamics for the orca agents
        self.casadi_orca = casadiORCA(self, joint_state, X)


        # see if we are using MID predictions
        # make dummy variables for the MID samples at all the time steps for each human to be used as the parameters in the functions that generate the OCP constraints.
        # NB this is actually useful for Acados since we need to set the parameters separately for each time step, and we define the problem equations for one timestep only.

        # make dummy variables for the MID samples at the current time (t) and the next time step (t+1) for each human to be used in the functions that make the ORCA constraints.
        if self.hum_model == 'orca_casadi_kkt' and self.human_pred_MID:

            self.MID_x_samples_t_all_hums_stacked = cs.MX.sym('MID_x_samples_t_all_hums_stacked', self.num_MID_samples*self.num_hums, 1)
            self.MID_y_samples_t_all_hums_stacked = cs.MX.sym('MID_y_samples_t_all_hums_stacked', self.num_MID_samples*self.num_hums, 1)
            self.MID_x_samples_tp1_all_hums_stacked = cs.MX.sym('MID_x_samples_tp1_all_hums_stacked', self.num_MID_samples*self.num_hums, 1)
            self.MID_y_samples_tp1_all_hums_stacked = cs.MX.sym('MID_y_samples_tp1_all_hums_stacked', self.num_MID_samples*self.num_hums, 1)

            self.MID_samples_t_all_hums_stacked = cs.horzcat(self.MID_x_samples_t_all_hums_stacked, self.MID_y_samples_t_all_hums_stacked)
            self.MID_samples_tp1_all_hums_stacked = cs.horzcat(self.MID_x_samples_tp1_all_hums_stacked, self.MID_y_samples_tp1_all_hums_stacked)

            self.MID_samples_t_all_hums = cs.vertsplit(self.MID_samples_t_all_hums_stacked, [self.num_MID_samples*h_idx for h_idx in range(self.num_hums+1)])
            self.MID_samples_tp1_all_hums = cs.vertsplit(self.MID_samples_tp1_all_hums_stacked, [self.num_MID_samples*h_idx for h_idx in range(self.num_hums+1)])

        if self.hum_model == 'orca_casadi_kkt' and self.human_pred_MID and self.MID_stateful_weights:
            next_X_hums, U = self.get_human_dyn_eqn(X, X_hums, U, self.MID_samples_t_all_hums, self.MID_samples_tp1_all_hums)
        else:
            next_X_hums, U = self.get_human_dyn_eqn(X, X_hums, U)

        # concatenating the next_X
        next_X = cs.vertcat(next_X_r, next_P_g, next_X_hums)

        # Setting object-level symbolic variables
        self.X = X
        self.U = U
        return X, U, next_X



    def get_stat_coll_const(self, comb_rad, stat_ob_idx, name_stub=None):
        if not self.outdoor_robot_setting:
            stat_ob = self.env_static_obs[stat_ob_idx]
            p_1 = cs.DM(stat_ob[0]).reshape((2,1))
            p_2 = cs.DM(stat_ob[1]).reshape((2,1))
        else:
            p_1 = self.stat_obs_params[stat_ob_idx, :2].reshape((2,1))
            p_2 = self.stat_obs_params[stat_ob_idx, 2:].reshape((2,1))

        P = self.X[:2]
        V = p_2 - p_1
        W =  P - p_1
        t = (W.T @ V) / (V.T @ V)
        t_clamped = cs.fmax(cs.fmin(t, 1.0), 0.0)
        closest_point = p_1 + t_clamped * V
        d_vec = (P - closest_point)
        const_val = -(d_vec.T @ d_vec - comb_rad**2)

        # Create NonlinearConstraint object for capsule
        if name_stub is None:
            name = 'stat_ob{:}_capsule'.format(stat_ob_idx)
        else:
            name = name_stub+'_stat_ob{:}_capsule'.format(stat_ob_idx)

        # Define CasADi function for the constraint equation
        if not self.outdoor_robot_setting:
            capsule_func = cs.Function(name, [self.X], [const_val], ['input'], ['const'])
        else:
            capsule_func = cs.Function(name, [cs.vertcat(self.X, self.stat_obs_params_vecced)], [const_val], ['input'], ['const'])
        capsule_const = NonlinearConstraint(env=self,
                                            sym_cs_func=capsule_func,
                                            constrained_variable=ConstrainedVariableType.STATE,
                                            name=name)

        return [capsule_const]

    def get_stat_coll_const_new(self, width_corr, stat_ob_idx, name_stub=None):
        # TODO resolve numerical issues: causes NaNs when used with CVMM human model.
        if not self.outdoor_robot_setting:
            stat_ob = self.env_static_obs[stat_ob_idx]
            p_1 = cs.DM(stat_ob[0]).reshape((2,1))
            p_2 = cs.DM(stat_ob[1]).reshape((2,1))
        else:
            p_1 = self.stat_obs_params[stat_ob_idx, :2].reshape((2,1))
            p_2 = self.stat_obs_params[stat_ob_idx, 2:].reshape((2,1))


        pos_cur_rob = P = self.X[:2]

        v_r_prev = cs.if_else(cs.fabs(self.X[4]) > 1e-3, self.X[4], 1e-3, True)
        # theta = self.X[2]
        pos_prev_rob = pos_cur_rob - self.time_step*cs.vertcat(v_r_prev*self.X[3],
                                                               v_r_prev*self.X[2])

        p_1_rob = pos_prev_rob - cs.sign(v_r_prev)*cs.vertcat((0.5*self.length-0.5*width_corr)*self.X[3],
                                                              (0.5*self.length-0.5*width_corr)*self.X[2])

        p_2_rob = pos_cur_rob + cs.sign(v_r_prev)*cs.vertcat((0.5*self.length-0.5*width_corr)*self.X[3],
                                                             (0.5*self.length-0.5*width_corr)*self.X[2])

        # NB we assume that neither line has zero length!!

        # adapted closest point between two lines from https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
        A = p_2 - p_1
        B = p_2_rob - p_1_rob

        magA = cs.sqrt(A.T @ A)
        magB = cs.sqrt(B.T @ B)

        _A = A / magA
        _B = B / magB

        cross = cs.cross(cs.vertcat(_A, 0.0), cs.vertcat(_B, 0.0))
        # denom = cs.norm_2(cross)**2
        denom = cross.T @ cross

        # If denom is 0 lines A and B are parallel
        cond_parallel = denom <= 1e-6

        def parallel_lines():
            # print('parallel')
            d0 = _A.T @ (p_1_rob - p_1)

            d1 = _A.T @ (p_2_rob - p_1)

            cond_b_before_a = (d0 <= 0.0) * (d1 <= 0.0)
            cond_b_after_a = (d0 >= magA) * (d1 >= magA)
            cond_no_overlap = cond_b_before_a + cond_b_after_a

            def no_overlap():
                def seg_b_before_a():
                    # closest distance between first point of A and the closest point of B
                    return cs.if_else(cs.fabs(d0) < cs.fabs(d1),
                                    cs.sqrt((p_1 - p_1_rob).T @ (p_1 - p_1_rob)),
                                    cs.sqrt((p_1 - p_2_rob).T @ (p_1 - p_2_rob)),
                                    True)

                def seg_b_after_a():
                    # closest distance between second point of A and the closest point of B
                    return cs.if_else(cs.fabs(d0) < cs.fabs(d1),
                                    cs.sqrt((p_2 - p_1_rob).T @ (p_2 - p_1_rob)),
                                    cs.sqrt((p_2 - p_2_rob).T @ (p_2 - p_2_rob)),
                                    True)

                return cs.if_else(cond_b_before_a,
                                  seg_b_before_a(),
                                  seg_b_after_a(),
                                  True)


            def overlap():
                # closest distance between parallel segments
                val = ((d0*_A)+p_1)-p_1_rob
                return cs.sqrt(val.T @ val)

            return cs.if_else(cond_no_overlap,
                              no_overlap(),
                              overlap(),
                              True)
        symb_mtx = cs.SX.sym('symb_mtx', 3, 3)
        det_cs_func = cs.Function('get_det', [symb_mtx], [cs.det(symb_mtx)])

        def not_parallel():
            t = (p_1_rob - p_1)
            detA = det_cs_func(cs.vertcat(cs.vertcat(t, 0.0).T, cs.vertcat(_B, 0.0).T, cross.T))
            detB = det_cs_func(cs.vertcat(cs.vertcat(t, 0.0).T, cs.vertcat(_A, 0.0).T, cross.T))

            t0 = detA/denom
            t1 = detB/denom

            cond_pA_a0 = t0 < 0.0
            cond_pA_a1 = t0 > magA

            def pA_one_of_two():
                return cs.if_else(cond_pA_a0,
                                  p_1,
                                  p_2,
                                  True)
            pA = cs.if_else(cond_pA_a1 + cond_pA_a0,
                            pA_one_of_two(),
                            p_1 + t0*_A,
                            True)

            cond_pB_b0 = t1 < 0.0
            cond_pB_b1 = t1 > magB

            def pB_one_of_two():
                return cs.if_else(cond_pB_b0,
                                  p_1_rob,
                                  p_2_rob,
                                  True)
            pB = cs.if_else(cond_pB_b1 + cond_pB_b0,
                            pB_one_of_two(),
                            p_1_rob + t1*_B,
                            True)

            def clamp_proj_a_get_pB_actual():
                dot_A = cs.fmax(cs.fmin(_B.T @ (pA - p_1_rob), magB), 0.0)
                return p_1_rob + (_B*dot_A)

            pB_actual = cs.if_else(cond_pA_a0 + cond_pA_a1,
                                   clamp_proj_a_get_pB_actual(),
                                   pB,
                                   True)


            def clamp_proj_b_get_pA_actual():
                dot_B = cs.fmax(cs.fmin(_A.T @ (pB_actual - p_1), magA), 0.0)
                return p_1 + (_A*dot_B)

            pA_actual = cs.if_else(cond_pB_b0 + cond_pB_b1,
                                   clamp_proj_b_get_pA_actual(),
                                   pA,
                                   True)

            # return cs.norm_2(pA_actual - pB_actual)
            return cs.sqrt((pA_actual - pB_actual).T @ (pA_actual - pB_actual))


        min_dist = cs.if_else(cond_parallel,
                              parallel_lines(),
                              not_parallel(),
                              True)


        const_val = -(min_dist**2 - width_corr**2)

        # Create NonlinearConstraint object for capsule
        if name_stub is None:
            name = 'stat_ob{:}_capsule'.format(stat_ob_idx)
        else:
            name = name_stub+'_stat_ob{:}_capsule'.format(stat_ob_idx)

        # Define CasADi function for the constraint equation
        if not self.outdoor_robot_setting:
            capsule_func = cs.Function(name, [self.X], [const_val], ['input'], ['const']).expand()
        else:
            capsule_func = cs.Function(name, [cs.vertcat(self.X, self.stat_obs_params_vecced)], [const_val], ['input'], ['const']).expand()
        capsule_const = NonlinearConstraint(env=self,
                                            sym_cs_func=capsule_func,
                                            constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                            name=name)

        return [capsule_const]

    def get_mpc_stat_coll_constraints(self, joint_state):
        logging.debug('[MPCEnv] Generating collision avoidance constraints for static obstacles')
        coll_consts = []
        if self.max_speed * self.time_step >= self.radius:
            stat_buffer = self.max_speed * self.time_step - self.radius + 0.01
        else:
            stat_buffer = 0.0 + 0.05
        logging.debug('[MPCEnv] static buffer: {:}'.format(stat_buffer))
        comb_rad = self.radius + stat_buffer
        for j_idx in range(self.num_stat_obs):
            logging.debug('[MPCEnv] Generating constraints for static obstacle {:}'.format(j_idx))
            cons = self.get_stat_coll_const(comb_rad, j_idx)
            coll_consts.extend(cons)
        return coll_consts
        # NEW way 1 capsule instead of just circle for robot, and do collision checking between timesteps
        # comb_rad = self.radius + self.rob_stat_buffer
        # for j_idx in range(self.num_stat_obs):
        #     cons = self.get_stat_coll_const(comb_rad, j_idx)
        #     coll_consts.extend(cons)

        # NEW way 2 capsule and do collision checking between timesteps
        # for j_idx in range(self.num_stat_obs):
        #     cons = self.get_stat_coll_const_new(0.5*self.width+self.rob_stat_buffer, j_idx)
        #     coll_consts.extend(cons)

        # return coll_consts


    def get_coll_const(self, active_dims, comb_rad, name='coll_const_unspecified'):
        A_cj_mtx = np.array([[1, 0, -1,  0],
                             [0, 1,  0, -1]])
        P_mtx = - (A_cj_mtx.T @ A_cj_mtx)
        b = - comb_rad**2
        return QuadraticConstraint(env=self,
                                    constrained_variable=ConstrainedVariableType.STATE,
                                    P=P_mtx,
                                    b=b,
                                    strict=True,
                                    active_dims=active_dims,
                                    name=name)


    def get_rob_coll_capsule(self, humA_idx, name_stub=None):
        corrected_width = self.width + self.rob_wid_buffer
        corrected_buffer = self.rob_len_buffer - corrected_width/2.0
        p_1 = self.X[:2] - cs.vertcat(
            (0.5*self.length + corrected_buffer) * self.X[3],
            (0.5*self.length + corrected_buffer) * self.X[2]
            )
        p_2 = self.X[:2] + cs.vertcat(
            (0.5*self.length + corrected_buffer) * self.X[3],
            (0.5*self.length + corrected_buffer) * self.X[2]
            )
        dim_offset = self.nx_r+self.np_g # start index of other agent stats
        dim_len = self.nx_hum
        comb_rad = self.casadi_orca.agent_radii[humA_idx] +  0.5*corrected_width
        P = self.X[dim_offset+dim_len*humA_idx:dim_offset+dim_len*humA_idx+2]
        V = p_2 - p_1
        mag_V2 = V.T @ V
        def V_large():
            W =  P - p_1
            t = (W.T @ V) / mag_V2
            return cs.fmax(cs.fmin(t, 1.0), 0.0)

        t_clamped = cs.if_else(mag_V2 > 1e-3,
                               V_large(),
                               0.0,
                               True)

        closest_point = p_1 + t_clamped * V
        d_vec = (P - closest_point)
        const_val = -(d_vec.T @ d_vec - comb_rad**2)

        # Create NonlinearConstraint object for capsule
        if name_stub is None:
            name = 'rob_hum{:}_capsule'.format(humA_idx)
        else:
            name = name_stub+'_rob_hum{:}_capsule'.format(humA_idx)

        # Define CasADi function for the constraint equation
        capsule_func = cs.Function(name, [self.X], [const_val], ['input'], ['const'])
        capsule_const = NonlinearConstraint(env=self,
                                            sym_cs_func=capsule_func,
                                            constrained_variable=ConstrainedVariableType.STATE,
                                            name=name)

        return capsule_const


    def get_mpc_coll_constraints(self):
        # Constraints for collision avoidance with simulated humans
        # Old way, just circles
        coll_consts = []
        dim_offset = self.nx_r+self.np_g # start index of other agent stats
        dim_len = self.nx_hum
        for j_idx in range(self.num_hums):
            comb_rad = self.callback_orca.human_radii[j_idx] + self.radius + self.rob_len_buffer + 0.01
            # dims of [pxr, pyr, pxj, pyj]^T
            active_dims = [0, 1, dim_offset+dim_len*j_idx, dim_offset+dim_len*j_idx+1]
            name = 'coll_const_rob_hum{:}'.format(j_idx)
            con = self.get_coll_const(active_dims, comb_rad, name)
            coll_consts.append(con)


        # # NEW: use capsule instead of just circles
        # coll_consts = []

        # for j_idx in range(self.num_hums):
        #     coll_consts.append(self.get_rob_coll_capsule(j_idx))


        return coll_consts


    def get_term_constraints(self, b_term=None):
        # Terminal Constraint for convergence:
        active_dims = [0, 1, self.nx_r, self.nx_r+2]
        A_cpg_mtx = np.array([[1, 0, -1,  0],
                              [0, 1,  0, -1]])
        P_mtx_term = (A_cpg_mtx.T @ A_cpg_mtx)
        if b_term is None:
            b_term = 0.0


        term_con_leq = QuadraticConstraint(env=self,
                                       constrained_variable=ConstrainedVariableType.STATE,
                                       P=P_mtx_term,
                                       b=b_term,
                                       strict=True,
                                       active_dims=active_dims,
                                       name='term_con_leq')
        term_con_geq = QuadraticConstraint(env=self,
                                       constrained_variable=ConstrainedVariableType.STATE,
                                       P=-P_mtx_term,
                                       b=-b_term,
                                       strict=True,
                                       active_dims=active_dims,
                                       name='term_con_geq')
        term_consts = [term_con_leq, term_con_geq]
        return term_consts


    def get_kin_acc_constraint(self):
        XU_symb = cs.vertcat(self.X, self.U)

        assert self.max_l_dcc <= 0, 'max_l_dcc should be negative'
        assert self.max_l_acc >= 0, 'max_l_acc should be positive'

        # Old way, not smooth. Also limited deceleration, which we don't want to do.
        # # diff of mags is the total change in the magnitude of the velocity
        # # regardless of the robot's motion of travel,
        # # if diff_of_mags is positive then the robot is accelerating, if it is negative it is braking
        # diff_of_mags = cs.fabs(self.U[0]) - cs.fabs(self.X[3])

        # # we need the change in magnitude to be upper bounded by max(max_l_acc*dt) by the maximum acceleration
        # const_eqn_upper = diff_of_mags  - self.max_l_acc * self.time_step #<=0
        # # we need the change in magnitude to be lower bounded by the maximum braking, max_l_dcc*dt.
        # # But we also need to ensure that the direction of travel does not change within the timestep so we instead take the lowerbound to be,
        # lb = cs.fmax(self.max_l_dcc * self.time_step, -cs.fabs(self.X[3]))
        # const_eqn_lower = -diff_of_mags +  lb #<=0 i.e. diff_of_mags >= lb
        # # Finally ensure that the sign of the
        # sign_const = -cs.sign(self.X[3])*self.U[0] - 1e-5 #<=0
        # row_names = ['magv_change_upper', 'magv_change_lower', 'v_sign_const']
        # symcon_func = cs.Function('kin_acc_const', [XU_symb], [cs.vertcat(const_eqn_upper, const_eqn_lower, sign_const)], ['input'], ['con'])

        # New way, smooth, and allows for sudden braking all the way to zero
        # we want to say v <= (abs(v_prev) + max_acc*dt). But we smooth it and get rid of the abs completely by making it a quartic function

        inc_in_mags_fun = (self.U[0]**2 - self.X[4]**2 - (self.max_l_acc*self.time_step)**2)**2 - 4*self.X[4]**2*(self.max_l_acc*self.time_step)**2 # <=0

        decrease_in_mags_fun = (self.U[0] - self.X[4])**2 - cs.fmax((self.max_l_acc*self.time_step)**2-0.01, cs.fmin(self.X[4]**2, (-self.max_l_dcc*self.time_step)**2)) # <=0

        symcon_func = cs.Function('kin_acc_const', [XU_symb], [cs.vertcat(inc_in_mags_fun, decrease_in_mags_fun)], ['input'], ['con'])
        row_names = ['inc_in_mags_fun', 'decrease_in_mags_fun']
        acc_con = NonlinearConstraint(env=self,
                                    sym_cs_func=symcon_func,
                                    constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                    name='kin_acc_const',
                                    row_names=row_names)


        return [acc_con]


    def get_kin_bound_constraint(self):
        # Constraints for robot kinematic limits
        print('[MPCEnv] Adding kinematic bound constraints')
        bound_con = BoundedConstraint(env=self,
                                    lower_bounds=np.array([-self.max_rev_speed, -self.max_rot+0.001]),
                                    upper_bounds=np.array([self.max_speed, self.max_rot]),
                                    constrained_variable=ConstrainedVariableType.INPUT,
                                    active_dims=[0,1],
                                    strict=True,
                                    name='rob_act_bounds'
                                    )
        return bound_con


    def get_symcon_ORCA_humA_humB(self, X, U, humA_idx, humB_idx, casadi_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:2]
        ksi_humA = U_ksi_humA[2]
        if humB_idx == -1:
            # i.e. if we are dealing with the robot
            _, _, line_norm_checked, line_scalar_checked = self.casadi_orca.get_ORCA_pairwise_humrob(X, humA_idx=humA_idx, casadi_dict=casadi_dict)
        else:
            _, _, line_norm_checked, line_scalar_checked = self.casadi_orca.get_ORCA_pairwise_humhum(X, humA_idx=humA_idx, humB_idx=humB_idx, casadi_dict=casadi_dict)

        # rand_adjustment = 1 + self.rng.uniform(-5e-4, 5e-4)
        # const_eqn = - line_norm_checked.T @ (self.orca_vxy_scaling*U_humA) + line_scalar_checked - rand_adjustment*self.orca_ksi_scaling*ksi_humA #<= 0
        const_eqn = - line_norm_checked.T @ (self.orca_vxy_scaling*U_humA) + line_scalar_checked - self.orca_ksi_scaling*ksi_humA #<= 0
        if casadi_dict is not None:
            # casadi_dict['rand_adj'] = rand_adjustment
            ag_name = 'hum{:}'.format(humB_idx) if humB_idx >= 0 else 'rob'
            casadi_dict['const_fn_of_humA'] = cs.Function('pairwise_hum{:}_{:}'.format(humA_idx, ag_name),
                                                      [U_ksi_humA, X],
                                                      [const_eqn],
                                                      ['U_ksi_humA', 'X'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_stat_list(self, X, U, humA_idx, casadi_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:2]
        ksi_humA = U_ksi_humA[2]
        orca_con_list = []
        line_norms_stat, line_scalars_stat = self.casadi_orca.get_ORCA_stat_set_list(X, humA_idx, casadi_dict=casadi_dict)
        for idx in range(len(line_norms_stat)):
            const_eqn = -line_norms_stat[idx].T @ (self.orca_vxy_scaling*U_humA) + line_scalars_stat[idx]
            if casadi_dict is not None:
                agA_txt = 'hum{:}'.format(humA_idx) if humA_idx > -1 else 'rob'
                debug_text = '{:}_stat{:}'.format(agA_txt, idx)
                if debug_text not in casadi_dict.keys():
                    casadi_dict[debug_text] = {}
                casadi_dict_it = casadi_dict[debug_text]
                # casadi_dict_it['rand_adj'] = rand_adjustment
                if not self.outdoor_robot_setting:
                    casadi_dict_it['const_fn_of_humA'] = cs.Function('pairwise_hum{:}_stat{:}'.format(humA_idx, idx),
                                                                [U_ksi_humA, X],
                                                                [const_eqn],
                                                                ['U_ksi_humA', 'X'],
                                                                ['const'],
                                                                )
                else:
                    casadi_dict_it['const_fn_of_humA'] = cs.Function('pairwise_hum{:}_stat{:}'.format(humA_idx, idx),
                                                                [U_ksi_humA, X, self.stat_obs_params_vecced],
                                                                [const_eqn],
                                                                ['U_ksi_humA', 'X', 'stat_obs_params'],
                                                                ['const'],
                                                                )
            orca_con_list.append(const_eqn)
        return orca_con_list

    def get_symcon_ORCA_humA_maxvel(self, X, U, humA_idx, casadi_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:2]


        X_hums = X[self.nx_r+self.np_g:]
        X_humA = X_hums[humA_idx*self.nx_hum:(humA_idx+1)*self.nx_hum]
        vel_const_b = self.casadi_orca.v_max_prefs[humA_idx] ** 2

        const_eqn = (self.orca_vxy_scaling*U_humA[0])**2 + (self.orca_vxy_scaling*U_humA[1])**2 - vel_const_b
        if casadi_dict is not None:
            casadi_dict['maxvel_const_fn_of_humA'] = cs.Function('maxvel_hum{:}'.format(humA_idx),
                                                      [U_ksi_humA, X],
                                                      [const_eqn],
                                                      ['U_ksi_humA', 'X'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_maxacc(self, X, U, humA_idx, casadi_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:2]
        ksi_2_humA = U_ksi_humA[3]

        X_hums = X[self.nx_r+self.np_g:]
        X_humA = X_hums[self.nx_hum*humA_idx:(self.nx_hum*humA_idx+self.nx_hum)]

        v_current = X_humA[2:4]

        v_next = self.orca_vxy_scaling*U_humA[:2]
        delta_v = v_next - v_current
        const_eqn = delta_v.T @ delta_v - (self.time_step * self.max_hum_acc) ** 2 - self.orca_ksi_scaling*ksi_2_humA

        if casadi_dict is not None:
            casadi_dict['maxacc_const_fn_of_humA'] = cs.Function('maxacc_hum{:}'.format(humA_idx),
                                                      [U_ksi_humA, X],
                                                      [const_eqn],
                                                      ['U_ksi_humA', 'X'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_ksi(self, U, humA_idx, casadi_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:2]
        ksi_humA = U_ksi_humA[2]

        const_eqn = - self.orca_ksi_scaling*ksi_humA
        if casadi_dict is not None:
            # casadi_dict['rand_adj_vx'] = vx_coeff
            # casadi_dict['rand_adj_vy'] = vy_coeff
            casadi_dict['ksi_const_fn_of_humA'] = cs.Function('ksicon_hum{:}'.format(humA_idx),
                                                      [U_ksi_humA],
                                                      [const_eqn],
                                                      ['U_ksi_humA'],
                                                      ['const'],
                                                      )

        return const_eqn

    def get_symcon_ORCA_humA_ksi_2(self, U, humA_idx, casadi_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:2]
        ksi_humA = U_ksi_humA[2]
        ksi_2_humA = U_ksi_humA[3]

        const_eqn = - self.orca_ksi_scaling*ksi_2_humA
        if casadi_dict is not None:
            casadi_dict['ksi_2_const_fn_of_humA'] = cs.Function('ksi2con_hum{:}'.format(humA_idx),
                                                      [U_ksi_humA],
                                                      [const_eqn],
                                                      ['U_ksi_humA'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_kkt_const(self, X, U, humA_idx, hums_orca_consts_humA, hums_max_vel_const_humA, hums_ksi_const_humA, hums_max_acc_const_humA, hums_ksi_2_const_humA, stat_obs_params_vecced=None):
        if self.outdoor_robot_setting:
            XU_symb = cs.vertcat(X, U, stat_obs_params_vecced)
        else:
            XU_symb = cs.vertcat(X, U)
        Lam = U[self.nu_r+self.nVars_hums:]

        vars_humA = U[self.nu_r+humA_idx*self.nvars_hum:self.nu_r+(humA_idx+1)*self.nvars_hum]
        Lam_humA = Lam[(self.nlambda_hum)*humA_idx:(self.nlambda_hum)*(humA_idx+1)]

        v_pref = self.casadi_orca.get_v_pref_fromstate(humA_idx, X)
        cost_l = self.casadi_orca.cost_func(U_humA=self.orca_vxy_scaling*vars_humA[:2], U_humA_pref=v_pref)['l'] + self.casadi_orca.ksi_penal_func(ksi_humA=self.orca_ksi_scaling*vars_humA[2])['l'] + self.casadi_orca.ksi_2_penal_func(ksi_2_humA=self.orca_ksi_scaling*vars_humA[3])['l']

        const_g = cs.vertcat(*tuple([const.get_cs_fn()(input=XU_symb)['const'] for const in hums_orca_consts_humA]+[hums_max_vel_const_humA.get_cs_fn()(input=XU_symb)['const']]+[hums_max_acc_const_humA.get_cs_fn()(input=XU_symb)['const']]+[hums_ksi_const_humA.get_cs_fn()(input=U)['const']]+[hums_ksi_2_const_humA.get_cs_fn()(input=U)['const']]))
        const_g_names = list()
        const_g_names.extend(['primfeas_'+const.name[5:] for const in hums_orca_consts_humA]+['primfeas_'+hums_max_vel_const_humA.name]+['primfeas_'+hums_max_acc_const_humA.name]+['primfeas_'+hums_ksi_const_humA.name]+['primfeas_'+hums_ksi_2_const_humA.name])




        Lagr = cost_l + Lam_humA.T @ const_g

        grad_Lagr = cs.gradient(Lagr, vars_humA)
        ineq_con_eqn = cs.vertcat(const_g, -Lam_humA) # <=0
        eq_con_eqn = cs.vertcat(grad_Lagr, Lam_humA * const_g) # ==0

        # All orca kkt conds as one inequality constraint + one equality constraint
        ineq_names = const_g_names + \
                    ['dual_feas_{:}'.format(i) for i in range(Lam_humA.shape[0])]

        eq_names = ['eq_grad_Lagr_{:}'.format(i) for i in range(vars_humA.shape[0])] + \
                   ['eq_comp_slack_{:}'.format(i) for i in range(Lam_humA.shape[0])]


        return ineq_con_eqn, eq_con_eqn, ineq_names, eq_names


    def get_human_constraints(self):
        """Generates the human constraints based on the configured human model

        :return: lists of human constraints
        """
        print('[MPCEnv] Getting human constraints, human model = {:}'.format(self.hum_model))
        hums_orca_consts = []
        hums_orca_consts_list = []
        hums_max_vel_consts = []
        hums_max_acc_consts = []
        hums_ksi_consts = []
        hums_ksi_2_consts = []

        if self.hum_model == 'orca_casadi_simple' or self.hum_model == 'orca_casadi_kkt':
            print('[MPCEnv] Generating ORCA constraints')
            if self.outdoor_robot_setting:
                XU_symb = cs.vertcat(self.X, self.U, self.stat_obs_params_vecced)
            else:
                XU_symb = cs.vertcat(self.X, self.U)
            casadi_dicts = {}
            self.casadi_dicts = casadi_dicts
            self.debug_dicts = casadi_dicts
            for humA_idx in range(self.num_hums):
                humA_casadi_dict = {'X' : self.X}
                casadi_dicts['hum{:}'.format(humA_idx)] = humA_casadi_dict
                # get all the pairwise orca constraints
                humA_orca_consts = []
                for humB_idx in self.casadi_orca.humB_idcs_list[humA_idx]:
                    if humB_idx == humA_idx:
                        print('[MPCEnv] humB_idx == humA_idx in orca constraints list. Agent does not need to avoid itself! Skipping.')
                        continue
                    humA_pairwise_casadi_dict = {'X' : self.X}
                    debug_text = 'hum{:}_hum{:}'.format(humA_idx, humB_idx) if humB_idx>=0 else 'hum{:}_rob'.format(humA_idx, humB_idx)
                    humA_casadi_dict[debug_text] = humA_pairwise_casadi_dict
                    const_eqn = self.get_symcon_ORCA_humA_humB(self.X, self.U, humA_idx, humB_idx, casadi_dict=humA_pairwise_casadi_dict)
                    const_func_name = 'orca_hum{:}_hum{:}_const'.format(humA_idx, humB_idx) if humB_idx != -1 else 'orca_hum{:}_rob_const'.format(humA_idx)
                    symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                    orca_con = NonlinearConstraint(env=self,
                                                    sym_cs_func=symcon_func,
                                                    constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                    name=const_func_name,
                                                    casadi_dict=humA_pairwise_casadi_dict)
                    hums_orca_consts.append(orca_con)
                    humA_orca_consts.append(orca_con)


                # static obs constraints

                stat_const_eqn_list = self.get_symcon_ORCA_humA_stat_list(self.X, self.U, humA_idx, casadi_dict=humA_casadi_dict)
                for s_idx, const_eqn in enumerate(stat_const_eqn_list):
                    const_func_name = 'orca_hum{:}_stat{:}_const'.format(humA_idx, s_idx)
                    if not self.outdoor_robot_setting:
                        symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                    else:
                        symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                    orca_con = NonlinearConstraint(env=self,
                                                   sym_cs_func=symcon_func,
                                                   constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                   name=const_func_name,
                                                   casadi_dict=humA_casadi_dict['hum{:}_stat{:}'.format(humA_idx, s_idx)])
                    hums_orca_consts.append(orca_con)
                    humA_orca_consts.append(orca_con)

                hums_orca_consts_list.append(humA_orca_consts)


                const_func_name = 'orca_hum{:}_vel_const'.format(humA_idx)
                const_eqn = self.get_symcon_ORCA_humA_maxvel(self.X, self.U, humA_idx, casadi_dict=humA_casadi_dict)
                symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                maxvel_con = NonlinearConstraint(env=self,
                                                sym_cs_func=symcon_func,
                                                constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                name=const_func_name,
                                                casadi_dict=humA_casadi_dict)
                hums_max_vel_consts.append(maxvel_con)

                const_func_name = 'orca_hum{:}_acc_const'.format(humA_idx)
                const_eqn = self.get_symcon_ORCA_humA_maxacc(self.X, self.U, humA_idx, casadi_dict=humA_casadi_dict)
                symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                maxacc_con = NonlinearConstraint(env=self,
                                                sym_cs_func=symcon_func,
                                                constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                name=const_func_name,
                                                casadi_dict=humA_casadi_dict)
                hums_max_acc_consts.append(maxacc_con)

                const_func_name = 'ksi_con_hum{:}'.format(humA_idx)
                const_eqn = self.get_symcon_ORCA_humA_ksi(self.U, humA_idx, casadi_dict=humA_casadi_dict)
                symcon_func = cs.Function(const_func_name, [self.U], [const_eqn], ['input'], ['const'])
                ksi_con = NonlinearConstraint(env=self,
                                                sym_cs_func=symcon_func,
                                                constrained_variable=ConstrainedVariableType.INPUT,
                                                name=const_func_name,
                                                casadi_dict=humA_casadi_dict)
                hums_ksi_consts.append(ksi_con)

                const_func_name = 'ksi_2_con_hum{:}'.format(humA_idx)
                const_eqn = self.get_symcon_ORCA_humA_ksi_2(self.U, humA_idx, casadi_dict=humA_casadi_dict)
                symcon_func = cs.Function(const_func_name, [self.U], [const_eqn], ['input'], ['const'])
                ksi_2_con = NonlinearConstraint(env=self,
                                                sym_cs_func=symcon_func,
                                                constrained_variable=ConstrainedVariableType.INPUT,
                                                name=const_func_name,
                                                casadi_dict=humA_casadi_dict)
                hums_ksi_2_consts.append(ksi_2_con)

        if self.hum_model == 'orca_casadi_kkt':
            print('[MPCEnv] Generating ORCA KKT constraints')
            hum_kkt_ineq_consts = []
            hum_kkt_eq_consts = []
            for humA_idx in range(self.num_hums):
                ineq_con_eqn, eq_con_eqn, ineq_names, eq_names = self.get_symcon_ORCA_humA_kkt_const(self.X, self.U, humA_idx, hums_orca_consts_list[humA_idx], hums_max_vel_consts[humA_idx], hums_ksi_consts[humA_idx], hums_max_acc_consts[humA_idx], hums_ksi_2_consts[humA_idx], self.stat_obs_params_vecced)

                ineq_const_func_name = 'orca_kkt_hum{:}_ineq_con_func'.format(humA_idx)
                ineq_const_func = cs.Function(ineq_const_func_name, [XU_symb], [ineq_con_eqn], ['input'], ['const'])
                ineq_kkt_con = NonlinearConstraint(env=self,
                                                   sym_cs_func=ineq_const_func,
                                                   constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                   name=ineq_const_func_name,
                                                   row_names=ineq_names)
                hum_kkt_ineq_consts.append(ineq_kkt_con)

                eq_const_func_name = 'orca_kkt_hum{:}_eq_con_func'.format(humA_idx)
                eq_const_func = cs.Function(eq_const_func_name, [XU_symb], [eq_con_eqn], ['input'], ['const'])
                eq_kkt_con = NonlinearConstraint(env=self,
                                                 sym_cs_func=eq_const_func,
                                                 constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                 name=eq_const_func_name,
                                                 row_names=eq_names,
                                                 contype=ConstraintType.EQ
                                                 )
                hum_kkt_eq_consts.append(eq_kkt_con)
            hum_numstab_consts = []

            print('[MPCEnv] No. hums_orca_consts_list: {}, No. hums_max_vel_consts: {}, No. hum_numstab_consts: {}'.format(len(hums_orca_consts_list), len(hums_max_vel_consts), len(hum_numstab_consts)))

            # NB orca primal feasibilty constraints must be included in the kkt consts
            return hums_orca_consts_list, hums_max_vel_consts, hums_max_acc_consts, hums_ksi_consts, hums_ksi_2_consts, hum_kkt_ineq_consts, hum_kkt_eq_consts, hum_numstab_consts



        print('[MPCEnv] No. hums_orca_consts_list: {}, No. hums_max_vel_consts: {}'.format(len(hums_orca_consts_list), len(hums_max_vel_consts)))
        return hums_orca_consts, hums_max_vel_consts, hums_max_acc_consts, hums_ksi_consts, hums_ksi_2_consts, []

    def get_hums_close_to_preds_constraints(self, X, U):
        """Generates soft constraints on the humans orca solutions in the bilevel mpc so that the ORCA solutions stay close to the predicted samples. This is to ensure that the ORCA solutions are as close as feasible to the predicted samples.

        :return: lists of human constraints
        """
        X_hums_sym = X[self.nx_r+self.np_g:]
        split_up_U = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_hums_sym = cs.vertcat(*tuple([split_up_U[1+hum_idx] for hum_idx in range(self.num_hums)]))

        MID_samples_allhums_t_sym = cs.MX.sym('MID_samples_allhums_t_sym', self.num_MID_samples, 2*self.num_hums)

        # find the maximum distance between any two samples
        closest_samp_dists_list = []
        for s_idx in range(self.num_MID_samples):
            sq_dists_list = []
            for s2_idx in range(self.num_MID_samples):
                if s_idx == s2_idx:
                    continue
                sq_diffs = (MID_samples_allhums_t_sym[s_idx, :] - MID_samples_allhums_t_sym[s2_idx, :]) ** 2
                sq_dist_list = []
                for hum_idx in range(self.num_hums):
                    sq_dist_list.append(sq_diffs[2*hum_idx] + sq_diffs[2*hum_idx+1])
                sq_dists_list.append(cs.vertcat(*tuple(sq_dist_list)))
            sq_dists = cs.horzcat(*tuple(sq_dists_list))
            per_human_min_list = []
            for hum_idx in range(self.num_hums):
                per_human_min_list.append(cs.mmin(sq_dists[hum_idx, :]))
            closest_samp_dists_list.append(cs.vertcat(*tuple(per_human_min_list)))
        closest_samp_dists_prefunc = cs.horzcat(*tuple(closest_samp_dists_list))
        # intermediate function
        max_sq_dists_fn = cs.Function('max_sq_dists_fn', [MID_samples_allhums_t_sym], [closest_samp_dists_prefunc], ['MID_samples_allhums_t_sym'], ['max_sq_dists']).expand()

        closest_samp_dists = max_sq_dists_fn(MID_samples_allhums_t_sym)

        max_sq_dist_between_pair_of_samples_list = []
        for hum_idx in range(self.num_hums):
            potential_max = cs.mmax(closest_samp_dists[hum_idx, :])
            # if all the samples are dummy samples and identical, then the max distance will be zero. In this case, set it to 5.0m to allow ORCA to deviate from the samples.
            max_dist = cs.if_else(potential_max > 1e-3,
                       potential_max,
                       10.0,
                       True)
            max_sq_dist_between_pair_of_samples_list.append(max_dist)

        max_sq_dist_between_pair_of_samples_prefunc = cs.vertcat(*tuple(max_sq_dist_between_pair_of_samples_list)) # shape (num_hums, 1)

        max_sq_dists_between_pair_fn = cs.Function('max_sq_dists_between_pair_fn', [MID_samples_allhums_t_sym], [max_sq_dist_between_pair_of_samples_prefunc], ['MID_samples_allhums_t_sym'], ['max_sq_dists_between_pair']).expand()

        max_sq_dist_between_pair_of_samples = max_sq_dists_between_pair_fn(MID_samples_allhums_t_sym)

        # now find the distance between the ORCA solution and the predicted samples
        next_P_hums_list = []
        for humA_idx_sym in range(self.num_hums):
            idx_x = humA_idx_sym*self.nx_hum
            idx_u = humA_idx_sym*self.nvars_hum
            idx_x_next = (humA_idx_sym+1)*self.nx_hum
            X_humA_sym2 = X_hums_sym[self.nx_hum*humA_idx_sym:(self.nx_hum*humA_idx_sym+self.nx_hum)]
            U_humA_sym2 = U_hums_sym[idx_u:idx_u+self.nvars_hum]
            next_P_humj_sym3 = self.get_next_P_humj_cs_func(X_humA_sym2, U_humA_sym2)
            next_P_hums_list.append(next_P_humj_sym3)

        next_P_hums_sym = cs.vertcat(*tuple(next_P_hums_list))
        sq_dists_t = (MID_samples_allhums_t_sym.T - next_P_hums_sym)**2
        sq_dist_to_closest_sample_list = []
        for hum_idx in range(self.num_hums):
            sq_dist_to_closest_sample_list.append(cs.mmin(sq_dists_t[2*hum_idx, :] + sq_dists_t[2*hum_idx+1, :]))

        sq_dists_to_closest_sample = cs.vertcat(*tuple(sq_dist_to_closest_sample_list))

        const_eqn = sq_dists_to_closest_sample - 0.5 * max_sq_dist_between_pair_of_samples # <= 0

        XU_symb = cs.vertcat(X, U)

        const_func = cs.Function('hums_close_to_preds', [XU_symb, MID_samples_allhums_t_sym], [const_eqn], ['XU', 'MID_samples_allhums_t_sym'], ['const']).expand()

        return const_func