from copy import deepcopy
import logging
from math import ceil
from re import M
from unicodedata import name
import numpy as np
import casadi as cs
from sicnav.utils.mpc_utils.orca_callback import callbackORCA
from sicnav.utils.mpc_utils.orca_casadi import casadiORCA
from sicnav.utils.mpc_utils.system_model import SystemModel
from sicnav.utils.mpc_utils.constraints import BoundedConstraint, LinearConstraint, NonlinearConstraint, QuadraticConstraint, ConstrainedVariableType, ConstraintType

class MPCEnv(object):

    def __init__(self, time_step, joint_state, num_hums, K, dummy_human, config=None):
        self.dummy_human = dummy_human
        self.time_step = time_step

        # constrain the robot to not get within this buffer distance to the other agents
        self.rob_rad_buffer = 0.0

        # model mpc uses for humans can be 'cvmm' (mpc-cvmm baseline), 'orca_casadi_kkt' (our sicnav-*)
        self.hum_model = 'orca_casadi_kkt'

        # horizon that we use the kkt reformulation, after this, we would use cvmm
        self.horiz = K
        # if human model is orca_casadi_kkt, the subset of the horizon where ORCA constraints are used to model humans
        self.orca_kkt_horiz = 0

        # pref speed used for generatig refernce and warmstart
        self.pref_speed = 0.95

        # robot physical constraints
        self.max_speed = 1.0
        self.max_rot = np.pi/4
        self.max_l_acc = 1.0
        self.max_l_dcc = -1.0

        if config is not None:
            self.configure(config)
        else:
            logging.info('[MPCEnv] No config. file, using default values')

        if self.max_l_dcc > -self.max_speed / (time_step*max(1,ceil(K*0.5))):
            logging.info('[MPCEnv] Stopping from max speed will take more than 0.5K, max decc = {:}'.format(self.max_l_dcc))
        if self.max_l_dcc > -self.max_speed / (time_step*max(1,ceil(K))):
            logging.info('[MPCEnv] POTENTIAL PROBLEM Stopping from max speed will take more than K, max decc = {:}'.format(self.max_l_dcc))
        # other agent state attributes
        self.num_hums = num_hums
        # self.nx_hum = 4
        self.num_stat_obs = len(joint_state.static_obs)
        self.env_static_obs = deepcopy(joint_state.static_obs)

        # if the hums are inputs to the MPC optimization, this will get overwritten
        if self.hum_model == 'cvmm':
            self.nx_hum = 6 # pos, vel, goal_pos
            self.nvars_hum = 2
            self.nVars_hums = 0
            self.nlambda_hum = 0
            self.nLambda = 0
        elif self.hum_model == 'orca_casadi_kkt':
            self.nx_hum = 6 # pos, vel, goal_pos
            self.nvars_hum = 3 # vx, vy, ksi <- for relaxed orca
            self.nVars_hums = self.nvars_hum * self.num_hums
            self.nlambda_hum = self.num_hums + self.num_stat_obs + 2 # also counts the lambdas for velocity + the relaxation slack variable for the relaxed orca
            self.nLambda = self.num_hums * self.nlambda_hum # one for each human
            self.rng = np.random.default_rng(5)
        else:
            raise NotImplementedError("Only cvmm (for mpc-cvmm) and orca_casadi_kkt (for sicnav-*) are supported for human model.")

        nX_hums = self.nx_hum * num_hums
        self.nX_hums = nX_hums

        nx_r, np_g = 4, 2
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
        # callback which calls o.g. rvo2 optimization
        self.callback_orca = callbackORCA('callback_orca', self.time_step, joint_state, self.nx_r, self.np_g, self.nX_hums, self.dummy_human.policy, {'enable_fd':True})

        # dynamics model of the environment
        X, U, next_X = self.gen_kin_model(joint_state)

        dynamics = {"dyn_eqn": next_X, "obs_eqn": None, "vars": {"X": X, "U": U}}

        # update final dimensions given dynamics type
        nx = self.nx_r + self.np_g + nX_hums
        self.state_dim = self.nx = nx
        self.action_dim = self.nu = self.nu_r + self.nVars_hums + self.nLambda


        # MPC cost parameters:
        term_q_coeff = 100.0
        self.Q = cs.sparsify(np.diag(np.hstack([np.ones(2), np.zeros(self.nx_r-2+self.np_g+self.nx_hum*self.num_hums)])))
        self.term_Q = cs.sparsify(np.diag(np.hstack([term_q_coeff*np.ones(2), np.zeros(self.nx_r-2+self.np_g+self.nx_hum*self.num_hums)])))
        self.R = cs.sparsify(np.diag(np.hstack([np.zeros(1), np.ones(1), np.zeros(self.nu-self.nu_r)]))*0.1)

        # From intialization:
        self.init_joint_state = joint_state

        # robot attributes
        self.radius = joint_state.self_state.radius


        # Defining the elements of the MPC cost function
        cost = self.get_cost_eqn(X, U)
        self.system_model = SystemModel(dynamics=dynamics, cost=cost, dt=self.time_step, linearize=False)

        # State constraints
        coll_consts = self.get_mpc_coll_constraints()
        stat_coll_consts = self.get_mpc_stat_coll_constraints(joint_state)
        # coll_consts += stat_coll_consts

        # Terminal constraints
        term_consts = self.get_term_constraints()

        bound_con = self.get_kin_bound_constraint()
        # Input and action constraint
        acc_cons = self.get_kin_acc_constraint()
        bound_cons_state = []

        # Constraints on human (if to be included in MPC model)
        if self.hum_model == 'orca_casadi_kkt':
            if self.orca_kkt_horiz == 0:
                self.orca_kkt_horiz = self.horiz
            self.hums_orca_consts, self.hums_max_vel_consts, self.hums_ksi_consts, self.hum_orca_kkt_ineq_consts, self.hum_orca_kkt_eq_consts, self.hum_numstab_consts = self.get_human_constraints()
            hums_orca_consts, hums_max_vel_consts, hums_ksi_consts, hum_numstab_consts = [], [], [], [] # NB, requires that the constraints be included in the kkt constraints in this case.
        else:
            hums_orca_consts, hums_max_vel_consts, hums_ksi_consts, hum_numstab_consts = self.get_human_constraints()

        self.state_constraints  = [con for con in coll_consts] + bound_cons_state
        self.state_constraints_sym = [con.get_symbolic_model() for con in self.state_constraints]
        self.stat_coll_consts = stat_coll_consts
        self.stat_coll_consts_sym = [con.get_symbolic_model() for con in self.stat_coll_consts]

        self.term_constraints  = [con for con in term_consts]
        self.term_constraints_sym = [con.get_symbolic_model() for con in self.term_constraints]

        self.input_constraints  = [bound_con] + [con for con in hums_max_vel_consts+hums_ksi_consts]
        self.input_constraints_sym = [con.get_symbolic_model() for con in self.input_constraints]

        self.input_state_constraints  = [acc_con for acc_con in acc_cons] + [con for con in hums_orca_consts]
        self.input_state_constraints_sym = [con.get_symbolic_model() for con in self.input_state_constraints]

        self.numstab_state_consts = [con for con in hum_numstab_consts]
        self.numstab_state_consts_sym = [con.get_symbolic_model() for con in self.numstab_state_consts]


    def configure(self, config):
        self.hum_model = config.get('mpc_env', 'hum_model')
        self.orca_kkt_horiz = config.getint('mpc_env', 'orca_kkt_horiz')
        self.rob_rad_buffer = config.getfloat('mpc_env', 'rob_rad_buffer')
        self.pref_speed =  config.getfloat('mpc_env', 'pref_speed')
        self.max_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rev_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rot = config.getfloat('mpc_env', 'max_rot_degrees') * np.pi / 180.0
        self.max_l_acc = config.getfloat('mpc_env', 'max_l_acc')
        self.max_l_dcc = config.getfloat('mpc_env', 'max_l_dcc')

        self.orca_ksi_scaling = config.getfloat('mpc_env', 'orca_ksi_scaling')
        self.orca_vxy_scaling = config.getfloat('mpc_env', 'orca_vxy_scaling')
        self.human_max_speed = config.getfloat('mpc_env', 'human_v_max_assumption')

        logging.info('[MPCEnv] Config {:} = {:}'.format('hum_model', self.hum_model))
        logging.info('[MPCEnv] Config {:} = {:}'.format('orca_kkt_horiz', self.orca_kkt_horiz))
        logging.info('[MPCEnv] Config {:} = {:}'.format('rob_rad_buffer', self.rob_rad_buffer))
        logging.info('[MPCEnv] Config {:} = {:}'.format('pref_speed', self.pref_speed))
        logging.info('[MPCEnv] Config {:} = {:}'.format('max_speed', self.max_speed))
        logging.info('[MPCEnv] Config {:} = {:}'.format('max_rev_speed', self.max_rev_speed))
        logging.info('[MPCEnv] Config {:} = {:}'.format('max_rot', self.max_rot))
        logging.info('[MPCEnv] Config {:} = {:}'.format('max_l_acc', self.max_l_acc))
        logging.info('[MPCEnv] Config {:} = {:}'.format('max_l_dcc', self.max_l_dcc))
        logging.info('[MPCEnv] Config {:} = {:}'.format('orca_ksi_scaling', self.orca_ksi_scaling))
        logging.info('[MPCEnv] Config {:} = {:}'.format('orca_vxy_scaling', self.orca_vxy_scaling))


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
        Q = cs.MX.sym('Q', self.nx, self.nx)
        R = cs.MX.sym('R', self.nu, self.nu)
        cost_eqn = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)

        # if the human objective is to be included in MPC objective
        cost_eqn = self.get_human_cost_eqn(X, U, cost_eqn)

        cost = {"cost_eqn": cost_eqn, "vars": {"X": X, "U": U, "Xr": Xr, "Ur": Ur, "Q": Q, "R": R}}
        return cost


    def get_human_cost_eqn(self, X, U, cost_eqn):
        """Augment the MPC cost with the cost for humans' objectives, if to be included

        :param X: symbolic variable for current state of the environment
        :param U: symbolic variable for the input to the system
        :param cost_eqn: symbolic equation for the cost to be autmented
        :return: symbolic equation for the augmented cost function
        """
        return cost_eqn


    def get_human_dyn_eqn(self, X, X_hums, U_rob):
        """Dynamics equation for the human agents

        :param X: symbolic variable for state of environment
        :param X_hums: symbolic variable for state of human agents (included in X)
        :param U_rob: symbolic variable for input to the robot
        :return: next_X_hums (equation to dynamics of robot), U (input to entire system)
        """
        if self.hum_model == 'cvmm':
            next_X_hums = self.get_CVMM_human_eqn(X_hums)
            U = U_rob
        elif self.hum_model == 'stationary':
            next_X_hums = X_hums
            U = U_rob
        elif self.hum_model == 'orca_casadi_kkt':
            next_X_hums, U_temp = self.get_ORCA_human_dynamics_eqn(X_hums, U_rob)
            lambdas_list = []
            for humA_idx in range(self.num_hums):
                lambda_humA = cs.MX.sym('lambda_hum{:}'.format(humA_idx), self.nlambda_hum, 1)
                lambdas_list.append(lambda_humA)
            lambdas = cs.vertcat(*tuple(lambdas_list))
            self.lambdas_list = lambdas_list
            U = cs.vertcat(U_temp, lambdas)

        return next_X_hums, U



    def get_ORCA_human_dynamics_eqn(self, X_hums, U_rob):
        """Get equation of next human dynamics when human input is solved by the optimizer

        :param X_hums: symbolic variable for current state of the humans
        :param U_rob: symbolic variable for the input to the robot
        :return: next_X_hums (equation to the dynamics of the next humans), U (inputs to the entire system)
        """
        nu_hum = 2
        nksi_hum = 1
        self.nvars_hum = nu_hum + nksi_hum
        self.nVars_hums = self.nvars_hum * self.num_hums
        U_hums_list = []
        for humA_idx in range(self.num_hums):
            U_ksi_humA = cs.MX.sym('U_ksi_hum{:}'.format(humA_idx), self.nvars_hum, 1)
            U_hums_list.append(U_ksi_humA)
        self.U_hums_list = U_hums_list
        U_hums = cs.vertcat(*tuple(U_hums_list))
        idx_x = 0
        idx_u = 0
        idx_x_next = (0+1)*self.nx_hum
        next_X_hums = cs.vertcat(X_hums[idx_x] + self.orca_vxy_scaling*U_hums[idx_u] * self.time_step,
                                 X_hums[idx_x+1] + self.orca_vxy_scaling*U_hums[idx_u+1] * self.time_step,
                                 self.orca_vxy_scaling*U_hums[idx_u],
                                 self.orca_vxy_scaling*U_hums[idx_u+1],
                                 X_hums[idx_x_next-2], # human goal posns
                                 X_hums[idx_x_next-1]) # human goal posns
        for j in range(1, self.num_hums):
            idx_x = j*self.nx_hum
            idx_u = j*self.nvars_hum
            idx_x_next = (j+1)*self.nx_hum
            next_X_hums = cs.vertcat(next_X_hums,
                                     X_hums[idx_x] + self.orca_vxy_scaling*U_hums[idx_u] * self.time_step,
                                     X_hums[idx_x+1] + self.orca_vxy_scaling*U_hums[idx_u+1] * self.time_step,
                                     self.orca_vxy_scaling*U_hums[idx_u],
                                     self.orca_vxy_scaling*U_hums[idx_u+1],
                                     X_hums[idx_x_next-2], # human goal posns
                                     X_hums[idx_x_next-1]) # human goal posns

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
        th_r = cs.MX.sym('theta_r')
        v_r_prev = cs.MX.sym('v_r_prev')
        X_r = cs.vertcat(x_r, y_r, th_r, v_r_prev)

        # The goal position
        P_g = cs.MX.sym('P_goal', self.np_g)

        # the human agent positions and velocities
        X_hums = cs.MX.sym('X_hums', self.nX_hums)

        # the symbolic joint state
        X = cs.vertcat(X_r, P_g, X_hums)

        # the actions
        v_r = cs.MX.sym('v_r')
        om_r = cs.MX.sym('om_r')
        U = cs.vertcat(v_r, om_r)

        # Defining discrete-time dynamics equations
        # the robot's dynamics
        next_x_r = x_r + self.time_step * v_r * cs.cos(th_r + self.time_step * om_r)
        next_y_r = y_r + self.time_step * v_r * cs.sin(th_r + self.time_step * om_r)
        next_th_r = th_r + self.time_step * om_r
        next_v_r_prev = v_r
        next_X_r = cs.vertcat(next_x_r, next_y_r, next_th_r, next_v_r_prev)

        # the goal position next
        next_P_g = P_g

        # the next step dynamics for the orca agents
        self.casadi_orca = casadiORCA(self, joint_state, X)
        next_X_hums, U = self.get_human_dyn_eqn(X, X_hums, U)

        # concatenating the next_X
        next_X = cs.vertcat(next_X_r, next_P_g, next_X_hums)

        # Setting object-level symbolic variables
        self.X = X
        self.U = U
        return X, U, next_X




    def get_stat_coll_const(self, comb_rad, stat_ob_idx, name_stub=None):
        stat_ob = self.env_static_obs[stat_ob_idx]
        p_1 = cs.DM(stat_ob[0]).reshape((2,1))
        p_2 = cs.DM(stat_ob[1]).reshape((2,1))

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
        capsule_func = cs.Function(name, [self.X], [const_val], ['input'], ['const'])
        capsule_const = NonlinearConstraint(env=self,
                                            sym_cs_func=capsule_func,
                                            constrained_variable=ConstrainedVariableType.STATE,
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


    def get_mpc_coll_constraints(self):
        # Constraints for collision avoidance with simulated humans
        coll_consts = []
        dim_offset = self.nx_r+self.np_g # start index of other agent stats
        dim_len = self.nx_hum
        for j_idx in range(self.num_hums):
            comb_rad = self.callback_orca.human_radii[j_idx] + self.radius + self.rob_rad_buffer + 0.01
            # dims of [pxr, pyr, pxj, pyj]^T
            active_dims = [0, 1, dim_offset+dim_len*j_idx, dim_offset+dim_len*j_idx+1]
            name = 'coll_const_rob_hum{:}'.format(j_idx)
            con = self.get_coll_const(active_dims, comb_rad, name)
            coll_consts.append(con)

        return coll_consts


    def get_term_constraints(self, b_term=None):
        # Terminal Constraint for MPC (untested)
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
        # upper bound on change in magnitude
        XU_symb = cs.vertcat(self.X, self.U)

        assert self.max_l_dcc <= 0, 'max_l_dcc should be negative'
        assert self.max_l_acc >= 0, 'max_l_acc should be positive'

        # diff of mags is the total change in the magnitude of the velocity
        # regardless of the robot's motion of travel,
        # if diff_of_mags is positive then the robot is accelerating, if it is negative it is braking
        diff_of_mags = cs.fabs(self.U[0]) - cs.fabs(self.X[3])

        # we need the change in magnitude to be upper bounded by max(max_l_acc*dt) by the maximum acceleration
        const_eqn_upper = diff_of_mags  - self.max_l_acc * self.time_step #<=0
        # we need the change in magnitude to be lower bounded by the maximum braking, max_l_dcc*dt.
        # But we also need to ensure that the direction of travel does not change within the timestep so we instead take the lowerbound to be,
        lb = cs.fmax(self.max_l_dcc * self.time_step, -cs.fabs(self.X[3]))
        const_eqn_lower = -diff_of_mags +  lb #<=0 i.e. diff_of_mags >= lb
        # Finally ensure that the sign of the
        sign_const = -cs.sign(self.X[3])*self.U[0] - 1e-5 #<=0
        row_names = ['magv_change_upper', 'magv_change_lower', 'v_sign_const']
        symcon_func = cs.Function('kin_acc_const', [XU_symb], [cs.vertcat(const_eqn_upper, const_eqn_lower, sign_const)], ['input'], ['con'])

        acc_con = NonlinearConstraint(env=self,
                                    sym_cs_func=symcon_func,
                                    constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                    name='kin_acc_const',
                                    row_names=row_names)


        return [acc_con]


    def get_kin_bound_constraint(self):
        # Constraints for robot kinematic limits
        logging.info('[MPCEnv] Adding kinematic bound constraints')
        bound_con = BoundedConstraint(env=self,
                                    lower_bounds=np.array([-self.max_rev_speed, -self.max_rot+0.001]),
                                    upper_bounds=np.array([self.max_speed, self.max_rot]),
                                    constrained_variable=ConstrainedVariableType.INPUT,
                                    active_dims=[0,1],
                                    strict=True,
                                    name='rob_act_bounds'
                                    )
        return bound_con



    def get_symcon_ORCA_humA_humB(self, X, U, humA_idx, humB_idx, debug_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:-1]
        ksi_humA = U_ksi_humA[-1]
        if humB_idx == -1:
            # i.e. if we are dealing with the robot
            _, _, line_norm_checked, line_scalar_checked = self.casadi_orca.get_ORCA_pairwise_humrob(X, humA_idx=humA_idx, debug_dict=debug_dict)
        else:
            _, _, line_norm_checked, line_scalar_checked = self.casadi_orca.get_ORCA_pairwise_humhum(X, humA_idx=humA_idx, humB_idx=humB_idx, debug_dict=debug_dict)

        rand_adjustment = 1 + self.rng.uniform(-5e-4, 5e-4)
        const_eqn = - line_norm_checked.T @ (self.orca_vxy_scaling*U_humA) + line_scalar_checked - rand_adjustment*self.orca_ksi_scaling*ksi_humA #<= 0
        if debug_dict is not None:
            debug_dict['rand_adj'] = rand_adjustment
            ag_name = 'hum{:}'.format(humB_idx) if humB_idx >= 0 else 'rob'
            debug_dict['const_fn_of_humA'] = cs.Function('pairwise_hum{:}_{:}'.format(humA_idx, ag_name),
                                                      [U_ksi_humA, X],
                                                      [const_eqn],
                                                      ['U_ksi_humA', 'X'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_stat_list(self, X, U, humA_idx, debug_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:-1]
        ksi_humA = U_ksi_humA[-1]
        orca_con_list = []
        line_norms_stat, line_scalars_stat = self.casadi_orca.get_ORCA_stat_set_list(X, humA_idx, debug_dict=debug_dict)
        for idx in range(len(line_norms_stat)):
            rand_adjustment = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(1e-4, 9e-4)
            const_eqn = -line_norms_stat[idx].T @ (self.orca_vxy_scaling*U_humA) + line_scalars_stat[idx] - rand_adjustment*self.orca_ksi_scaling*ksi_humA
            if debug_dict is not None:
                agA_txt = 'hum{:}'.format(humA_idx) if humA_idx > -1 else 'rob'
                debug_text = '{:}_stat{:}'.format(agA_txt, idx)
                if debug_text not in debug_dict.keys():
                    debug_dict[debug_text] = {}
                debug_dict_it = debug_dict[debug_text]
                debug_dict_it['rand_adj'] = rand_adjustment
                debug_dict_it['const_fn_of_humA'] = cs.Function('pairwise_hum{:}_stat{:}'.format(humA_idx, idx),
                                                             [U_ksi_humA, X],
                                                             [const_eqn],
                                                             ['U_ksi_humA', 'X'],
                                                             ['const'],
                                                            )
            orca_con_list.append(const_eqn)
        return orca_con_list


    def get_symcon_ORCA_humA_maxvel(self, U, humA_idx, debug_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:-1]

        vel_const_b = self.casadi_orca.v_max_prefs[humA_idx] ** 2

        const_eqn = (self.orca_vxy_scaling*U_humA[0])**2 + (self.orca_vxy_scaling*U_humA[1])**2 - vel_const_b
        if debug_dict is not None:
            debug_dict['maxvel_const_fn_of_humA'] = cs.Function('maxvel_hum{:}'.format(humA_idx),
                                                      [U_ksi_humA],
                                                      [const_eqn],
                                                      ['U_ksi_humA'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_ksi(self, U, humA_idx, debug_dict=None):
        split_up = cs.vertsplit(U, np.cumsum([0, self.nu_r, *tuple([self.nvars_hum for _ in range(self.num_hums)]), *tuple([self.nlambda_hum for _ in range(self.num_hums)])]).tolist())
        U_ksi_humA = split_up[1+humA_idx]
        U_humA = U_ksi_humA[:-1]
        ksi_humA = U_ksi_humA[-1]

        vx_coeff = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(1e-5, 9e-4)
        vy_coeff = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(1e-5, 9e-4)

        const_eqn = vx_coeff*self.orca_vxy_scaling*U_humA[0] + vy_coeff*self.orca_vxy_scaling*U_humA[1] - self.orca_ksi_scaling*ksi_humA
        if debug_dict is not None:
            debug_dict['rand_adj_vx'] = vx_coeff
            debug_dict['rand_adj_vy'] = vy_coeff
            debug_dict['ksi_const_fn_of_humA'] = cs.Function('ksicon_hum{:}'.format(humA_idx),
                                                      [U_ksi_humA],
                                                      [const_eqn],
                                                      ['U_ksi_humA'],
                                                      ['const'],
                                                      )

        return const_eqn


    def get_symcon_ORCA_humA_kkt_const(self, X, U, humA_idx, hums_orca_consts_humA, hums_max_vel_const_humA, hums_ksi_const_humA):
        Lam = U[self.nu_r+self.nVars_hums:]

        vars_humA = U[self.nu_r+humA_idx*self.nvars_hum:self.nu_r+(humA_idx+1)*self.nvars_hum]
        Lam_humA = Lam[(self.nlambda_hum)*humA_idx:(self.nlambda_hum)*(humA_idx+1)]

        v_pref = self.casadi_orca.get_v_pref_fromstate(humA_idx, X)
        cost_l = self.casadi_orca.cost_func(U_humA=self.orca_vxy_scaling*vars_humA[:-1], U_humA_pref=v_pref)['l'] + self.casadi_orca.ksi_penal_func(ksi_humA=self.orca_ksi_scaling*vars_humA[-1])['l']
        const_g = cs.vertcat(*tuple([const.get_cs_fn()(input=cs.vertcat(X,U))['const'] for const in hums_orca_consts_humA]+[hums_max_vel_const_humA.get_cs_fn()(input=U)['const']]+[hums_ksi_const_humA.get_cs_fn()(input=U)['const']]))

        const_g_names = list()
        const_g_names.extend(['primfeas_'+const.name[5:] for const in hums_orca_consts_humA]+['primfeas_'+hums_max_vel_const_humA.name]+['primfeas_'+hums_ksi_const_humA.name])

        Lagr = cost_l + Lam_humA.T @ const_g

        grad_Lagr = cs.gradient(Lagr, vars_humA)
        # grad_Lagr = cs.jacobian(Lagr, vars_humA).T
        ineq_con_eqn = cs.vertcat(const_g, -Lam_humA) # <=0
        rho = 1e-10 # called tau in Nocedal and Wright 2006 page 397, complementary slackness
        # needs rho -> 0, but a small value should be good enough for us.
        eq_con_eqn = cs.vertcat(grad_Lagr, Lam_humA * const_g - rho) # ==0

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
        logging.info('[MPCEnv] Getting human constraints, human model = {:}'.format(self.hum_model))
        hums_orca_consts = []
        hums_orca_consts_list = []
        hums_max_vel_consts = []
        hums_ksi_consts = []

        if self.hum_model == 'orca_casadi_simple' or self.hum_model == 'orca_casadi_kkt':
            logging.info('[MPCEnv] Generating ORCA constraints')
            XU_symb = cs.vertcat(self.X, self.U)
            debug_dicts = {}
            self.debug_dicts = debug_dicts
            for humA_idx in range(self.num_hums):
                humA_debug_dict = {'X' : self.X}
                debug_dicts['hum{:}'.format(humA_idx)] = humA_debug_dict
                # get all the pairwise orca constraints
                humA_orca_consts = []
                for humB_idx in self.casadi_orca.humB_idcs_list[humA_idx]:
                    if humB_idx == humA_idx:
                        logging.warn('[MPCEnv] humB_idx == humA_idx in orca constraints list. Agent does not need to avoid itself! Skipping.')
                        continue
                    humA_pairwise_debug_dict = {'X' : self.X}
                    debug_text = 'hum{:}_hum{:}'.format(humA_idx, humB_idx) if humB_idx>=0 else 'hum{:}_rob'.format(humA_idx, humB_idx)
                    humA_debug_dict[debug_text] = humA_pairwise_debug_dict
                    const_eqn = self.get_symcon_ORCA_humA_humB(self.X, self.U, humA_idx, humB_idx, debug_dict=humA_pairwise_debug_dict)
                    const_func_name = 'orca_hum{:}_hum{:}_const'.format(humA_idx, humB_idx) if humB_idx != -1 else 'orca_hum{:}_rob_const'.format(humA_idx)
                    symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                    orca_con = NonlinearConstraint(env=self,
                                                    sym_cs_func=symcon_func,
                                                    constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                    name=const_func_name,
                                                    debug_dict=humA_pairwise_debug_dict)
                    hums_orca_consts.append(orca_con)
                    humA_orca_consts.append(orca_con)


                # static obs constraints
                stat_const_eqn_list = self.get_symcon_ORCA_humA_stat_list(self.X, self.U, humA_idx, debug_dict=humA_debug_dict)
                for s_idx, const_eqn in enumerate(stat_const_eqn_list):
                    const_func_name = 'orca_hum{:}_stat{:}_const'.format(humA_idx, s_idx)
                    symcon_func = cs.Function(const_func_name, [XU_symb], [const_eqn], ['input'], ['const'])
                    orca_con = NonlinearConstraint(env=self,
                                                   sym_cs_func=symcon_func,
                                                   constrained_variable=ConstrainedVariableType.INPUT_AND_STATE,
                                                   name=const_func_name,
                                                   debug_dict=humA_debug_dict['hum{:}_stat{:}'.format(humA_idx, s_idx)])
                    hums_orca_consts.append(orca_con)
                    humA_orca_consts.append(orca_con)

                hums_orca_consts_list.append(humA_orca_consts)


                const_func_name = 'orca_hum{:}_vel_const'.format(humA_idx)
                const_eqn = self.get_symcon_ORCA_humA_maxvel(self.U, humA_idx, debug_dict=humA_debug_dict)
                symcon_func = cs.Function(const_func_name, [self.U], [const_eqn], ['input'], ['const'])
                maxvel_con = NonlinearConstraint(env=self,
                                                sym_cs_func=symcon_func,
                                                constrained_variable=ConstrainedVariableType.INPUT,
                                                name=const_func_name,
                                                debug_dict=humA_debug_dict)
                hums_max_vel_consts.append(maxvel_con)

                const_func_name = 'ksi_con_hum{:}'.format(humA_idx)
                const_eqn = self.get_symcon_ORCA_humA_ksi(self.U, humA_idx, debug_dict=humA_debug_dict)
                symcon_func = cs.Function(const_func_name, [self.U], [const_eqn], ['input'], ['const'])
                ksi_con = NonlinearConstraint(env=self,
                                                sym_cs_func=symcon_func,
                                                constrained_variable=ConstrainedVariableType.INPUT,
                                                name=const_func_name,
                                                debug_dict=humA_debug_dict)
                hums_ksi_consts.append(ksi_con)

        if self.hum_model == 'orca_casadi_kkt':
            logging.info('[MPCEnv] Generating ORCA KKT constraints')
            hum_kkt_ineq_consts = []
            hum_kkt_eq_consts = []
            for humA_idx in range(self.num_hums):
                ineq_con_eqn, eq_con_eqn, ineq_names, eq_names = self.get_symcon_ORCA_humA_kkt_const(self.X, self.U, humA_idx, hums_orca_consts_list[humA_idx], hums_max_vel_consts[humA_idx], hums_ksi_consts[humA_idx])
                XU_symb = cs.vertcat(self.X, self.U)

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

            logging.info('[MPCEnv] No. hums_orca_consts_list: {}, No. hums_max_vel_consts: {}, No. hum_numstab_consts: {}'.format(len(hums_orca_consts_list), len(hums_max_vel_consts), len(hum_numstab_consts)))

            # NB orca primal feasibilty constraints must be included in the kkt consts
            return hums_orca_consts_list, hums_max_vel_consts, hums_ksi_consts, hum_kkt_ineq_consts, hum_kkt_eq_consts, hum_numstab_consts



        logging.info('[MPCEnv] No. hums_orca_consts_list: {}, No. hums_max_vel_consts: {}'.format(len(hums_orca_consts_list), len(hums_max_vel_consts)))
        return hums_orca_consts, hums_max_vel_consts, hums_ksi_consts, []



    def convert_to_mpc_state_vector(self, state, nx_r, np_g, nX_hums, get_numpy=False):
        """Made to test the system model. from state object, return the state in the format that the orca solver would expect.

        :param state: _description_
        :param nx_r: _description_
        :param np_g: _description_
        :param nX_hums: _description_
        :return: _description_
        """
        val = np.zeros(nx_r+np_g+nX_hums)
        val[0] = state.self_state.px
        val[1] = state.self_state.py
        val[2] = state.self_state.theta
        v_coeff = 1 if np.abs(state.self_state.theta - np.arctan2(state.self_state.velocity[1], state.self_state.velocity[0])) < 1e-6 else -1
        val[3] = v_coeff * np.sqrt(state.self_state.velocity[0]**2 + state.self_state.velocity[1]**2)
        if not np.abs(val[3] * np.cos(val[2]) - state.self_state.velocity[0])<1e-5 or not np.abs(val[3] * np.sin(val[2]) - state.self_state.velocity[1]) < 1e-5:
            logging.warn('[MPC ENV] PROBLEM with vel vs. heading')
        val[4] = state.self_state.gx
        val[5] = state.self_state.gy
        offset = self.nx_r+self.np_g
        for i, h_state in enumerate(state.human_states):
            val[offset+i*self.nx_hum] = h_state.px
            val[offset+i*self.nx_hum+1] = h_state.py
            val[offset+i*self.nx_hum+2] = h_state.vx
            val[offset+i*self.nx_hum+3] = h_state.vy
            val[offset+i*self.nx_hum+4] = h_state.gx
            val[offset+i*self.nx_hum+5] = h_state.gy
        if get_numpy:
            return val.reshape(val.shape[0], 1)
        return val.tolist()