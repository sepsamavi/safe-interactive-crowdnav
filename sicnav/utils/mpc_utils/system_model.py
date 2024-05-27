import logging

"""Symbolic System Model.
    Adapted from: https://github.com/utiasDSL/safe-control-gym
    (safe-control-gym/safe_control_gym/math_and_models/symbolic_systems.py)
"""
import numpy as np
import casadi as cs


class SystemModel():
    """Implements a discrete-time dynamics model with symbolic variables.
    x_kp1 = f(x,u), y = g(x,u), with other pre-defined, symbolic functions
    (e.g. cost, constraints)

    Notes:
        * naming convention on symbolic variable and functions.
            * for single-letter symbol, use {}_sym, otherwise use underscore for delimiter.
            * for symbolic functions to be exposed, use {}_func.
    """

    def __init__(self,
                 dynamics,
                 cost,
                 dt=0.25,
                 funcs=None,
                 **kwargs):
        """
        """
        self.linearize = kwargs.get('linearize', "default False")

        # Setup for dynamics.
        self.x_sym = dynamics["vars"]["X"]
        self.u_sym = dynamics["vars"]["U"]
        self.x_kp1 = dynamics["dyn_eqn"]

        if dynamics["obs_eqn"] is None:
            self.y_sym = self.x_sym
        else:
            self.y_sym = dynamics["obs_eqn"]

        # Sampling time.
        self.dt = dt

        # Other symbolic functions.
        if funcs is not None:
            for name, func in funcs.items():
                assert name not in self.__dict__
                self.__dict__[name] = func
        # Variable dimensions.
        self.nx = self.x_sym.shape[0]
        self.nu = self.u_sym.shape[0]
        self.ny = self.y_sym.shape[0]
        # Setup cost function.
        self.cost_eqn = cost["cost_eqn"]
        logging.debug("cost equation: {:}".format(self.cost_eqn))
        self.Q = cost["vars"]["Q"]
        self.R = cost["vars"]["R"]
        self.Xr = cost["vars"]["Xr"]
        self.Ur = cost["vars"]["Ur"]

        # Setup symbolic model.
        self.setup_model()

        # Setup Jacobian and Hessian of the dynamics and cost functions.
        if self.linearize:
            self.setup_linearization()
        else:
            self.setup_nonlinear()


    def setup_model(self):
        """Exposes functions to evaluate the model.
        """
        # Continuous time dynamics.
        self.f_func_nonlin = cs.Function('f', [self.x_sym, self.u_sym], [self.x_kp1], ['x', 'u'], ['f'])

        # Observation model.
        # self.g_func = cs.Function('g', [self.x_sym, self.u_sym], [self.y_sym], ['x', 'u'], ['g'])

    def setup_nonlinear(self):
        l_inputs = [self.x_sym, self.u_sym, self.Xr, self.Ur, self.Q, self.R]
        l_inputs_str = ['x', 'u', 'Xr', 'Ur', 'Q', 'R']
        l_outputs = [self.cost_eqn]
        l_outputs_str = ['l']
        self.cost_func = cs.Function('loss', l_inputs, l_outputs, l_inputs_str, l_outputs_str)
        self.f_func = self.f_func_nonlin

    def setup_linearization(self):
        """Exposes functions for the linearized model.
        """

        # Jacobians w.r.t state & input.
        self.dfdx = cs.jacobian(self.x_kp1, self.x_sym)
        self.dfdu = cs.jacobian(self.x_kp1, self.u_sym)
        self.df_func = cs.Function('df', [self.x_sym, self.u_sym],
                                   [self.dfdx, self.dfdu], ['x', 'u'],
                                   ['dfdx', 'dfdu'])
        self.dgdx = cs.jacobian(self.y_sym, self.x_sym)
        self.dgdu = cs.jacobian(self.y_sym, self.u_sym)
        self.dg_func = cs.Function('dg', [self.x_sym, self.u_sym],
                                   [self.dgdx, self.dgdu], ['x', 'u'],
                                   ['dgdx', 'dgdu'])
        # Evaluation point for linearization.
        self.x_eval = cs.MX.sym('x_eval', self.nx, 1)
        self.u_eval = cs.MX.sym('u_eval', self.nu, 1)
        # Linearized dynamics model.
        self.x_kp1_linear = self.x_kp1 + self.dfdx @ (
             self.x_eval - self.x_sym) + self.dfdu @ ( self.u_eval - self.u_sym)
        self.f_linear_func = cs.Function(
            'f', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.x_kp1_linear], ['x_eval', 'u_eval', 'x', 'u'], ['f_linear'])
        self.f_func = self.f_linear_func
        self.f_func_nonlin = cs.Function('f', [self.x_sym, self.u_sym], [self.x_kp1], ['x', 'u'], ['f'])

        # Linearized observation model.
        self.y_linear = self.y_sym + self.dgdx @ (
            self.x_eval - self.x_sym) + self.dgdu @ (self.u_eval - self.u_sym)
        self.g_linear_func = cs.Function(
            'g_linear', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.y_linear], ['x_eval', 'u_eval', 'x', 'u'], ['g_linear'])
        # Jacobian and Hessian of cost function.
        self.l_x = cs.jacobian(self.cost_eqn, self.x_sym)
        self.l_xx = cs.jacobian(self.l_x, self.x_sym)
        self.l_u = cs.jacobian(self.cost_eqn, self.u_sym)
        self.l_uu = cs.jacobian(self.l_u, self.u_sym)
        self.l_xu = cs.jacobian(self.l_x, self.u_sym)
        l_inputs = [self.x_sym, self.u_sym, self.Xr, self.Ur, self.Q, self.R]
        l_inputs_str = ['x', 'u', 'Xr', 'Ur', 'Q', 'R']
        l_outputs = [self.cost_eqn, self.l_x, self.l_xx, self.l_u, self.l_uu, self.l_xu]
        l_outputs_str = ['l', 'l_x', 'l_xx', 'l_u', 'l_uu', 'l_xu']
        # TEMP Do not linearize cost, just linearize dynamics
        # l_inputs = [self.x_sym, self.u_sym, self.Xr, self.Ur, self.Q, self.R]
        # l_inputs_str = ['x', 'u', 'Xr', 'Ur', 'Q', 'R']
        # l_outputs = [self.cost_eqn]
        # l_outputs_str = ['l']
        self.cost_func = cs.Function('loss', l_inputs, l_outputs, l_inputs_str, l_outputs_str)