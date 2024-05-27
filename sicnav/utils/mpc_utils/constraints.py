"""Constraints module.
Classes for constraints and lists of constraints.

# Adapted from https://github.com/utiasDSL/safe-control-gym/
# (file safe-control-gym/safe_control_gym/envs/constraints.py)
"""
import casadi as cs
from enum import Enum
import numpy as np


class ConstraintType(str, Enum):
    """Allowable constraints, equality or inequality.
    """

    EQ = EQUALITY = "equality"  # Constraints who are a function of the state X.
    INEQ = INEQUALITY = "inequality"  # Constraints who are a function of the input U.

class ConstrainedVariableType(str, Enum):
    """Allowable constraint type specifiers.
    """

    STATE = "state"  # Constraints who are a function of the state X.
    INPUT = "input"  # Constraints who are a function of the input U.
    INPUT_AND_STATE = "input_and_state"  # Constraints who are a function of the input U and state X.


class Constraint:
    """Implements a (state-wise/trajectory-wise/stateful) constraint.
    A constraint can contain multiple scalar-valued constraint functions.
    Each should be represented as g(x) <= 0.
    Attributes:
        constrained_variable: the variable(s) from env to be constrained.
        dim (int): Total number of input dimensions to be constrained, i.e. dim of x.
        num_constraints (int): total number of output dimensions or number of constraints, i.e. dim of g(x).
        sym_func (Callable): the symbolic function of the constraint, can take in np.array or CasADi variable.

    """

    def __init__(self,
                 env,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 name : str='unspecified_const',
                 row_names : list=[''],
                 debug_dict : dict=None,
                 contype : ConstraintType=ConstraintType.INEQ,
                 **kwargs
                 ):
        """Defines params (e.g. bounds) and state.
        Args:
            joint_state (MPCenv): The MPC environment this problem is for
            constrained_variable (ConstrainedVariableType): Specifies the input type to the constraint as a constraint
                                                         that acts on the state, input, or both.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions.
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True.
            name (str): The distance from the constraint at which is_almost_active returns True.
        """
        self.constrained_variable = ConstrainedVariableType(constrained_variable)
        if self.constrained_variable == ConstrainedVariableType.STATE:
            self.og_dim = env.state_dim
            self.dim = env.state_dim
        elif self.constrained_variable == ConstrainedVariableType.INPUT:
            self.og_dim = env.action_dim
            self.dim = env.action_dim
        elif self.constrained_variable == ConstrainedVariableType.INPUT_AND_STATE:
            self.og_dim = env.state_dim + env.action_dim
            self.dim = env.state_dim + env.action_dim
        else:
            raise NotImplementedError('[ERROR] invalid constrained_variable (use STATE, INPUT or INPUT_AND_STATE).')
        # Save the strictness attribute
        self.strict = strict
        # Only want to select specific dimensions, implemented via a filter matrix.
        if active_dims is not None:
            if isinstance(active_dims, int):
                active_dims = [active_dims]
            assert isinstance(active_dims, (list, np.ndarray)), '[ERROR] active_dims is not a list/array.'
            assert (len(active_dims) <= self.dim), '[ERROR] more active_dim than constrainable self.dim'
            assert all(isinstance(n, int) for n in active_dims), '[ERROR] non-integer active_dim.'
            assert all((n < self.dim) for n in active_dims), '[ERROR] active_dim not stricly smaller than self.dim.'
            assert (len(active_dims) == len(set(active_dims))), '[ERROR] duplicates in active_dim'
            self.constraint_filter = cs.sparsify(np.eye(self.dim)[active_dims])
            self.dim = len(active_dims)
        else:
            self.constraint_filter = np.eye(self.dim)
        if tolerance is not None:
            self.tolerance = np.array(tolerance, ndmin=1)
        else:
            self.tolerance = None
        self.name = name
        self.row_names = row_names
        self.contype = contype
        if debug_dict is not None:
            self.debug_dict = debug_dict

    def reset(self):
        """Clears up the constraint state (if any).
        """
        pass

    def get_symbolic_model(self,
                           env
                           ):
        """Gets the symbolic form of the constraint function.
        Args:
            env: The environment to constrain.
        Returns:
            obj: The symbolic form of the constraint.
        """
        return NotImplementedError

    def check_tolerance_shape(self):
        if self.tolerance is not None and len(self.tolerance) != self.num_constraints:
            raise ValueError('[ERROR] the tolerance dimension does not match the number of constraints.')

    def create_casadi_fn(self):
        if not hasattr(self, 'sym_cs_func'):
            input=cs.MX.sym('in', self.og_dim)
            self.sym_cs_func = cs.Function(self.name+'_func', [input], [self.get_symbolic_model()(input)], ['input'], ['const'])

    def get_cs_fn(self):
        return self.sym_cs_func

    def get_cs_fn_map(self,K, *args):
        if not len(self.row_names) == self.sym_cs_func.size1_out(0):
            # assert len(self.row_names) == 1
            self.row_names = ['row_{:}'.format(idx) for idx in range(self.sym_cs_func.size1_out(0))]
        mapped_row_names = [row_name+'_k{:}'.format(k) for k in range(K) for row_name in self.row_names]
        cs_fn_map = self.sym_cs_func.map(K, *args)
        return cs_fn_map, mapped_row_names



class LinearConstraint(Constraint):
    """Constraint class for constraints of the form A @ x <= b.
    """

    def __init__(self,
                 env,
                 A: np.ndarray,
                 b: np.ndarray,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 **kwargs
                 ):
        """Initialize the class.
        Args:
            env (BenchmarkEnv): The environment to constraint.
            A (np.array or list): A matrix of the constraint (self.num_constraints by self.dim).
            b (np.array or list): b matrix of the constraint (1D array self.num_constraints)
                                  constrained_variable (ConstrainedVariableType): Type of constraint.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.
        """
        super().__init__(env, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance, **kwargs)
        A = np.array(A, ndmin=1)
        b = np.array(b, ndmin=1)
        assert A.shape[1] == self.dim, '[ERROR] A has the wrong dimension!'
        self.A = cs.sparsify(A)
        assert b.shape[0] == A.shape[0], '[ERROR] Dimension 0 of b does not match A!'
        self.b = cs.sparsify(b)
        self.num_constraints = A.shape[0]
        # self.sym_func = lambda x: self.A @ self.constraint_filter @ x - self.b
        self.sym_func = lambda x: cs.mtimes([self.A, self.constraint_filter, x]) - self.b
        self.check_tolerance_shape()
        self.create_casadi_fn()

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.
        Returns:
            lambda: The symbolic form of the constraint.
        """
        return self.sym_func


class QuadraticConstraint(Constraint):
    """Constraint class for constraints of the form x.T @ P @ x + q.T @ x <= b.
    """

    def __init__(self,
                 env,
                 P: np.ndarray,
                 b: float,
                 constrained_variable: ConstrainedVariableType,
                 q=None,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 **kwargs,
                 ):
        """Initializes the class.
        Args:
            env (MPC): The environment the constraint is for.
            P (np.array): The square matrix representing the quadratic.
            q (np.array): The square matrix representing the quadratic.
            b (float): The scalar limit for the quadatic constraint.
            constrained_variable (ConstrainedVariableType): Specifies the input type to the constraint as a constraint
                                                        that acts on the state, input, or both.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions.
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True.
        """
        super().__init__(env, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance, **kwargs)
        P = np.array(P, ndmin=1)
        assert P.shape == (self.dim, self.dim), '[ERROR] P has the wrong dimension!'
        if q is None:
            q = np.zeros((self.dim,1))
        self.q = q
        assert q.shape == (self.dim, 1), '[ERROR] q has the wrong dimension!'
        self.P = cs.sparsify(P)
        self.q = cs.sparsify(q)
        assert isinstance(b, float), '[ERROR] b is not a scalar!'
        self.b = cs.sparsify(b)
        self.num_constraints = 1  # Always scalar.
        self.sym_func = lambda x: cs.mtimes([x.T, self.constraint_filter.T, self.P, self.constraint_filter, x]) + cs.mtimes([x.T, self.constraint_filter.T, self.q]) - self.b
        self.check_tolerance_shape()
        self.create_casadi_fn()

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.
        Returns:
            lambda: The symbolic form of the constraint.
        """
        return self.sym_func

class NonlinearConstraint(Constraint):
    """Constraint class for constraints of the form g(x) <= b, where g(x) as a symbolic function.
    """

    def __init__(self,
                 env,
                 sym_cs_func,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 tolerance=None,
                 **kwargs
                 ):
        """Initializes the class.
        Args:
            env (MPC): The environment the constraint is for.
            sym_func (): A casadi Function symbolic function for g(x)
            b (float): The scalar limit for the quadatic constraint.
            constrained_variable (ConstrainedVariableType): Specifies the input type to the constraint as a constraint
                                                        that acts on the state, input, or both.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list of ints): Filters the constraint to only act only select certian dimensions.
            tolerance (list or np.array): The distance from the constraint at which is_almost_active returns True.
        """
        super().__init__(env, constrained_variable, strict=strict, tolerance=tolerance, **kwargs)

        self.num_constraints = 1  # Always scalar.
        self.sym_cs_func = sym_cs_func
        self.sym_func = lambda input : self.sym_cs_func(input=input)['const']
        self.check_tolerance_shape()
        self.create_casadi_fn()

    def get_symbolic_model(self):
        """Gets the symbolic form of the constraint function.
        Returns:
            lambda: The symbolic form of the constraint.
        """
        return self.sym_func


class BoundedConstraint(LinearConstraint):
    """ Class for bounded constraints lb <= x <= ub as polytopic constraints -Ix + b <= 0 and Ix - b <= 0.
    """

    def __init__(self,
                 env,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray,
                 constrained_variable: ConstrainedVariableType,
                 strict: bool=False,
                 active_dims=None,
                 tolerance=None,
                 **kwargs):
        """Initialize the constraint.
        Args:
            env (MPCEnv): The environment to constraint.
            lower_bounds (np.array or list): Lower bound of constraint.
            upper_bounds (np.array or list): Uppbound of constraint.
            constrained_variable (ConstrainedVariableType): Type of constraint.
            strict (optional, bool): Whether the constraint is violated also when equal to its threshold.
            active_dims (list or int): List specifying which dimensions the constraint is active for.
            tolerance (float): The distance at which is_almost_active(env) triggers.
        """
        self.lower_bounds = np.array(lower_bounds, ndmin=1)
        self.upper_bounds = np.array(upper_bounds, ndmin=1)
        dim = self.lower_bounds.shape[0]
        A = np.vstack((-np.eye(dim), np.eye(dim)))
        b = np.hstack((-self.lower_bounds, self.upper_bounds))
        super().__init__(env, A, b, constrained_variable, strict=strict, active_dims=active_dims, tolerance=tolerance, **kwargs)
        self.check_tolerance_shape()
        self.create_casadi_fn()