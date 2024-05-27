import logging
import numpy as np
import casadi as cs
from casadi import Callback

import rvo2
from crowd_sim_plus.envs.policy.orca import ORCA
from crowd_sim_plus.envs.utils.state_plus import FullyObservableJointState # just for hardcoded parameters

def get_human_radii(state, human_radii=None):
    if not human_radii:
        human_radii = np.zeros(len(state.human_states))

    for i, h_state in enumerate(state.human_states):
        human_radii[i] = h_state.radius

    return human_radii

def get_human_goals(state, human_gxs=None, human_gys=None):
    if not human_gxs:
        human_gxs = np.zeros(len(state.human_states))
    if not human_gys:
        human_gys = np.zeros(len(state.human_states))
    for i, h_state in enumerate(state.human_states):
        human_gxs[i] = h_state.gx
        human_gys[i] = h_state.gy
    return human_gxs, human_gys

class callbackORCA(Callback):
    def __init__(self, name, time_step, state, nx_r, np_g, nX_hums, dummy_orca_pol, opts={}):
        Callback.__init__(self)
        self.orca_pol = dummy_orca_pol
        self.time_step = time_step

        self.reset_humans(state)
        self.nx_r = nx_r
        self.np_g = np_g
        self.nX_hums = nX_hums
        self.nx_hum = int(nX_hums/len(state.human_states))
        self.construct(name, opts)

    def reset_humans(self, state, new_h_gxs=None, new_h_gys=None):
        robot_state = state.self_state
        self.num_humans = len(state.human_states)
        if new_h_gxs is None or new_h_gys is None:
            if isinstance(state, FullyObservableJointState):
                self.human_gxs, self.human_gys = get_human_goals(state)
            self.human_radii = get_human_radii(state)
        else:
            if isinstance(state, FullyObservableJointState):
                self.human_gxs, self.human_gys = new_h_gxs, new_h_gys
            self.human_radii = get_human_radii(state)


        params = self.orca_pol.neighbor_dist, self.orca_pol.max_neighbors, self.orca_pol.time_horizon, self.orca_pol.time_horizon_obst
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.orca_pol.radius, self.orca_pol.max_speed)
        # Add the static obstacles:
        for _, line in enumerate(state.static_obs):
            self.sim.addObstacle(line)
        if state.static_obs:
            self.sim.processObstacles()

        # Add the robot agent:
        self.sim.addAgent(robot_state.position, *params, robot_state.radius + 0.01 + self.orca_pol.safety_space,
                              self.orca_pol.max_speed, robot_state.velocity)
        # Add the other agents:
        for human_state in state.human_states:
            self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.orca_pol.safety_space,
                            human_state.v_pref, human_state.velocity)

    # Number of inputs and outputs
    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return cs.Sparsity.dense(self.nx_r+self.np_g+self.nX_hums, 1)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.nX_hums, 1)

    # Initialize the object
    def init(self):
        logging.info('initializing callbackORCA object')

    # Evaluate the function numerically
    def eval(self, arg):
        X = arg[0].toarray()[:,0]
        X_hums = X[self.nx_r+self.np_g:]

        # Set robot's position now and velocity at prev. time step (ie vel now)
        rob_lin_vel = (X[3]*np.cos(X[2]), X[3]*np.sin(X[2]))
        self.sim.setAgentPosition(0, (float(X[0]), float(X[1])))
        self.sim.setAgentVelocity(0, rob_lin_vel)


        for i in range(self.num_humans):
            px = X_hums[i*self.nx_hum]
            py = X_hums[i*self.nx_hum+1]
            vx = X_hums[i*self.nx_hum+2]
            vy = X_hums[i*self.nx_hum+3]
            gx = X_hums[i*self.nx_hum+4]
            gy = X_hums[i*self.nx_hum+5]

            self.sim.setAgentPosition(i + 1, (px, py))
            self.sim.setAgentVelocity(i + 1, (vx, vy))

            # each agent is aware of its own pref vel
            velocity = np.array((gx - px, gy - py))
            speed = np.linalg.norm(velocity)
            epsilon = 1e-3
            agent_vmax = self.sim.getAgentMaxSpeed(i + 1)
            pref_vel = velocity / speed * (agent_vmax - epsilon) if speed > (agent_vmax - epsilon) else velocity
            self.sim.setAgentPrefVelocity(i + 1, tuple(pref_vel))

        self.sim.doStep()
        return_list = np.zeros(self.nX_hums, dtype=float)
        for i in range(self.num_humans):
            px, py = self.sim.getAgentPosition(i+1)
            vx, vy = self.sim.getAgentVelocity(i+1)
            return_list[i*self.nx_hum] = px
            return_list[i*self.nx_hum+1]  = py
            return_list[i*self.nx_hum+2] = vx
            return_list[i*self.nx_hum+3] = vy
            return_list[i*self.nx_hum+4] = X_hums[i*self.nx_hum+4]
            return_list[i*self.nx_hum+5] = X_hums[i*self.nx_hum+5]

        ret_Val = cs.DM(return_list.tolist())
        return [ret_Val]

    def eval_sx(self, arg):
        return self.eval(arg)

