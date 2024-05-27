# CrowdSimPlus Simulation Framework
## Environment
The environment is built on top of OpenAI gym library, and has implemented two abstract methods.
* reset(): the environment resets the positions for all the agents and returns the observation
that the robot expects.
* step(action): the environment takes the action of the robot as input, then computes the actions for the simulated human agents and moves the simulation one time step forward. This function detects whether there is a collision between agents, or if there are collisions between an agent and any static obstacle, then correctly adjusts the movement of the agents to ensure the agent does not penetrate any static obstacle.

## Agents
AgentPlus is a base class for both the human and robot agents.

## Simulated Human Policies

Policy takes state as input and output an action. Current available policies:
* ORCA_plus: computes human actions in environments with static obstacles based on the Optimal Reciprocal Collision Avoidance scheme.
* SFM: computes human actions in environments with static obstacles based on the Social Forces Model scheme.


## States
* `ObservableState`: position, velocity, radius of one agent.
* `FullState`: position, velocity, radius, goal position, preferred velocity, rotation.
* `JointState`: concatenation of one agent's full state and all other agents' observable states.
* `FullyObservableFullState`: concatenation of all agents' full states.


## Actions
* `ActionXY`: (vx, vy) if `kinematics` == `holonomic`.
* `ActionRot`: (velocity, rotation angle) if `kinematics` == `unicycle`.