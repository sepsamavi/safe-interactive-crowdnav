from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from sicnav.policy.dwa import DynamicWindowApproach
from sicnav.policy.campc import CollisionAvoidMPC

policy_factory['dwa'] = DynamicWindowApproach
policy_factory['campc'] = CollisionAvoidMPC