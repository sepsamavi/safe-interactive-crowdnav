from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from sicnav.policy.dwa import DynamicWindowApproach
from sicnav.policy.campc import CollisionAvoidMPC
from sicnav_diffusion.policy.sicnav_acados import SICNavAcados

policy_factory['dwa'] = DynamicWindowApproach
policy_factory['campc'] = CollisionAvoidMPC
policy_factory['sicnav_acados'] = SICNavAcados