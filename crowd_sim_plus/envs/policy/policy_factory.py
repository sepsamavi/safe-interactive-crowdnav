from crowd_sim_plus.envs.policy.linear import Linear
from crowd_sim_plus.envs.policy.orca import ORCA
from crowd_sim_plus.envs.policy.orca_plus import ORCAPlus
from crowd_sim_plus.envs.policy.SB3_policy import SB3
from crowd_sim_plus.envs.policy.social_force import SFM

def none_policy():
    return None

policy_factory = dict()
policy_factory['none'] = none_policy
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['SB3'] = SB3
policy_factory['orca_plus'] = ORCAPlus
policy_factory['sfm'] = SFM
