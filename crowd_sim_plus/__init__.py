from gym.envs.registration import register

register(
    id='CrowdSimPlus-v0',
    entry_point='crowd_sim_plus.envs:CrowdSimPlus',
)