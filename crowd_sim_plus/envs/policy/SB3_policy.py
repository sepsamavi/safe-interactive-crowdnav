from crowd_sim_plus.envs.policy.policy import Policy


class SB3(Policy):
    def __init__(self):
        super().__init__()
        self.name = None
        self.trainable = True
        self.multiagent_training = False
        self.kinematics = False
        self.attention = False

    def set_model(self, model):
        self.model = model.policy.q_net

    def set_attention(self, attention):
        self.attention = attention

    def configure(self, config, env_config):
        self.name = config.get('rl', 'model')

        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = env_config.getboolean('robot', 'holonomic')
        self.kinematics = "holonomic" if self.kinematics is True else "unicycle"
        self.sampling = "exponential"
        self.speed_samples = env_config.getint('robot', 'speed_samples')
        self.rotation_samples = env_config.getint('robot', 'rotation_samples')


