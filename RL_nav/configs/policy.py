import os
from RL_nav.SB3_Support.callbacks import CustomCallback
import torch

class SB3_policy():

    def __init__(self, env, policy_config, curr_dir):
        self.env = env
        self.policy_config = policy_config
        self.curr_dir = curr_dir

        # retreive values common to all models
        self.model_name = policy_config.get("rl","model")
        self.gamma= policy_config.getfloat('rl', 'gamma')
        self.adjusted_gamma = self.gamma **  (env.time_step * env.vpref)
        self.exploration_fraction = policy_config.getfloat('rl', 'exploration_fraction')
        self.learning_rate = policy_config.getfloat('rl', 'learning_rate')

        self.save_freq = policy_config.getint('rl', 'save_freq')
        self.num_actions = int(env.speed_samples * env.rotation_samples + 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

    def get_model_name(self):
        name = f"{self.model_name}_sim_{self.env.train_val_sim}_occlusion_{self.env.occlusion}_rot_bound_{self.env.rotation_bound}_speed_samps_{self.env.speed_samples}_rot_samps_{self.env.rotation_samples}_humans_{self.env.human_num}_hmin_{self.env.training_schema[0]}_hmax_{self.env.training_schema[-1]}"
        # name = f"{self.model_name}_sim_{self.env.train_val_sim}_occlusion_{self.env.occlusion}_speed_samps_{self.env.speed_samples}_rot_samps_{self.env.rotation_samples}_vpref_{self.env.vpref}_humans_{self.env.human_num}_hmin_{self.env.training_schema[0]}_hmax_{self.env.training_schema[-1]}_rs_{self.env.success_reward}_rc_{self.env.collision_penalty}_rd_{self.env.discomfort_dist}_{self.env.discomfort_penalty_factor}_rf_{self.env.freezing_penalty}_rp_{self.env.progress_factor}"
        
        return name

    def set_name_and_log_dirs(self):
        name = self.get_model_name()
        save_path = os.path.join(self.curr_dir, f"./logs/{name}")

        i = 0
        while os.path.exists(f"{save_path}_{i}"):
            i+=1
        save_path = f"{save_path}_{i}"


        logs_path = os.path.join(self.curr_dir, "./logs")
        tensorboard_path = os.path.join(logs_path, "./tensorboard_logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
            os.makedirs(tensorboard_path)
        elif not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)

        tensorboard_log = os.path.join(self.curr_dir, f"./logs/tensorboard_logs/{name}_{i}")

        self.name = name
        self.save_path = save_path
        self.tensorboard_log = tensorboard_log

    def get_custom_callback(self, env_config_file, policy_config_file):
        self.set_name_and_log_dirs()
        callback = CustomCallback(
                                save_freq=self.save_freq,
                                save_path= self.save_path,

                                env_config_path=env_config_file,
                                policy_config_path=policy_config_file,

                                name_prefix=self.name,
                                save_replay_buffer=True,
                                save_vecnormalize=True,
                                )
        return callback

    def get_policy_kwargs(self):
        """
        Return policy kwargs for SB3 model. SB3_vars contains all the necessary variables. 
        This function is used to expand out what is needed for a particular SB3 model form the config.
        """
        env = self.env
        SB3_vars = self

        policy_config = self.policy_config
        SB3_vars.vpref = env.vpref
        if SB3_vars.model_name == "qsarl":
            SB3_vars.mlp1_dims = [int(x) for x in policy_config.get('qsarl', 'mlp1_dims').split(', ')]
            SB3_vars.mlp2_dims = [int(x) for x in policy_config.get('qsarl', 'mlp2_dims').split(', ')]
            SB3_vars.mlp3_dims = [int(x) for x in policy_config.get('qsarl', 'mlp3_dims').split(', ')]
            SB3_vars.attention_dims = [int(x) for x in policy_config.get('qsarl', 'attention_dims').split(', ')]
            SB3_vars.with_global_state = policy_config.getboolean('qsarl', 'with_global_state')
        elif SB3_vars.model_name == "sarl":
            SB3_vars.mlp1_dims = [int(x) for x in policy_config.get('sarl', 'mlp1_dims').split(', ')]
            SB3_vars.mlp2_dims = [int(x) for x in policy_config.get('sarl', 'mlp2_dims').split(', ')]
            SB3_vars.mlp3_dims = [int(x) for x in policy_config.get('sarl', 'mlp3_dims').split(', ')]
            SB3_vars.attention_dims = [int(x) for x in policy_config.get('sarl', 'attention_dims').split(', ')]
            SB3_vars.with_global_state = policy_config.getboolean('sarl', 'with_global_state')
        elif SB3_vars.model_name == "rgl" or SB3_vars.model_name == "rgl_multistep":
            SB3_vars.num_layer = policy_config.getint('rgl', 'num_layer')
            SB3_vars.X_dim = policy_config.getint('rgl', 'X_dim')
            SB3_vars.wr_dims = [int(x) for x in policy_config.get('rgl', 'wr_dims').split(', ')]
            SB3_vars.wh_dims = [int(x) for x in policy_config.get('rgl', 'wh_dims').split(', ')]
            SB3_vars.final_state_dim = policy_config.getint('rgl', 'final_state_dim')
            SB3_vars.gcn2_w1_dim = policy_config.getint('rgl', 'gcn2_w1_dim')
            SB3_vars.planning_dims = [int(x) for x in policy_config.get('rgl', 'planning_dims').split(', ')]
            SB3_vars.similarity_function = policy_config.get('rgl', 'similarity_function')
            SB3_vars.layerwise_graph = policy_config.getboolean('rgl', 'layerwise_graph')
            SB3_vars.skip_connection = policy_config.getboolean('rgl', 'skip_connection')

            SB3_vars.robot_state_dim = policy_config.getint('rgl', 'robot_state_dim')
            SB3_vars.human_state_dim = policy_config.getint('rgl', 'human_state_dim')
            # self.set_common_parameters(config)
        else:
            raise NotImplementedError
        
        policy_kwargs = {"env": env,
                         "SB3_vars": SB3_vars}
        
        return policy_kwargs