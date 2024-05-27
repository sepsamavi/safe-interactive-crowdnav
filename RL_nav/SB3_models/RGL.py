from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import softmax, relu
import numpy as np
from numpy.linalg import norm

from stable_baselines3 import PPO
from stable_baselines3.dqn import MultiInputPolicy
from stable_baselines3.common.policies import BasePolicy


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor

from crowd_sim_plus.envs.utils.action import ActionXY
from crowd_sim_plus.envs.utils.utils_plus import point_to_segment_dist_vectorized
# can't use have to use torch
# from crowd_sim_plus.envs.utils.utils_plus import point_to_segment_dist

# this is trying to abstract out the Robot Plus class
class RGLNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        env,
        SB3_vars,

    ):
        super(RGLNetwork, self).__init__()

        self.env=env
        self.SB3_vars = SB3_vars

        self.gamma = SB3_vars.gamma
        self.time_step = env.time_step
        self.device = SB3_vars.device
        self.vpref = SB3_vars.vpref
        self.action_map = env.action_space_map
        self.num_actions = len(self.action_map)
        self.kinematics = "unicycle" if env.robot_holonomic is False else None

        self.adjusted_gamma  = self.gamma **  (self.time_step * self.vpref)

        # define indicies for needed elements
        # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
        self.robot_px = 0
        self.robot_py = 1
        self.robot_vx = 2
        self.robot_vy = 3
        self.robot_r = 4
        self.robot_gx = 5
        self.robot_gy = 6
        self.robot_v_pref = 7
        self.robot_theta = 8
        # self.px, self.py, self.vx, self.vy, self.radius
        self.human_px = 0
        self.human_py = 1
        self.human_vx = 2
        self.human_vy = 3
        self.human_r = 4

        # define architecture
        self.self_state_dim = SB3_vars.robot_state_dim
        self.human_state_dim = SB3_vars.human_state_dim
        self.input_dim = self.self_state_dim + self.human_state_dim
        self.model = self.construct_value_network(self.input_dim, self.self_state_dim, SB3_vars.num_layer, SB3_vars.X_dim, SB3_vars.wr_dims, SB3_vars.wh_dims, SB3_vars.final_state_dim, SB3_vars.gcn2_w1_dim, SB3_vars.planning_dims, SB3_vars.similarity_function, SB3_vars.layerwise_graph, SB3_vars.skip_connection)
        self.ran = 0
    def construct_value_network(self, input_dim, self_state_dim, num_layer, X_dim, wr_dims, wh_dims, final_state_dim,
                 gcn2_w1_dim, planning_dims, similarity_function, layerwise_graph, skip_connection):
        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        human_state_dim = input_dim - self_state_dim
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)

        if num_layer == 1:
            self.w1 = Parameter(torch.randn(self.X_dim, final_state_dim))
        elif num_layer == 2:
            self.w1 = Parameter(torch.randn(self.X_dim, gcn2_w1_dim))
            self.w2 = Parameter(torch.randn(gcn2_w1_dim, final_state_dim))
        else:
            raise NotImplementedError

        self.value_net = mlp(final_state_dim, planning_dims)


    def forward(self, features):

        if isinstance(features, dict):

            # collect raw inputs
            self_state = features["robot_state"] # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta

            batch_size = self_state.shape[0]

            # predicted_q_vals = torch.zeros((batch_size, self.num_actions), device=self.device)
            predicted_q_vals = []
            for action_index, action in enumerate(self.action_map):
                next_state_robot = self.get_robot_state(features, action_index)
                next_state_humans = self.get_human_states(features, action_index)
                rewards = self.get_rewards(features, action_index)

                next_state_value = self.calc_value(next_state_robot, next_state_humans)
                value = rewards + self.adjusted_gamma * next_state_value
                predicted_q_vals.append(value)

            predicted_q_vals = torch.cat(predicted_q_vals, dim=1)
            return predicted_q_vals

        else:

            # robot self state
            #self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
            # followed by human state
            # self.px, self.py, self.vx, self.vy, self.radius
            state = self.input_transformation(features)
            return self.value_calculation(state)

    def calc_value(self, robot_state, human_states):
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        batch_states = torch.stack([torch.cat([robot_state, next_human_state], dim=1) for next_human_state in human_states], dim=1)#.to(self.device)
        batch_input_states = self.input_transformation(batch_states)
        state_value = self.value_calculation(batch_input_states)

        return state_value

    def value_calculation(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        self.ran+=1
        self_state = state[:, 0, :self.self_state_dim]
        human_states = state[:, :, self.self_state_dim:]

        # compute feature matrix X
        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings], dim=1)

        # compute matrix A
        normalized_A = self.compute_similarity_matrix(X)
        self.A = normalized_A[0, :, :].data.cpu().numpy()

        # graph convolution
        if self.num_layer == 0:
            feat = X[:, 0, :]
        elif self.num_layer == 1:
            h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            feat = h1[:, 0, :]
        else:
            # compute h1 and h2
            if not self.skip_connection:
                h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            else:
                h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1)) + X
            if self.layerwise_graph:
                normalized_A2 = self.compute_similarity_matrix(h1)
            else:
                normalized_A2 = normalized_A
            if not self.skip_connection:
                h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
            else:
                h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2)) + h1
            feat = h2[:, 0, :]

        # do planning using only the final layer feature of the agent
        value = self.value_net(feat)

        return value

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def calc_target_values(self, next_observations, rewards, dones, gamma):
        next_state_value = self.calc_default_state(next_observations)
        target_q_values = rewards + (1 - dones) * gamma * next_state_value
        return target_q_values

    def calc_train_values(self, curr_observations, actions):
        return self.calc_default_state(curr_observations)

    def calc_default_state(self, state):
        robot_state = self.get_robot_state(state, -1, default=True)
        human_states = self.get_human_states(state, -1, default=True)
        state_value = self.calc_value(robot_state, human_states)

        return state_value

    def get_robot_state(self, features, action_index, default=False):
        if default is False:
            return features[f"robot_state_{action_index}"]
        else:
            return features[f"robot_state"]

    def get_human_states(self, features, action_index, default=False):
        num_active_humans = features["num_humans"][0]
        if default is False:
            human_states = [features[f"human_state_{i}_{action_index}"] for i in range(num_active_humans)]
            return human_states
        else:
            human_states = [features[f"human_state_{i}"] for i in range(num_active_humans)]
            return human_states

    def get_rewards(self, features, action_index):
        return features[f"reward_{action_index}"]

    def input_transformation(self, state):
        """
        Difference from rotate is that we do not rotate x-axis to align with goal
        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch_size = state.shape[0]
        batch = state.shape[1]

        dx = (state[:,:, 5] - state[:, :, 0]).reshape((batch_size, batch, -1))
        dy = (state[:, :, 6] - state[:, :, 1]).reshape((batch_size, batch, -1))

        v_pref = state[:,:, 7].reshape((batch_size, batch, -1))
        vx = state[:, :, 2].reshape((batch_size, batch, -1))
        vy = state[:, :, 3].reshape((batch_size, batch, -1))

        radius = state[:, :, 4].reshape((batch_size, batch, -1))

        vx1 = state[:,:, 11].reshape((batch_size, batch, -1))
        vy1 = state[:,:, 12].reshape((batch_size, batch, -1))
        px1 = (state[:,:, 9] - state[:,:, 0]) .reshape((batch_size, batch, -1))
        py1 = (state[:,:, 10] - state[:,:, 1]).reshape((batch_size, batch, -1))
        radius1 = state[:, :, 13].reshape((batch_size, batch, -1))
        radius_sum = radius + radius1

        da = torch.norm(torch.cat([(state[:,:, 0] - state[:,:, 9]).reshape((batch_size, batch, -1)), (state[:,:, 1] - state[:,:, 10]).reshape((batch_size, batch, -1))], dim=2), 2, dim=2, keepdim=True)

        # theta not used in point-turn robot
        new_state = torch.cat([dx, dy, v_pref, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=2)

        return new_state

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def _predict(self, observation, deterministic: bool = True):
        # print(observation)
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

class RGL(MultiInputPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        self.policy_init = kwargs
        super(RGL, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            #**kwargs,
        )

        # Disable orthogonal initialization
        self.ortho_init = False

    def make_q_net(self) -> None:
        RGLNet  = RGLNetwork(**self.policy_init)
        # Load in pretrained weights here
        SB3_vars = self.policy_init["SB3_vars"]
        if SB3_vars.init_weights is not False:
            RGLNet.load_state_dict(torch.load(SB3_vars.init_weights))
        return RGLNet



def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net