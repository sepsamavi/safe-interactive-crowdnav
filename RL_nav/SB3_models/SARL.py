from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn
import numpy as np
from numpy.linalg import norm

from stable_baselines3.dqn import MultiInputPolicy
from stable_baselines3.common.policies import BasePolicy


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import CombinedExtractor

from crowd_sim_plus.envs.utils.action import ActionXY
from crowd_sim_plus.envs.utils.utils_plus import point_to_segment_dist_vectorized


class SARLNetwork(nn.Module):
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
        super(SARLNetwork, self).__init__()

        # self.env=env
        # self.SB3_vars = SB3_vars

        # dim of rotated states
        self.self_state_dim = 6
        self.human_state_dim = 7

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
        self.input_dim = self.self_state_dim + self.human_state_dim
        self.model = self.construct_value_network(self.input_dim, self.self_state_dim, SB3_vars.mlp1_dims, SB3_vars.mlp2_dims, SB3_vars.mlp3_dims, SB3_vars.attention_dims, SB3_vars.with_global_state)

        self.softmax = torch.nn.functional.softmax

    def construct_value_network(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state):
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)

        self.attention_weights = None


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
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output

        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        # remove possible instability
        min_score = torch.min(scores, dim=1, keepdim=True)[0]
        adjusted_scores = scores - min_score
        adjusted_scores = torch.clamp(adjusted_scores, max=50)
        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # scores_exp = torch.exp(adjusted_scores) * (adjusted_scores != 0).float()

        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        weights = self.softmax(adjusted_scores, dim=1).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)

        return value

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

    def propagate(self, state, action, robot_move):
        next_state = state

        if robot_move:
            try:
                next_state[:, self.robot_theta] = state[:, self.robot_theta] + action.r
                next_state[:, self.robot_vx] = action.v * torch.cos(next_state[:, self.robot_theta])
                next_state[:, self.robot_vy] = action.v * torch.sin(next_state[:, self.robot_theta])
                next_state[:, self.robot_px] = state[:, self.robot_px] + next_state[:, self.robot_vx] * self.time_step
                next_state[:, self.robot_py] = state[:, self.robot_py] + next_state[:, self.robot_vy] * self.time_step
            except:
                next_state[self.robot_theta] = state[self.robot_theta] + action.r
                next_state[self.robot_vy] = action.v * torch.sin(next_state[self.robot_theta])
                next_state[self.robot_vx] = action.v * torch.cos(next_state[self.robot_theta])
                next_state[self.robot_px] = state[self.robot_px] + next_state[self.robot_vx] * self.time_step
                next_state[self.robot_py] = state[self.robot_py] + next_state[self.robot_vy] * self.time_step

        else:
            # propagate state of humans
            next_state[:, self.human_px] = state[:, self.human_px] + action.vx * self.time_step
            next_state[:, self.human_py] = state[:, self.human_py] + action.vy * self.time_step

        return next_state

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

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        # batch = state.shape[0]
        batch_size = state.shape[0]
        batch = state.shape[1]
        dx = (state[:,:, 5] - state[:, :, 0]).reshape((batch_size, batch, -1))
        dy = (state[:, :, 6] - state[:, :, 1]).reshape((batch_size, batch, -1))
        rot = torch.atan2(state[:, :, 6] - state[:, :, 1], state[:, :, 5] - state[:,:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=2), 2, dim=2, keepdim=True)
        v_pref = state[:, :, 7].reshape((batch_size, batch, -1))
        vx = (state[:, :, 2] * torch.cos(rot) + state[:, :, 3] * torch.sin(rot)).reshape((batch_size, batch, -1))
        vy = (state[:, :, 3] * torch.cos(rot) - state[:, :, 2] * torch.sin(rot)).reshape((batch_size, batch, -1))

        radius = state[:, :, 4].reshape((batch_size, batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, :, 8] - rot).reshape((batch_size, batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:,:, 11] * torch.cos(rot) + state[:,:, 12] * torch.sin(rot)).reshape((batch_size, batch, -1))
        vy1 = (state[:,:, 12] * torch.cos(rot) - state[:,:, 11] * torch.sin(rot)).reshape((batch_size, batch, -1))
        px1 = (state[:,:, 9] - state[:,:, 0]) * torch.cos(rot) + (state[:,:, 10] - state[:,:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch_size, batch, -1))
        py1 = (state[:,:, 10] - state[:,:, 1]) * torch.cos(rot) - (state[:,:, 9] - state[:,:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch_size, batch, -1))
        radius1 = state[:, :, 13].reshape((batch_size, batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:,:, 0] - state[:,:, 9]).reshape((batch_size, batch, -1)), (state[:,:, 1] - state[:,:, 10]).
                                  reshape((batch_size, batch, -1))], dim=2), 2, dim=2, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=2)
        return new_state

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def _predict(self, observation, deterministic: bool = True):
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def compute_reward(self, nav, humans, wall_collisions=None):

        batch_size = nav.shape[0]

        # collision detection with other humans
        dmin = float('inf')
        # collision = False
        collision = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        dmin_locations = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # Note that 0.2 is hard coded for right now
        dmin_values = torch.ones(batch_size, device=self.device)*0.2

        for i, human in enumerate(humans):
            # dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            dist = torch.norm(torch.stack([nav[:, self.robot_px] - human[:, self.human_px], nav[:, self.robot_py] - human[:, self.human_py]], dim=1), dim=1) - nav[:, self.robot_r] - human[:, self.human_r]

            # if dist < 0:
            #     collision = True
            #     break
            # if dist < dmin:
                # dmin = dist
            collisions = dist < 0
            collision = torch.logical_or(collision, collisions)
            dmin_locations = collision != 1
            dmin_to_replace = torch.minimum(dmin_values[dmin_locations], dist[dmin_locations])
            dmin_values[dmin_locations] = dmin_to_replace

        # collision detection with walls
        # stat_collision = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        # for idx, line in enumerate(self.env.static_obstacles):
        #     closest_dist = point_to_segment_dist_vectorized(line[0][0], line[0][1], line[1][0], line[1][1], nav[:,self.robot_px], nav[:,self.robot_py], self.device)
        #     stat_collisions = closest_dist < (nav[:, self.robot_r])
        #     stat_collision = torch.logical_or(stat_collision, stat_collisions)
        # collision = torch.logical_or(collision, stat_collision)

        if wall_collisions != None:
            # stat_collision = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            stat_collisions = torch.from_numpy(np.array(wall_collisions)).to(self.device)
            collision = torch.logical_or(collision, stat_collisions)

        # check if reaching the goal
        reaching_goal = torch.norm(torch.stack([nav[:, self.robot_px] - nav[:, self.robot_gx], nav[:, self.robot_py] - nav[:, self.robot_gy]], dim=1), dim=1) < nav[:, self.robot_r]


        reward = torch.zeros(batch_size, device=self.device) \
                + collision * -0.25 \
                + torch.logical_not(collision) * reaching_goal * 1 \
                + torch.logical_not(reaching_goal) * dmin_locations * (dmin_values - 0.2) * 0.5 * self.time_step

        # if collision:
        #     reward = -0.25
        # elif reaching_goal:
        #     reward = 1
        # elif dmin < 0.2:
        #     reward = (dmin - 0.2) * 0.5 * self.time_step
        # else:
        #     reward = 0

        return reward

class SARL(MultiInputPolicy):
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
        super(SARL, self).__init__(
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
        SARLNet  = SARLNetwork(**self.policy_init)
        # Load in pretrained weights here
        SB3_vars = self.policy_init["SB3_vars"]
        if SB3_vars.init_weights is not False:
            SARLNet.load_state_dict(torch.load(SB3_vars.init_weights))
        return SARLNet



def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net