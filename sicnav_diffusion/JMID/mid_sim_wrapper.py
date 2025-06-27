from .MID.environment import Environment, Scene, Node, derivative_of
from .MID.mid import MID

import numpy as np

import einops
import scipy
import pandas as pd
import torch

import configparser
from threading import Lock

def get_most_likely_samples(forecasts, mid_model, num_ret_samples):
    preds_swap = einops.rearrange(
        forecasts, "samples humans horiz xy -> humans horiz samples xy"
    )
    if hasattr(mid_model, "cfg"):
        cond_joint_prediction = 'Agent' in mid_model.cfg.get('description')
    else:
        cond_joint_prediction = True
    if cond_joint_prediction:
        preds = einops.rearrange(
            preds_swap, "humans horiz samples xy -> horiz samples (humans xy)"
        )
        bandwidth = torch.exp(
            torch.linspace(
                np.log(0.01), np.log(0.1), steps=forecasts.size(2), device = "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        # adaptive bandwidth
        # find the two closest points
    else:
        preds = einops.rearrange(
            preds_swap, "humans horiz samples xy -> (humans horiz) samples xy"
        )
        bandwidth = torch.tensor(0.05, device = "cuda" if torch.cuda.is_available() else "cpu")

    # KDE adapted from https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py

    num_samples, num_humans, horiz, xy = forecasts.size()
    n, d = forecasts.size(0), xy * num_humans
    n = torch.tensor(n, dtype=torch.float32, device = "cuda" if torch.cuda.is_available() else "cpu")

    pi = torch.tensor(np.pi, device = "cuda" if torch.cuda.is_available() else "cpu")

    # Compute covariance matrix (ignoring the batch dimension)
    preds_mean = torch.mean(preds, dim=1, keepdim=True)
    preds_diff = preds - preds_mean
    cov = torch.bmm(preds_diff.transpose(1, 2), preds_diff) / (n - 1)

    # Scale covariance matrix by the squared inverse of bandwidth
    if cond_joint_prediction:
        scale_cov_inv = bandwidth[:, None, None] ** -2 * cov
    else:
        scale_cov_inv = bandwidth ** -2 * cov

    # Add a small constant to the diagonal to avoid singularity
    identity = torch.eye(d, device = "cuda" if torch.cuda.is_available() else "cpu").expand_as(cov)
    scale_cov_inv += identity * 1e-6
    scale_cov = torch.inverse(scale_cov_inv)

    # Compute Cholesky decomposition of the covariance matrix
    lower_triangular_matrix = torch.linalg.cholesky_ex(scale_cov)[0]

    # Compute difference between each x in preds with all preds
    test_Xs = preds.unsqueeze(2)
    train_Xs = preds.unsqueeze(1)
    diffs = test_Xs - train_Xs

    # Multiply differences with the inverse of lower_triangular_matrix
    inv_lower_triangular_matrix = torch.linalg.inv(lower_triangular_matrix).unsqueeze(1)
    if cond_joint_prediction:
        diffs = torch.matmul(diffs, inv_lower_triangular_matrix).div_(
            bandwidth[:, None, None, None]
        )
    else:
        diffs = torch.matmul(diffs, inv_lower_triangular_matrix).div_(bandwidth)

    # Compute the log likelihoods
    log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2
    log_det = 2 * torch.sum(
        torch.log(torch.diagonal(lower_triangular_matrix, dim1=-2, dim2=-1)), dim=-1
    )
    Z = 0.5 * d * torch.log(2 * pi) + 0.5 * log_det.unsqueeze(-1) + torch.log(n)

    # Compute log likelihoods, properly broadcast Z before subtracting
    likelihoods_all_ts = torch.logsumexp(log_exp - Z.unsqueeze(-1), dim=-1)

    # Normalize to importance weights using log sum exp along the samples dimension
    likelihoods_all_ts -= torch.logsumexp(likelihoods_all_ts, dim=1, keepdim=True)

    # Sum the log likelihoods accross the horiz dim
    if cond_joint_prediction:
        likelihoods_all = einops.reduce(
            likelihoods_all_ts, "horiz samples -> samples", reduction="sum"
        )
    else:
        likelihoods_all_ts_reshaped = einops.rearrange(
            likelihoods_all_ts,
            "(humans horiz) samples -> humans horiz samples",
            humans=num_humans,
            horiz=preds_swap.shape[1],
        )
        likelihoods_all = einops.reduce(
            likelihoods_all_ts_reshaped,
            "humans horiz samples -> humans samples",
            reduction="sum",
        )
    # normalize the likelihoods_all_unnormed

    # Get top k samples for each human in a vectorized way, without for loops
    k = num_ret_samples
    # shape of the likelihoods all is (num_humans, num_samples)
    # get sorted indices of the likelihoods along the samples dimension (axis=1)
    # sorted_indices = np.argsort(likelihoods_all, axis=1)
    sorted_indices = torch.argsort(likelihoods_all, axis=-1)
    top_k_indices = sorted_indices[..., -k:]  # num_humans x num_samples
    if cond_joint_prediction:
        new_forecasts = forecasts[top_k_indices]  # samples, humans, horiz, xy
        top_k_likelihoods_all_unnormed = likelihoods_all[top_k_indices]
    else:
        # Get the shape of forecasts and top_k_indices
        forecasts_swap = einops.rearrange(
            forecasts, "samples humans horiz xy -> humans samples horiz xy"
        )

        x = torch.broadcast_to(
            torch.arange(num_humans, device = "cuda" if torch.cuda.is_available() else "cpu")[:, None], top_k_indices.shape
        )
        y = top_k_indices

        # Index forecasts using index arrays
        new_forecasts = forecasts_swap[x, y]

        # also get the likelihoods_all for the top k samples, keep in mind that the likelihoods_all is of shape (num_humans, num_samples)
        top_k_likelihoods_all_unnormed = likelihoods_all[x, y]
    # top_k_likelihoods_all = top_k_likelihoods_all_unnormed - scipy.special.logsumexp(top_k_likelihoods_all_unnormed, axis=1)[:, None]
    top_k_likelihoods_all = top_k_likelihoods_all_unnormed - torch.logsumexp(
        top_k_likelihoods_all_unnormed, dim=-1, keepdim=True
    )

    # Reshape forecasts and likelihoods
    if cond_joint_prediction:
        assert new_forecasts.size() == (k, num_humans, horiz, xy)
        forecasts_topk_samples = einops.rearrange(
            new_forecasts, "samples humans horiz xy -> humans samples horiz xy"
        )
        top_k_likelihoods_all = torch.broadcast_to(
            top_k_likelihoods_all, (num_humans, k)
        )
    else:
        assert new_forecasts.shape == (
            num_humans,
            k,
            horiz,
            xy,
        )  # making sure dimensions are conserved
        # reshape preds_topk_samples from (humans samples (horiz xy)) to (humans samples horiz xy)
        forecasts_topk_samples = einops.rearrange(
            new_forecasts,
            "humans samples horiz xy -> humans samples horiz xy",
            humans=num_humans,
            samples=k,
            horiz=horiz,
            xy=2,
        )

    return forecasts_topk_samples, top_k_likelihoods_all


class ForecasterSimSuper():
    def init_super(self, env_config):
        self.prev_states_lock = Lock()
        if env_config is None:
            env_config_file = "./src/human_traj_forecaster/configs/env_utias_vicon.config"
            env_config = configparser.RawConfigParser()
            env_config.read(env_config_file)

        self.publish_freq = env_config.getfloat('human_trajectory_forecaster', 'publish_freq') # frequency at which to publish forecasts
        self.time_step = env_config.getfloat('env', 'time_step')  # Time interval between steps in a trajectory prediction and the history used for it
        assert (self.time_step*100).is_integer(), "please only specify human time step to a hundredth of a second"  # Constraint comes from subsample_df
        self.num_hist_frames = env_config.getint("human_trajectory_forecaster", "past_num_frames")  # Number of frames at human_time_step required for forecasting
        self.predict_horizon = env_config.getint("human_trajectory_forecaster", "prediction_horizon")  # Number of frames to forecast
        self.num_ret_samples = env_config.getint("human_trajectory_forecaster", "num_samples")  # Number of frames to forecast

        self.num_hums = env_config.getint('sim', 'human_num')

        # History of human frames
        self.prev_states = []  # Wait until we have 3.2 seconds of previous states before forecasting
        for i in range(self.num_hums):
            self.prev_states.append([])

        # History of robot positions

        self.prev_robot_states = []

    def update_state_hists(self, robot_state, human_states, time_stamp):
        for i in range(self.num_hums):
            self.prev_states[i].append([*human_states[i].position, time_stamp])
            if len(self.prev_states[i]) > self.num_hist_frames:
                self.prev_states[i].pop(0)

        self.prev_robot_states.append([*robot_state.position, time_stamp])


class HumanTrajectoryForecasterSim(ForecasterSimSuper):

    def __init__(self, env_config=None, mid_config_file=None):
        self.init_super(env_config)
        # for non-joint predictions
        # mid_config_file = "./src/human_traj_forecaster/configs/mid.yaml" if mid_config_file is None else mid_config_file
        # for joint predictions

        self._init_MID(mid_config_file=mid_config_file)

    def _init_MID(self, mid_config_file):
        # mid_config_file = "./src/human_traj_forecaster/configs/mid.yaml"
        standardization = {
            "PEDESTRIAN": {
                "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
                "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
                "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
            },
            # Use pedestrian standardization for robot since robot is assumed to move similarly to a pedestrian
            "JRDB_ROBOT": {
                "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
                "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
                "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
            }
        }
        self.mid_env = Environment(node_type_list=["PEDESTRIAN", "JRDB_ROBOT"], dt=self.time_step, standardization=standardization)
        attention_radius = dict()
        attention_radius[(self.mid_env.NodeType.PEDESTRIAN, self.mid_env.NodeType.PEDESTRIAN)] = 3.0
        attention_radius[(self.mid_env.NodeType.PEDESTRIAN, self.mid_env.NodeType.JRDB_ROBOT)] = 3.0
        attention_radius[(self.mid_env.NodeType.JRDB_ROBOT, self.mid_env.NodeType.PEDESTRIAN)] = 3.0
        attention_radius[(self.mid_env.NodeType.JRDB_ROBOT, self.mid_env.NodeType.JRDB_ROBOT)] = 3.0
        assert attention_radius[(self.mid_env.NodeType.PEDESTRIAN, self.mid_env.NodeType.PEDESTRIAN)] == attention_radius[(self.mid_env.NodeType.PEDESTRIAN, self.mid_env.NodeType.JRDB_ROBOT)], "Please make sure that the attention radius is the same for pedestrians and robots. Otherwise the logic in convert_mid_to_state_env will break."
        self.mid_env.attention_radius = attention_radius
        self.mid_model = MID(config=mid_config_file, init_env=self.mid_env, num_history=self.num_hist_frames, prediction_horizon=self.predict_horizon, sicnav_inference=True)
        self.model = self.mid_model


    def _gen_agent_df(self, prev_states, prev_robot_states):
        data_columns = pd.MultiIndex.from_product(
            [["position", "velocity", "acceleration"], ["x", "y"]]
        )

        individual_dfs = []

        self.prev_states_lock.acquire()

        for hum_idx in range(self.num_hums):
            df = pd.DataFrame(prev_states[hum_idx], columns=["pos_x_"+str(hum_idx), "pos_y_"+str(hum_idx), "time"])
            individual_dfs.append(df)
        robot_df = pd.DataFrame(prev_robot_states, columns=["pos_x_robot", "pos_y_robot", "time"])

        self.prev_states_lock.release()

        # Combine dataframes into a single one
        agent_df = individual_dfs[0].set_index("time")
        for hum_idx in range(1, self.num_hums):
            agent_df = agent_df.join(individual_dfs[hum_idx].set_index("time"))
        pose_estimates = agent_df.tail(1)  # Get pose estimates of pedestrians; exclude robot
        agent_df = agent_df.join(robot_df.set_index("time"))
        agent_df = agent_df.dropna()
        agent_df = agent_df.reset_index()
        agent_df = agent_df.sort_values(by=["time"])

        subsampled_df = self.subsample_df(agent_df).tail(self.num_hist_frames)
        data = self.expand_df(subsampled_df)
        data["frame_id"] = pd.to_numeric(
            data["frame_id"], downcast="integer"
        )
        data["track_id"] = pd.to_numeric(
            data["track_id"], downcast="integer"
        )
        data["node_id"] = data["track_id"].astype(str)
        data.sort_values("frame_id", inplace=True)

        return data, data_columns, pose_estimates

    def subsample_df(self, scene_df):
        """
        Assumes that the original fps is higher than the target fps.
        Adds a "datetime" column to scene_df in-place.
        """
        subsampled_df = scene_df

        # Convert time in seconds to datetime for resampling
        # Pandas datetime interprets floats as nanoseconds and only keeps track of whole nanoseconds
        # Multiply by 100 so the first two decimal points of our float will be kept
        subsampled_df["datetime"] = pd.to_datetime(subsampled_df["time"]*100)

        # Start from the end and keep the last row that falls in bins of int(self.time_step*100)
        subsampled_df = subsampled_df.resample(f"{int(round(self.time_step*100))}ns", on="datetime", origin="end").last()
        subsampled_df = subsampled_df.interpolate(method="linear", axis=0)
        return subsampled_df


    def expand_df(self, df):
        """Expands scene_df so each detection is its own row instead of having one row per frame with all the detections in it."""
        new_df_list = []
        i = 0
        for _, row in df.iterrows():
            for hum_idx in range(self.num_hums):
                new_df_list.append([i, hum_idx, row["pos_x_"+str(hum_idx)], row["pos_y_"+str(hum_idx)], "PEDESTRIAN"])
            new_df_list.append([i, -1, row["pos_x_robot"], row["pos_y_robot"], "ROBOT"])
            i += 1
        return pd.DataFrame(new_df_list, columns=["frame_id", "track_id", "pos_x", "pos_y", "node_type"])


    def convert_to_mid_state_env(self, prev_states, prev_robot_states):
        # prev_states is a multi-dim list N x T x 3 (num_agents x timesteps x (x, y, timestamp))
        # prev_robot_states is a multi-dim list T x 3 (timesteps x (x, y, timestamp))
        data, data_columns, pose_estimates = self._gen_agent_df(prev_states, prev_robot_states)
        pos_x_mean = 0.0
        pos_y_mean = 0.0
        data["pos_x"] = data["pos_x"] - pos_x_mean
        data["pos_y"] = data["pos_y"] - pos_y_mean
        max_timesteps = data["frame_id"].max()

        # get the relevant scene given the attention radius
        filtered_df = data[data['frame_id'] == max_timesteps]
        # sort filtered_df based on node_id
        filtered_df = filtered_df.sort_values(by=['track_id'])

        # Get positions as an N x 2 numpy array
        positions = filtered_df[['pos_x', 'pos_y']].to_numpy()

        # Get corresponding node_id array
        track_ids = filtered_df['track_id'].to_numpy()
        # get idx of node_id == -1

        sq_diffs = np.square(positions[:, np.newaxis] - positions[np.newaxis, :])
        dists = np.sqrt(np.sum(sq_diffs, axis=2))

        # Generate a mask based on the attention radius
        mask = dists < self.mid_env.attention_radius[(self.mid_env.NodeType.PEDESTRIAN, self.mid_env.NodeType.PEDESTRIAN)]

        cluster_sums = mask @ positions
        # Count the number of agents in each cluster
        cluster_counts = mask.sum(axis=1, keepdims=True)

        # Compute the mean positions of the clusters
        cluster_means = cluster_sums / cluster_counts

        # Find minimum cluster mean to the position of the robot (first row in positions)
        robot_dist = np.linalg.norm(cluster_means - positions[0], axis=1)

        chosen_cluster_idx = np.argmin(robot_dist[1:])+1  # ignore the robot

        track_ids_in_cluster = track_ids[mask[chosen_cluster_idx]]

        track_ids_not_in_cluster = track_ids[~mask[chosen_cluster_idx]]


        scene = Scene(
            timesteps=max_timesteps + 1,
            dt=self.time_step,
            name="posvel_tracker/pos",
            aug_func=None,
        )

        # for node_id in pd.unique(data["node_id"]):
        for track_id in track_ids_in_cluster:
            node_df = data[data["track_id"] == track_id]
            node_id = str(track_id)
            node_values = node_df[["pos_x", "pos_y"]].values
            if node_values.shape[0] < 2:
                continue
            new_first_idx = node_df["frame_id"].iloc[0]

            # Assume we don't need interpolation
            x = node_values[:, 0]
            y = node_values[:, 1]
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)
            data_dict = {
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            }

            node_data = pd.DataFrame(data_dict, columns=data_columns)
            if node_df.iloc[0]["node_type"] == "PEDESTRIAN":
                node_type = self.mid_env.NodeType.PEDESTRIAN
            else:
                node_type = self.mid_env.NodeType.JRDB_ROBOT
            node = Node(
                node_type=node_type,
                node_id=node_id,
                data=node_data,
                aux_data={
                    "pos_x_mean": pos_x_mean,
                    "pos_y_mean": pos_y_mean,
                },
            )
            node.first_timestep = new_first_idx

            # Store node specially if it is the robot; used for conditioning on robot future motion
            # if node_df.iloc[0]["node_type"] == "ROBOT":
            #     node.is_robot = True
            #     scene.robot = node

            scene.nodes.append(node)

        forecasts_dummy_dict = {}
        for track_id in track_ids_not_in_cluster:
            if track_id == -1:
                continue
            node_df = data[data["track_id"] == track_id]
            node_values = node_df[["pos_x", "pos_y"]].values

            # Assume we don't need interpolation
            x = node_values[:, 0]
            y = node_values[:, 1]
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)

            forecast = np.zeros((self.predict_horizon, 2))
            forecast[:, 0] = x[-1] + pos_x_mean + np.cumsum(np.tile(vx[-1] * scene.dt, self.predict_horizon))
            forecast[:, 1] = y[-1] + pos_y_mean + np.cumsum(np.tile(vy[-1] * scene.dt, self.predict_horizon))
            forecasts_dummy_dict[track_id] = forecast

        self.mid_env.scenes = [scene]

        # remove -1 from the ids
        track_ids_in_cluster = track_ids_in_cluster[track_ids_in_cluster != -1]
        track_ids_not_in_cluster = track_ids_not_in_cluster[track_ids_not_in_cluster != -1]

        return self.mid_env, pose_estimates, track_ids_in_cluster, track_ids_not_in_cluster, forecasts_dummy_dict


    def get_most_likely_samples(self, forecasts):
        return get_most_likely_samples(forecasts, self.model, self.num_ret_samples)


    def add_current_pose_to_forecasts(self, curr_poses, forecasts):
        """
        Prepends estimates of the current poses of agents to their forecasted trajectories
        """
        pose_estimates = []
        for hum_idx in range(self.num_hums):
            pose_estimates.append(np.array([curr_poses.iloc[0]["pos_x_"+str(hum_idx)], curr_poses.iloc[0]["pos_y_"+str(hum_idx)]]))
        pose_estimates = np.stack(pose_estimates)  # num_hum x 2
        pose_estimates = np.repeat(pose_estimates[:, None, None, :], forecasts.shape[1], axis=1)  #
        forecasts = np.concatenate((pose_estimates, forecasts), axis=2)
        return forecasts


    def predict(self):
        if len(self.prev_states[0]) < self.num_hist_frames:
            return None

        prev_state_env, last_poses = self.convert_to_mid_state_env(self.prev_states)

        # Generate trajectory forecasts
        forecasts = self.mid_model.eval(prev_state_env)  # num_hum x samples x future timesteps x 2
        forecasts = forecasts.cpu().detach().numpy()

        # Add current pose estimate
        pose_estimates = []
        fut_forecasts = []
        for hum_idx in range(self.num_hums):
            pose_estimates.append(np.array([last_poses.iloc[0]["pos_x_"+str(hum_idx)], last_poses.iloc[0]["pos_y_"+str(hum_idx)]]))
            fut_forecasts.append(forecasts[hum_idx])
        pose_estimates = np.stack(pose_estimates)  # num_hum x 2
        pose_estimates = np.repeat(pose_estimates[:, None, None, :], forecasts[0].shape[0], axis=1)  #  num_hum x num_preds x 1 x 2
        forecasts = np.stack(fut_forecasts)
        forecasts = np.concatenate((pose_estimates, forecasts), axis=2)

        return forecasts
    # def _get_human_state_from_sim(self, human_id, time):


    def predict_ret_best(self):
        prev_state_env, last_poses, track_ids_in_cluster, track_ids_not_in_cluster, forecasts_dummy_dict = self.convert_to_mid_state_env(self.prev_states, self.prev_robot_states)
        # Generate trajectory forecasts
        forecasts_all_samples = self.mid_model.eval(prev_state_env)  # samples x num_hums x future timesteps x 2
            # Check if we need to return fewer samples than we have. If so, find the highest likelihood samples to return
        if self.num_ret_samples < forecasts_all_samples.size(0):
            forecasts_torch, top_k_likelihoods_all = self.get_most_likely_samples(forecasts_all_samples)
            top_k_likelihoods_all_in_cluster = top_k_likelihoods_all.detach().cpu().numpy()
        else:
            forecasts_torch = einops.rearrange(forecasts_all_samples, "samples humans horiz xy -> humans samples horiz xy")
            top_k_likelihoods_all_in_cluster = np.log(np.ones((forecasts_torch.size(0), forecasts_all_samples.size(0)), dtype=np.float64) / forecasts_all_samples.shape[0])
        forecasts_in_cluster = forecasts_torch.detach().cpu().numpy()
        forecasts = np.zeros((self.num_hums, self.num_ret_samples, self.predict_horizon, 2), dtype=np.float64)
        top_k_likelihoods_all = np.zeros((self.num_hums, self.num_ret_samples), dtype=np.float64)
        forecasts[track_ids_in_cluster, :, :, :] = forecasts_in_cluster
        top_k_likelihoods_all[track_ids_in_cluster, :] = top_k_likelihoods_all_in_cluster

        for h_idx in track_ids_not_in_cluster:
            if h_idx == -1:
                continue
            # do cvmm for all the samples
            forecasts[h_idx, :, :, :] = forecasts_dummy_dict[h_idx][np.newaxis, :, :]
            top_k_likelihoods_all[h_idx, :] = top_k_likelihoods_all_in_cluster[0, :]

        # Add current pose estimate to prediction
        forecasts_top_k_with_t0 = self.add_current_pose_to_forecasts(last_poses, forecasts)



        return forecasts_top_k_with_t0, top_k_likelihoods_all