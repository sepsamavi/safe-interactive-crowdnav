import dill
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pickle
import scipy
from tensorboardX import SummaryWriter
import time
import torch
from torch import utils
from tqdm.auto import tqdm

from dataset import EnvironmentDataset, collate, get_timesteps_data
import environment
import evaluation
from utils.trajectron_hypers import get_traj_hypers


class Baseline:
    def __init__(self, config, test_dataset=None):
        self.config = config
        self.test_dataset = test_dataset
        torch.backends.cudnn.benchmark = True
        self._build()

    def eval(self):
        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_mean_ade_batch_errors = []
        eval_std_ade_batch_errors = []
        eval_fde_batch_errors = []
        eval_mean_fde_batch_errors = []
        eval_std_fde_batch_errors = []
        eval_kde_batch_errors = []
        eval_ade_list_batch_errors = []
        eval_fde_list_batch_errors = []
        eval_de_list_batch_errors = []

        ph = self.hyperparams["prediction_horizon"]
        max_hl = self.hyperparams["maximum_history_length"]
        total_num_gt_tracks = 0
        total_num_det_tracks = 0
        all_pred_time = 0
        all_gt_time = 0
        for i, scene in enumerate(self.eval_scenes):
            print(
                f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}: {scene.name} -----"
            )
            total_num_gt_tracks += len(scene.nodes)
            inference_scene = scene  # Initialize
            inference_env = self.eval_env
            inference_node = scene.nodes[0]
            inference_node_id_to_gt_node = {node.id: node for node in scene.nodes}
            inference_scene = self._get_inference_scene(scene)
            inference_env = self.inference_env
            inference_nodes = inference_scene.nodes
            inference_node = inference_nodes[0]
            gt_nodes = scene.nodes
            inference_idx, gt_idx = self._associate_nodes_with_other_nodes(
                inference_nodes,
                gt_nodes,
                inference_scene.normalized_px,  # Assume that inference_scene and scene both are or are not using normalized pixels
                inference_scene.img_width,  # Assume width and height of images are the same between inference_scene and scene if they are images
                inference_scene.img_height,
            )
            inference_node_id_to_gt_node = {
                inference_nodes[inference_idx[i]].id: gt_nodes[gt_idx[i]]
                for i in range(len(inference_idx))
            }
            total_num_det_tracks += len(inference_node_id_to_gt_node)

            if self.config.save_trajectories:
                time_to_node_id_to_preds = {}
            for t in tqdm(range(0, inference_scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                batch = get_timesteps_data(
                    env=inference_env,
                    scene=inference_scene,
                    t=timesteps,
                    node_type=node_type,
                    state=self.hyperparams["state"],
                    pred_state=self.hyperparams["pred_state"],
                    edge_types=inference_env.get_edge_types(),
                    min_ht=7,
                    max_ht=self.hyperparams["maximum_history_length"],
                    min_ft=12,
                    max_ft=12,
                    hyperparams=self.hyperparams,
                )
                if batch is None:
                    continue
                test_batch = batch[0]
                history = test_batch[
                    1
                ]  # B (num_pedestrian * timesteps) x history_timesteps x data (x pos, y pos, x vel, y vel, x accel, y accel)
                nodes = batch[1]
                timesteps_o = batch[2]
                traj_pred = self._generate(
                    history, num_points=12, dt=scene.dt
                )  # 1 * B * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                for i, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    gt_node = inference_node_id_to_gt_node.get(nodes[i].id)
                    if not isinstance(
                        gt_node, environment.Node
                    ):  # This case happens when there are more detected tracks than ground truth tracks, or if none of them overlap in time
                        continue
                    if not self.exists_at_t(
                        gt_node, ts
                    ):  # Only evaluate ADE when the ground truth exists now (at ts) to be consistent with OSPA(2) calculations
                        continue
                    predictions_dict[ts][gt_node] = np.transpose(
                        predictions[:, [i]], (1, 0, 2, 3)
                    )

                gt_node = inference_node_id_to_gt_node[
                    list(inference_node_id_to_gt_node.keys())[0]
                ]  # Use any gt node since we assume they all have the same data preprocessing
                batch_error_dict = evaluation.compute_batch_statistics(
                    predictions_dict,
                    scene.dt,  # Assume scene.dt is equal to inference_scene.dt
                    max_hl,
                    ph,
                    self.eval_env.NodeType,  # Assume self.eval_env.NodeType is equal to self.inference_env.NodeType
                    gt_node.get_mean_x_and_y(),
                    inference_node.get_mean_x_and_y(),  # Use any inference node since we assume they all have the same mean x and mean y
                    kde=False,
                    compute_ml=False,
                    map=None,
                    best_of=True,
                    all_de=True,
                    prune_ph_to_future=True,
                    normalized_px=inference_scene.normalized_px,
                    img_width=inference_scene.img_width,
                    img_height=inference_scene.img_height,
                )

                eval_ade_batch_errors = np.hstack(
                    (eval_ade_batch_errors, batch_error_dict[node_type]["ade"])
                )
                eval_mean_ade_batch_errors = np.hstack(
                    (
                        eval_mean_ade_batch_errors,
                        batch_error_dict[node_type]["ade_mean"],
                    )
                )
                eval_std_ade_batch_errors = np.hstack(
                    (eval_std_ade_batch_errors, batch_error_dict[node_type]["ade_std"])
                )
                eval_fde_batch_errors = np.hstack(
                    (eval_fde_batch_errors, batch_error_dict[node_type]["fde"])
                )
                eval_mean_fde_batch_errors = np.hstack(
                    (
                        eval_mean_fde_batch_errors,
                        batch_error_dict[node_type]["fde_mean"],
                    )
                )
                eval_std_fde_batch_errors = np.hstack(
                    (eval_std_fde_batch_errors, batch_error_dict[node_type]["fde_std"])
                )
                eval_kde_batch_errors = np.hstack(
                    (eval_kde_batch_errors, batch_error_dict[node_type]["kde"])
                )
                eval_ade_list_batch_errors = np.hstack(
                    (
                        eval_ade_list_batch_errors,
                        batch_error_dict[node_type]["ade_list"],
                    )
                )
                eval_fde_list_batch_errors = np.hstack(
                    (
                        eval_fde_list_batch_errors,
                        batch_error_dict[node_type]["fde_list"],
                    )
                )
                eval_de_list_batch_errors = np.hstack(
                    (eval_de_list_batch_errors, batch_error_dict[node_type]["de_list"])
                )

                if self.config.save_trajectories:
                    time_to_node_id_to_preds.update(
                        self._get_preds(
                            traj_pred,
                            nodes,
                            timesteps_o,
                            inference_scene.normalized_px,
                            inference_scene.img_width,
                            inference_scene.img_height,
                        )
                    )
            if self.config.save_trajectories:
                with open(
                    osp.join(self.model_dir, scene.name + "_predictions.pkl"), "wb"
                ) as f:
                    pickle.dump(time_to_node_id_to_preds, f)

        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)
        kde = np.mean(eval_kde_batch_errors)
        ade_mean = np.mean(eval_mean_ade_batch_errors)
        fde_mean = np.mean(eval_mean_fde_batch_errors)
        ade_std = np.mean(eval_std_ade_batch_errors)
        fde_std = np.mean(eval_std_fde_batch_errors)
        ade_max = np.max(eval_ade_list_batch_errors)
        fde_max = np.max(eval_fde_list_batch_errors)
        ade_min = np.min(eval_ade_list_batch_errors)
        fde_min = np.min(eval_fde_list_batch_errors)

        if self.config.dataset == "eth":
            ade = ade / 0.6
            fde = fde / 0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50

        print(
            "Below results optimize ADE rather than OSPA, so the ADE reported here should be less than or equal to the ADE reported by the OSPA evaluation script."
        )
        self.log.info(
            "Below results optimize ADE rather than OSPA, so the ADE reported here should be less than or equal to the ADE reported by the OSPA evaluation script."
        )
        print(f"Best Of 20: ADE: {ade} FDE: {fde} KDE: {kde}")
        self.log.info(f"Best Of 20: ADE: {ade} FDE: {fde} KDE: {kde}")
        print(
            f"Mean ADE: {ade_mean} (Std: {ade_std} (Min: {ade_min}, Max: {ade_max})) Mean FDE: {fde_mean} (Std: {fde_std} (Min: {fde_min}, Max: {fde_max}))"
        )
        self.log.info(
            f"Mean ADE: {ade_mean} (Std: {ade_std} (Min: {ade_min}, Max: {ade_max})) Mean FDE: {fde_mean} (Std: {fde_std} (Min: {fde_min}, Max: {fde_max}))"
        )
        print(
            f"Detected {total_num_det_tracks} tracks out of {total_num_gt_tracks} tracks"
        )
        self.log.info(
            f"Detected {total_num_det_tracks} tracks out of {total_num_gt_tracks} tracks"
        )
        # if self.config.eval_other:
        #     print(
        #         f"Percent time overlap between predictions and ground truth: {all_pred_time/all_gt_time*100}%"
        #     )
        #     self.log.info(
        #         f"Percent time overlap between predictions and ground truth: {all_pred_time/all_gt_time*100}%"
        #     )

        # self._plot_displacement_error_histogram(eval_de_list_batch_errors, 0.1)

        return ade, fde, kde

    def _generate(self, history, num_points=12):
        raise NotImplementedError("_generate method was not implemented")

    def _build(self):
        self._build_dir()
        self._build_encoder_config()
        self._build_eval_loader()
        self._build_inference_scenes()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments", self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        log_name = "{}.log".format(time.strftime("%Y-%m-%d-%H-%M"))
        if self.test_dataset is not None:
            log_name = f"{self.config.dataset}_{self.test_dataset}_{log_name}"
        else:
            log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(
            self.config.data_dir, self.config.dataset + "_train.pkl"
        )
        if self.test_dataset is not None:
            self.eval_data_path = osp.join(
                self.config.data_dir, self.test_dataset + "_val.pkl"
            )
        else:
            self.eval_data_path = osp.join(
                self.config.data_dir,
                self.config.dataset + "_" + self.config.test_split + ".pkl",
            )

        self.inference_data_path = osp.join(
            self.config.data_dir,
            self.config.inference_dataset + "_" + self.config.test_split + ".pkl",
        )

        print("> Directory built!")

    def _build_encoder_config(self):
        self.hyperparams = get_traj_hypers()
        self.hyperparams["enc_rnn_dim_edge"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_edge_influence"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_history"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_future"] = self.config.encoder_dim // 2

        with open(self.train_data_path, "rb") as f:
            self.train_env = dill.load(f, encoding="latin1")
        with open(self.eval_data_path, "rb") as f:
            self.eval_env = dill.load(f, encoding="latin1")

    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []

        if config.eval_every is not None:
            with open(self.eval_data_path, "rb") as f:
                self.eval_env = dill.load(f, encoding="latin1")

            for attention_radius_override in config.override_attention_radius:
                (
                    node_type1,
                    node_type2,
                    attention_radius,
                ) = attention_radius_override.split(" ")
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(
                    attention_radius
                )

            if self.eval_env.robot_type is None and self.hyperparams["incl_robot_node"]:
                self.eval_env.robot_type = self.eval_env.NodeType[
                    0
                ]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            self.eval_dataset = EnvironmentDataset(
                self.eval_env,
                self.hyperparams["state"],
                self.hyperparams["pred_state"],
                scene_freq_mult=self.hyperparams["scene_freq_mult_eval"],
                node_freq_mult=self.hyperparams["node_freq_mult_eval"],
                hyperparams=self.hyperparams,
                min_history_timesteps=self.hyperparams["minimum_history_length"],
                min_future_timesteps=self.hyperparams["prediction_horizon"],
                return_robot=not config.incl_robot_node,
            )
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(
                    node_type_data_set,
                    collate_fn=collate,
                    pin_memory=True,
                    batch_size=config.eval_batch_size,
                    shuffle=True,
                    num_workers=config.preprocess_workers,
                )
                self.eval_data_loader[
                    node_type_data_set.node_type
                ] = node_type_dataloader

        print("> Dataset built!")

    def _build_inference_scenes(self):
        config = self.config
        self.inference_scenes = []

        with open(self.inference_data_path, "rb") as f:
            self.inference_env = dill.load(f, encoding="latin1")

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                " "
            )
            self.inference_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius
            )

        if (
            self.inference_env.robot_type is None
            and self.hyperparams["incl_robot_node"]
        ):
            self.inference_env.robot_type = self.inference_env.NodeType[
                0
            ]  # TODO: Make more general, allow the user to specify?
            for scene in self.inference_env.scenes:
                scene.add_robot_from_nodes(self.inference_env.robot_type)

        self.inference_scenes = self.inference_env.scenes

    def _get_inference_scene(self, scene):
        # Assume scenes have unique names
        for i_scene in self.inference_scenes:
            if i_scene.name == scene.name:
                return i_scene
        return None

    def _add_mean(self, data, node):
        data = data + np.expand_dims(node.get_mean_x_and_y(), axis=0)
        return data

    def _unnormalize_pixels(self, data, normalized_px, img_width=0, img_height=0):
        if not normalized_px:
            return data
        data = data * np.array([[img_width, img_height]])
        return data

    def _un_preprocess_data(self, data, node, normalized_px, img_width=0, img_height=0):
        """Data (timesteps x 2). Assume second dimension of data is x-position then y-position. `node` is the node data came
        from. `normalized_px` is a boolean representing if the data is normalized pixel values or not.
        """
        data = self._add_mean(data, node)  # Data is mean-centered (see process_data.py)
        data = self._unnormalize_pixels(data, normalized_px, img_width, img_height)
        return data

    def _compute_all_ades(
        self, nodes_list_1, nodes_list_2, normalized_px, img_width, img_height
    ):
        """Computes the ade between all pairs of nodes between nodes_list_1 and nodes_list_2"""
        ade_matrix = np.full(
            (len(nodes_list_1), len(nodes_list_2)), np.finfo(np.float64).max
        )
        for i, node_1 in enumerate(nodes_list_1):
            for j, node_2 in enumerate(nodes_list_2):
                # Assume that timesteps are consistent across all nodes
                ts = node_1.get_overlapping_timesteps(node_2)
                if len(ts) == 0:
                    continue
                x_y_positions_1 = node_1.get_x_and_y(ts)
                x_y_positions_1 = self._un_preprocess_data(
                    x_y_positions_1, node_1, normalized_px, img_width, img_height
                )
                x_y_positions_2 = node_2.get_x_and_y(ts)
                x_y_positions_2 = self._un_preprocess_data(
                    x_y_positions_2, node_2, normalized_px, img_width, img_height
                )
                ade_matrix[i, j] = evaluation.compute_ade(
                    x_y_positions_1, x_y_positions_2
                )  # TODO(alem): Should be calculated in pixels
        return ade_matrix

    def _postprocess_optimal_indices(self, row_ind, col_ind, ade_matrix):
        """Remove matches with an ADE of the max value a float can hold since these are not actual matches. The max value a float can hold is meant to indicate there is no `edge.`"""
        adjusted_row_ind = []
        adjusted_col_ind = []
        for i in range(len(row_ind)):
            if ade_matrix[row_ind[i], col_ind[i]] > np.finfo(np.float64).max / 10:
                continue
            adjusted_row_ind.append(row_ind[i])
            adjusted_col_ind.append(col_ind[i])
        return np.array(adjusted_row_ind), np.array(adjusted_col_ind)

    def _get_optimal_node_matches(self, ade_matrix):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            ade_matrix
        )  # nodes, other_nodes
        row_ind, col_ind = self._postprocess_optimal_indices(
            row_ind, col_ind, ade_matrix
        )
        return row_ind, col_ind

    def _associate_nodes_with_other_nodes(
        self, nodes, other_nodes, normalized_px, img_width, img_height
    ):
        """Associates each node in `nodes` with a unique node in `other_nodes`. Each node can only be associated once.
        Nodes are associated such that the total ADE of all associations is minimized. Returns an array of nodes indices
        and an array of other_nodes indices defining the node associations (pairs)."""
        ade_matrix = self._compute_all_ades(
            nodes, other_nodes, normalized_px, img_width, img_height
        )
        row_ind, col_ind = self._get_optimal_node_matches(ade_matrix)
        return row_ind, col_ind

    def _calculate_total_overlap_time(
        self, inference_idx, gt_idx, inference_nodes, gt_nodes
    ):
        overlap_time = 0
        for inf_i, gt_i in zip(inference_idx, gt_idx):
            inf_node = inference_nodes[inf_i]
            gt_node = gt_nodes[gt_i]
            ts = inf_node.get_overlapping_timesteps(gt_node)
            overlap_time += len(ts)
        return overlap_time

    def _calculate_total_time(self, nodes):
        total_time = 0
        for node in nodes:
            total_time += node.timesteps
        return total_time

    def exists_at_t(self, gt_node, ts):
        _, paddingl, paddingu = gt_node.scene_ts_to_node_ts([ts, ts])
        return not ((paddingl > 0) or (paddingu > 0))

    def _get_preds(
        self, traj_pred, nodes, timesteps, normalized_px, img_width, img_height
    ):
        """Return predictions in the odometry frame"""
        predictions_dict = {}
        for i, ts in enumerate(timesteps):
            if ts not in predictions_dict.keys():
                predictions_dict[ts] = dict()
            predictions = np.transpose(traj_pred[:, [i]], (1, 0, 2, 3))
            predictions_shape = predictions.shape
            predictions = np.reshape(predictions, (-1, 2))
            predictions = self._un_preprocess_data(
                predictions, nodes[i], normalized_px, img_width, img_height
            )
            predictions = np.reshape(predictions, predictions_shape)
            predictions_dict[ts][int(nodes[i].id)] = predictions[0, :, :, :]
        return predictions_dict

    def _plot_displacement_error_histogram(self, de_list, bin_interval=0.1):
        """`de_list` is a list of the displaement erorrs. `bin_interval` is the width of each bin in meters"""
        plt.figure(figsize=(10, 8))
        plt.hist(
            de_list,
            bins=np.arange(min(de_list), max(de_list) + bin_interval, bin_interval),
        )
        plt.xlabel("Displacement Error (m)")
        plt.ylabel("Number of Errors")
        plt.title(
            "Histogram of Number of Errors per Displacement Error Binned in Intervals of 0.1m"
        )
        plt.xticks(ticks=range(0, math.ceil(max(de_list))))
        plt.tight_layout()
        plt.savefig(osp.join(self.model_dir, "de_distribution.jpg"))
