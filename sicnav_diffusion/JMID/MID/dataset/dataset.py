import os
import dill
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from .preprocessing import (
    get_node_timestep_data,
    generate_scene_in_attention_radius,
    augment_scene,
    concat_node_timestep_data,
    permute_nodes_timestemp_data_for_bs_one
)


class EnvironmentDataset(object):
    def __init__(
        self,
        env,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        is_joint_pred=False,
        regenerate_index=False,
        index_path="",
        is_dataloader_debug=False,
        **kwargs
    ):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.max_ft = kwargs["min_future_timesteps"]
        self.node_type_datasets = list()
        self._augment = False
        self.is_joint_pred = is_joint_pred
        self.is_dataloader_debug = is_dataloader_debug
        if self.is_joint_pred:
            for node_type in env.NodeType:
                if node_type not in hyperparams["pred_state"]:
                    continue
                self.node_type_datasets.append(
                    JointPredNodeTypeDataset(
                        env,
                        node_type,
                        state,
                        pred_state,
                        node_freq_mult,
                        scene_freq_mult,
                        hyperparams,
                        regenerate_index=regenerate_index,
                        index_path=index_path,
                        is_dataloader_debug=is_dataloader_debug,
                        **kwargs
                    )
                )
        else:
            for node_type in env.NodeType:
                if node_type not in hyperparams["pred_state"]:
                    continue
                self.node_type_datasets.append(
                    NodeTypeDataset(
                        env,
                        node_type,
                        state,
                        pred_state,
                        node_freq_mult,
                        scene_freq_mult,
                        hyperparams,
                        regenerate_index=regenerate_index,
                        index_path=index_path,
                        **kwargs
                    )
                )

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(
        self,
        env,
        node_type,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        regenerate_index=False,
        index_path="",
        augment=False,
        is_dataloader_debug=False,
        **kwargs
    ):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams["maximum_history_length"]
        self.max_ft = kwargs["min_future_timesteps"]

        self.is_dataloader_debug = is_dataloader_debug

        self.augment = augment

        self.node_type = node_type
        self.edge_types = [
            edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type
        ]
        self.index = self.index_env(node_freq_mult, scene_freq_mult, regenerate_index=regenerate_index, index_path=index_path, **kwargs)
        self.len = len(self.index)

    def index_env(self, node_freq_mult, scene_freq_mult, regenerate_index=False, index_path="", **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(
                np.arange(0, scene.timesteps), type=self.node_type, **kwargs
            )
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += (
                        [(scene, t, node)]
                        * (scene.frequency_multiplier if scene_freq_mult else 1)
                        * (node.frequency_multiplier if node_freq_mult else 1)
                    )

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        return get_node_timestep_data(
            self.env,
            scene,
            t,
            node,
            self.state,
            self.pred_state,
            self.edge_types,
            self.max_ht,
            self.max_ft,
            self.hyperparams,
        )


class JointPredNodeTypeDataset(NodeTypeDataset):
    def __init__(
        self,
        env,
        node_type,
        state,
        pred_state,
        node_freq_mult,
        scene_freq_mult,
        hyperparams,
        regenerate_index=False,
        index_path="",
        augment=False,
        is_dataloader_debug=False,
        **kwargs
    ):
        super().__init__(
            env,
            node_type,
            state,
            pred_state,
            node_freq_mult,
            scene_freq_mult,
            hyperparams,
            regenerate_index=regenerate_index,
            index_path=index_path,
            augment=False,
            is_dataloader_debug=is_dataloader_debug,
            **kwargs
        )
        self.aug_degs = np.arange(0, 360, 15)

    def index_env(self, node_freq_mult, scene_freq_mult, regenerate_index=False, index_path="", **kwargs):

        if regenerate_index is False:
            if not os.path.isfile(index_path):
                raise ValueError("Index was specified to not be regenerated and instead loaded from disk. The index file does not exist though (" + index_path + ")")
            with open(index_path, "rb") as f:
                return dill.load(f, encoding="latin1")

        index = list()
        max_nodes_len = 0
        min_nodes_len = float("inf")
        total_nodes_len = 0
        count_iter = 0
        # Index each scene
        for scene in tqdm(self.env.scenes):
            max_scene_nodes_len = 0
            min_scene_nodes_len = float("inf")
            total_scene_nodes_len = 0
            count_scene_iter = 0
            # Get all the nodes (agents) in the scene with data after the first timestep (due to the min_history_timesteps of 1) and from eight timesteps away from the last timestep of the scene
            # Gets all the nodes of type self.node_type (PEDESTRIAN)
            present_node_dict = scene.present_nodes(
                np.arange(0, scene.timesteps), type=self.node_type, **kwargs
            )
            # sample target agent
            for t, nodes in tqdm(present_node_dict.items()):
                for node in nodes:
                    # create new scene corresponding to target agent
                    # by using attention radius and sample neighbor agents
                    # Only store neighbour nodes with at least min_history_timesteps of history (1 in addition to the current time) and min_future_timesteps of data into the future
                    (
                        scene_in_att_rad,
                        neighbor_nodes,
                    ) = generate_scene_in_attention_radius(
                        node,
                        scene,
                        t,
                        env=self.env,
                        edge_types=self.edge_types,  # PEDESTRIAN -> x
                        hyperparams=self.hyperparams,
                        min_history_timesteps=kwargs["min_history_timesteps"],
                        min_future_timesteps=kwargs["min_future_timesteps"],
                    )
                    # store scene data
                    index += [(scene_in_att_rad, t)] * (
                        scene.frequency_multiplier if scene_freq_mult else 1
                    )
                    if len(scene_in_att_rad.nodes) > max_scene_nodes_len:
                        max_scene_nodes_len = len(scene_in_att_rad.nodes)
                    if len(scene_in_att_rad.nodes) < min_scene_nodes_len:
                        min_scene_nodes_len = len(scene_in_att_rad.nodes)
                    total_scene_nodes_len += len(scene_in_att_rad.nodes)
                    count_scene_iter += 1
                    total_nodes_len += len(scene_in_att_rad.nodes)
                    count_iter += 1
                    # break  # TODO: only if doing fast test
            print(scene.name)
            print(
                "Average Batch Number:{}, Max Batch Number:{}, Min Batch Number:{}".format(
                    int(total_scene_nodes_len / count_scene_iter),
                    max_scene_nodes_len,
                    min_scene_nodes_len,
                )
            )
            if max_scene_nodes_len > max_nodes_len:
                max_nodes_len = max_scene_nodes_len
            if min_scene_nodes_len < min_nodes_len:
                min_nodes_len = min_scene_nodes_len
            # break  # TODO: only if doing fast test
        print("All data")
        print(
            "Average Batch Number:{}, Max Batch Number:{}, Min Batch Number:{}".format(
                int(total_nodes_len / count_iter), max_nodes_len, min_nodes_len
            )
        )

        with open(index_path, "wb") as f:
            dill.dump(index, f, protocol=dill.HIGHEST_PROTOCOL)

        return index

    def __getitem__(self, i):
        (org_scene, t) = self.index[i]

        # if self.augment:
        #     scene = scene.augment()

        if self.augment:
            angle_deg = np.random.choice(self.aug_degs)
            scene = augment_scene(org_scene, angle_deg, t)
        else:
            scene = org_scene
        # If you would like to get non augment dataset, you should comment out.
        # angle_deg = 0
        # scene = org_scene

        all_nodes = scene.nodes

        nodes = [node for node in all_nodes if node.type.name in self.node_type.name]

        target_node_current_x = None
        nodes_timestep_data = []
        for iter_node, node in enumerate(nodes):
            node = scene.get_node_by_id(node.id)
            node_timestep_data = get_node_timestep_data(
                self.env,
                scene,
                t,
                node,
                self.state,
                self.pred_state,
                self.edge_types,
                self.max_ht,
                self.max_ft,
                self.hyperparams,
                target_node_current_x=target_node_current_x,
            )

            nodes_timestep_data = concat_node_timestep_data(
                nodes_timestep_data, node_timestep_data, iter_node
            )

        nodes_timestep_data = permute_nodes_timestemp_data_for_bs_one(
            nodes_timestep_data
        )

        if self.is_dataloader_debug:
            print("Data augmentation degree: {}".format(angle_deg))
            print("Number of agents: {}".format(nodes_timestep_data[0].shape[0]))

        return nodes_timestep_data
