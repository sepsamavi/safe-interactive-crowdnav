import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
import pandas as pd
from copy import copy, deepcopy
try:
    from environment import Scene, Node, derivative_of
except ImportError:
    from ..environment import Scene, Node, derivative_of
from itertools import chain

container_abcs = collections.abc


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data

def masked_fill(batch):
    for elem in batch:
        if isinstance(elem, torch.Tensor):
            elem.masked_fill_(torch.isnan(elem), 0)
    return batch

def generate_mask(batch, batch_size, pred_length):
    """
    Makes an attention mask for the Transformer and loss mask to indicate which model outputs
    should be back-propagated on.

    Returns:
    attn_mask: batch_size * agent_num * pred_length x batch_size * agent_num * pred_length
    loss_mask: batch_size x num_agents * pred_length
    """
    # shape of batch[2]: (batch_size * agent_num) x pred_length x 2
    # Reason for initializing with a diagonal matrix:
    # Because nn.TransformerEncoder is designed to return nan
    # if there is always at least one element in each row for which attention is calculated.
    # Calculate the number of agents per data item
    agent_num = batch[2].shape[0] // batch_size
    # Calculate the total length of the attention mask
    total_length = batch_size * agent_num * pred_length

    # Initialize the attention mask as an identity matrix
    attn_mask = torch.eye(total_length).float()

    # Process each data item
    for batch_id in range(batch_size):
        # Calculate the start and end indices for the current data item
        start_id = batch_id * agent_num
        end_id = start_id + agent_num

        # Determine which agents have NaN values
        nan_mask = torch.isnan(batch[2][start_id:end_id]).any(dim=2).any(dim=1)
        # Get the indices of agents with NaN values
        nan_indices = torch.nonzero(nan_mask).squeeze()

        # Calculate the start index for the mask
        mask_start_id = batch_id * agent_num * pred_length
        if nan_indices.numel() > 0:
            # Determine the first agent with NaN values
            nan_starts_at = nan_indices.item() if nan_indices.dim() == 0 else nan_indices[0].item()
            # Calculate the end index for the mask based on the first NaN agent
            mask_end_id = mask_start_id + nan_starts_at * pred_length
        else:
            # If no NaN agents, set the end index to include all agents
            mask_end_id = mask_start_id + agent_num * pred_length

        # Set the attention mask to 1 for the relevant range
        attn_mask[mask_start_id:mask_end_id, mask_start_id:mask_end_id] = 1

    # Convert the attention mask to float and apply masked_fill
    attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float("-inf")).masked_fill(attn_mask == 1, 0.0)

    # Compute the loss mask to identify sequences with all NaN values
    loss_mask = torch.isnan(batch[2]).all(dim=2).all(dim=1).view(batch_size, -1)
    # Repeat the loss mask for the prediction length
    loss_mask = loss_mask.repeat_interleave(pred_length, dim=1)

    return attn_mask, loss_mask

def calculate_max_num_agents(batch):
    max_num_agents = 0
    for data_item in batch:
        if data_item[0].shape[0] > max_num_agents:
            max_num_agents = data_item[0].shape[0]
    return max_num_agents

def get_dataitem_id_and_num_agents_without_max_num_agents(batch, max_num_agents):
    ids = []
    # lacks_of_num_agents = []
    for id, data_item in enumerate(batch):
        if data_item[0].shape[0] < max_num_agents:
            ids.append(id)
            # lacks_of_num_agents.append(max_num_agents - data_item[0].shape[0])
    return ids # , lacks_of_num_agents

def fill_nan_agents(batch, max_num_agents):
    fill_nan_batch = batch

    batch_ids = get_dataitem_id_and_num_agents_without_max_num_agents(batch, max_num_agents)

    for batch_id in batch_ids:
        for i, elem in enumerate(batch[batch_id]):
            if elem is None:
                continue
            elif isinstance(elem, dict):
                if i == 5:
                    for key in elem.keys():
                        for j in range(max_num_agents - len(elem[tuple(key)])):
                            elem[tuple(key)].append([])
                else:
                    for key in elem.keys():
                        for _ in range(max_num_agents - len(elem[tuple(key)])):
                            elem[tuple(key)].append(torch.zeros(1))
            else:
                if i == 0: # for information of first time step
                    nan_filled_elem = torch.full((max_num_agents, *elem.size()[1:]), int(0))
                else:
                    nan_filled_elem = torch.full((max_num_agents, *elem.size()[1:]), float('nan'))
                nan_filled_elem[:elem.shape[0]] = elem
                batch[batch_id][i] = nan_filled_elem

    return fill_nan_batch

def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if (
            len(elem) == 4
        ):  # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(
                scene_map,
                scene_pts=torch.Tensor(scene_pts),
                patch_size=patch_size[0],
                rotation=heading_angle,
            )
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return (
            dill.dumps(neighbor_dict)
            if torch.utils.data.get_worker_info()
            else neighbor_dict
        )
    return default_collate(batch)

def joint_pred_train_collate(batch):
    if len(batch) == 0:
        return batch

    if isinstance(batch, list):
        max_num_agents = calculate_max_num_agents(batch)
        # max_num_agents = 20
        batch = fill_nan_agents(batch, max_num_agents)

    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if (
            len(elem) == 4
        ):  # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(
                scene_map,
                scene_pts=torch.Tensor(scene_pts),
                patch_size=patch_size[0],
                rotation=heading_angle,
            )
            return map
        transposed = zip(*batch)
        # return [joint_pred_train_collate(samples) for samples in transposed]
        output_batch = [joint_pred_train_collate(samples) for samples in transposed]
        batch_size = output_batch[0].shape[0]
        output_batch = reshape_batch(output_batch)
        return output_batch, batch_size
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return neighbor_dict
        # return (
        #     dill.dumps(neighbor_dict)
        #     if torch.utils.data.get_worker_info()
        #     else neighbor_dict
        # )
    return default_collate(batch)


def reshape_tensor(tensor):
    if tensor.dim() == 2:  # shape [4, 30]
        return tensor.view(-1)  # shape [120]
    elif tensor.dim() == 4:  # shape [4, 30, 6, 6]
        return tensor.view(-1, tensor.size(2), tensor.size(3))  # shape [120, 6, 6]
    else:
        return tensor

def flatten_list(nested_list):
    return list(chain.from_iterable(nested_list))

def process_element(element):
    if isinstance(element, dict):
        for key in element:
            element[key] = list(chain.from_iterable(element[key]))
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        if torch.utils.data.get_worker_info():
            element = dill.dumps(element)
        return element
    elif torch.is_tensor(element):
        return reshape_tensor(element)
    else:
        return element

def reshape_batch(batch):
    return [process_element(element) for element in batch]


def get_list_data_items_from_batch(batch, batch_size):
    list_data_items = []
    if batch_size == 1:
        list_data_items.append(batch)
    else:
        agent_num = int(batch[0].shape[0] / batch_size)
        for batch_id in range(batch_size):
            data_item = deepcopy(batch)
            start_id = (batch_id * agent_num)
            end_id = ((batch_id * agent_num) + agent_num)
            for i, elem in enumerate(batch):
                if isinstance(elem, torch.Tensor):
                    data_item[i] = data_item[i][start_id:end_id]
                elif isinstance(elem, dict):
                    for key in batch[i].keys():
                        data_item[i][tuple(key)] = data_item[i][tuple(key)][start_id:end_id]
                else:
                    continue
            list_data_items.append(data_item)
    return list_data_items

def remove_nan_rows(array):
    return array[~np.isnan(array).any(axis=1)]

def check_padding_value(st_hist, target_current_pos):
    # Introduce a checker for the neighborhood history since it is 0-padded.
    out_st_hist = np.zeros(st_hist[:, 0:2].shape)
    out_st_hist = np.full(st_hist.shape, float('nan'))
    is_pad = np.all(np.isclose(st_hist[:, 0:2] + target_current_pos, 0.0, atol=1e-4), axis=1)
    if not np.any(is_pad):
        last_pad_index = -1
    else:
        last_pad_index = np.argmax(np.diff(is_pad.astype(int)) == -1)

    out_st_hist[last_pad_index+1:] = st_hist[last_pad_index+1:]
    return out_st_hist

def allclose_with_nan(a, b, rtol=0, atol=1e-4):
    mask = np.isnan(a) & np.isnan(b)
    return np.allclose(a[~mask], b[~mask], rtol=rtol, atol=atol) and np.array_equal(np.isnan(a), np.isnan(b))

def get_node_current_x(node, time_step, state):
    timestep_range = np.array([time_step])
    node_current_x = node.get(timestep_range, state)
    return node_current_x


def rotate_pc(pc, alpha, base_pos=np.array([0, 0])):
    shift_pc = pc - base_pos
    M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    shift_rotated_pc = M @ shift_pc
    rotated_pc = shift_rotated_pc + base_pos
    return rotated_pc


def augment_scene(scene, angle_deg, time_step):
    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    scene_aug = Scene(
        timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene
    )

    target_node = scene.nodes[0]

    target_node_current_pos = get_node_current_x(
        target_node, time_step, {"position": ["x", "y"]}
    )
    target_node_current_pos = target_node_current_pos.transpose([1, 0])

    alpha = angle_deg * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        # rotate all agents based on the direction of target agent at current time step
        x, y = rotate_pc(np.array([x, y]), alpha, base_pos=target_node_current_pos)

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

        node = Node(
            node_type=node.type,
            node_id=node.id,
            data=node_data,
            first_timestep=node.first_timestep,
        )

        scene_aug.nodes.append(node)
    return scene_aug


def generate_scene_in_attention_radius(
    node,
    scene,
    t,
    env,
    edge_types,
    hyperparams,
    min_history_timesteps=0,
    min_future_timesteps=0,
):
    scene_in_att_rad = copy(scene)
    # get scene graph
    scene_graph = scene.get_scene_graph(
        t,
        env.attention_radius,
        hyperparams["edge_addition_filter"],
        hyperparams["edge_removal_filter"],
    )
    # get neighbor agents
    all_valid_neighbor_nodes = []
    for edge_type in edge_types:
        neighbor_nodes = scene_graph.get_neighbors(node, edge_type[1])
        if neighbor_nodes.size == 0:
            continue
        # filtered list
        valid_nodes = scene.present_nodes(
            np.array([t]),
            type=edge_type[1],
            min_history_timesteps=min_history_timesteps,
            min_future_timesteps=min_future_timesteps,
        )
        if len(valid_nodes) == 0:
            continue
        valid_neighbor_nodes = neighbor_nodes[
            np.isin(neighbor_nodes, np.array(valid_nodes[t]))
        ]

        all_valid_neighbor_nodes.extend(valid_neighbor_nodes.tolist())

    # update scene including target and neighbor agents
    target_neighbor_agents = []
    target_neighbor_agents.append(node)
    target_neighbor_agents.extend(all_valid_neighbor_nodes)
    scene_in_att_rad.nodes = target_neighbor_agents

    return scene_in_att_rad, neighbor_nodes


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(
        robot_traj, state[robot_type], node_type=robot_type, mean=node_traj, std=std
    )
    robot_traj_st = (robot_traj_st.reshape(-1, 2) @ rot_mat).reshape(
        -1, robot_traj.shape[1]
    )  # Post-multiply to rotate coordinates by -heading radians
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def compute_heading(x_vel, y_vel):
    """
    heading ranges from [-pi, pi].
    """
    heading = np.arctan2(y_vel, x_vel)
    return heading


def get_node_timestep_data(
    env,
    scene,
    t,
    node,
    state,
    pred_state,
    edge_types,
    max_ht,
    max_ft,
    hyperparams,
    scene_graph=None,
    target_node_current_x=None,
):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    # get position of GT
    pos_y = node.get(timestep_range_y, state[node.type])[:, 0:2]
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    # heading = compute_heading(
    #     np.array(x)[-1, 2], np.array(x)[-1, 3]
    # )  # Estimate heading from current x and y velocities
    # rot_mat = np.array(
    #     [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
    # )

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(
        x, state[node.type], node.type, mean=rel_state, std=std
    )  # hist x state
    # x_st = (x_st.reshape(-1, 2) @ rot_mat).reshape(
    #     -1, x.shape[1]
    # )  # Post-multiply to rotate coordinates by -heading radians
    if (
        list(pred_state[node.type].keys())[0] == "position"
    ):  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)
    # y_st = y_st @ rot_mat  # Post-multiply to rotate coordinates by -heading radians

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    pos_y_t = torch.tensor(pos_y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams["edge_encoding"]:  # True in trajectron_hypers.py
        # Scene Graph
        scene_graph = (
            scene.get_scene_graph(
                t,
                env.attention_radius,
                hyperparams["edge_addition_filter"],
                hyperparams["edge_removal_filter"],
            )
            if scene_graph is None
            else scene_graph
        )

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:  # PEDESTRIAN -> x
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams["dynamic_edges"] == "yes":  # "yes" in trajectron_hypers.py
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(
                    scene_graph.get_edge_scaling(node), dtype=torch.float
                )
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(
                    np.array([t - max_ht, t]), state[connected_node.type], padding=0.0
                )

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(
                    state[connected_node.type], node_type=connected_node.type
                )
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(
                    neighbor_state_np,
                    state[connected_node.type],
                    node_type=connected_node.type,
                    mean=rel_state,
                    std=std,
                )
                # neighbor_state_np_st = (
                #     neighbor_state_np_st.reshape(-1, 2) @ rot_mat
                # ).reshape(
                #     -1, neighbor_state_np.shape[1]
                # )  # Post-multiply to rotate coordinates by -heading radians

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    timestep_range_r = np.array([t, t + max_ft])
    if hyperparams["incl_robot_node"]:  # False in trajectron_hypers.py
        print(
            "[get_node_timestep_data] WARNING: the robot frame needs to be oriented correctly"
        )
        x_node = node.get(timestep_range_r, state[node.type])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        robot_traj_st_t = get_relative_robot_traj(
            env, state, x_node, robot_traj, node.type, robot_type
        )

    # Map
    map_tuple = None
    if hyperparams["use_map_encoding"]:  # False in trajectron_hypers.py
        if node.type in hyperparams["map_encoder"]:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams["map_encoder"][node.type]
            if "heading_state_index" in me_hyp:
                heading_state_index = me_hyp["heading_state_index"]
                # We have to rotate the map in the opposit direction of the agent to match them
                if (
                    type(heading_state_index) is list
                ):  # infer from velocity or heading vector
                    heading_angle = (
                        -np.arctan2(
                            x[-1, heading_state_index[1]], x[-1, heading_state_index[0]]
                        )
                        * 180
                        / np.pi
                    )
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams["map_encoder"][node.type]["patch_size"]
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    return (
        first_history_index,  # int
        x_t,  # 6 (history) x 6 (state)
        y_t,  # 8 (future) x 2 (state)
        x_st_t,  # 6 (history) x 6 (state)
        y_st_t,  # 8 (future) x 2 (state)
        neighbors_data_st,  # {edge (PEDESTRIAN -> X): [6 (history) x 6 (state)]}
        neighbors_edge_value,  # {edge (PEDESTRIAN -> X): [num_neighbors]}
        robot_traj_st_t,  # None
        map_tuple,  # None
        pos_y_t,  # 8 (future) x 2 (state)
    )


def get_timesteps_data(
    env,
    scene,
    t,
    node_type,
    state,
    pred_state,
    edge_types,
    min_ht,
    max_ht,
    min_ft,
    max_ft,
    hyperparams,
):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    # Get valid pedestrian nodes
    nodes_per_ts = scene.present_nodes(
        t,
        type=node_type,
        min_history_timesteps=min_ht,
        min_future_timesteps=max_ft,
        return_robot=not hyperparams["incl_robot_node"],
    )
    batch = list()
    nodes = list()
    out_timesteps = list()
    # sample timestamp (0,1,2,...)
    for timestep in nodes_per_ts.keys():
        # Get scene graph for Trajectron++
        scene_graph = scene.get_scene_graph(
            timestep,
            env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"],
        )
        present_nodes = nodes_per_ts[timestep]
        # sample history and future trajectory and neghbor pedestrian information in last frame per track id
        for node in present_nodes:
            nodes.append(node)
            out_timesteps.append(timestep)
            batch.append(
                get_node_timestep_data(
                    env,
                    scene,
                    timestep,
                    node,
                    state,
                    pred_state,
                    edge_types,
                    max_ht,
                    max_ft,
                    hyperparams,
                    scene_graph=scene_graph,
                )
            )
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps


def find_indices_of_missing_elements(list_A, list_B):
    missing_elements_with_indices = [
        (index, item) for index, item in enumerate(list_B) if item not in list_A
    ]
    return missing_elements_with_indices


def add_unique_elements(list_A, list_B):
    for item in list_A:
        if item not in list_B:
            list_B.append(item)


# import time
def get_scenes_in_att_rad(
    env,
    scene,
    t,
    node_type,
    state,
    pred_state,
    edge_types,
    min_ht,
    max_ht,
    min_ft,
    max_ft,
    hyperparams,
    gen_one_scene_per_node=False,
):
    # start = time.time()
    # Get valid pedestrian nodes
    nodes_per_ts = scene.present_nodes(
        t,
        type=node_type,
        min_history_timesteps=min_ht,
        min_future_timesteps=max_ft,
        return_robot=not hyperparams["incl_robot_node"],
    )
    if len(nodes_per_ts) == 0:
        if gen_one_scene_per_node:
            return None, None
        return None
    scenes_in_att_rad = []
    are_scenes_unique = []
    processed_nodes = []
    unique_scene_node_ids = []
    timestep = t[0]
    # print("present_nodes", time.time() - start)
    for node in nodes_per_ts[timestep]:
        # create new scene corresponding to target agent
        # by using attention radius and sample neighbor agents
        # TODO: min_future_timesteps以下のノードに対して，シーングラフに含める？実験によって判断．
        # TODO: generate_scene_in_attention_radious should be faster.
        # s1 = time.time()
        selected_edge_type = [
            edge_type for edge_type in edge_types if edge_type[0] == node.type
        ]
        (
            scene_in_att_rad,
            neighbor_nodes,
        ) = generate_scene_in_attention_radius(
            node,
            scene,
            timestep,
            env=env,
            edge_types=selected_edge_type,
            hyperparams=hyperparams,
            min_history_timesteps=min_ht,  # TODO: True?
            min_future_timesteps=min_ft,  # TODO: True?
        )
        # print("generate_scene_in_attetion_radius", time.time() - s1)
        # s2 = time.time()

        if gen_one_scene_per_node:
            scenes_in_att_rad.append(scene_in_att_rad)
            unique_scene = True
            scene_node_id_set = set([node.id for node in scene_in_att_rad.nodes])
            for node_id_set in unique_scene_node_ids:
                if node_id_set == scene_node_id_set:
                    unique_scene = False
            if unique_scene:
                are_scenes_unique.append(True)
                unique_scene_node_ids.append(scene_node_id_set)
            else:
                are_scenes_unique.append(False)
        # Only if there is any un-handled nodes.
        # is_including_needed_nodes = any(item not in processed_nodes for item in scene_in_att_rad.nodes)
        # if is_including_needed_nodes:
        else:
            missing_elements = find_indices_of_missing_elements(
                processed_nodes, scene_in_att_rad.nodes
            )
            # print("find_indices_of_missing_elements", time.time() - s2)
            # s3 = time.time()
            if len(missing_elements) > 0:
                scenes_in_att_rad.append(scene_in_att_rad)
                add_unique_elements(scene_in_att_rad.nodes, processed_nodes)
            # print("add_unique_elements", time.time() - s3)

    if gen_one_scene_per_node:
        return scenes_in_att_rad, are_scenes_unique
    return scenes_in_att_rad


def get_timesteps_each_scene_data(
    env,
    scene,
    t,
    node_type,
    state,
    pred_state,
    edge_types,
    min_ht,
    max_ht,
    min_ft,
    max_ft,
    hyperparams,
):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    timestep = t[0]

    batch = list()
    nodes = list()
    out_timesteps = list()
    # Get normalizing value based on target node
    target_node_current_x = None
    # Get scene graph for Trajectron++
    scene_graph = scene.get_scene_graph(
        timestep,
        env.attention_radius,
        hyperparams["edge_addition_filter"],
        hyperparams["edge_removal_filter"],
    )
    all_present_nodes = scene.nodes
    present_nodes = [node for node in all_present_nodes if node.type.name in node_type]
    # sample history and future trajectory and neghbor pedestrian information in last frame per track id
    for node in present_nodes:
        nodes.append(node)
        out_timesteps.append(timestep)
        batch.append(
            get_node_timestep_data(
                env,
                scene,
                timestep,
                node,
                state,
                pred_state,
                edge_types,
                max_ht,
                max_ft,
                hyperparams,
                scene_graph=scene_graph,
                target_node_current_x=target_node_current_x,
            )
        )
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps

def concat_node_timestep_data(
    nodes_timestep_data, node_timestep_data, iter_node
):
    # Initialize output data
    if iter_node == 0:
        for k in range(len(node_timestep_data)):
            # print(node_timestep_data[k])
            if node_timestep_data[k] is None:
                nodes_timestep_data.append(None)
            elif type(node_timestep_data[k]) is dict:
                dict_list = {}
                for key in node_timestep_data[k].keys():
                    dict_list[tuple(key)] = [node_timestep_data[k][tuple(key)]]
                nodes_timestep_data.append(dict_list)
            elif not torch.is_tensor(node_timestep_data[k]):
                nodes_timestep_data.append(torch.tensor([node_timestep_data[k]]))
            else:
                nodes_timestep_data.append(
                    node_timestep_data[k].reshape(
                        node_timestep_data[k].shape[0],
                        node_timestep_data[k].shape[1],
                        1,
                    )
                )
        return nodes_timestep_data

    # Concatnate output data
    for k in range(len(node_timestep_data)):
        if node_timestep_data[k] is None:
            nodes_timestep_data[k] = None
        else:
            if type(node_timestep_data[k]) is dict:
                for key in node_timestep_data[k].keys():
                    insert_element = node_timestep_data[k][tuple(key)]
                    nodes_timestep_data[k][tuple(key)].append(insert_element)
            else:
                if torch.is_tensor(node_timestep_data[k]):
                    insert_element = node_timestep_data[k]
                    insert_element = insert_element.reshape(
                        insert_element.shape[0], insert_element.shape[1], 1
                    )
                else:
                    insert_element = torch.tensor([node_timestep_data[k]])
                nodes_timestep_data[k] = torch.cat(
                    (nodes_timestep_data[k], insert_element.clone().detach()),
                    dim=-1,
                )
    return nodes_timestep_data

def permute_nodes_timestemp_data_for_bs_one(nodes_timestep_data):
    for k in range(len(nodes_timestep_data)):
        if torch.is_tensor(nodes_timestep_data[k]):
            if nodes_timestep_data[k].dim() == 3:
                nodes_timestep_data[k] = nodes_timestep_data[k].permute(2, 0, 1)
    return nodes_timestep_data
