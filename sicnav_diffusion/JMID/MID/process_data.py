import argparse
import dill
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from environment import Environment, Scene, Node, derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
# default sampling interval
default_dt = 0.4

# If raw data does not have class information, all agents are treated as pedestrians and below standardization is used
standardization_just_ped = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}
# If raw data has class information, below standardization is used
standardization_all_class = {
    "PEDESTRIAN": {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    },
    # Trajectron++ did not consider bicycles, so for now we use PEDESTRIAN standardization for BICYCLE standardization
    # Update BICYCLE standardization to follow vehicle if we decide to predict bicycle trajectories
    "BICYCLE": {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process raw data into the proper format for MID"
    )
    parser.add_argument(
        "--desired_sources",
        help="Specify sources of raw data to process. If multiple sources, separate with commas",
        default="SICNav_TRO_MID_data",
    )
    parser.add_argument(
        "--splits",
        help="Specify data splits to process. If multiple splits, separate with commas",
    )
    return parser.parse_args()


def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def load_pos(full_data_path):
    data = pd.read_csv(full_data_path, sep="\t", index_col=False, header=None)
    data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
    return data[["pos_x", "pos_y"]]


def augment_scene(scene, angle_deg):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    scene_aug = Scene(
        timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene
    )

    alpha = angle_deg * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

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
            aux_data=node._aux_data,
        )

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


def check_has_class_info(split_list, desired_source):
    """
    Check if the raw data contains the class information or not based on the
    number of columns in the raw data. Assumes that checking just one file in
    the raw data is sufficient.
    """
    for data_class in split_list:
        for subdir, dirs, files in os.walk(
            os.path.join("raw_data", desired_source, data_class)
        ):
            for file in tqdm(files):
                if file.endswith(".txt") or file.endswith(".csv"):
                    full_data_path = os.path.join(subdir, file)

                    if file.endswith(".csv"):
                        data = pd.read_csv(
                            full_data_path, index_col=False, header=None, skiprows=1
                        )
                    else:
                        data = pd.read_csv(
                            full_data_path, sep="\t", index_col=False, header=None
                        )
                    if (data.shape[1] == 4 and desired_source != "jrdb_bev_hst") or desired_source == "jrdb_bev_hst":
                        return False
                    elif data.shape[1] == 5 and desired_source != "jrdb_bev_hst":
                        return True
                    else:
                        raise ValueError("Raw data has incorrect number of columns")


def main():
    args = parse_args()

    data_folder_name = "processed_data"
    # data_folder_name = 'yolov8_bytetracktrack_data'

    maybe_makedirs(data_folder_name)
    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    # XXX: The below dictionary is incomplete
    # Image dimensions are width in pixels by height in pixels
    # If a source is not included in this dictionary, then it is assumed that its data is not in normalized pixel values (0 to 1)
    source_to_img_dim_px = {
        "jrdb_pifenet_simple_track_ego": (3760, 480)
    }  # JRDB stitched image dimensions

    dt_source = {"jrdb_bev_hst": 0.33}  # JRDB stitched image dimensions

    # Process ETH-UCY / PersonPath22 / JRDB
    desired_sources = [
        "eth",
        "hotel",
        "univ",
        "zara1",
        "zara2",
        "personpath",
        "jrdb",
        "jrdb_stitched",
        "jrdb_yolov8x_stitched_bytetracker",
        "jrdb_yolov8n_stitched_bytetracker",
        "jrdb_bev",
        "jrdb_pifenet_simple_track",
        "jrdb_pifenet_simple_track_ego",
        "jrdb_bev_hst",
        "pixel_eth",
        "jrdb_bev_inx2_yolov8x_bytetracker",
        "SICNav_TRO_MID_data",
        "sim",
    ]
    if args.desired_sources is not None:
        desired_sources = args.desired_sources.split(",")
    for desired_source in desired_sources:
        if desired_source in [
            "jrdb",
            "jrdb_stitched",
            "jrdb_yolov8x_stitched_bytetracker",
            "jrdb_yolov8n_stitched_bytetracker",
            "jrdb_pifenet_simple_track_ego",
            "personpath",
        ]:
            split_list = ["train", "test"]
        elif desired_source in [
            "jrdb_bev",
            "jrdb_bev_hst",
            "jrdb_bev_0_4_multi_class_clean",
            "jrdb_bev_0.25",
            "jrdb_bev_0_25_multi_class",
            "SICNav_TRO_MID_data",
        ]:
            split_list = ["train", "val"]
        elif desired_source in ["jrdb_bev_inx2_yolov8x_bytetracker"]:
            split_list = ["val"]
        elif desired_source in ["utias_rosbags_vicon"]:
            split_list = ["test"]
        else:
            split_list = ["train", "val", "test"]
        if args.splits is not None:
            split_list = args.splits.split(",")

        # Set flag to indicate if there is class information or not in the raw data
        has_class_info = check_has_class_info(split_list, desired_source)
        for data_class in split_list:
            print("Processing " + data_class + " split")
            if has_class_info:
                env = Environment(
                    node_type_list=["PEDESTRIAN", "BICYCLE", "JRDB_ROBOT"],
                    standardization=standardization_all_class
                )
            else:
                env = Environment(
                    node_type_list=["PEDESTRIAN"], standardization=standardization_just_ped
                )
            # IMPORTANT
            attention_radius = dict()
            if has_class_info:
                # attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0

                # # Trajectron++ did not consider bicycles, so I just doubled the attention radius for pedestrians for bicycles
                # attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.BICYCLE)] = 10.0
                # attention_radius[(env.NodeType.BICYCLE, env.NodeType.PEDESTRIAN)] = 10.0
                # attention_radius[(env.NodeType.BICYCLE, env.NodeType.BICYCLE)] = 10.0

                # attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.JRDB_ROBOT)] = 10.0
                # attention_radius[(env.NodeType.JRDB_ROBOT, env.NodeType.PEDESTRIAN)] = 10.0

                # attention_radius[(env.NodeType.BICYCLE, env.NodeType.JRDB_ROBOT)] = 10.0
                # attention_radius[(env.NodeType.JRDB_ROBOT, env.NodeType.BICYCLE)] = 10.0
                # attention_radius[(env.NodeType.JRDB_ROBOT, env.NodeType.JRDB_ROBOT)] = 10.0
                # env.attention_radius = attention_radius
                # env.robot_type = env.NodeType.PEDESTRIAN  # treat robot as pedestrian since I assume its speeds are about as fast as a pedestrian's; this is used if we want to condition on future robot motion

                # Let's try a smaller attention radius
                attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0

                # Trajectron++ did not consider bicycles, so I just doubled the attention radius for pedestrians for bicycles
                attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.BICYCLE)] = 3.0
                attention_radius[(env.NodeType.BICYCLE, env.NodeType.PEDESTRIAN)] = 3.0
                attention_radius[(env.NodeType.BICYCLE, env.NodeType.BICYCLE)] = 3.0

                attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.JRDB_ROBOT)] = 3.0
                attention_radius[(env.NodeType.JRDB_ROBOT, env.NodeType.PEDESTRIAN)] = 3.0

                attention_radius[(env.NodeType.BICYCLE, env.NodeType.JRDB_ROBOT)] = 3.0
                attention_radius[(env.NodeType.JRDB_ROBOT, env.NodeType.BICYCLE)] = 3.0
                attention_radius[(env.NodeType.JRDB_ROBOT, env.NodeType.JRDB_ROBOT)] = 3.0
            else:
                attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
            env.attention_radius = attention_radius

            scenes = []
            data_dict_path = os.path.join(
                data_folder_name, "_".join([desired_source, data_class]) + ".pkl"
            )

            for subdir, dirs, files in os.walk(
                os.path.join(os.getcwd(), "raw_data", desired_source, data_class)
            ):
                for file in tqdm(files):
                    if file.endswith(".txt") or file.endswith(".csv"):
                        full_data_path = os.path.join(subdir, file)

                        if file.endswith(".csv"):
                            data = pd.read_csv(
                                full_data_path, index_col=False, header=None, skiprows=1
                            )
                        else:
                            data = pd.read_csv(
                                full_data_path, sep="\t", index_col=False, header=None
                            )
                        # Create flag to check if there is class information or not
                        # If there are five columns and the desired source is not jrdb_bev_hst, process according to RA-L branch
                            # This just needs custom class processing instead of setting everything to PEDESTRIAN
                        # If there are four columns and the desired source is not jrdb_bev_hst or if the desired source is jrdb_bev_hst, process according to main branch
                            # Set everything to PEDESTRIAN class
                        if desired_source == "jrdb_bev_hst":
                            data.columns = [
                                "frame_id",
                                "track_id",
                                "pos_x",
                                "pos_y",
                                "interpolated",
                            ]
                        else:
                            if has_class_info:
                                data.columns = ["frame_id", "track_id", "pos_x", "pos_y", "node_type"]  # node_type is either PEDESTRIAN, BICYCLE, or ROBOT
                            else:
                                data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
                        data["frame_id"] = pd.to_numeric(
                            data["frame_id"], downcast="integer"
                        )
                        data["track_id"] = pd.to_numeric(
                            data["track_id"], downcast="integer"
                        )

                        if "sim" not in desired_source:
                            data["frame_id"] = data["frame_id"] // 10

                        # data['frame_id'] -= data['frame_id'].min()

                        if not has_class_info:
                            data["node_type"] = "PEDESTRIAN"
                        data["node_id"] = data["track_id"].astype(str)

                        data.sort_values("frame_id", inplace=True)

                        # calculate a mean of position each scene
                        pos_x_mean = data["pos_x"].mean()
                        pos_y_mean = data["pos_y"].mean()
                        data["pos_x"] = data["pos_x"] - pos_x_mean
                        data["pos_y"] = data["pos_y"] - pos_y_mean

                        max_timesteps = data["frame_id"].max()

                        dt = dt_source.get(desired_source)
                        if not dt:
                            dt = default_dt
                        img_dim = source_to_img_dim_px.get(desired_source)
                        if img_dim:
                            scene = Scene(
                                timesteps=max_timesteps + 1,
                                dt=dt,
                                name=file.split(".")[0],
                                aug_func=augment if data_class == "train" else None,
                                normalized_px=True,
                                img_width=img_dim[0],
                                img_height=img_dim[1],
                            )
                        else:
                            scene = Scene(
                                timesteps=max_timesteps + 1,
                                dt=dt,
                                name=file.split(".")[0],
                                aug_func=augment if data_class == "train" else None,
                            )

                        for node_id in pd.unique(data["node_id"]):
                            node_df = data[data["node_id"] == node_id]

                            node_values = node_df[["pos_x", "pos_y"]].values

                            if node_values.shape[0] < 2:
                                continue

                            new_first_idx = node_df["frame_id"].iloc[0]

                            # TODO: Interporate? Zero-padding?
                            # We can not pad zero because it does not make sense for evaluation.
                            # Interpration is good, but it has hypothesis that is there is no re-id annotation for people returing to the camera again.
                            # We have two option.
                            # Check above.
                            # Rename tracking ID for too long tracklet.
                            ##

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
                            if has_class_info:
                                if node_df.iloc[0]["node_type"] == "PEDESTRIAN":
                                    node_type = env.NodeType.PEDESTRIAN
                                elif node_df.iloc[0]["node_type"] == "BICYCLE":
                                    node_type = env.NodeType.BICYCLE
                                else:
                                    node_type = env.NodeType.JRDB_ROBOT
                            else:
                                node_type = env.NodeType.PEDESTRIAN
                            aux_data = {
                                "pos_x_mean": pos_x_mean,
                                "pos_y_mean": pos_y_mean,
                            }
                            if desired_source == "jrdb_bev_hst":
                                aux_data["interpolated"] = node_df[
                                    ["interpolated"]
                                ].values
                            node = Node(
                                node_type=node_type,
                                node_id=node_id,
                                data=node_data,
                                aux_data=aux_data,
                            )
                            node.first_timestep = new_first_idx

                            # Store node specially if it is the robot; used if we want to condition on robot future motion
                            # if node_df.iloc[0]["node_type"] == "ROBOT":
                            # node.is_robot = True
                            # scene.robot = node

                            scene.nodes.append(node)
                        if data_class == "train":
                            scene.augmented = list()
                            angles_deg = (
                                np.arange(0, 360, 15) if data_class == "train" else [0]
                            )
                            for angle_deg in angles_deg:
                                scene.augmented.append(augment_scene(scene, angle_deg))

                        scenes.append(scene)
            print(f"Processed {len(scenes):.2f} scene for data class {data_class}")

            env.scenes = scenes

            if len(scenes) > 0:
                with open(data_dict_path, "wb") as f:
                    dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
