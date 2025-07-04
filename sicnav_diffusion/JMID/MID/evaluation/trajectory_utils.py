import numpy as np


def prediction_output_to_trajectories(
    prediction_output_dict,
    dt,
    max_h,
    ph,
    map=None,
    prune_ph_to_future=False,
    is_eval_hst=False,
):
    """
    `output_dict` is a dictionary of predictions {timestamp: {pedestrian: predictions (np.array: num_samples x pred_horiz x 2)}}
    `futures_dict` is a dictionary of ground truth futures {timestamp: {pedestrian: future (np.array: pred_horiz x 2)}}
    """
    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()
    interpolated_futures_dict = dict()
    interpolated_histories_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        interpolated_futures_dict[t] = dict()
        interpolated_histories_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {"position": ["x", "y"]}

            history = node.get(
                np.array([t - max_h, t]), position_state
            )  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]
            future = node.get(np.array([t + 1, t + ph]), position_state)
            # replace nan to 0
            # future[np.isnan(future)] = 0
            future = future[~np.isnan(future.sum(axis=1))]

            if is_eval_hst:
                if "interpolated" in node._aux_data.keys():
                    interpolated_future = np.full((ph), fill_value=True)
                    org_interpolated_future = node._aux_data["interpolated"][
                        t + 1 : t + ph + 1
                    ].flatten()
                    interpolated_future[
                        0 : org_interpolated_future.shape[0]
                    ] = org_interpolated_future
                    interpolated_history = np.full((max_h + 1), fill_value=True)
                    org_interpolated_history = node._aux_data["interpolated"][
                        t - max_h : t + 1
                    ].flatten()
                    interpolated_history[
                        0 : org_interpolated_history.shape[0]
                    ] = org_interpolated_history
                else:
                    print("Check the train/val data if it includes 'interpolated' key.")
                    exit()

                interpolated_futures_dict[t][int(node.id)] = interpolated_future
                interpolated_histories_dict[t][int(node.id)] = interpolated_history

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, : future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future)

    return (
        output_dict,
        histories_dict,
        futures_dict,
        interpolated_futures_dict,
        interpolated_histories_dict,
    )
