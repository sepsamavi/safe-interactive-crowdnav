import einops
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
import torch

from .trajectory_utils import prediction_output_to_trajectories


def compute_ade(predicted_trajs, gt_traj, interpolated_future=None, cutoff_idx=None):
    if isinstance(interpolated_future, np.ndarray):
        if cutoff_idx is not None:
            if interpolated_future[cutoff_idx] == True:
                return np.array([None])
            ade = np.linalg.norm(
                predicted_trajs[:, :, cutoff_idx, :] - gt_traj[cutoff_idx, :], axis=-1
            )
        else:
            error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
            error = np.compress(~interpolated_future, error, axis=-1)
            ade = np.mean(error, axis=-1)
    else:
        error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
        ade = np.mean(error, axis=-1)

    return ade.flatten()


# When GT of final timestep is interpolated, pass the caluculation of FDE.
def compute_fde(predicted_trajs, gt_traj, interpolated_future=None):
    if isinstance(interpolated_future, np.ndarray):
        if interpolated_future[-1] == True:
            return np.array([None])
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde(forecasts, joint_pred=True, k=20):
    """
    For joint predictions, fits a separate KDE per timestep on all agents across samples.
    For individual predictions, fits a separate KDE per timestep per agent across samples.
    :params:
    predicted_trajs (np.ndarray): num_agents x num_samples x timesteps x position (x,y)  XXX: Assume batch size is always 1
    gt_traj (np.ndarray): timesteps x position
    k (int): number of most-likely samples to return
    :returns:
    top-k most-likely samples (np.ndarray), likelihoods of the top-k most-likely samples (np.ndarray)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecasts = torch.from_numpy(forecasts).to(device)
    forecasts = einops.rearrange(
        forecasts, "humans samples horiz xy -> samples humans horiz xy"
    )
    preds_swap = einops.rearrange(
        forecasts, "samples humans horiz xy -> humans horiz samples xy"
    )
    if joint_pred:
        preds = einops.rearrange(
            preds_swap, "humans horiz samples xy -> horiz samples (humans xy)"
        )
    else:
        preds = einops.rearrange(
            preds_swap, "humans horiz samples xy -> (humans horiz) samples xy"
        )

    num_samples, num_humans, horiz, xy = forecasts.size()
    n, d = forecasts.size(0), xy * num_humans
    n = torch.tensor(n, dtype=torch.float32, device=device)

    bandwidth = torch.tensor(0.05, device=device)
    pi = torch.tensor(np.pi, device=device)

    # Compute covariance matrix (ignoring the batch dimension)
    preds_mean = torch.mean(preds, dim=1, keepdim=True)
    preds_diff = preds - preds_mean
    cov = torch.bmm(preds_diff.transpose(1, 2), preds_diff) / (n - 1)

    # Scale covariance matrix by the squared inverse of bandwidth
    scale_cov_inv = bandwidth**-2 * cov

    # Add a small constant to the diagonal to avoid singularity
    identity = torch.eye(d, device=device).expand_as(cov)
    scale_cov_inv += identity * 1e-6
    scale_cov = torch.inverse(scale_cov_inv)

    # Compute Cholesky decomposition of the covariance matrix
    lower_triangular_matrix = torch.linalg.cholesky(scale_cov)

    # Compute difference between each x in preds with all preds
    test_Xs = preds.unsqueeze(2)
    train_Xs = preds.unsqueeze(1)
    diffs = test_Xs - train_Xs

    # Multiply differences with the inverse of lower_triangular_matrix
    inv_lower_triangular_matrix = torch.linalg.inv(lower_triangular_matrix).unsqueeze(1)
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
    if joint_pred:
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
    # shape of the likelihoods all is (num_humans, num_samples)
    # get sorted indices of the likelihoods along the samples dimension (axis=1)
    sorted_indices = torch.argsort(likelihoods_all, axis=-1)
    top_k_indices = sorted_indices[..., -k:]  # num_humans x num_samples
    if joint_pred:
        new_forecasts = forecasts[top_k_indices]  # samples, humans, horiz, xy
        top_k_likelihoods_all_unnormed = likelihoods_all[top_k_indices]
    else:
        # Get the shape of forecasts and top_k_indices
        forecasts_swap = einops.rearrange(
            forecasts, "samples humans horiz xy -> humans samples horiz xy"
        )

        # Create an indices grid
        x = torch.broadcast_to(
            torch.arange(num_humans, device=device)[:, None], top_k_indices.shape
        )
        y = top_k_indices

        # Index forecasts using index arrays
        new_forecasts = forecasts_swap[x, y]

        # also get the likelihoods_all for the top k samples, keep in mind that the likelihoods_all is of shape (num_humans, num_samples)
        top_k_likelihoods_all_unnormed = likelihoods_all[x, y]
    top_k_likelihoods_all = top_k_likelihoods_all_unnormed - torch.logsumexp(
        top_k_likelihoods_all_unnormed, dim=-1, keepdim=True
    )

    # Reshape forecasts and likelihoods
    if joint_pred:
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

    forecasts_topk_samples = forecasts_topk_samples.cpu().numpy()
    top_k_likelihoods_all = top_k_likelihoods_all.cpu().numpy()
    return forecasts_topk_samples, top_k_likelihoods_all


def compute_kde_nll(predicted_trajs, gt_traj, joint_pred=True, interpolated_future=None, cutoff_idx=None):
    """
    For joint predictions, fits a separate KDE per timestep on all agents across samples.
    For individual predictions, fits a separate KDE per timestep per agent across samples.
    :params:
    predicted_trajs (np.ndarray): num_agents x num_samples x timesteps x position (x,y)  XXX: Assume batch size is always 1
    gt_traj (np.ndarray): timesteps x position
    joint_pred (bool): If predictions are joint (all the agent together) or not (individual predictions)
    :returns:
    KDE_NLL, top-k most-likely samples, likelihoods of the top-k most-likely samples
    """
    kde_ll = 0.0
    log_pdf_lower_bound = -20

    if isinstance(interpolated_future, np.ndarray):
        if cutoff_idx is not None:
            if interpolated_future[cutoff_idx] == True:
                return None
            gt_traj = gt_traj[cutoff_idx]
            predicted_trajs = predicted_trajs[:, :, cutoff_idx]
        else:
            # Delete interpolated data.
            predicted_trajs = np.compress(~interpolated_future, predicted_trajs, axis=2)
            gt_traj = np.compress(~interpolated_future, gt_traj, axis=0)

    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(
                    kde.logpdf(gt_traj[timestep].T),
                    a_min=log_pdf_lower_bound,
                    a_max=None,
                )[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(
        range(obs_map.shape[1]),
        range(obs_map.shape[0]),
        binary_dilation(obs_map.T, iterations=4),
        kx=1,
        ky=1,
    )

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False
    )
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


# TODO (alem): Copied. Move to a common file.
def _calc_kde_nll_for_each_traj(trajectories, interpolated_future=None):
    """`trajectories: batch x 12 x 2 array"""
    squeezed_trajectories = np.squeeze(trajectories)
    if isinstance(interpolated_future, np.ndarray):
        # Delete interpolated data.
        squeezed_trajectories = np.compress(
            ~interpolated_future, squeezed_trajectories, axis=1
        )

    num_batches = squeezed_trajectories.shape[0]
    num_timesteps = squeezed_trajectories.shape[1]
    kde_lls = {i: 0 for i in range(num_batches)}
    log_pdf_lower_bound = -20

    for timestep in range(num_timesteps):
        kde = gaussian_kde(squeezed_trajectories[:, timestep].T)
        pdf = np.clip(
            kde.logpdf(squeezed_trajectories[:, timestep].T),
            a_min=log_pdf_lower_bound,
            a_max=None,
        )
        for i in range(num_batches):
            kde_lls[i] += pdf[i]
    for i in range(num_batches):
        kde_lls[i] /= num_timesteps
    kde_nlls = {i: -kde_lls[i] for i in range(num_batches)}
    return kde_nlls


def get_most_likely_samples(forecasts, mid_model, num_ret_samples):
    preds_swap = einops.rearrange(
        forecasts, "samples humans horiz xy -> humans horiz samples xy"
    )
    if "_jp" in mid_model.config["method"]:
        preds = einops.rearrange(
            preds_swap, "humans horiz samples xy -> horiz samples (humans xy)"
        )
        # bandwidth = torch.tensor(0.05, device = "cuda" if torch.cuda.is_available() else "cpu")
        # bandwidth = torch.linspace(0.01, 0.05, steps=forecasts.size(2), device = "cuda" if torch.cuda.is_available() else "cpu")
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
    if "_jp" in mid_model.config["method"]:
        scale_cov_inv = bandwidth[:, None, None] ** -2 * cov
    else:
        scale_cov_inv = bandwidth**-2 * cov

    # Add a small constant to the diagonal to avoid singularity
    identity = torch.eye(d, device = "cuda" if torch.cuda.is_available() else "cpu").expand_as(cov)
    scale_cov_inv += identity * 1e-6
    scale_cov = torch.inverse(scale_cov_inv)

    # Compute Cholesky decomposition of the covariance matrix
    lower_triangular_matrix = torch.linalg.cholesky(scale_cov)

    # Compute difference between each x in preds with all preds
    test_Xs = preds.unsqueeze(2)
    train_Xs = preds.unsqueeze(1)
    diffs = test_Xs - train_Xs

    # Multiply differences with the inverse of lower_triangular_matrix
    inv_lower_triangular_matrix = torch.linalg.inv(lower_triangular_matrix).unsqueeze(1)
    if "_jp" in mid_model.config["method"]:
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
    if "_jp" in mid_model.config["method"]:
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
    if "_jp" in mid_model.config["method"]:
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
    if "_jp" in mid_model.config["method"]:
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


# TODO (alem): Move to a common file.
def get_most_likely_trajectory_idx(trajectories, interpolated_future=None):
    """Returns the most likely trajectory from the set of `trajectories` based on KDE-NLL.

    Inputs:
        trajectories (np.ndarray): 20 (batch) x 12 (timesteps) x 2 (position)"""

    kde_nlls = _calc_kde_nll_for_each_traj(trajectories, interpolated_future)
    most_likely_traj_idx = min(kde_nlls, key=kde_nlls.get)
    return most_likely_traj_idx


def compute_batch_statistics(
    prediction_output_dict,
    dt,
    max_hl,
    ph,
    node_type_enum,
    gt_node_means,
    inf_node_means,
    target_node_id=None,
    kde=True,
    compute_ml=True,
    obs=False,
    map=None,
    prune_ph_to_future=False,
    best_of=False,
    all_de=False,
    normalized_px=False,
    img_width=0,
    img_height=0,
    is_eval_hst=False,
    joint_pred=True,

):
    """
    `prediction_output_dict` is a dictionary of predictions: {ts: {agent: np.array (num_agents x num_samples x ph x 2)}}
    `compute_ml` specifies if the statistics for the most-likely trajectories should be computed
    `target_node_id` is the ID of the node that the joint data item was constructed for
    """

    (
        prediction_dict,
        _,
        futures_dict,
        interpolated_futures_dict,
        interpolated_histories_dict,
    ) = prediction_output_to_trajectories(
        prediction_output_dict,
        dt,
        max_hl,
        ph,
        prune_ph_to_future=prune_ph_to_future,
        is_eval_hst=is_eval_hst,
    )

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {
            "ade": list(),
            "ade_one_fourth": list(),
            "ade_two_fourth": list(),
            "ade_three_fourth": list(),
            "ade_most_likely": list(),
            "ade_most_likely_one_fourth": list(),
            "ade_most_likely_two_fourth": list(),
            "ade_most_likely_three_fourth": list(),
            "ade_mean": list(),
            "ade_std": list(),
            "ade_mean_one_fourth": list(),
            "ade_mean_two_fourth": list(),
            "ade_mean_three_fourth": list(),
            "ade_list": list(),
            "fde": list(),
            "fde_most_likely": list(),
            "fde_mean": list(),
            "fde_std": list(),
            "fde_list": list(),
            "kde": list(),
            "kde_one_fourth": list(),
            "kde_two_fourth": list(),
            "kde_three_fourth": list(),
            "obs_viols": list(),
            "de_list": list(),
            "sade": list(),
            "sade_mean": list(),
            "sade_std": list(),
            "sfde": list(),
            "sfde_mean": list(),
            "sfde_std": list(),
        }

    for t in prediction_dict.keys():
        ade_per_node = []
        fde_per_node = []
        for node in prediction_dict[t].keys():
            if is_eval_hst:
                # When GT of all history or future is interpolated, pass the caluculation of all metrics.
                interpolated_history = interpolated_histories_dict[t][int(node.id)]
                interpolated_future = interpolated_futures_dict[t][int(node.id)]
                if np.all(interpolated_history) or np.all(interpolated_future):
                    continue
            predictions = prediction_dict[t][node] # 1 x num_samples x pred_horiz x 2
            gt = futures_dict[t][node]  # pred_horiz x 2
            predictions = predictions + np.expand_dims(
                np.expand_dims(np.expand_dims(inf_node_means, axis=0), axis=0), axis=0
            )
            gt = gt + np.expand_dims(gt_node_means, axis=0)

            if normalized_px:
                predictions = predictions * np.array([[[[img_width, img_height]]]])
                gt = gt * np.array([[img_width, img_height]])
            if is_eval_hst:
                ade_errors = compute_ade(predictions, gt, interpolated_future)  # ADE per sample
                fde_errors = compute_fde(predictions, gt, interpolated_future)
                ade_per_node.append(ade_errors)
                fde_per_node.append(fde_errors)
            else:
                ade_errors = compute_ade(predictions, gt)  # ADE per sample
                fde_errors = compute_fde(predictions, gt)
                ade_per_node.append(ade_errors)
                fde_per_node.append(fde_errors)
            if kde:
                if is_eval_hst:
                    kde_ll = compute_kde_nll(predictions, gt, joint_pred=joint_pred, interpolated_future=interpolated_future)
                else:
                    kde_ll = compute_kde_nll(predictions, gt, joint_pred=joint_pred)
            else:
                kde_ll = 0
            if compute_ml:
                if is_eval_hst:
                    most_likely_traj_idx = get_most_likely_trajectory_idx(
                        predictions, interpolated_future
                    )
                else:
                    most_likely_traj_idx = get_most_likely_trajectory_idx(predictions)
                most_likely_ade_errors = ade_errors[most_likely_traj_idx]
                if (fde_errors == None).any() != True:
                    most_likely_fde_errors = fde_errors[most_likely_traj_idx]
            else:
                most_likely_ade_errors = 0
                most_likely_fde_errors = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                min_ade_errors = np.min(ade_errors, keepdims=True)
                if (fde_errors == None).any() != True:
                    min_fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            if all_de:  # Compute all displacement errors
                de_array = np.linalg.norm(predictions - gt, axis=-1).flatten()

            mean_ade_errors = np.mean(ade_errors, keepdims=True)
            std_ade_errors = np.std(ade_errors, keepdims=True)
            if (fde_errors == None).any() != True:
                mean_fde_errors = np.mean(fde_errors, keepdims=True)
                std_fde_errors = np.std(fde_errors, keepdims=True)

            if node.id == target_node_id:
                batch_error_dict[node.type]["ade"].extend(list(min_ade_errors))
                batch_error_dict[node.type]["ade_most_likely"].extend(
                    [most_likely_ade_errors]
                )
                batch_error_dict[node.type]["ade_mean"].extend(list(mean_ade_errors))
                batch_error_dict[node.type]["ade_std"].extend(list(std_ade_errors))
                batch_error_dict[node.type]["ade_list"].extend(list(ade_errors))
                batch_error_dict[node.type]["fde"].extend(list(min_fde_errors))
                batch_error_dict[node.type]["fde_most_likely"].extend(
                    [most_likely_fde_errors]
                )
                batch_error_dict[node.type]["fde_mean"].extend(list(mean_fde_errors))
                batch_error_dict[node.type]["fde_std"].extend(list(std_fde_errors))
                batch_error_dict[node.type]["fde_list"].extend(list(fde_errors))
                batch_error_dict[node.type]["kde"].extend([kde_ll])
                batch_error_dict[node.type]["obs_viols"].extend([obs_viols])
                if all_de:
                    batch_error_dict[node.type]["de_list"].extend(list(de_array))

            if is_eval_hst:
                ade_errors_one_fourth = compute_ade(
                    predictions, gt, interpolated_future, cutoff_idx=2
                )
                ade_errors_two_fourth = compute_ade(
                    predictions, gt, interpolated_future, cutoff_idx=5
                )
                ade_errors_three_fourth = compute_ade(
                    predictions, gt, interpolated_future, cutoff_idx=8
                )
                if (ade_errors_one_fourth == None).any() != True:
                    min_ade_errors_one_fourth = np.min(
                        ade_errors_one_fourth, keepdims=True
                    )
                    batch_error_dict[node.type]["ade_one_fourth"].extend(
                        list(min_ade_errors_one_fourth)
                    )
                    mean_ade_errors_one_fourth = np.mean(
                        ade_errors_one_fourth, keepdims=True
                    )
                    batch_error_dict[node.type]["ade_mean_one_fourth"].extend(
                        list(mean_ade_errors_one_fourth)
                    )
                    if compute_ml:
                        mlade_errors_one_fourth = ade_errors_one_fourth[
                            most_likely_traj_idx
                        ]
                        batch_error_dict[node.type][
                            "ade_most_likely_one_fourth"
                        ].extend([mlade_errors_one_fourth])
                if (ade_errors_two_fourth == None).any() != True:
                    min_ade_errors_two_fourth = np.min(
                        ade_errors_two_fourth, keepdims=True
                    )
                    batch_error_dict[node.type]["ade_two_fourth"].extend(
                        list(min_ade_errors_two_fourth)
                    )
                    mean_ade_errors_two_fourth = np.mean(
                        ade_errors_two_fourth, keepdims=True
                    )
                    batch_error_dict[node.type]["ade_mean_two_fourth"].extend(
                        list(mean_ade_errors_two_fourth)
                    )
                    if compute_ml:
                        mlade_errors_two_fourth = ade_errors_two_fourth[
                            most_likely_traj_idx
                        ]
                        batch_error_dict[node.type][
                            "ade_most_likely_two_fourth"
                        ].extend([mlade_errors_two_fourth])
                if (ade_errors_three_fourth == None).any() != True:
                    min_ade_errors_three_fourth = np.min(
                        ade_errors_three_fourth, keepdims=True
                    )
                    batch_error_dict[node.type]["ade_three_fourth"].extend(
                        list(min_ade_errors_three_fourth)
                    )
                    mean_ade_errors_three_fourth = np.mean(
                        ade_errors_three_fourth, keepdims=True
                    )
                    batch_error_dict[node.type]["ade_mean_three_fourth"].extend(
                        list(mean_ade_errors_three_fourth)
                    )
                    if compute_ml:
                        mlade_errors_three_fourth = ade_errors_three_fourth[
                            most_likely_traj_idx
                        ]
                        batch_error_dict[node.type][
                            "ade_most_likely_three_fourth"
                        ].extend([mlade_errors_three_fourth])
                if kde:
                    kde_ll_one_fourth = compute_kde_nll(
                        predictions, gt, joint_pred=joint_pred, interpolated_future=interpolated_future, cutoff_idx=2
                    )
                    kde_ll_two_fourth = compute_kde_nll(
                        predictions, gt, joint_pred=joint_pred, interpolated_future=interpolated_future, cutoff_idx=5
                    )
                    kde_ll_three_fourth = compute_kde_nll(
                        predictions, gt, joint_pred=joint_pred, interpolated_future=interpolated_future, cutoff_idx=8
                    )
                    if kde_ll_one_fourth != None:
                        batch_error_dict[node.type]["kde_one_fourth"].extend(
                            [kde_ll_one_fourth]
                        )
                    if kde_ll_two_fourth != None:
                        batch_error_dict[node.type]["kde_two_fourth"].extend(
                            [kde_ll_two_fourth]
                        )
                    if kde_ll_three_fourth != None:
                        batch_error_dict[node.type]["kde_three_fourth"].extend(
                            [kde_ll_three_fourth]
                        )

        sade = ade_per_node[0]  # keeps track of sade per sample
        sfde = fde_per_node[0]  # keeps track of sfde per sample
        if len(ade_per_node) > 1:
            for ade in ade_per_node[1:]:
                sade = sade + ade
            for fde in fde_per_node[1:]:
                sfde = sfde + fde
        sade = sade/len(ade_per_node)
        sfde = sfde/len(fde_per_node)
        min_sade = np.min(sade, keepdims=True)
        min_sfde = np.min(sfde, keepdims=True)
        mean_sade = np.mean(sade, keepdims=True)
        std_sade = np.std(sade, keepdims=True)
        mean_sfde = np.mean(sfde, keepdims=True)
        std_sfde = np.std(sfde, keepdims=True)
        batch_error_dict[node.type]["sade"].extend(list(min_sade))
        batch_error_dict[node.type]["sade_mean"].extend(list(mean_sade))
        batch_error_dict[node.type]["sade_std"].extend(list(std_sade))
        batch_error_dict[node.type]["sfde"].extend(list(min_sfde))
        batch_error_dict[node.type]["sfde_mean"].extend(list(mean_sfde))
        batch_error_dict[node.type]["sfde_std"].extend(list(std_sfde))

    return batch_error_dict, interpolated_futures_dict, interpolated_histories_dict


# def compute_batch_statistics(
#     prediction_output_dict,
#     dt,
#     max_hl,
#     ph,
#     node_type_enum,
#     gt_node_means,
#     inf_node_means,
#     kde=True,
#     compute_ml=True,
#     obs=False,
#     map=None,
#     prune_ph_to_future=False,
#     best_of=False,
#     all_de=False,
#     normalized_px=False,
#     img_width=0,
#     img_height=0,
#     joint_pred=True,
# ):
#     """
#     `compute_ml` specifies if the statistics for the most-likely trajectories should be computed.
#     `joint_pred` specifies if the predictions being evaluated are joint predictions or not. This is controlled by the config by appending "_jp" to the end of the method field.
#     """

#     (prediction_dict, _, futures_dict) = prediction_output_to_trajectories(
#         prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future
#     )

#     batch_error_dict = dict()
#     for node_type in node_type_enum:
#         batch_error_dict[node_type] = {
#             "ade": list(),
#             "ade_most_likely": list(),
#             "ade_mean": list(),
#             "ade_std": list(),
#             "ade_list": list(),
#             "fde": list(),
#             "fde_most_likely": list(),
#             "fde_mean": list(),
#             "fde_std": list(),
#             "fde_list": list(),
#             "kde": list(),
#             "obs_viols": list(),
#             "de_list": list(),
#         }

#     for t in prediction_dict.keys():
#         # Store all predictions and ground truths across agents to compute KDE-NLL
#         all_agent_predictions = []
#         all_agent_gts = []
#         for node in prediction_dict[t].keys():
#             predictions = prediction_dict[t][node]
#             gt = futures_dict[t][node]
#             predictions = predictions + np.expand_dims(
#                 np.expand_dims(np.expand_dims(inf_node_means, axis=0), axis=0), axis=0
#             )
#             gt = gt + np.expand_dims(gt_node_means, axis=0)
#             if normalized_px:
#                 predictions = predictions * np.array([[[[img_width, img_height]]]])
#                 gt = gt * np.array([[img_width, img_height]])

#             all_agent_predictions.append(predictions)
#             all_agent_gts.append(gt)

#         all_agent_predictions = np.stack(all_agent_predictions).squeeze(
#             axis=1
#         )  # humans x samples x horiz x (xy)
#         all_agent_gts = np.stack(all_agent_gts)  # humans x horiz x (xy)
#         if kde:
#             # kde_ll = compute_kde_nll(all_agent_predictions, all_agent_gts)
#             # XXX: Assume the order of humans is the same as input
#             # Forecasts are sorted from most-likely to least
#             forecasts_topk_samples, top_k_likelihoods_all = compute_kde(
#                 all_agent_predictions, joint_pred=joint_pred
#             )  # humans x samples x horiz x (xy)
#             kde_ll = 0  # For now do not compute KDE-NLL
#         else:
#             kde_ll = 0

#         for agent in range(all_agent_gts.shape[0]):
#             ade_errors = compute_ade(forecasts_topk_samples, all_agent_gts[agent])
#             fde_errors = compute_fde(forecasts_topk_samples, all_agent_gts[agent])
#             most_likely_ade_errors = ade_errors[0]
#             most_likely_fde_errors = fde_errors[0]
#             if obs:
#                 obs_viols = compute_obs_violations(forecasts_topk_samples, map)
#             else:
#                 obs_viols = 0
#             min_ade_errors = np.min(ade_errors, keepdims=True)
#             min_fde_errors = np.min(fde_errors, keepdims=True)
#             if all_de:  # Compute all displacement errors
#                 de_array = np.linalg.norm(
#                     forecasts_topk_samples - gt, axis=-1
#                 ).flatten()

#             mean_ade_errors = np.mean(ade_errors, keepdims=True)
#             std_ade_errors = np.std(ade_errors, keepdims=True)
#             mean_fde_errors = np.mean(fde_errors, keepdims=True)
#             std_fde_errors = np.std(fde_errors, keepdims=True)

#             batch_error_dict[node.type]["ade"].extend(list(min_ade_errors))
#             batch_error_dict[node.type]["ade_most_likely"].extend(
#                 [most_likely_ade_errors]
#             )
#             batch_error_dict[node.type]["ade_mean"].extend(list(mean_ade_errors))
#             batch_error_dict[node.type]["ade_std"].extend(list(std_ade_errors))
#             batch_error_dict[node.type]["ade_list"].extend(list(ade_errors))
#             batch_error_dict[node.type]["fde"].extend(list(min_fde_errors))
#             batch_error_dict[node.type]["fde_most_likely"].extend(
#                 [most_likely_fde_errors]
#             )
#             batch_error_dict[node.type]["fde_mean"].extend(list(mean_fde_errors))
#             batch_error_dict[node.type]["fde_std"].extend(list(std_fde_errors))
#             batch_error_dict[node.type]["fde_list"].extend(list(fde_errors))
#             batch_error_dict[node.type]["kde"].extend([kde_ll])
#             batch_error_dict[node.type]["obs_viols"].extend([obs_viols])
#             if all_de:
#                 batch_error_dict[node.type]["de_list"].extend(list(de_array))

#     return batch_error_dict


# def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
#     for node_type in batch_errors_list[0].keys():
#         for metric in batch_errors_list[0][node_type].keys():
#             metric_batch_error = []
#             for batch_errors in batch_errors_list:
#                 metric_batch_error.extend(batch_errors[node_type][metric])

#             if len(metric_batch_error) > 0:
#                 log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

#                 if metric in bar_plot:
#                     pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

#                 if metric in box_plot:
#                     mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean",
                    np.mean(metric_batch_error),
                )
                print(
                    f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median",
                    np.median(metric_batch_error),
                )


def batch_pcmd(
    prediction_output_dict,
    dt,
    max_hl,
    ph,
    node_type_enum,
    kde=True,
    obs=False,
    map=None,
    prune_ph_to_future=False,
    best_of=False,
):
    (prediction_dict, _, futures_dict, _, _) = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=prune_ph_to_future
    )

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {
            "ade": list(),
            "fde": list(),
            "kde": list(),
            "obs_viols": list(),
        }

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(
                    prediction_dict[t][node], futures_dict[t][node]
                )
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]["ade"].append(np.array(ade_errors))
            batch_error_dict[node.type]["fde"].append(np.array(fde_errors))

    return batch_error_dict
