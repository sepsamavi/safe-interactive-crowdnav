import torch
import torch.nn.functional as F


def get_diffs_pred(traj):
    """Same order of ped pairs as pdist.
    Input:
        - traj: (ts, n_ped, 2)"""
    num_peds = traj.shape[1]
    return torch.cat(
        [
            torch.tile(traj[:, ped_i : ped_i + 1], (1, num_peds - ped_i - 1, 1))
            - traj[:, ped_i + 1 :]
            for ped_i in range(num_peds)
        ],
        dim=1,
    )


def lineseg_dist(a, b):
    """
    Computes how far away the lines starting at `a` and ending at `b` are away from 0.
    https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
    https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
    a: N x 2
    b: N x 2
    """
    # reduce computation
    if torch.all(a == b):
        return torch.linalg.norm(-a, dim=1)

    # normalized tangent vector
    d = torch.zeros_like(a)
    a_eq_b = torch.all(a == b, axis=-1)
    d[~a_eq_b] = (b - a)[~a_eq_b] / torch.linalg.norm(
        b[~a_eq_b] - a[~a_eq_b], axis=-1, keepdims=True
    )

    # signed parallel distance components
    s = (a * d).sum(axis=-1)
    t = (-b * d).sum(axis=-1)

    # clamped parallel distance
    distances = torch.stack([s, t, torch.zeros_like(t)])  # 3 x N
    h = torch.max(distances, dim=0).values

    # perpendicular distance component
    c = torch.cross(F.pad(-a, (0, 1)), F.pad(d, (0, 1)), dim=-1)[..., 2]

    ans = torch.hypot(h, torch.abs(c))

    # edge case where agent stays still
    ans[a_eq_b] = torch.linalg.norm(-a, dim=1)[a_eq_b]

    return ans


def calc_min_dists(positions):
    """
    Calculates the minimum distance between agents over all timesteps where the trajectory
    for each agent is between timesteps is line parameterized by two end-points.
    positions: (n_agents, ts, 2)
    """
    positions = positions.permute(1, 0, 2)
    ts, num_peds, _ = positions.size()
    num_ped_pairs = (num_peds * (num_peds - 1)) // 2

    ped_pair_diffs_pred = get_diffs_pred(positions)  # ts x agent_pairs x 2
    if ped_pair_diffs_pred.size(0) == 1:
        pxy = ped_pair_diffs_pred[0].reshape(-1, 2)
        exy = ped_pair_diffs_pred[0].reshape(-1, 2)
    else:
        pxy = ped_pair_diffs_pred[:-1].reshape(-1, 2)  # (ts - 1) x agent_pairs
        exy = ped_pair_diffs_pred[1:].reshape(-1, 2)
    # Compute how far the lines are from 0. This is the minimum distance between agents.
    collision_t_pred = lineseg_dist(pxy, exy)  # (ts - 1) * agent_pairs
    if ts > 1:
        collision_t_pred = collision_t_pred.reshape(ts - 1, num_ped_pairs)
    collision_t_pred = torch.min(collision_t_pred, 0).values
    return collision_t_pred


def get_agents_in_collision(pairwise_distances, num_peds):
    """
    Returns IDs of agents in collisions
    pairwise_distances (np.array): (num_peds * (num_peds - 1)) // 2
    """
    in_collisions = torch.nonzero(pairwise_distances < 0.2)  # N x 1
    if torch.numel(in_collisions) == 0:
        return in_collisions  # empty tensor
    in_collisions = in_collisions.squeeze(1)
    mat_ids = torch.triu_indices(
        num_peds, num_peds, 1, device=in_collisions.device
    )  # 2 x M
    collision_mat_ids = torch.index_select(mat_ids, 1, in_collisions)
    agent_ids = torch.unique(collision_mat_ids)
    return agent_ids


def check_collision_velocity(vel, dynamics):
    """
    Checks if there is a collision for given velocities or not
    vel (torch.tensor): n_agents, ts, 2 (x, y)
    """
    pos = dynamics.integrate_samples(vel.unsqueeze(0)).squeeze(0)  # (n_agents, ts, 2)
    dists = calc_min_dists(pos)
    colliding_agent_ids = get_agents_in_collision(dists, pos.size(0))
    return torch.numel(colliding_agent_ids) > 0
