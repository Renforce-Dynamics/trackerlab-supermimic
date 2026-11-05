"""Terrain downstream objectives shared by Warp collection and imagination."""

import torch

from trackerLab.kinematics.supermimic.rotations import mat_to_6d, yaw_mat
from trackerLab.tracker_env.mdp.supermimic import invariant_features

TERMS = ("velocity", "progress", "alive", "upright", "lane", "height",
         "posture", "action_rate", "joint_velocity", "joint_limit")
SUCCESS_BONUS = 0.5
FALL_PENALTY = 0.5
MAX_STEP_REWARD = 0.05
OBJECTIVE_VERSION = 2


def success_terminal_value(gamma):
    if not 0 < gamma < 1:
        raise ValueError("Discount factor must be in (0, 1)")
    # Replace the positive continuation lost at the goal, then add a success bonus.
    return SUCCESS_BONUS + gamma * MAX_STEP_REWARD / (1 - gamma)


def terminal_masks(reached_goal, fallen, timeout):
    success = reached_goal & ~fallen
    return success, fallen | success | timeout, timeout & ~fallen & ~success


def update_terrain_levels(levels, success, fallen, max_level=5):
    change = success.to(levels.dtype) - fallen.to(levels.dtype)
    return (levels + change).clamp(0, max_level)


class HeightField:
    """Piecewise-linear MuJoCo heightfield with derivatives in XY coordinates."""

    def __init__(self, terrain, height, device):
        self.terrain, self.height, self.scale = terrain, height, height / .76
        self.values = torch.as_tensor(terrain.heights, device=device)
        self.rows, self.cols = self.values.shape
        self.constant_surface = bool((terrain.heights == terrain.heights[0, 0]).all())
        x, y = torch.meshgrid(torch.linspace(-.3, 1.6, 17, device=device),
                              torch.linspace(-.55, .55, 11, device=device), indexing="ij")
        self.offsets = torch.stack((x.flatten(), y.flatten()), -1) * self.scale
        self.model_offsets = torch.stack((x[::2, ::2].flatten(), y[::2, ::2].flatten()), -1) * self.scale
        self.model_indices = torch.arange(self.offsets.shape[0], device=device).reshape(17, 11)[::2, ::2].flatten()

    def sample(self, xy):
        if self.constant_surface:
            # Constant terrain has zero XY derivatives; avoid grid lookup in
            # every imagined observation while retaining the height tensor.
            return xy[..., 0] * 0. + self.values[0, 0]
        t = self.terrain
        x = ((xy[..., 0] - t.x_min) / (t.x_max - t.x_min) * (self.cols - 1)).clamp(0, self.cols - 1)
        y = ((xy[..., 1] - t.y_min) / (t.y_max - t.y_min) * (self.rows - 1)).clamp(0, self.rows - 1)
        ix, iy = x.long().clamp(max=self.cols - 2), y.long().clamp(max=self.rows - 2)
        fx, fy = x - ix, y - iy
        h00, h10 = self.values[iy, ix], self.values[iy, ix + 1]
        h01, h11 = self.values[iy + 1, ix], self.values[iy + 1, ix + 1]
        # MuJoCo splits each cell along the (0, 0) to (1, 1) diagonal.
        lower = h00 + fx * (h10 - h00) + fy * (h11 - h10)
        upper = h00 + fx * (h11 - h01) + fy * (h01 - h00)
        return torch.where(fx >= fy, lower, upper)

    def inside(self, xy):
        t = self.terrain
        return ((xy[..., 0] >= t.x_min) & (xy[..., 0] <= t.x_max) &
                (xy[..., 1] >= t.y_min) & (xy[..., 1] <= t.y_max))

    def scan_xy(self, S, offsets=None):
        H = yaw_mat(S[1])
        offsets = self.offsets if offsets is None else offsets
        return S[0][..., None, :2] + (H[..., None, :2, :2] @ offsets[..., None]).squeeze(-1)

    def features(self, S, offsets=None):
        offsets = self.offsets if offsets is None else offsets
        xy = self.scan_xy(S, offsets)
        scan = (S[0][..., 2:3] - self.sample(xy) - self.height) / self.scale
        return torch.where(self.inside(xy), scan, torch.ones_like(scan)).clamp(-1, 1) * 2

    def model_features(self, S):
        return self.features(S, self.model_offsets)

    def invariant(self, S):
        value = invariant_features(S)
        clearance = S[0][..., 2:3] - self.sample(S[0][..., :2]).unsqueeze(-1)
        return torch.cat((clearance, value[..., 1:]), -1)


class TerrainMDP:
    # c = (lane center Y in meters, commanded world-forward speed in m/s).
    def __init__(self, field, default_q, qlo, qhi, gamma=.99):
        self.field, self.default_q = field, default_q
        self.success_reward = success_terminal_value(gamma)
        mid, half = (qlo + qhi) / 2, (qhi - qlo) * .45
        self.soft_lo, self.soft_hi = mid - half, mid + half
        self.num_joints = len(default_q)
        self.obs_dim, self.goal_dim = 3 * self.num_joints + 15 + len(field.offsets), 6
        n, dev = self.num_joints, default_q.device
        self.obs_noise = torch.cat((torch.full((n,), .01), torch.full((n,), .5),
                                    torch.full((3,), .05), torch.full((3,), .05),
                                    torch.zeros(n), torch.full((3,), .05), torch.full((6,), .02),
                                    torch.full((len(field.offsets),), .01 / field.scale))).to(dev)

    def obs(self, S, a_prev, xi, c):
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        desired = torch.eye(3, device=p.device).expand_as(R)
        proprio = torch.cat((q - self.default_q - xi[..., 1:1+self.num_joints], qd * .1,
                             (Rt @ w.unsqueeze(-1)).squeeze(-1) * .25, -R[..., 2, :], a_prev,
                             (Rt @ v.unsqueeze(-1)).squeeze(-1) / self.field.scale,
                             mat_to_6d(Rt @ desired)), -1)
        return torch.cat((proprio, self.field.features(S)), -1)

    def add_noise(self, o):
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    def goal(self, S, c):
        p, R = S[:2]
        return torch.stack((c[..., 1] / self.field.scale, R[..., 0, 0], -R[..., 1, 0],
                            (c[..., 0] - p[..., 1]) / self.field.scale,
                            ((self.field.terrain.goal_x - p[..., 0]) / self.field.scale).clamp(-1, 3),
                            torch.full_like(c[..., 0], .76)), -1)

    def reward(self, S, a, a_prev, c, terms=False):
        p, R, v, w, q, qd = S
        scale, height = self.field.scale, self.field.height
        clearance = p[..., 2] - self.field.sample(p[..., :2])
        lane = (p[..., 1] - c[..., 0]) / scale
        ev = ((v[..., 0] - c[..., 1]) / scale).square() + (v[..., 1] / scale).square()
        values = ( .03 * torch.exp(-ev / .25), .01 * (v[..., 0] / scale).clamp(-1, 1.5),
                   torch.full_like(lane, .005), -.01 * (1 - R[..., 2, 2]), -.001 * lane.square(),
                   -.002 * (clearance / height - 1).square(),
                   -.0006 * (q - self.default_q).square().mean(-1),
                   -.0008 * (a - a_prev).square().mean(-1), -.00002 * qd.square().mean(-1),
                   -.04 * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1))
        reward = sum(values)
        return (reward, dict(zip(TERMS, values))) if terms else reward

    def termination(self, S, c):
        p, R = S[:2]
        clearance = p[..., 2] - self.field.sample(p[..., :2])
        lane = (p[..., 1] - c[..., 0]) / self.field.scale
        finite = torch.isfinite(p).all(-1) & torch.isfinite(R).all((-1, -2))
        for value in S[2:]:
            finite = finite & torch.isfinite(value).all(-1)
        return ((clearance < .42 * self.field.height) | (R[..., 2, 2] < .35)
                | (lane.abs() > 1.75) | ~self.field.inside(p[..., :2]) | ~finite)

    def soft_outcome_probabilities(self, S, c):
        p, R = S[:2]
        clearance = p[..., 2] - self.field.sample(p[..., :2])
        lane = (p[..., 1] - c[..., 0]) / self.field.scale
        fall = torch.sigmoid((.42 * self.field.height - clearance) / (.02 * self.field.scale))
        tilt = torch.sigmoid((.35 - R[..., 2, 2]) / .04)
        lateral = torch.sigmoid((lane.abs() - 1.75) / .05)
        goal = torch.sigmoid((S[0][..., 0] - self.field.terrain.goal_x) / (.1 * self.field.scale))
        fall = 1 - (1 - fall) * (1 - tilt) * (1 - lateral)
        success = goal * (1 - fall)
        return success, fall

    def soft_terminal_reward(self, S, c):
        success, fall = self.soft_outcome_probabilities(S, c)
        return self.terminal_reward(success, fall)

    def terminal_reward(self, success, fall):
        return self.success_reward * success - FALL_PENALTY * fall

    def imagined_reward(self, S, a, a_prev, c):
        success, fall = self.soft_outcome_probabilities(S, c)
        return (self.reward(S, a, a_prev, c) + self.terminal_reward(success, fall),
                (success + fall).clamp(0, 1))

    def soft_termination(self, S, c):
        success, fall = self.soft_outcome_probabilities(S, c)
        return (success + fall).clamp(0, 1)

    def critic_features(self, S, xi, c):
        return torch.cat((self.field.invariant(S), self.normalize_context(xi),
                          self.goal(S, c), self.field.model_features(S)), -1)

    def normalize_context(self, xi):
        return torch.cat(((xi[..., :1] - .75) / .45,
                          xi[..., 1:1+self.num_joints] / .01,
                          xi[..., -3:] / self.field.scale), -1)
