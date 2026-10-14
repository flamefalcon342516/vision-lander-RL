"""Classical baseline: PD visual servoing + gated descent, on the SAME observation.

Acting on the policy's observation (not on ground truth) makes it a fair
baseline: same perception, same filter, same action interface, evaluated by
the same code in Stage A, Stage B and on hardware.

  v_xy = ff * rover_vel_est + kp * xy_err - kd * (v_drone_xy - rover_vel_est)
  descend at clip(k * h, vmin, vmax) only inside an alignment cone
  marker lost: hold height, then climb to re-acquire
"""
from __future__ import annotations

import numpy as np

from lander.envs import task as tk


class PIDLander:
    def __init__(self, cfg: dict, task_cfg: dict) -> None:
        self.c = cfg
        self.t = task_cfg

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs (N, OBS_DIM) -> action (N, 3) in [-1, 1]."""
        o, c = self.t["observation"], self.c
        obs = np.atleast_2d(obs)
        xy = obs[:, 0:2] * o["pos_xy_scale_m"]
        h = obs[:, 2] * o["height_scale_m"]
        rover_v = obs[:, 3:5] * o["vel_scale_mps"]
        drone_v = obs[:, 5:7] * o["vel_scale_mps"]
        lost = obs[:, 9] * o["lost_time_clip_s"]
        never_seen = (np.abs(obs[:, 0:3]).sum(axis=1) == 0) & (obs[:, 8] == 0)

        v_xy = c["feedforward"] * rover_v + c["kp_xy"] * xy - c["kd_xy"] * (drone_v - rover_v)
        aligned = np.linalg.norm(xy, axis=1) < c["align_radius_m"] + c["align_per_m"] * h
        vz = np.where(aligned, np.clip(c["descent_gain"] * h, c["descent_min_mps"],
                                       c["descent_max_mps"]), 0.0)
        holding = lost > c["lost_hold_s"]
        vz = np.where(holding, 0.0, vz)
        climbing = ((lost > c["lost_climb_s"]) & (h < c["climb_ceiling_m"])) | never_seen
        vz = np.where(climbing, -c["climb_mps"], vz)
        v_xy = np.where(climbing[:, None], 0.0, v_xy)
        v = np.concatenate([v_xy, vz[:, None]], axis=1)
        scale = np.array([self.t["action"]["v_xy_max_mps"]] * 2 + [self.t["action"]["v_z_max_mps"]])
        return np.clip(v / scale, -1, 1).astype(np.float32)
