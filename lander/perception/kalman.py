"""Vectorised Kalman filter for the marker's position and the rover's velocity.

Why this state, not "relative position + relative velocity":
  We KNOW the drone's own velocity (ArduPilot EKF). The rover's velocity changes
  slowly. So we estimate
      x = [px, py, pz, vrx, vry]       (all in the level frame)
  p = marker position relative to the drone, z = height above the marker (down +),
  vr = rover horizontal velocity, and use the drone velocity v_d as a known
  control input:
      p' = [vr - v_d_xy, -v_d_z],   vr' = white noise (random-walk velocity).
  During marker dropouts the filter keeps predicting with correct drone motion,
  which a constant-relative-velocity model cannot do once the drone changes
  its velocity.

Assumption: yaw is held constant (our action has no yaw), so the level frame
does not rotate between steps.

Batched over N filters (N envs in Stage A; N=1 in Stage B and on hardware).
"""
from __future__ import annotations

import numpy as np


class MarkerKF:
    def __init__(self, n: int, cfg: dict) -> None:
        self.n = n
        self.q_vel = cfg["q_rover_accel"]        # rover accel noise (m/s^2)^2/s
        self.q_pos = cfg["q_pos"]                # unmodelled motion (m^2/s)
        self.r_base = cfg["r_pos_base"]          # measurement std at 1 m height (m)
        self.r_scale = cfg["r_pos_per_m"]        # std grows with height (m per m)
        self.init_vel_std = cfg["init_rover_vel_std"]
        self.x = np.zeros((n, 5))
        self.P = np.tile(np.eye(5), (n, 1, 1))
        self.initialised = np.zeros(n, dtype=bool)
        self.since_meas = np.full(n, np.inf)     # seconds since last detection

    def reset(self, idx: np.ndarray) -> None:
        self.x[idx] = 0.0
        self.P[idx] = np.eye(5)
        self.initialised[idx] = False
        self.since_meas[idx] = np.inf

    def predict(self, dt: float, v_drone: np.ndarray) -> None:
        """v_drone: (N,3) drone velocity in the level frame."""
        f = np.eye(5)
        f[0, 3] = f[1, 4] = dt
        self.x[:, 0] += (self.x[:, 3] - v_drone[:, 0]) * dt
        self.x[:, 1] += (self.x[:, 4] - v_drone[:, 1]) * dt
        self.x[:, 2] += -v_drone[:, 2] * dt
        q = np.diag([self.q_pos * dt] * 3 + [self.q_vel * dt] * 2)
        self.P = f @ self.P @ f.T + q
        self.since_meas += dt

    def update(self, idx: np.ndarray, z: np.ndarray) -> None:
        """z: (len(idx),3) measured relative position (level frame)."""
        if len(idx) == 0:
            return
        new = idx[~self.initialised[idx]]
        if len(new):  # first detection: snap position, keep velocity uncertain
            sel = np.isin(idx, new)
            self.x[new, :3] = z[sel]
            self.x[new, 3:] = 0.0
            self.P[new] = np.diag([0.05] * 3 + [self.init_vel_std ** 2] * 2)
            self.initialised[new] = True
        old = idx[np.isin(idx, new, invert=True)]
        zo = z[np.isin(idx, new, invert=True)]
        if len(old):
            h = np.zeros((3, 5))
            h[:, :3] = np.eye(3)
            std = self.r_base + self.r_scale * np.abs(zo[:, 2])
            r = (std ** 2)[:, None, None] * np.eye(3)
            p = self.P[old]
            s = h @ p @ h.T + r
            k = p @ h.T @ np.linalg.inv(s)
            innov = zo - self.x[old, :3]
            self.x[old] += np.einsum("nij,nj->ni", k, innov)
            self.P[old] = (np.eye(5) - k @ h) @ p
        self.since_meas[idx] = 0.0

    @property
    def rel_pos(self) -> np.ndarray:
        return self.x[:, :3]

    @property
    def rover_vel(self) -> np.ndarray:
        return self.x[:, 3:]
