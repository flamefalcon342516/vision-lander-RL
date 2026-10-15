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

Optional upgrades (configs/perception.yaml, both off by default = the published results):
  model: ctrv    constant turn rate: adds the rover's turn rate w to the state
                 (x = [px, py, pz, vrx, vry, w]) and rotates vr by w*dt each step (EKF),
                 so predictions follow circles and figure-8s instead of tangents.
  gate_chi2: g   innovation gate: a detection with Mahalanobis distance^2 > g is rejected
                 (PnP outliers); after `gate_reinit` rejections in a row the filter snaps
                 to the measurement, so a diverged filter can't lock itself out.

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
        self.ctrv = cfg.get("model", "cv") == "ctrv"
        self.q_turn = cfg.get("q_turn_rate", 0.3)           # (rad/s)^2 per s
        self.init_turn_std = cfg.get("init_turn_rate_std", 0.3)
        self.gate = float(cfg.get("gate_chi2", 0.0))        # 0 = no gating
        self.gate_reinit = int(cfg.get("gate_reinit", 3))
        self.dim = 6 if self.ctrv else 5
        self.x = np.zeros((n, self.dim))
        self.P = np.tile(np.eye(self.dim), (n, 1, 1))
        self.initialised = np.zeros(n, dtype=bool)
        self.since_meas = np.full(n, np.inf)     # seconds since last detection
        self.rejected = np.zeros(n, dtype=int)   # consecutive gated-out detections
        self.n_rejected = 0                      # total, for diagnostics

    def reset(self, idx: np.ndarray) -> None:
        self.x[idx] = 0.0
        self.P[idx] = np.eye(self.dim)
        self.initialised[idx] = False
        self.since_meas[idx] = np.inf
        self.rejected[idx] = 0

    def predict(self, dt: float, v_drone: np.ndarray) -> None:
        """v_drone: (N,3) drone velocity in the level frame."""
        if self.ctrv:
            self._predict_ctrv(dt, v_drone)
            return
        f = np.eye(5)
        f[0, 3] = f[1, 4] = dt
        self.x[:, 0] += (self.x[:, 3] - v_drone[:, 0]) * dt
        self.x[:, 1] += (self.x[:, 4] - v_drone[:, 1]) * dt
        self.x[:, 2] += -v_drone[:, 2] * dt
        q = np.diag([self.q_pos * dt] * 3 + [self.q_vel * dt] * 2)
        self.P = f @ self.P @ f.T + q
        self.since_meas += dt

    def _predict_ctrv(self, dt: float, v_drone: np.ndarray) -> None:
        vx, vy, w = self.x[:, 3], self.x[:, 4], self.x[:, 5]
        c, s = np.cos(w * dt), np.sin(w * dt)
        # position uses the velocity at mid-step (half the rotation): better on tight turns
        ch, sh = np.cos(w * dt / 2), np.sin(w * dt / 2)
        vmx, vmy = ch * vx - sh * vy, sh * vx + ch * vy
        f = np.tile(np.eye(6), (self.n, 1, 1))
        f[:, 0, 3], f[:, 0, 4], f[:, 0, 5] = ch * dt, -sh * dt, (-sh * vx - ch * vy) * dt * dt / 2
        f[:, 1, 3], f[:, 1, 4], f[:, 1, 5] = sh * dt, ch * dt, (ch * vx - sh * vy) * dt * dt / 2
        f[:, 3, 3], f[:, 3, 4], f[:, 3, 5] = c, -s, (-s * vx - c * vy) * dt
        f[:, 4, 3], f[:, 4, 4], f[:, 4, 5] = s, c, (c * vx - s * vy) * dt
        self.x[:, 0] += (vmx - v_drone[:, 0]) * dt
        self.x[:, 1] += (vmy - v_drone[:, 1]) * dt
        self.x[:, 2] += -v_drone[:, 2] * dt
        self.x[:, 3], self.x[:, 4] = c * vx - s * vy, s * vx + c * vy
        q = np.diag([self.q_pos * dt] * 3 + [self.q_vel * dt] * 2 + [self.q_turn * dt])
        self.P = f @ self.P @ np.transpose(f, (0, 2, 1)) + q
        self.since_meas += dt

    def update(self, idx: np.ndarray, z: np.ndarray) -> np.ndarray:
        """z: (len(idx),3) measured relative position (level frame).
        Returns a bool mask over idx: True where the detection was accepted."""
        idx = np.asarray(idx)
        accepted = np.ones(len(idx), dtype=bool)
        if len(idx) == 0:
            return accepted
        h = np.zeros((3, self.dim))
        h[:, :3] = np.eye(3)
        std = self.r_base + self.r_scale * np.abs(z[:, 2])
        r = (std ** 2)[:, None, None] * np.eye(3)
        snap = ~self.initialised[idx]
        if self.gate > 0:
            old = ~snap
            if old.any():
                io = idx[old]
                s_mat = h @ self.P[io] @ h.T + r[old]
                innov = z[old] - self.x[io, :3]
                d2 = np.einsum("ni,nij,nj->n", innov, np.linalg.inv(s_mat), innov)
                bad = d2 > self.gate
                ob = np.flatnonzero(old)
                self.rejected[io[bad]] += 1
                self.rejected[io[~bad]] = 0
                self.n_rejected += int(bad.sum())
                accepted[ob[bad]] = False
                # too many rejections in a row: the filter, not the detector, is wrong
                re = bad & (self.rejected[io] >= self.gate_reinit)
                snap[ob[re]] = True
                accepted[ob[re]] = True
        new = idx[snap]
        if len(new):  # first detection (or re-init): snap position, keep velocity uncertain
            self.x[new, :3] = z[snap]
            self.x[new, 3:] = 0.0
            d = [0.05] * 3 + [self.init_vel_std ** 2] * 2 + ([self.init_turn_std ** 2] if self.ctrv else [])
            self.P[new] = np.diag(d)
            self.initialised[new] = True
            self.rejected[new] = 0
        upd = accepted & ~snap
        if upd.any():
            old = idx[upd]
            p = self.P[old]
            s_mat = h @ p @ h.T + r[upd]
            k = p @ h.T @ np.linalg.inv(s_mat)
            innov = z[upd] - self.x[old, :3]
            self.x[old] += np.einsum("nij,nj->ni", k, innov)
            self.P[old] = (np.eye(self.dim) - k @ h) @ p
        self.since_meas[idx[accepted]] = 0.0
        return accepted

    @property
    def turn_rate(self) -> np.ndarray:
        return self.x[:, 5] if self.ctrv else np.zeros(self.n)

    @property
    def rel_pos(self) -> np.ndarray:
        return self.x[:, :3]

    @property
    def rover_vel(self) -> np.ndarray:
        return self.x[:, 3:]
