"""The landing task: observation layout, action scaling, reward, termination.

Pure NumPy and batched over N envs, so Stage A (N = thousands), Stage B
(N = 1) and the companion-computer runtime all use literally the same code.
That is how the two environments get IDENTICAL observation/action spaces and
the same reward.

Quantities (all in the level frame: x forward, y right, z down, see
perception/frames.py):
  d      : marker position relative to the drone (TRUE value, reward only)
  v_rel  : rover velocity minus drone velocity (TRUE, reward only)
  h      : height of the drone origin above the deck (TRUE)
The policy never sees true values: only the Kalman-filtered estimates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBS_NAMES = (
    "kf_x", "kf_y", "kf_h",           # estimated marker position (level frame)
    "kf_rover_vx", "kf_rover_vy",     # estimated rover velocity
    "drone_vx", "drone_vy", "drone_vz",  # drone velocity from the autopilot EKF
    "visible",                         # marker detected this step (0/1)
    "lost_time",                       # seconds since last detection (clipped, scaled)
    "prev_ax", "prev_ay", "prev_az",   # previous action (lets the policy be smooth)
)
OBS_DIM = len(OBS_NAMES)
ACT_DIM = 3

# Terminal outcome codes (for metrics).
RUNNING, SUCCESS, MISS, CRASH, OUT_OF_BOUNDS, TIMEOUT = 0, 1, 2, 3, 4, 5
OUTCOME_NAMES = {SUCCESS: "success", MISS: "miss", CRASH: "crash",
                 OUT_OF_BOUNDS: "out_of_bounds", TIMEOUT: "timeout"}


def action_to_velocity(a: np.ndarray, cfg: dict) -> np.ndarray:
    """a in [-1,1]^3 -> level-frame velocity command (m/s), z down positive."""
    a = np.clip(a, -1.0, 1.0)
    s = cfg["action"]
    return a * np.array([s["v_xy_max_mps"], s["v_xy_max_mps"], s["v_z_max_mps"]])


def build_obs(kf_pos: np.ndarray, kf_rover_vel: np.ndarray, kf_init: np.ndarray,
              drone_vel: np.ndarray, visible: np.ndarray, lost_time: np.ndarray,
              prev_action: np.ndarray, cfg: dict) -> np.ndarray:
    o = cfg["observation"]
    pos = np.where(kf_init[:, None], kf_pos, 0.0)
    rv = np.where(kf_init[:, None], kf_rover_vel, 0.0)
    lost = np.minimum(lost_time, o["lost_time_clip_s"]) / o["lost_time_clip_s"]
    obs = np.concatenate([
        pos[:, :2] / o["pos_xy_scale_m"],
        pos[:, 2:3] / o["height_scale_m"],
        rv / o["vel_scale_mps"],
        drone_vel / o["vel_scale_mps"],
        visible[:, None].astype(float),
        lost[:, None],
        prev_action,
    ], axis=1)
    return obs.astype(np.float32)


@dataclass
class StepTruth:
    d: np.ndarray          # (N,3) true marker position relative to drone, level frame
    v_rel: np.ndarray      # (N,3) true rover-minus-drone velocity, level frame
    h: np.ndarray          # (N,) true drone-origin height above the deck
    ground_clearance: np.ndarray  # (N,) drone-origin height above the GROUND


def potential(t: StepTruth, cfg: dict) -> np.ndarray:
    r = cfg["reward"]
    dxy = np.linalg.norm(t.d[:, :2], axis=1)
    near = np.exp(-np.maximum(t.h, 0) / r["near_scale_m"])
    vxy = np.linalg.norm(t.v_rel[:, :2], axis=1)
    return -(r["phi_xy"] * dxy + r["phi_h"] * np.maximum(t.h, 0) + r["phi_v"] * vxy * near)


def check_termination(t: StepTruth, elapsed_s: np.ndarray, cfg: dict) -> np.ndarray:
    """Outcome code per env (RUNNING if the episode continues)."""
    td, b = cfg["touchdown"], cfg["bounds"]
    dxy = np.linalg.norm(t.d[:, :2], axis=1)
    out = np.full(len(t.h), RUNNING)
    touch = t.h <= td["height_m"]
    good = ((dxy <= td["max_xy_err_m"])
            & (np.linalg.norm(t.v_rel[:, :2], axis=1) <= td["max_rel_vxy_mps"])
            & (-t.v_rel[:, 2] <= td["max_vz_mps"]))   # v_rel z = -drone vz (rover vz = 0)
    out[touch & good] = SUCCESS
    out[touch & ~good] = MISS
    # Ground crash: low over the ground but not over the deck.
    crash = (t.ground_clearance <= b["min_ground_clearance_m"]) & ~touch
    out[(out == RUNNING) & crash] = CRASH
    oob = (dxy > b["max_xy_dist_m"]) | (t.h > b["max_height_m"])
    out[(out == RUNNING) & oob] = OUT_OF_BOUNDS
    out[(out == RUNNING) & (elapsed_s >= cfg["max_episode_s"] - 1e-9)] = TIMEOUT
    return out


def reward(prev: StepTruth, cur: StepTruth, outcome: np.ndarray, visible: np.ndarray,
           action: np.ndarray, prev_action: np.ndarray, cfg: dict,
           terms: dict | None = None) -> np.ndarray:
    """Per-step reward. Pass `terms={}` to get each component (for ablations/plots)."""
    r = cfg["reward"]
    gamma = cfg["gamma"]
    phi_next = potential(cur, cfg)
    # Terminal states have potential 0 by convention, so shaping can't be
    # "cashed in" by ending the episode.
    ended = outcome != RUNNING
    phi_next = np.where(ended & (outcome != TIMEOUT), 0.0, phi_next)
    shape = gamma * phi_next - potential(prev, cfg)
    lost = -r["w_lost"] * (~visible.astype(bool))
    smooth = -r["w_smooth"] * np.sum((action - prev_action) ** 2, axis=1)
    time_pen = -r["w_time"] * np.ones(len(outcome))
    dxy = np.linalg.norm(cur.d[:, :2], axis=1)
    precision = np.clip(1 - dxy / cfg["touchdown"]["max_xy_err_m"], 0, 1)
    terminal = np.where(outcome == SUCCESS, r["r_success"] + r["r_precision"] * precision, 0.0)
    terminal = np.where(np.isin(outcome, (MISS, CRASH, OUT_OF_BOUNDS)), r["r_fail"], terminal)
    if terms is not None:
        terms.update(shape=shape, lost=lost, smooth=smooth, time=time_pen, terminal=terminal)
    return shape + lost + smooth + time_pen + terminal
