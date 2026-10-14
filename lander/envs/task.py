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
    "rel_vx", "rel_vy",                # estimated rover velocity - drone velocity (horizontal)
    "drone_vz",                        # drone vertical velocity from the autopilot EKF
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


def action_to_velocity(a: np.ndarray, cfg: dict, rover_vel_est: np.ndarray | None = None) -> np.ndarray:
    """a in [-1,1]^3 -> level-frame velocity command (m/s), z down positive.

    The horizontal action is a velocity RELATIVE TO THE ESTIMATED ROVER
    VELOCITY (feed-forward): v_cmd_xy = v_rover_est + a_xy * v_max. Together
    with the relative-velocity observation this makes the policy invariant to
    rover speed by construction: "close the gap, damp the relative velocity"
    learned on a static rover is already the right behaviour on a moving one.
    (With absolute velocity actions + drone-velocity observations, PPO learned
    to damp its OWN velocity on static rovers and then failed at 0.1 m/s - 8%
    success where the PID got 84%; Step 9, reward/obs v5.)
    """
    a = np.clip(a, -1.0, 1.0)
    s = cfg["action"]
    v = a * np.array([s["v_xy_max_mps"], s["v_xy_max_mps"], s["v_z_max_mps"]])
    if rover_vel_est is not None:
        v[:, :2] += rover_vel_est
    return v


def feedforward(kf_rover_vel: np.ndarray, kf_init: np.ndarray) -> np.ndarray:
    """Rover velocity estimate used as feed-forward (zero before the first detection)."""
    return np.where(kf_init[:, None], kf_rover_vel, 0.0)


def build_obs(kf_pos: np.ndarray, kf_rover_vel: np.ndarray, kf_init: np.ndarray,
              drone_vel: np.ndarray, visible: np.ndarray, lost_time: np.ndarray,
              prev_action: np.ndarray, cfg: dict) -> np.ndarray:
    o = cfg["observation"]
    pos = np.where(kf_init[:, None], kf_pos, 0.0)
    rv = feedforward(kf_rover_vel, kf_init)
    rel_v = rv - drone_vel[:, :2]
    lost = np.minimum(lost_time, o["lost_time_clip_s"]) / o["lost_time_clip_s"]
    obs = np.concatenate([
        pos[:, :2] / o["pos_xy_scale_m"],
        pos[:, 2:3] / o["height_scale_m"],
        rv / o["vel_scale_mps"],
        rel_v / o["vel_scale_mps"],
        drone_vel[:, 2:3] / o["vel_scale_mps"],
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
    h = np.maximum(t.h, 0)
    xy_weight = 1.0 + r["k_low"] / (h + r["h0_m"])
    return -(r["phi_xy"] * dxy * xy_weight + r["phi_h"] * h + r["phi_v"] * vxy * near)


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
    terminal = np.where(np.isin(outcome, (CRASH, OUT_OF_BOUNDS)), r["r_fail"], terminal)
    td = cfg["touchdown"]
    vxy = np.linalg.norm(cur.v_rel[:, :2], axis=1)
    miss_pen = (r["r_miss"]
                + r["r_miss_per_m"] * np.clip(dxy - td["max_xy_err_m"], 0, 2)
                + r["r_miss_per_mps"] * np.clip(vxy - td["max_rel_vxy_mps"], 0, 2))
    terminal = np.where(outcome == MISS, miss_pen, terminal)
    terminal = np.where(outcome == TIMEOUT, r["r_timeout"], terminal)
    if terms is not None:
        terms.update(shape=shape, lost=lost, smooth=smooth, time=time_pen, terminal=terminal)
    return shape + lost + smooth + time_pen + terminal
