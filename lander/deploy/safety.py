"""Safety layer: a small, deterministic filter between the policy and MAVLink.

It is deliberately separate from the policy. A learned policy is a black box,
so every hard limit lives here, in code that can be unit-tested exhaustively
and reviewed line by line. The policy proposes; the safety layer disposes.

Priority (highest first):
  1. Pilot/RC override: flight mode is not GUIDED -> send NOTHING (the pilot's
     mode switch always wins; the FC is flying, not us).
  2. Lost link to the FC (no HEARTBEAT) -> send nothing (FC failsafes act).
  3. Invalid policy output (NaN/inf) -> hover.
  4. Geofence breach -> RTL.
  5. Marker lost too long -> hover, then RTL.
  6. Altitude ceiling -> no climbing; altitude floor without a confirmed
     marker under us -> no descending.
  7. Velocity limits -> clamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Decision(Enum):
    SEND = "send"            # send the (possibly clamped) velocity
    HOVER = "hover"          # send zero velocity
    RTL = "rtl"              # request RTL mode, stop commanding
    SILENT = "silent"        # send nothing at all


@dataclass
class VehicleState:
    mode: str
    armed: bool
    heartbeat_age_s: float
    alt_m: float                      # height above home (m, up positive)
    dist_from_home_m: float           # horizontal
    marker_lost_s: float              # time since last detection
    marker_xy_err_m: float            # |estimated horizontal offset|, inf if never seen


@dataclass
class SafetyConfig:
    v_xy_max_mps: float = 4.0
    v_up_max_mps: float = 1.5
    v_down_max_mps: float = 1.5
    alt_ceiling_m: float = 15.0
    alt_floor_m: float = 1.5          # below this, only descend with the marker confirmed under us
    floor_marker_max_lost_s: float = 0.5
    floor_marker_max_xy_m: float = 0.5
    geofence_radius_m: float = 50.0
    lost_hover_s: float = 2.0
    lost_rtl_s: float = 8.0
    heartbeat_timeout_s: float = 1.5
    required_mode: str = "GUIDED"


@dataclass
class SafetyOutput:
    decision: Decision
    velocity: np.ndarray
    reasons: list[str] = field(default_factory=list)


def check(state: VehicleState, v_cmd: np.ndarray, cfg: SafetyConfig) -> SafetyOutput:
    """v_cmd: proposed level-frame velocity (x fwd, y right, z DOWN), m/s."""
    zero = np.zeros(3)
    if state.mode != cfg.required_mode or not state.armed:
        return SafetyOutput(Decision.SILENT, zero, [f"mode {state.mode} / armed={state.armed}: pilot or FC in control"])
    if state.heartbeat_age_s > cfg.heartbeat_timeout_s:
        return SafetyOutput(Decision.SILENT, zero, [f"no heartbeat for {state.heartbeat_age_s:.1f} s"])
    if not np.all(np.isfinite(v_cmd)):
        return SafetyOutput(Decision.HOVER, zero, ["non-finite policy output"])
    if state.dist_from_home_m > cfg.geofence_radius_m:
        return SafetyOutput(Decision.RTL, zero, [f"geofence: {state.dist_from_home_m:.1f} m from home"])
    if state.marker_lost_s > cfg.lost_rtl_s:
        return SafetyOutput(Decision.RTL, zero, [f"marker lost {state.marker_lost_s:.1f} s"])
    if state.marker_lost_s > cfg.lost_hover_s:
        return SafetyOutput(Decision.HOVER, zero, [f"marker lost {state.marker_lost_s:.1f} s: hover"])

    v = np.asarray(v_cmd, dtype=float).copy()
    reasons: list[str] = []
    if state.alt_m >= cfg.alt_ceiling_m and v[2] < 0:
        v[2] = 0.0
        reasons.append("ceiling: climb blocked")
    marker_ok = (state.marker_lost_s <= cfg.floor_marker_max_lost_s
                 and state.marker_xy_err_m <= cfg.floor_marker_max_xy_m)
    if state.alt_m <= cfg.alt_floor_m and v[2] > 0 and not marker_ok:
        v[2] = 0.0
        reasons.append("floor: descent blocked (marker not confirmed below)")
    n = np.linalg.norm(v[:2])
    if n > cfg.v_xy_max_mps:
        v[:2] *= cfg.v_xy_max_mps / n
        reasons.append("xy speed clamped")
    if v[2] > cfg.v_down_max_mps or v[2] < -cfg.v_up_max_mps:
        v[2] = float(np.clip(v[2], -cfg.v_up_max_mps, cfg.v_down_max_mps))
        reasons.append("vz clamped")
    return SafetyOutput(Decision.SEND, v, reasons)
