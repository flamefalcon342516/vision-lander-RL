"""Coordinate frames used everywhere in the project.

  cam   : OpenCV optical frame. x = image right, y = image down, z = out of lens.
  body  : FRD body frame (x forward, y right, z down), ArduPilot's convention.
  level : body frame with roll and pitch removed: x = forward along the
          horizon, y = right, z = straight down. It turns with yaw only, which
          is exactly the frame ArduPilot uses for MAV_FRAME_BODY_NED velocity
          commands (yaw-only rotation, verified in Step 0). Observations AND
          actions live in this frame.
  NED   : local North-East-Down (ArduPilot LOCAL_POSITION_NED).
  ENU   : Gazebo world (East-North-Up).

Rotations use the aerospace ZYX convention: R_ned_body = Rz(yaw) Ry(pitch) Rx(roll).
"""
from __future__ import annotations

import numpy as np

# Camera mounted looking down, image top = nose, image right = right wing
# (Step 3 SDF, verified in Step 4 from the rover's orientation in the image).
#   body x (forward) = -cam y,  body y (right) = cam x,  body z (down) = cam z
R_BODY_CAM = np.array([[0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])


def rot_x(a: np.ndarray | float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1),
                     np.stack([z, s, c], -1)], -2)


def rot_y(a: np.ndarray | float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def rot_z(a: np.ndarray | float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def r_level_body(roll: np.ndarray | float, pitch: np.ndarray | float) -> np.ndarray:
    """Rotation taking body-frame vectors into the level frame: Ry(pitch) Rx(roll)."""
    return rot_y(pitch) @ rot_x(roll)


def cam_to_body(p_cam: np.ndarray, mount_offset_body: np.ndarray) -> np.ndarray:
    """Point in camera frame -> body frame (adds the lens offset from the CoG)."""
    return p_cam @ R_BODY_CAM.T + mount_offset_body


def body_to_level(p_body: np.ndarray, roll: np.ndarray | float,
                  pitch: np.ndarray | float) -> np.ndarray:
    """Batch-friendly: p_body (..., 3), roll/pitch broadcastable to (...)."""
    r = r_level_body(roll, pitch)
    return np.einsum("...ij,...j->...i", r, p_body)


def level_to_body(p_level: np.ndarray, roll: np.ndarray | float,
                  pitch: np.ndarray | float) -> np.ndarray:
    r = r_level_body(roll, pitch)
    return np.einsum("...ji,...j->...i", r, p_level)  # R^T p


def ned_to_level(v_ned: np.ndarray, yaw: np.ndarray | float) -> np.ndarray:
    """Rotate a NED vector into the level frame (undo yaw only)."""
    return np.einsum("...ji,...j->...i", rot_z(yaw), v_ned)


def level_to_ned(v_level: np.ndarray, yaw: np.ndarray | float) -> np.ndarray:
    return np.einsum("...ij,...j->...i", rot_z(yaw), v_level)


def enu_to_ned(v_enu: np.ndarray) -> np.ndarray:
    """(E, N, U) -> (N, E, D). Verified against SITL in Step 2."""
    v = np.asarray(v_enu, dtype=float)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], -1)


def quat_to_euler_frd(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Gazebo ENU/FLU quaternion of a model -> ArduPilot (roll, pitch, yaw) NED/FRD.

    FLU->FRD and ENU->NED flip y and z; this gives exactly
    roll_frd = roll_flu, pitch_frd = -pitch_flu, yaw_ned = pi/2 - yaw_enu.
    """
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw_ned = (np.pi / 2 - yaw + np.pi) % (2 * np.pi) - np.pi
    return float(roll), float(-pitch), float(yaw_ned)
