"""Generated Gazebo assets: the ArUco marker texture for the rover."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_marker_texture(cfg: dict, out: Path) -> Path:
    """Write a PNG of the marker plus a white quiet zone.

    The texture covers (size + 2*border) metres on the rover deck, so the
    pixel scale is texture_px / that width, and the black square is exactly
    marker_size_m wide in the world.
    """
    total_m = cfg["marker_size_m"] + 2 * cfg["border_m"]
    px_per_m = cfg["texture_px"] / total_m
    marker_px = round(cfg["marker_size_m"] * px_per_m)
    border_px = (cfg["texture_px"] - marker_px) // 2
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    marker = cv2.aruco.generateImageMarker(dictionary, cfg["marker_id"], marker_px)
    img = np.full((cfg["texture_px"], cfg["texture_px"]), 255, np.uint8)
    img[border_px:border_px + marker_px, border_px:border_px + marker_px] = marker
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out
