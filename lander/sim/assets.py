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
    nest = cfg.get("nested", {})
    if nest.get("enabled"):
        cell_px = marker_px / 6
        r, c = nest["cell"]
        x0, y0 = border_px + c * cell_px, border_px + r * cell_px
        if marker[round(r * cell_px + cell_px / 2), round(c * cell_px + cell_px / 2)] < 128:
            raise ValueError(f"nested.cell {nest['cell']} is black in marker {cfg['marker_id']}")
        inner_px = round(cell_px * nest["inner_fraction"])
        off = (cell_px - inner_px) / 2
        img[round(y0 + off):round(y0 + off) + inner_px, round(x0 + off):round(x0 + off) + inner_px] = \
            cv2.aruco.generateImageMarker(dictionary, nest["inner_id"], inner_px)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out
