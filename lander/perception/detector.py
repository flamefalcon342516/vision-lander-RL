"""ArUco marker detection and pose estimation (pinhole model + solvePnP)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    k: np.ndarray            # 3x3 intrinsic matrix
    dist: np.ndarray         # distortion coefficients (zeros in sim)

    @classmethod
    def from_config(cls, cfg: dict) -> "CameraModel":
        """Ideal pinhole from resolution + horizontal FOV (what Gazebo renders)."""
        w, h = cfg["width"], cfg["height"]
        f = (w / 2) / np.tan(cfg["hfov_rad"] / 2)
        k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        return cls(w, h, k, np.zeros(5))

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """(N,3) camera-frame points -> (N,2) pixels. Assumes z > 0."""
        uvw = p_cam @ self.k.T
        return uvw[:, :2] / uvw[:, 2:3]


@dataclass(frozen=True)
class Detection:
    p_cam: np.ndarray        # marker centre in the camera frame (m)
    yaw_cam: float           # marker x-axis angle in the image plane (rad)
    corners: np.ndarray      # (4,2) pixel corners
    reproj_err_px: float     # RMS reprojection error: a detection-quality metric
    side_px: float           # mean side length in pixels (another quality cue)


class MarkerDetector:
    def __init__(self, cam: CameraModel, marker_cfg: dict) -> None:
        self.cam = cam
        self.marker_id = marker_cfg["marker_id"]
        s = marker_cfg["marker_size_m"]
        # Object points in the marker frame, in OpenCV's corner order
        # (TL, TR, BR, BL); z = 0 plane. Required order for SOLVEPNP_IPPE_SQUARE.
        self.obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0],
                             [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], dtype=np.float64)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_cfg["dictionary"]))
        params = cv2.aruco.DetectorParameters()
        # Corner refinement dominates PnP accuracy for small (far) markers.
        # Measured on synthetic renders (Step 5), mean depth error at 2/6/10/12 m:
        #   NONE +1.2/+3.0/+5.5/+3.5 %   SUBPIX +0.4/+0.6/+1.8/+3.7 %
        #   APRILTAG +0.0/-0.8/-0.8/-0.2 %  for 1.14 ms vs 0.91 ms per frame.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, image: np.ndarray) -> Detection | None:
        """Return the target marker's pose relative to the camera, or None."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        hits = np.flatnonzero(ids.ravel() == self.marker_id)
        if len(hits) == 0:
            return None
        c = corners[hits[0]].reshape(4, 2).astype(np.float64)
        # IPPE_SQUARE: the analytic solver for a planar square. A planar target
        # always has TWO pose solutions (the second is a "flipped" plane). When
        # viewed nearly head-on -- i.e. most of a landing -- they become
        # almost equally likely, and solvePnP's single answer was the mirrored
        # one in an exactly frontal synthetic test (reprojection error = one
        # full side length, 80 px). So: get both, keep the one that (a) faces
        # the camera (marker +z towards the lens: R[2,2] < 0) and (b) has the
        # lowest reprojection error.
        # IPPE's homography decomposition also degenerates for an EXACTLY
        # fronto-parallel view (both returned solutions were the same wrong
        # one). SQPnP (globally optimal, ~0.05 ms) is added as a third
        # candidate so the selection below always has a correct option.
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(self.obj, c, self.cam.k, self.cam.dist,
                                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
        ok, rv_sq, tv_sq = cv2.solvePnP(self.obj, c, self.cam.k, self.cam.dist,
                                        flags=cv2.SOLVEPNP_SQPNP)
        candidates = list(zip(rvecs, tvecs)) + ([(rv_sq, tv_sq)] if ok else [])
        if not candidates:
            return None
        best = None
        for rv, tv in candidates:
            proj, _ = cv2.projectPoints(self.obj, rv, tv, self.cam.k, self.cam.dist)
            e = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - c) ** 2, axis=1))))
            facing = cv2.Rodrigues(rv)[0][2, 2] < 0
            key = (not facing, e)
            if best is None or key < best[0]:
                best = (key, rv, tv, e)
        _, rvec, tvec, err = best
        r, _ = cv2.Rodrigues(rvec)
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))  # marker x-axis in the image plane
        side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
        return Detection(tvec.ravel(), yaw, c, err, side)
