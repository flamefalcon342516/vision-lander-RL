"""Frames, detector accuracy on synthetic images, and the Kalman filter."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from lander.perception import frames as fr
from lander.perception.detector import CameraModel, MarkerDetector
from lander.perception.kalman import MarkerKF
from lander.sim.assets import make_marker_texture

ROOT = Path(__file__).resolve().parents[1]
CAM = CameraModel.from_config(yaml.safe_load((ROOT / "configs/camera.yaml").read_text()))
MARKER = yaml.safe_load((ROOT / "configs/marker.yaml").read_text())


# ---------------------------------------------------------------- frames
def test_body_cam_rotation_is_proper() -> None:
    assert np.isclose(np.linalg.det(fr.R_BODY_CAM), 1.0)


def test_image_top_is_body_forward_and_right_is_right() -> None:
    off = np.zeros(3)
    assert np.allclose(fr.cam_to_body(np.array([0, -1.0, 0]), off), [1, 0, 0])  # image up
    assert np.allclose(fr.cam_to_body(np.array([1.0, 0, 0]), off), [0, 1, 0])   # image right
    assert np.allclose(fr.cam_to_body(np.array([0, 0, 1.0]), off), [0, 0, 1])   # optical axis


def test_nose_up_pitch_makes_marker_below_appear_behind() -> None:
    # Marker straight below in the level frame; drone pitched nose-up 10 deg.
    p_level = np.array([0.0, 0.0, 5.0])
    p_body = fr.level_to_body(p_level, 0.0, np.radians(10))
    assert p_body[0] < 0 and np.isclose(p_body[0], -5 * np.sin(np.radians(10)))
    assert np.allclose(fr.body_to_level(p_body, 0.0, np.radians(10)), p_level)


def test_roll_right_makes_marker_below_appear_right() -> None:
    # Right wing down tilts the camera axis to the LEFT, so a marker straight
    # below shows up on the right side of the image (+y body).
    p_body = fr.level_to_body(np.array([0.0, 0.0, 5.0]), np.radians(10), 0.0)
    assert p_body[1] > 0 and np.isclose(p_body[1], 5 * np.sin(np.radians(10)))


def test_level_ned_round_trip_and_heading() -> None:
    v = np.array([1.0, 0.0, 0.0])          # 1 m/s north
    # Facing east (yaw 90): north is to the drone's left (-y in level frame).
    assert np.allclose(fr.ned_to_level(v, np.pi / 2), [0, -1, 0], atol=1e-12)
    assert np.allclose(fr.level_to_ned(fr.ned_to_level(v, 0.7), 0.7), v)


def test_enu_to_ned_matches_step2_verification() -> None:
    assert np.allclose(fr.enu_to_ned([3.0, 1.0, 2.0]), [1, 3, -2])


def test_gazebo_quat_nose_north_is_yaw_zero() -> None:
    # Iris spawned with ENU yaw +90 deg (Step 2) => NED heading 0.
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    roll, pitch, yaw = fr.quat_to_euler_frd(*q)
    assert np.allclose([roll, pitch, yaw], 0.0, atol=1e-9)


# ---------------------------------------------------------------- detector
def _render(tmp: Path, p_cam: np.ndarray, yaw: float, tilt: float, marker: dict = MARKER) -> np.ndarray:
    """Render the marker texture onto a grey image at a known pose (homography)."""
    tex = cv2.imread(str(make_marker_texture(marker, tmp / "m.png")), cv2.IMREAD_GRAYSCALE)
    total = marker["marker_size_m"] + 2 * marker["border_m"]
    px = tex.shape[0]
    # texture pixel (u,v) -> marker-plane metres (x right, y up), centred
    s = total / px
    t_plane = np.array([[s, 0, -total / 2], [0, -s, total / 2], [0, 0, 1]])
    # OpenCV marker frame: x right, y up, z OUT of the marker towards the
    # viewer. A marker facing the camera is therefore rotated by pi about x
    # relative to the camera frame (whose z looks INTO the scene). Using the
    # identity here renders a mirror image, which ArUco correctly rejects.
    r = fr.rot_z(yaw) @ fr.rot_x(np.pi + tilt)
    h = CAM.k @ np.column_stack([r[:, 0], r[:, 1], p_cam]) @ t_plane
    img = cv2.warpPerspective(tex, h, (CAM.width, CAM.height), borderValue=120,
                              flags=cv2.INTER_AREA)
    return img


@pytest.mark.parametrize("p_cam,yaw,tilt", [
    ((0.0, 0.0, 2.0), 0.0, 0.0),
    ((0.5, -0.3, 4.0), 0.7, 0.1),
    ((-1.5, 1.0, 6.0), -2.0, -0.15),
    ((0.0, 0.0, 10.0), 0.3, 0.0),
])
def test_detector_pose_accuracy_on_synthetic_images(tmp_path, p_cam, yaw, tilt) -> None:
    p = np.array(p_cam)
    det = MarkerDetector(CAM, MARKER).detect(_render(tmp_path, p, yaw, tilt))
    assert det is not None
    err = np.linalg.norm(det.p_cam - p)
    # PnP error grows with range; 1.5% of distance is a generous bound.
    assert err < 0.015 * np.linalg.norm(p) + 0.01, (det.p_cam, p)
    assert abs((det.yaw_cam - yaw + np.pi) % (2 * np.pi) - np.pi) < np.radians(3)
    assert det.reproj_err_px < 1.0


def test_detector_returns_none_without_marker() -> None:
    assert MarkerDetector(CAM, MARKER).detect(np.full((480, 640), 120, np.uint8)) is None


# ---------------------------------------------------------------- Kalman filter
def _kf(n: int = 1) -> MarkerKF:
    return MarkerKF(n, yaml.safe_load((ROOT / "configs/perception.yaml").read_text())["kf"])


def test_kf_estimates_rover_velocity_while_drone_moves() -> None:
    rng = np.random.default_rng(0)
    kf, dt = _kf(), 0.1
    rover_v, drone_v = np.array([1.5, -0.5]), np.array([[0.5, 0.2, 0.3]])
    p = np.array([2.0, 1.0, 6.0])
    est = []
    for i in range(300):
        p = p + np.r_[rover_v - drone_v[0, :2], -drone_v[0, 2]] * dt
        kf.predict(dt, drone_v)
        kf.update(np.array([0]), (p + rng.normal(0, 0.05, 3))[None])
        if i >= 100:
            est.append(kf.rover_vel[0].copy())
    # Single estimates are noisy (steady-state std ~0.16 m/s at 8 cm position
    # noise, measured); the estimator must be UNBIASED.
    assert np.allclose(np.mean(est, axis=0), rover_v, atol=0.05)
    assert np.allclose(kf.rel_pos[0], p, atol=0.15)


def test_kf_predicts_through_dropout() -> None:
    kf, dt = _kf(), 0.1
    rover_v, drone_v = np.array([1.0, 0.0]), np.zeros((1, 3))
    p = np.array([0.0, 0.0, 5.0])
    for _ in range(50):
        p = p + np.r_[rover_v, 0] * dt
        kf.predict(dt, drone_v)
        kf.update(np.array([0]), p[None])
    for _ in range(10):  # 1 s with no detections: prediction only
        p = p + np.r_[rover_v, 0] * dt
        kf.predict(dt, drone_v)
    assert np.linalg.norm(kf.rel_pos[0] - p) < 0.05
    assert np.isclose(kf.since_meas[0], 1.0)


def _kf_cfg(**over) -> dict:
    cfg = yaml.safe_load((ROOT / "configs/perception.yaml").read_text())["kf"]
    return {**cfg, **over}


def _circle_run(cfg: dict, speed: float = 1.5, radius: float = 3.0, dropout_s: float = 1.0):
    """Rover on a circle, static drone 5 m up; track 20 s, then predict through a dropout.
    Returns the position error at the end of the dropout."""
    rng = np.random.default_rng(1)
    kf, dt, w = MarkerKF(1, cfg), 0.1, speed / radius
    for i in range(int(20 / dt) + int(dropout_s / dt)):
        t = (i + 1) * dt
        p = np.array([radius * np.sin(w * t), radius * (1 - np.cos(w * t)), 5.0])
        kf.predict(dt, np.zeros((1, 3)))
        if t <= 20:
            kf.update(np.array([0]), (p + rng.normal(0, 0.03, 3))[None])
    return float(np.linalg.norm(kf.rel_pos[0, :2] - p[:2]))


def test_kf_defaults_are_the_published_filter() -> None:
    """New options off by default: identical numbers to the constant-velocity filter."""
    a, b = _kf(), MarkerKF(1, _kf_cfg(model="cv", gate_chi2=0))
    rng = np.random.default_rng(3)
    for _ in range(100):
        v = rng.normal(0, 0.5, (1, 3))
        z = rng.normal(0, 2, (1, 3))
        a.predict(0.1, v), b.predict(0.1, v)
        a.update(np.array([0]), z), b.update(np.array([0]), z)
    assert a.dim == 5 and np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P)


def test_ctrv_predicts_curves_better_than_cv() -> None:
    err_cv = _circle_run(_kf_cfg(model="cv"))
    err_ctrv = _circle_run(_kf_cfg(model="ctrv"))
    assert err_ctrv < 0.5 * err_cv, (err_ctrv, err_cv)


def test_ctrv_estimates_turn_rate() -> None:
    kf = MarkerKF(1, _kf_cfg(model="ctrv"))
    rng, dt, w = np.random.default_rng(2), 0.1, 0.5
    for i in range(300):
        t = (i + 1) * dt
        p = np.array([3 * np.sin(w * t), 3 * (1 - np.cos(w * t)), 5.0])
        kf.predict(dt, np.zeros((1, 3)))
        kf.update(np.array([0]), (p + rng.normal(0, 0.03, 3))[None])
    assert abs(kf.turn_rate[0] - w) < 0.1


def test_gate_rejects_a_pose_jump_and_recovers_from_a_real_one() -> None:
    kf, dt = MarkerKF(1, _kf_cfg(gate_chi2=16.3, gate_reinit=3)), 0.1
    p = np.array([0.0, 0.0, 2.2])
    for _ in range(30):
        kf.predict(dt, np.zeros((1, 3)))
        kf.update(np.array([0]), p[None])
    kf.predict(dt, np.zeros((1, 3)))
    ok = kf.update(np.array([0]), (p + [-0.6, 0.0, 0.0])[None])   # the rehearsal's 0.6 m jump
    assert not ok[0] and np.linalg.norm(kf.rel_pos[0] - p) < 0.05
    for _ in range(3):   # the jump persists -> it was real: re-initialise on it
        kf.predict(dt, np.zeros((1, 3)))
        kf.update(np.array([0]), (p + [-0.6, 0.0, 0.0])[None])
    assert np.linalg.norm(kf.rel_pos[0] - (p + [-0.6, 0, 0])) < 0.05


NESTED = {**MARKER, "texture_px": 1200, "error_correction_rate": 1.0,
          "nested": {**MARKER["nested"], "enabled": True}}


@pytest.mark.parametrize("p_cam,yaw,tilt,want_id", [
    ((0.0, 0.0, 4.0), 0.4, 0.05, 0),       # high: the big marker
    ((0.05, -0.03, 0.8), 1.2, 0.08, 1),    # low: only the small one resolves cleanly
    ((-0.04, 0.02, 0.3), -2.5, -0.1, 1),   # below 0.47 m: the big one is out of frame
    ((0.02, 0.02, 0.15), 0.0, 0.0, 1),
])
def test_nested_marker_gives_pad_centre_at_every_height(tmp_path, p_cam, yaw, tilt, want_id) -> None:
    p = np.array(p_cam)
    det = MarkerDetector(CAM, NESTED).detect(_render(tmp_path, p, yaw, tilt, NESTED))
    assert det is not None and det.marker_id == want_id
    # pose of the PAD centre (inner marker corrected by its 4.2 cm offset), not of the inner marker
    assert np.linalg.norm(det.p_cam - p) < 0.015 * np.linalg.norm(p) + 0.008, (det.p_cam, p)


def test_nested_cell_must_be_white(tmp_path) -> None:
    bad = {**NESTED, "nested": {**NESTED["nested"], "cell": [2, 3]}}   # a black cell of id 0
    with pytest.raises(ValueError):
        make_marker_texture(bad, tmp_path / "bad.png")
