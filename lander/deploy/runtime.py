"""Companion-computer runtime: camera -> detector -> KF -> policy -> safety -> MAVLink.

Threads (and why):
  * camera thread    : grabs frames as fast as the camera delivers and keeps
                       only the NEWEST (stamped at capture). A queue would
                       hand the controller ever-older images.
  * MAVLink thread   : `lander.mav.MavLink` reads every message, caches the
                       newest of each type and an attitude history.
  * control loop     : the main thread, at a FIXED rate (10 Hz = the rate the
                       policy was trained at). Fixed rate matters because the
                       policy's dynamics knowledge (and the KF's dt) assume it:
                       running at 7 Hz or with jitter is a different MDP. It
                       never blocks on I/O; it only reads cached state.

Latency budget per tick (measured and logged): image age at use + detection
+ KF + policy + safety + send. Must stay well inside 100 ms; the policy was
trained with image ages up to 0.165 s (measured in Gazebo) x up to 1.5 (DR).

The same code runs against Gazebo+SITL (GzFrameSource, sim clock) and on the
real drone (CvFrameSource, monotonic clock, serial MAVLink).
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from lander.deploy.export import onnx_session
from lander.deploy.safety import Decision, SafetyConfig, VehicleState, check
from lander.envs import task as tk
from lander.mav import MavLink
from lander.perception import frames as fr
from lander.perception.detector import CameraModel, MarkerDetector
from lander.perception.kalman import MarkerKF


class FrameSource(Protocol):
    def latest(self) -> tuple[float, np.ndarray] | None: ...
    def clock(self) -> float: ...


class CvFrameSource:
    """USB/CSI camera through OpenCV, frames stamped with time.monotonic()."""

    def __init__(self, device: int | str, width: int, height: int) -> None:
        import cv2

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # don't let the driver queue stale frames
        self._lock = threading.Lock()
        self._frame: tuple[float, np.ndarray] | None = None
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        import cv2

        while self._run:
            ok, bgr = self.cap.read()
            t = time.monotonic()
            if ok:
                with self._lock:
                    self._frame = (t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def latest(self):
        with self._lock:
            return self._frame

    def clock(self) -> float:
        return time.monotonic()


class GzFrameSource:
    """Gazebo camera topic; stamps and clock are SIM time (lockstep)."""

    def __init__(self, topic: str, sim_clock: Callable[[], float]) -> None:
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node

        self._lock = threading.Lock()
        self._msg = None
        self._clock = sim_clock
        self.node = Node()
        self.node.subscribe(Image, topic, self._cb)

    def _cb(self, m) -> None:
        with self._lock:
            self._msg = m

    def latest(self):
        with self._lock:
            m = self._msg
        if m is None:
            return None
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        return t, np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)

    def clock(self) -> float:
        return self._clock()


@dataclass
class RuntimeConfig:
    rate_hz: float = 10.0
    cut_height_m: float = 0.40        # est. height above deck at which motors are cut (touchdown)
    cut_max_xy_m: float = 0.20
    cut_max_lost_s: float = 0.6       # marker must have been seen this recently (KF still fresh)
    max_duration_s: float = 60.0


class Runtime:
    def __init__(self, cfg: dict, onnx_path: Path, frames: FrameSource, mav: MavLink,
                 safety: SafetyConfig, rt: RuntimeConfig, log_path: Path,
                 residual_scale: float | None = None) -> None:
        """residual_scale: if set, the ONNX model is a residual on top of the PID
        baseline (training/residual.py) and the action is PID + scale * residual."""
        self.residual_scale = residual_scale
        if residual_scale is not None:
            from lander.baselines.pid import PIDLander

            self.pid = PIDLander(cfg["baseline"], cfg["task"])
        self.cfg, self.task = cfg, cfg["task"]
        self.frames, self.mav, self.safety, self.rt = frames, mav, safety, rt
        self.sess = onnx_session(onnx_path, threads=1)
        self.cam = CameraModel.from_config(cfg["camera"])
        self.det = MarkerDetector(self.cam, cfg["marker"])
        self.mount = np.array(cfg["camera"]["mount_offset_body_frd_m"])
        self.kf = MarkerKF(1, cfg["perception"]["kf"])
        self.log_path = log_path

    def _vehicle_state(self) -> VehicleState:
        p = self.mav.pos_ned()
        est = self.kf.rel_pos[0]
        return VehicleState(
            mode=self.mav.mode, armed=self.mav.armed, heartbeat_age_s=self.mav.age_s("HEARTBEAT"),
            alt_m=float(-p[2]), dist_from_home_m=float(np.linalg.norm(p[:2])),
            marker_lost_s=float(self.kf.since_meas[0]),
            marker_xy_err_m=float(np.linalg.norm(est[:2])) if self.kf.initialised[0] else float("inf"))

    def run(self, stop: Callable[[], bool] = lambda: False) -> dict:
        period = 1.0 / self.rt.rate_hz
        clock = self.frames.clock
        # don't take control before the camera delivers: with no frame the marker counts as lost
        # forever and the safety layer would RTL on the first tick (seen in rehearsal 2026-10-15)
        t_wait = time.monotonic()
        while self.frames.latest() is None:
            if time.monotonic() - t_wait > 10.0:
                return {"outcome": "no_camera", "ticks": 0, "loop_ms_mean": 0.0, "loop_ms_p99": 0.0,
                        "img_age_mean_s": float("nan"), "overruns": 0}
            time.sleep(0.01)
        t_start = clock()
        next_tick = t_start
        last_img_t = -1.0
        last_tick = t_start
        prev_action = np.zeros(3)
        rows: list[dict] = []
        outcome = "timeout"
        overruns = 0
        while not stop():
            # ---- fixed-rate scheduling in the source's clock (sim or wall)
            while clock() < next_tick:
                time.sleep(0.0005)
            now = clock()
            if now - next_tick > period:      # we missed a whole tick: report, don't try to catch up
                overruns += 1
                next_tick = now
            next_tick += period
            w0 = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            # ---- state estimation
            _, _, yaw = self.mav.attitude()
            v_level = fr.ned_to_level(self.mav.vel_ned(), yaw)
            self.kf.predict(dt, v_level[None])
            frame = self.frames.latest()
            visible, img_age, det_ms = False, float("nan"), 0.0
            if frame is not None and frame[0] > last_img_t:
                last_img_t, img = frame
                img_age = now - last_img_t
                t0 = time.perf_counter()
                d = self.det.detect(img)
                det_ms = (time.perf_counter() - t0) * 1e3
                if d is not None:
                    roll, pitch, _ = self.mav.attitude_at(last_img_t)   # image-time attitude
                    p = fr.body_to_level(fr.cam_to_body(d.p_cam, self.mount), roll, pitch)
                    self.kf.update(np.array([0]), p[None])
                    visible = True
            obs = tk.build_obs(self.kf.rel_pos, self.kf.rover_vel, self.kf.initialised, v_level[None],
                               np.array([visible]), self.kf.since_meas, prev_action[None], self.task)
            t0 = time.perf_counter()
            action = self.sess.run(None, {"obs": obs})[0][0]
            if self.residual_scale is not None:
                from lander.training.residual import compose

                action = compose(self.pid(obs), action[None], self.residual_scale)[0]
            pol_ms = (time.perf_counter() - t0) * 1e3
            v_cmd = tk.action_to_velocity(action[None], self.task,
                                          tk.feedforward(self.kf.rover_vel, self.kf.initialised))[0]
            # ---- touchdown: motor cut on the platform
            est = self.kf.rel_pos[0]
            state = self._vehicle_state()
            if (self.kf.initialised[0] and est[2] <= self.rt.cut_height_m
                    and np.linalg.norm(est[:2]) <= self.rt.cut_max_xy_m
                    and self.kf.since_meas[0] <= self.rt.cut_max_lost_s and state.mode == "GUIDED"):
                # MAV_CMD_COMPONENT_ARM_DISARM param2=21196 = force (disarm in flight)
                self.mav.command(400, 0, 21196)
                outcome = "touchdown_cut"
                rows.append({"t": now - t_start, "event": "motor_cut", "est_h": est[2]})
                break
            out = check(state, v_cmd, self.safety)
            if out.decision == Decision.SEND:
                self.mav.send_velocity_level(out.velocity)
            elif out.decision == Decision.HOVER:
                self.mav.send_velocity_level(np.zeros(3))
            elif out.decision == Decision.RTL:
                self.mav.set_mode("RTL")
                outcome = "rtl:" + ";".join(out.reasons)
                break
            loop_ms = (time.perf_counter() - w0) * 1e3
            prev_action = np.asarray(action, dtype=float)
            rows.append({"t": now - t_start, "dt": dt, "img_age_s": img_age, "det_ms": det_ms,
                         "policy_ms": pol_ms, "loop_ms": loop_ms, "visible": visible,
                         "est_x": est[0], "est_y": est[1], "est_h": est[2],
                         "vx": out.velocity[0], "vy": out.velocity[1], "vz": out.velocity[2],
                         "decision": out.decision.value, "reasons": "|".join(out.reasons),
                         "mode": state.mode})
            if out.decision == Decision.SILENT and state.mode != "GUIDED":
                outcome = f"pilot_override:{state.mode}"
                break
            if now - t_start > self.rt.max_duration_s:
                break
        self._write_log(rows)
        timing = [r for r in rows if "loop_ms" in r]
        ages = [r["img_age_s"] for r in timing if np.isfinite(r["img_age_s"])]
        return {"outcome": outcome, "ticks": len(timing), "overruns": overruns,
                "loop_ms_mean": float(np.mean([r["loop_ms"] for r in timing])) if timing else 0.0,
                "loop_ms_p99": float(np.percentile([r["loop_ms"] for r in timing], 99)) if timing else 0.0,
                "det_ms_mean": float(np.mean([r["det_ms"] for r in timing if r["det_ms"] > 0] or [0])),
                "policy_ms_mean": float(np.mean([r["policy_ms"] for r in timing])) if timing else 0.0,
                "img_age_mean_s": float(np.mean(ages)) if ages else float("nan"),
                "dt_std_s": float(np.std([r["dt"] for r in timing])) if timing else 0.0}

    def _write_log(self, rows: list[dict]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(self.log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
