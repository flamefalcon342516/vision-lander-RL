"""MAVLink client shared by sysid, the Gazebo env and the companion runtime.

A background thread reads every message and keeps the newest of each type, so
control code never blocks on the link and never processes stale backlog. The
same class talks to SITL (tcp:127.0.0.1:5760) or a real flight controller
(serial:/dev/ttyAMA0:921600, udp:...).
"""
from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np
from pymavlink import mavutil

ML = mavutil.mavlink

# SET_POSITION_TARGET_LOCAL_NED type_mask bits (1 = ignore). See LEARNING Step 2.
MASK_VEL_ONLY = 0b0000_1111_1100_0111      # 4039: use vx, vy, vz only
MASK_POS_ONLY = 0b0000_1111_1111_1000      # 4088: use x, y, z only


class MavLink:
    def __init__(self, url: str, timeout_s: float = 30.0, rates_hz: dict[str, float] | None = None,
                 clock: Callable[[], float] = time.monotonic, history_s: float = 1.0):
        """clock: the time base for the attitude history. Must be the same clock
        the camera frames are stamped with (Gazebo sim time in Stage B,
        time.monotonic on hardware)."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.mav = mavutil.mavlink_connection(url, source_system=255, autoreconnect=True)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"nothing listening on {url}")
                time.sleep(0.5)
        if self.mav.wait_heartbeat(timeout=timeout_s) is None:
            raise TimeoutError("no HEARTBEAT")
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._rx_time: dict[str, float] = {}
        self._acks: list[Any] = []
        self._clock = clock
        # (t, roll, pitch, yaw) samples, newest last; long enough to cover
        # camera latency (~0.17 s measured) with margin.
        self._att_hist: deque[tuple[float, float, float, float]] = deque(maxlen=int(history_s * 200))
        self.statustext: list[str] = []
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        for name, hz in (rates_hz or {}).items():
            self.set_rate(name, hz)

    # ------------------------------------------------------------ plumbing
    def _reader(self) -> None:
        while self._running:
            m = self.mav.recv_match(blocking=True, timeout=0.1)
            if m is None:
                continue
            kind = m.get_type()
            with self._lock:
                self._latest[kind] = m
                self._rx_time[kind] = time.monotonic()
                if kind == "COMMAND_ACK":
                    self._acks.append(m)
                elif kind == "ATTITUDE":
                    self._att_hist.append((self._clock(), m.roll, m.pitch, m.yaw))
                elif kind == "STATUSTEXT":
                    self.statustext.append(m.text)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.mav.close()

    def latest(self, kind: str) -> Any:
        with self._lock:
            return self._latest.get(kind)

    def age_s(self, kind: str) -> float:
        with self._lock:
            t = self._rx_time.get(kind)
        return float("inf") if t is None else time.monotonic() - t

    def set_rate(self, name: str, hz: float) -> None:
        msg_id = getattr(ML, f"MAVLINK_MSG_ID_{name}")
        self.mav.mav.command_long_send(self.mav.target_system, self.mav.target_component,
                                       ML.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                       msg_id, 1e6 / hz, 0, 0, 0, 0, 0)

    def command(self, cmd: int, *params: float, timeout_s: float = 3.0) -> int:
        with self._lock:
            self._acks.clear()
        p = list(params) + [0.0] * (7 - len(params))
        self.mav.mav.command_long_send(self.mav.target_system, self.mav.target_component,
                                       cmd, 0, *p)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                for a in self._acks:
                    if a.command == cmd:
                        return a.result
            time.sleep(0.01)
        return -1

    # ------------------------------------------------------------ state
    @property
    def armed(self) -> bool:
        hb = self.latest("HEARTBEAT")
        return bool(hb and hb.base_mode & ML.MAV_MODE_FLAG_SAFETY_ARMED)

    @property
    def mode(self) -> str:
        hb = self.latest("HEARTBEAT")
        return mavutil.mode_string_v10(hb) if hb else "UNKNOWN"

    def attitude(self) -> tuple[float, float, float]:
        a = self.latest("ATTITUDE")
        return (a.roll, a.pitch, a.yaw) if a else (0.0, 0.0, 0.0)

    def attitude_at(self, t: float) -> tuple[float, float, float]:
        """Attitude interpolated at time t (camera timestamp, same clock).

        De-rotating an image with the attitude at processing time instead of
        capture time is a large error: up to ~165 ms of latency during a 15 deg
        tilt manoeuvre moves a marker seen from 6 m by ~1.6 m (Step 8).
        """
        with self._lock:
            hist = list(self._att_hist)
        if not hist:
            return 0.0, 0.0, 0.0
        times = [h[0] for h in hist]
        i = bisect.bisect_left(times, t)
        if i <= 0:
            return hist[0][1:]
        if i >= len(hist):
            return hist[-1][1:]
        (t0, *a0), (t1, *a1) = hist[i - 1], hist[i]
        w = (t - t0) / max(t1 - t0, 1e-9)
        return tuple(float(x0 + w * (x1 - x0)) for x0, x1 in zip(a0, a1))  # type: ignore[return-value]

    def vel_ned(self) -> np.ndarray:
        lp = self.latest("LOCAL_POSITION_NED")
        return np.array([lp.vx, lp.vy, lp.vz]) if lp else np.zeros(3)

    def pos_ned(self) -> np.ndarray:
        lp = self.latest("LOCAL_POSITION_NED")
        return np.array([lp.x, lp.y, lp.z]) if lp else np.zeros(3)

    # ------------------------------------------------------------ commands
    def set_mode(self, mode: str) -> None:
        self.mav.set_mode(mode)

    def arm_guided(self, timeout_s: float) -> None:
        """GUIDED + arm, retrying while pre-arm checks (EKF/GPS) are not ready."""
        deadline = time.monotonic() + timeout_s
        while True:
            self.set_mode("GUIDED")
            if self.command(ML.MAV_CMD_COMPONENT_ARM_DISARM, 1) == ML.MAV_RESULT_ACCEPTED:
                return
            if time.monotonic() > deadline:
                raise TimeoutError("could not arm: " + " | ".join(self.statustext[-3:]))
            time.sleep(1.0)

    def takeoff(self, alt_m: float) -> bool:
        return self.command(ML.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt_m) == ML.MAV_RESULT_ACCEPTED

    def send_velocity_level(self, v: np.ndarray) -> None:
        """Velocity in the level frame (x fwd, y right, z down). MAV_FRAME_BODY_NED
        is rotated by yaw only inside ArduPilot, i.e. it IS the level frame."""
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component, ML.MAV_FRAME_BODY_NED,
            MASK_VEL_ONLY, 0, 0, 0, float(v[0]), float(v[1]), float(v[2]), 0, 0, 0, 0, 0)

    def send_position_ned(self, p: np.ndarray) -> None:
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component, ML.MAV_FRAME_LOCAL_NED,
            MASK_POS_ONLY, float(p[0]), float(p[1]), float(p[2]), 0, 0, 0, 0, 0, 0, 0, 0)

    def set_param(self, name: str, value: float) -> None:
        self.mav.mav.param_set_send(self.mav.target_system, self.mav.target_component,
                                    name.encode(), float(value), ML.MAV_PARAM_TYPE_REAL32)
