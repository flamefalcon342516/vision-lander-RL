"""`python -m lander sim fly-check`: automated proof that Gazebo + SITL fly.

Brings up the world, spawns the iris, starts SITL, then GUIDED -> arm ->
takeoff -> hover -> LAND -> disarm with pymavlink. Along the way it checks two
things we will rely on for the rest of the project:

1. Frames: the drone's Gazebo pose (ENU) mapped to NED agrees with SITL's
   LOCAL_POSITION_NED. If this fails, every later transform is suspect.
2. Lockstep speed: sim seconds per wall second with SITL in the loop.

This is a check, not the control code: Step 6 builds the real velocity
controller.
"""
from __future__ import annotations

import math
import time

import numpy as np
from pymavlink import mavutil

from lander.sim.demo import running_world
from lander.sim.gz_client import GzWorld, Vec3, project_root
from lander.sim.sitl import connect, launch_sitl, stop_process

MSG_RATES_HZ = {  # what we ask SITL to stream on this link
    "LOCAL_POSITION_NED": 20,
    "ATTITUDE": 20,
    "SYS_STATUS": 2,
}


def _set_rates(mav: mavutil.mavfile) -> None:
    # MAV_CMD_SET_MESSAGE_INTERVAL: param1 = message id, param2 = interval in us.
    for name, hz in MSG_RATES_HZ.items():
        msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}")
        mav.mav.command_long_send(mav.target_system, mav.target_component,
                                  mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                  msg_id, 1e6 / hz, 0, 0, 0, 0, 0)


def _command(mav: mavutil.mavfile, cmd: int, *params: float, timeout_s: float = 5) -> int:
    """Send COMMAND_LONG and return the COMMAND_ACK result code."""
    p = list(params) + [0.0] * (7 - len(params))
    mav.mav.command_long_send(mav.target_system, mav.target_component, cmd, 0, *p)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
        if ack is not None and ack.command == cmd:
            return ack.result
    return -1


def _latest(mav: mavutil.mavfile, kind: str, timeout_s: float = 2.0):
    """Drain the link and return the newest message of `kind` (None if none)."""
    msg, deadline = None, time.monotonic() + timeout_s
    while True:
        m = mav.recv_match(type=kind, blocking=msg is None, timeout=0.2)
        if m is None:
            if msg is not None or time.monotonic() > deadline:
                return msg
            continue
        msg = m


def _wait(cond, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for: {what}")


def _gz_to_ned(p: Vec3, origin: Vec3) -> np.ndarray:
    """Gazebo ENU world position -> NED relative to the spawn point.

    ENU (x=E, y=N, z=U) -> NED (N, E, D) = (y, x, -z).
    """
    return np.array([p.y - origin.y, p.x - origin.x, -(p.z - origin.z)])


def fly_check(sim_cfg: dict, sitl_cfg: dict, paths: dict, gui: bool, rtf: float) -> int:
    d, fc = sitl_cfg["drone"], sitl_cfg["fly_check"]
    origin = Vec3(0.0, 0.0, d["spawn_z_m"])
    root = project_root()
    with running_world(sim_cfg, gui) as w:
        w.spawn(d["model_uri"], d["name"], origin, math.radians(d["spawn_yaw_deg"]))
        if rtf != 1.0:
            w.set_real_time_factor(rtf)
        w.pause(False)
        sitl = launch_sitl(sitl_cfg, paths, root)
        try:
            return _fly(w, sitl_cfg, d["name"], origin, fc)
        finally:
            stop_process(sitl)


def _fly(w: GzWorld, sitl_cfg: dict, name: str, origin: Vec3, fc: dict) -> int:
    mav = connect(sitl_cfg["mavlink_url"], timeout_s=30)
    print(f"connected: sysid {mav.target_system}, autopilot heartbeat OK")
    _set_rates(mav)

    # 1. GUIDED + arm. Arming is refused until the EKF has converged and GPS has
    # a fix (pre-arm checks), so retry instead of sleeping a magic duration.
    t_wall0 = time.monotonic()

    def armed_ok() -> bool:
        # GUIDED itself is refused until there is a position estimate, so the
        # mode request is retried together with arming.
        mav.set_mode("GUIDED")
        res = _command(mav, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if res != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            time.sleep(1.0)
        return res == mavutil.mavlink.MAV_RESULT_ACCEPTED

    _wait(armed_ok, fc["ekf_ready_timeout_s"], "pre-arm checks to pass and arm")
    print(f"armed after {time.monotonic() - t_wall0:.1f} s wall (EKF/GPS convergence)")

    # 2. Takeoff. param7 = altitude (m, relative to home).
    if _command(mav, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0,
                fc["takeoff_alt_m"]) != mavutil.mavlink.MAV_RESULT_ACCEPTED:
        print("FAIL: takeoff rejected")
        return 1
    sim0, wall0 = w.sim_time, time.monotonic()
    target = fc["takeoff_alt_m"] - fc["alt_tolerance_m"]
    _wait(lambda: -_latest(mav, "LOCAL_POSITION_NED").z >= target,
          fc["phase_timeout_s"], f"altitude >= {target} m")
    print(f"reached {fc['takeoff_alt_m']} m in {w.sim_time - sim0:.1f} s sim")

    # 3. Hover: compare Gazebo ground truth (mapped to NED) with SITL's estimate.
    errs, hover_end = [], w.sim_time + fc["hover_s"]
    while w.sim_time < hover_end:
        lp = _latest(mav, "LOCAL_POSITION_NED")
        gz = w.pose(name)
        if lp is not None and gz is not None:
            errs.append(_gz_to_ned(gz, origin) - np.array([lp.x, lp.y, lp.z]))
    err = np.array(errs)
    att = _latest(mav, "ATTITUDE")
    speed = (w.sim_time - sim0) / (time.monotonic() - wall0)

    # 4. Land and wait for auto-disarm.
    mav.set_mode("LAND")
    _wait(lambda: not (_latest(mav, "HEARTBEAT").base_mode
                       & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
          fc["phase_timeout_s"], "landing and disarm")
    print("landed and disarmed")

    mean, worst = err.mean(axis=0), np.abs(err).max()
    print(f"frame check     : mean (gz_NED - sitl_NED) = [{mean[0]:+.3f} {mean[1]:+.3f} "
          f"{mean[2]:+.3f}] m, worst |err| {worst:.3f} m over {len(err)} samples")
    print(f"heading at hover: {math.degrees(att.yaw):+.1f} deg (expect ~0 = north)")
    print(f"lockstep speed  : {speed:.2f}x real time during climb + hover")
    ok = worst <= fc["frame_tolerance_m"]
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1
