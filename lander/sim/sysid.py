"""System identification: how does ArduCopter's GUIDED velocity loop respond?

Stage A needs a cheap model of "command velocity u -> actual velocity v".
We excite SITL (in Gazebo, lockstep) with velocity steps in the level frame,
log the response, and fit per axis:

    a_des = clip( (u(t - delay) - v) / tau, -a_max, a_max )
    a'    = clip( (a_des - a) / dt, -jerk, jerk )          (jerk-limited)
    v'    = a

i.e. a first-order lag with time constant tau, a pure transport delay, an
acceleration limit (ArduPilot's WPNAV_ACCEL = 2.5 m/s^2) and a jerk limit
(PSC_JERK_NE = 5 m/s^3; both defaults verified in the source). The jerk limit
matters beyond velocity fit quality: acceleration -> tilt -> camera motion, and
an instantaneous acceleration step makes the simulated camera tilt 15 deg in
one substep, which a real copter never does (found in Stage A, Step 8). We also log attitude to fit tilt = atan(a / g), which Stage A uses to
tilt the simulated camera.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from lander.mav import MavLink
from lander.perception.frames import ned_to_level
from lander.sim.demo import running_world
from lander.sim.gz_client import Vec3, project_root
from lander.sim.sitl import launch_sitl, stop_process


def simulate_axis(u: np.ndarray, dt: float, tau: float, delay: float, a_max: float,
                  jerk: float, v0: float = 0.0) -> np.ndarray:
    """The Stage A velocity model on one axis (same equations as fast_env)."""
    k = int(round(delay / dt))
    ud = np.concatenate([np.full(k, u[0]), u[:len(u) - k]]) if k else u
    v = np.empty_like(u)
    cur, acc = v0, 0.0
    for i, cmd in enumerate(ud):
        a_des = np.clip((cmd - cur) / tau, -a_max, a_max)
        acc += np.clip(a_des - acc, -jerk * dt, jerk * dt)
        cur = cur + acc * dt
        v[i] = cur
    return v


def fit_axis(t: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict:
    """Grid search (tau, delay, a_max, jerk) minimising velocity RMSE.

    Vectorised over all parameter combinations at once: one time loop, each
    step updating every candidate model in parallel.
    """
    dt = 0.02
    grid_t = np.arange(t[0], t[-1], dt)
    ug = np.interp(grid_t, t, u)
    vg = np.interp(grid_t, t, v)
    taus = np.arange(0.05, 1.01, 0.05)
    delays = np.arange(0.0, 0.41, 0.02)
    amaxs = np.arange(0.5, 5.01, 0.25)
    jerks = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 50.0])
    tt, dd, aa, jj = (x.ravel() for x in np.meshgrid(taus, delays, amaxs, jerks, indexing="ij"))
    k = np.rint(dd / dt).astype(int)
    cur = np.full(tt.shape, vg[0])
    acc = np.zeros(tt.shape)
    sq = np.zeros(tt.shape)
    for i in range(len(ug)):
        cmd = ug[np.maximum(i - k, 0)]
        a_des = np.clip((cmd - cur) / tt, -aa, aa)
        acc += np.clip(a_des - acc, -jj * dt, jj * dt)
        cur = cur + acc * dt
        sq += (cur - vg[i]) ** 2
    j = int(np.argmin(sq))
    return {"tau_s": round(float(tt[j]), 3), "delay_s": round(float(dd[j]), 3),
            "a_max_mps2": round(float(aa[j]), 3), "jerk_mps3": float(jj[j]),
            "rmse_mps": float(np.sqrt(sq[j] / len(ug)))}


def fit_log(a: np.ndarray) -> dict:
    """Fit all axes + tilt gains from a logged experiment array."""
    t = a[:, 0]
    result: dict = {axis: fit_axis(t, a[:, 1 + i], a[:, 4 + i]) for i, axis in enumerate("xyz")}
    ax = np.gradient(a[:, 4], t)
    ay = np.gradient(a[:, 5], t)
    g = 9.81
    sel = np.abs(ax) > 0.3
    result["pitch_per_accel_gain"] = float(np.polyfit(-np.arctan(ax[sel] / g), a[sel, 8], 1)[0])
    sel = np.abs(ay) > 0.3
    result["roll_per_accel_gain"] = float(np.polyfit(np.arctan(ay[sel] / g), a[sel, 7], 1)[0])
    return result


def run_sysid(sim_cfg: dict, sitl_cfg: dict, paths: dict, out_dir: Path, rtf: float) -> dict:
    sid = sitl_cfg["sysid"]
    d = sitl_cfg["drone"]
    sitl_cfg = {**sitl_cfg, "speedup": rtf}
    log: list[tuple] = []
    with running_world(sim_cfg, gui=False) as w:
        w.spawn(d["model_uri"], d["name"], Vec3(0, 0, d["spawn_z_m"]), math.radians(d["spawn_yaw_deg"]))
        w.set_real_time_factor(rtf)
        w.pause(False)
        sitl = launch_sitl(sitl_cfg, paths, project_root())
        try:
            mav = MavLink(sitl_cfg["mavlink_url"], rates_hz={"LOCAL_POSITION_NED": 50, "ATTITUDE": 50})
            mav.arm_guided(sitl_cfg["fly_check"]["ekf_ready_timeout_s"])
            mav.takeoff(sid["altitude_m"])
            t_end = w.sim_time + sid["settle_s"] * 3
            while w.sim_time < t_end:
                time.sleep(0.01)
            # Sequence of (axis, value, duration): step up, back to zero, etc.
            plan = []
            for axis, amps in (("x", sid["xy_steps_mps"]), ("y", sid["xy_steps_mps"]),
                               ("z", sid["z_steps_mps"])):
                for a in amps:
                    plan += [(axis, a, sid["hold_s"]), (axis, 0.0, sid["settle_s"])]
            for axis, val, dur in plan:
                u = np.zeros(3)
                u["xyz".index(axis)] = val
                t_stop = w.sim_time + dur
                next_send = w.sim_time
                while (now := w.sim_time) < t_stop:
                    if now >= next_send:  # stream at control rate, like the policy will
                        mav.send_velocity_level(u)
                        next_send = now + sid["send_dt_s"]
                    roll, pitch, yaw = mav.attitude()
                    v = ned_to_level(mav.vel_ned(), yaw)
                    log.append((now, *u, *v, roll, pitch))
                    time.sleep(0.002)
            mav.set_mode("LAND")
            mav.close()
        finally:
            stop_process(sitl)

    a = np.array(log)
    # Drop duplicate timestamps (we poll faster than messages arrive).
    _, keep = np.unique(a[:, 0], return_index=True)
    a = a[keep]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sysid_log.npy", a)
    result = fit_log(a)
    (out_dir / "sysid.json").write_text(json.dumps(result, indent=2))
    return result
