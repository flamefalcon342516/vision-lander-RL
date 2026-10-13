"""Step 1 demos: bring the world up, and a free-fall physics check.

The drop test is our first "measure, don't assume" check of the simulator:
release a crate, step physics deterministically, and compare the touchdown
time and the fitted gravity against h = g t^2 / 2.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from lander.sim.gz_client import GzWorld, Vec3, launch_gz, project_root


@contextmanager
def running_world(cfg: dict, gui: bool) -> Iterator[GzWorld]:
    """Launch gz (paused), yield a connected client, always shut down cleanly."""
    world_file = str(project_root() / cfg["world_file"])
    proc = launch_gz(world_file, headless=not gui, verbosity=cfg["verbosity"])
    try:
        with GzWorld(cfg["world_name"], cfg["service_timeout_ms"], cfg["physics_step_s"]) as w:
            w.wait_until_up(cfg["startup_timeout_s"])
            yield w
    finally:
        _stop(proc)


def _stop(proc: subprocess.Popen) -> None:
    # SIGINT = Ctrl-C: gz shuts down its server/GUI threads gracefully.
    # Signalling the process group also reaches the GUI child process.
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def world_up(cfg: dict, gui: bool, drone: dict | None = None) -> int:
    """Start the world, unpause, and keep it running until Ctrl-C / SIGTERM.

    With `drone` (the `drone:` block of configs/sitl.yaml), the iris is spawned
    so SITL can connect: run `make sitl-manual` in a second terminal.
    """
    stop = threading.Event()
    # Install handlers explicitly: a process started in the background by a
    # non-interactive shell inherits SIGINT as *ignored*, so relying on the
    # default KeyboardInterrupt would leave gz running forever (measured).
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with running_world(cfg, gui) as w:
        if drone is not None:
            w.spawn(drone["model_uri"], drone["name"], Vec3(0.0, 0.0, drone["spawn_z_m"]),
                    math.radians(drone["spawn_yaw_deg"]))
        w.pause(False)
        print(f"world '{w.world}' running. Try:  gz topic -l   |   gz service -l")
        print("Ctrl-C to stop.")
        stop.wait()
    print("stopped cleanly")
    return 0


def drop_test(cfg: dict, gui: bool) -> int:
    d = cfg["drop_test"]
    dt = cfg["physics_step_s"]
    half, h, g = d["half_size_m"], d["drop_height_m"], d["gravity_mps2"]
    chunk = 10  # physics steps per sample: 10 ms resolution

    with running_world(cfg, gui) as w:
        # Unthrottled stepping: we don't care about wall-clock pacing here.
        w.set_real_time_factor(cfg["max_speed_rtf"])
        w.spawn(d["model_uri"], d["name"], Vec3(0.0, 0.0, h + half))
        t0 = w.sim_time
        w.wait_for_pose(d["name"], t0, timeout_s=5)

        ts, zs = [], []
        wall0 = time.perf_counter()
        t = t0
        # Run long enough to fall, touch down and settle.
        t_theory = float(np.sqrt(2 * h / g))
        while t < t0 + 1.5 * t_theory + 0.3:
            w.step(chunk)
            t += chunk * dt
            w.wait_for_sim_time(t, timeout_s=5)
            zs.append(w.wait_for_pose(d["name"], t, timeout_s=5).z)
            ts.append(t - t0)
        wall = time.perf_counter() - wall0

    ts_a, zs_a = np.array(ts), np.array(zs)
    bottom = zs_a - half
    # Touchdown: first sample whose bottom face reached the ground (1 mm slack:
    # the contact solver allows a tiny penetration).
    touched = np.flatnonzero(bottom <= 1e-3)
    if len(touched) == 0:
        print("FAIL: crate never reached the ground")
        return 1
    t_touch = ts_a[touched[0]]
    # Fit z(t) = z0 - g t^2/2 on the free-fall part only -> measured gravity.
    free = ts_a < t_touch - chunk * dt
    coeffs = np.polyfit(ts_a[free], zs_a[free], 2)
    g_fit = -2 * coeffs[0]
    rest = zs_a[-1] - half

    ok = abs(t_touch - t_theory) <= d["tolerance_s"]
    print(f"touchdown       : {t_touch:.3f} s   (theory sqrt(2h/g) = {t_theory:.3f} s, "
          f"sampled every {chunk * dt * 1e3:.0f} ms)")
    print(f"fitted gravity  : {g_fit:.3f} m/s^2 (configured {g})")
    print(f"rest height err : {rest * 1e3:+.2f} mm (negative = contact penetration)")
    print(f"speed           : {ts_a[-1]:.2f} s sim in {wall:.2f} s wall "
          f"= {ts_a[-1] / wall:.1f}x real time")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1
