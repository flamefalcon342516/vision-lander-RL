"""ArduCopter SITL process management (JSON backend <-> ardupilot_gazebo).

We run the `arducopter` binary directly instead of `sim_vehicle.py`: the latter
wraps MAVProxy, which wants a TTY and is awkward to script headless. For
*manual* flying, sim_vehicle.py is the right tool (see `make sitl-manual`).

Ports for instance I (default 0), all on localhost:
  tcp 5760 + 10*I : SERIAL0, MAVLink (what pymavlink / MAVProxy connect to)
  udp 9002 + 10*I : JSON servo packets to Gazebo; Gazebo replies with state
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from pymavlink import mavutil


def launch_sitl(sitl_cfg: dict, paths: dict, root: Path) -> subprocess.Popen:
    ardupilot = Path(paths["ardupilot_dir"]).expanduser()
    binary = Path(paths["arducopter_bin"]).expanduser()
    defaults = ",".join(
        str(root / p.removeprefix("PROJECT:")) if p.startswith("PROJECT:") else str(ardupilot / p)
        for p in sitl_cfg["default_params"])
    workdir = root / sitl_cfg["workdir"]
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--model", "JSON",           # external physics speaking the JSON protocol
        "--speedup", str(sitl_cfg["speedup"]),
        "--instance", str(sitl_cfg["instance"]),
        "--home", sitl_cfg["home"],
        "--defaults", defaults,
        "--wipe",                    # fresh eeprom: params = defaults every run (reproducible)
        "--sim-address", "127.0.0.1",
    ]
    log = open(workdir / "sitl.log", "w")
    # cwd=workdir: SITL writes eeprom.bin, logs/ and terrain/ into its cwd.
    return subprocess.Popen(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop_process(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def connect(url: str, timeout_s: float) -> mavutil.mavfile:
    """Connect to SITL's MAVLink port, retrying until it listens."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            # source_system 255 = the conventional GCS id; ArduPilot's own
            # MAVLink system id is 1.
            mav = mavutil.mavlink_connection(url, source_system=255, autoreconnect=True)
            break
        except OSError:  # ConnectionRefused while SITL is still starting
            if time.monotonic() > deadline:
                raise TimeoutError(f"SITL not listening on {url}")
            time.sleep(0.5)
    if mav.wait_heartbeat(timeout=timeout_s) is None:
        raise TimeoutError("no HEARTBEAT from SITL")
    return mav
