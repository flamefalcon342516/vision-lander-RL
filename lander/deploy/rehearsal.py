"""SITL rehearsal of the REAL deployment runtime (Step 13).

The Gazebo env is used only as a stage manager (spawn, put the drone above the
rover, keep the rover driving). Control is done by `deploy/runtime.Runtime` -
the same object, ONNX policy, safety layer and motor-cut logic that would run
on the companion computer - over its OWN MAVLink connection on SITL's SERIAL1
(tcp:5762), like a companion computer on a TELEM port.

A rehearsal ends with a physical touchdown: motors are cut over the deck and
the drone drops onto the moving platform. We then check where it came to rest.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from lander.deploy.runtime import GzFrameSource, Runtime, RuntimeConfig
from lander.deploy.safety import SafetyConfig
from lander.envs.gazebo_env import GazeboLandingEnv
from lander.mav import MavLink


def rehearse(cfg: dict, onnx_path: Path, kind: str, speed: float, episodes: int, out_dir: Path,
             rtf: float = 5.0, seed: int = 7, log=print, residual_scale: float | None = None) -> list[dict]:
    results = []
    for ep in range(episodes):
        # a fresh world per episode: after RTL the drone is disarmed at home, and removing it
        # for a respawn failed ("not a direct child of the world") -> "Arm: Leaning"
        env = GazeboLandingEnv(cfg, rtf=rtf, seed=seed + ep)
        env.set_scenario(kind, speed)
        try:
            env.reset()
            stop = threading.Event()

            def drive_rover() -> None:  # keep the rover on its path in sim time
                t = env.world.sim_time
                while not stop.is_set():
                    t += cfg["task"]["control_dt_s"]
                    while env.world.sim_time < t and not stop.is_set():
                        time.sleep(0.0005)
                    env.rover.step(cfg["task"]["control_dt_s"])
                    env._rover_cmd()

            rover_thread = threading.Thread(target=drive_rover, daemon=True)
            rover_thread.start()
            mav = MavLink(cfg["deploy"]["sitl_companion_url"],
                          rates_hz={"LOCAL_POSITION_NED": 50, "ATTITUDE": 50, "HEARTBEAT": 2},
                          clock=lambda: env.world.sim_time)
            frames = GzFrameSource(cfg["camera"]["image_topic"], lambda: env.world.sim_time)
            rt = Runtime(cfg, onnx_path, frames, mav, SafetyConfig(**cfg["deploy"]["safety"]),
                         RuntimeConfig(**cfg["deploy"]["runtime"]), out_dir / f"rehearsal_{kind}_{speed}_{ep}.csv",
                         residual_scale=residual_scale)
            res = rt.run()
            # let the drone settle on the deck, then measure where it is
            t_end = env.world.sim_time + 3.0
            while env.world.sim_time < t_end:
                time.sleep(0.001)
            truth, _ = env._true_state()
            stop.set()
            rover_thread.join(timeout=2)
            mav.close()
            dxy = float(np.linalg.norm(truth.d[0, :2]))
            on_deck = bool(res["outcome"] == "touchdown_cut" and dxy < 0.5 and truth.h[0] < 0.35)
            res.update(kind=kind, speed=speed, rest_xy_m=dxy, rest_h_m=float(truth.h[0]), on_deck=on_deck)
            results.append(res)
            log(f"  rehearsal {kind}@{speed} ep{ep}: {res['outcome']} on_deck={on_deck} "
                f"rest_xy={dxy:.2f} m loop {res['loop_ms_mean']:.2f} ms (p99 {res['loop_ms_p99']:.2f}) "
                f"img_age {res['img_age_mean_s'] * 1e3:.0f} ms overruns {res['overruns']}")
        finally:
            env.close()
    return results
