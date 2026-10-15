"""Capture README media from the Gazebo GUI: stills (arena, rover, drone, camera feed) and
a landing GIF. Needs an X display; drives one episode with the released policy.

    source env.sh && .venv/bin/python scripts/capture_media.py [--kind circle --speed 0.5]
"""
import argparse
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "media"
HEADER_PX = 48          # Gazebo window top bar
PANEL_W = 520           # camera panel width (gazebo/gui/showcase.config)


def window_id() -> str:
    tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
    for line in tree.splitlines():
        if '"Gazebo Sim"' in line:
            return line.split()[0]
    raise RuntimeError("Gazebo window not found")


def grab(wid: str) -> Image.Image:
    png = subprocess.run(["import", "-window", wid, "png:-"], capture_output=True).stdout
    from io import BytesIO
    return Image.open(BytesIO(png)).convert("RGB")


def scene_crop(im: Image.Image) -> Image.Image:
    """3D view without the header and the camera panel."""
    w, h = im.size
    return im.crop((0, HEADER_PX, w - PANEL_W - 10, h - 75))   # drop the play/step buttons


def gz_service(name: str, req_type: str, req: str) -> None:
    subprocess.run(["gz", "service", "-s", name, "--reqtype", req_type, "--reptype", "gz.msgs.Boolean",
                    "--timeout", "2000", "--req", req], capture_output=True, text=True)


def look_at(eye, target) -> None:
    """Move the GUI camera to `eye` looking at `target` (ENU)."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))   # +pitch looks down
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    q = (cy * cp, -sy * sp, cy * sp, sy * cp)            # w, x, y, z for rpy = (0, pitch, yaw)
    gz_service("/gui/follow", "gz.msgs.StringMsg", 'data: ""')
    gz_service("/gui/move_to/pose", "gz.msgs.GUICamera",
               f"pose: {{position: {{x: {eye[0]}, y: {eye[1]}, z: {eye[2]}}}, "
               f"orientation: {{w: {q[0]}, x: {q[1]}, y: {q[2]}, z: {q[3]}}}}}")
    time.sleep(2.5)   # the GUI animates the move


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="circle")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--gif-fps", type=float, default=6.0)
    args = ap.parse_args()

    from lander.config import load_all
    from lander.envs.gazebo_env import GazeboLandingEnv
    from lander.eval.evaluate import load_policy
    from lander.sim.gz_client import follow_in_gui

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_all()
    pol = load_policy(cfg, ROOT / "results/runs/main_s0")
    env = GazeboLandingEnv(cfg, gui=True, rtf=1.0, seed=3)
    env.set_scenario(args.kind, args.speed)
    try:
        obs, _ = env.reset()
        time.sleep(3)
        wid = window_id()
        drone = env.truth.get("drone")[1]
        rover = env.truth.get("rover")[1]
        env.world.pause(True)
        time.sleep(1.5)
        shots = {
            "arena": ((rover[0] - 13, rover[1] - 9, 7), ((rover[0] + drone[0]) / 2, (rover[1] + drone[1]) / 2, 0.8)),
            "rover": ((rover[0] - 2.6, rover[1] - 2.0, 1.7), (rover[0], rover[1], 0.25)),
            "drone": ((drone[0] - 1.5, drone[1] - 1.2, drone[2] + 0.6), tuple(drone)),
        }
        for name, (eye, tgt) in shots.items():
            look_at(eye, tgt)
            im = grab(wid)
            scene_crop(im).save(OUT / f"{name}.png")
            if name == "drone":   # the raw downward-camera frame the perception pipeline gets
                m = env._img
                rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
                Image.fromarray(rgb).save(OUT / "camera_feed.png")
            print("saved", OUT / f"{name}.png")
        # landing GIF with the follow camera
        follow_in_gui(cfg["sitl"]["drone"]["name"], offset=(-3.5, -3.5, 2.6))
        time.sleep(3)
        env.world.pause(False)
        frames, stop = [], threading.Event()

        def rec() -> None:
            while not stop.is_set():
                t0 = time.monotonic()
                im = scene_crop(grab(wid))
                frames.append(im.resize((720, im.height * 720 // im.width)))
                time.sleep(max(0.0, 1 / args.gif_fps - (time.monotonic() - t0)))

        th = threading.Thread(target=rec, daemon=True)
        th.start()
        info = {}
        while True:
            obs, _, term, trunc, info = env.step(pol(obs[None])[0])
            if term or trunc:
                break
        time.sleep(2.0)
        stop.set()
        th.join()
        frames += [frames[-1]] * int(args.gif_fps * 1.5)   # hold the touchdown
        frames[0].save(OUT / "landing.gif", save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        print("saved", OUT / "landing.gif", len(frames), "frames; outcome", info.get("outcome"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
