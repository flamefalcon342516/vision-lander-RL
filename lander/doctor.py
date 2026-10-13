"""`python -m lander doctor`: verify every piece of the toolchain we rely on.

Each check returns (ok, detail). The point is reproducibility: on a new machine
(e.g. the DGX Spark) run this first instead of discovering a missing piece
halfway through a training run.
"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import yaml

Check = Callable[[], tuple[bool, str]]


def _python() -> tuple[bool, str]:
    ok = sys.version_info >= (3, 10)
    return ok, f"{platform.python_version()} on {platform.machine()}"


def _torch() -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        # Not fatal: Stage A PPO with small MLPs trains fine on CPU.
        return True, f"{torch.__version__} (CUDA NOT available - CPU only)"
    return True, f"{torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}"


def _versions(*mods: str) -> Check:
    def check() -> tuple[bool, str]:
        out = [f"{m}={importlib.import_module(m).__version__}" for m in mods]
        return True, ", ".join(out)

    return check


def _aruco() -> tuple[bool, str]:
    import cv2

    # ArucoDetector is the >=4.7 API; older tutorials use the removed
    # cv2.aruco.detectMarkers free function.
    cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    return True, f"OpenCV {cv2.__version__} with cv2.aruco.ArucoDetector"


def _ppo_smoke() -> tuple[bool, str]:
    """One tiny PPO rollout+update on CartPole: proves the SB3 loop runs end to end."""
    from stable_baselines3 import PPO

    t0 = time.perf_counter()
    # n_steps=256 so learn() completes exactly one rollout + one update.
    model = PPO("MlpPolicy", "CartPole-v1", n_steps=256, batch_size=64, n_epochs=1,
                device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=256)
    return True, f"1 rollout + update in {time.perf_counter() - t0:.1f} s (CPU)"


def _gz_transport() -> tuple[bool, str]:
    """Publish and receive on a loopback topic: proves bindings + protobuf work."""
    from gz.msgs10.stringmsg_pb2 import StringMsg
    from gz.transport13 import Node

    got: list[str] = []
    node = Node()
    topic = "/lander/doctor"
    node.subscribe(StringMsg, topic, lambda m: got.append(m.data))
    pub = node.advertise(topic, StringMsg)
    time.sleep(0.5)  # discovery is asynchronous; publishing earlier is dropped
    for _ in range(3):
        pub.publish(StringMsg(data="ping"))
        time.sleep(0.05)
    time.sleep(0.2)
    return bool(got), f"received {len(got)}/3 messages on {topic}"


def _gz_cli() -> tuple[bool, str]:
    if shutil.which("gz") is None:
        return False, "gz not on PATH"
    out = subprocess.run(["gz", "sim", "--versions"], capture_output=True, text=True, timeout=30)
    ver = out.stdout.strip()
    return ver.startswith("8."), f"gz sim {ver} (Harmonic = 8.x)"


def _file(path: str) -> Check:
    def check() -> tuple[bool, str]:
        p = Path(path).expanduser()
        return p.exists(), str(p)

    return check


def _disk(min_free_gb: float) -> Check:
    def check() -> tuple[bool, str]:
        free = shutil.disk_usage(Path.home()).free / 1e9
        # Warn-only: low disk degrades us later but doesn't block Step 0.
        return True, f"{free:.1f} GB free" + ("  <-- LOW" if free < min_free_gb else "")

    return check


def run_doctor(paths_yaml: str) -> int:
    paths = yaml.safe_load(Path(paths_yaml).read_text())
    checks: dict[str, Check] = {
        "python": _python,
        "torch": _torch,
        "rl libs": _versions("gymnasium", "stable_baselines3", "tensorboard"),
        "deploy libs": _versions("onnx", "onnxruntime", "pymavlink", "numpy"),
        "sb3 ppo smoke": _ppo_smoke,
        "opencv aruco": _aruco,
        "gz-transport": _gz_transport,
        "gz cli": _gz_cli,
        "arducopter": _file(paths["arducopter_bin"]),
        "ardupilot plugin": _file(paths["ardupilot_gazebo_plugin"]),
        "disk": _disk(paths["min_free_disk_gb"]),
    }
    failed = 0
    for name, check in checks.items():
        try:
            ok, detail = check()
        except Exception as exc:  # report every failure, don't stop at the first
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        failed += not ok
        print(f"[{'OK' if ok else 'FAIL':4}] {name:17} {detail}")
    print("all checks passed" if not failed else f"{failed} check(s) failed")
    return 1 if failed else 0
