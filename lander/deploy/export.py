"""Export the PPO actor to ONNX and time inference (Step 13).

Only the deterministic action is exported: action = clip(mu(obs), -1, 1).
The value network and the Gaussian's log-std are training-only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from lander.envs.task import OBS_DIM


class _Actor(torch.nn.Module):
    def __init__(self, policy) -> None:
        super().__init__()
        self.features = policy.mlp_extractor.policy_net
        self.mu = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.mu(self.features(obs)), -1.0, 1.0)


def export_onnx(model_zip: Path, out: Path) -> Path:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_zip), device="cpu")
    actor = _Actor(model.policy).eval()
    dummy = torch.zeros(1, OBS_DIM)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(actor, dummy, str(out), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    # parity check against SB3's own deterministic prediction
    sess = onnx_session(out, threads=1)
    obs = np.random.default_rng(0).uniform(-1, 1, (256, OBS_DIM)).astype(np.float32)
    a_onnx = sess.run(None, {"obs": obs})[0]
    a_sb3, _ = model.predict(obs, deterministic=True)
    err = float(np.max(np.abs(a_onnx - a_sb3)))
    if err > 1e-4:
        raise RuntimeError(f"ONNX/SB3 mismatch: max |diff| = {err}")
    return out


def onnx_session(path: Path, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def time_inference(path: Path, threads: int, n: int = 2000) -> dict:
    sess = onnx_session(path, threads)
    obs = np.zeros((1, OBS_DIM), np.float32)
    for _ in range(50):
        sess.run(None, {"obs": obs})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {"obs": obs})
        ts.append(time.perf_counter() - t)
    ts_ms = np.array(ts) * 1e3
    return {"threads": threads, "mean_ms": float(ts_ms.mean()), "p99_ms": float(np.percentile(ts_ms, 99)),
            "max_ms": float(ts_ms.max())}
