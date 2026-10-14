"""Step 15 (stretch): end-to-end pixel policy by teacher-student distillation.

Teacher: the trained PPO policy, which acts on the pose-based observation
(ArUco -> PnP -> KF). Student: a CNN that sees only two consecutive low-res
grayscale camera frames + its own velocity + previous action. No detector, no
Kalman filter: it must learn localisation and motion estimation from pixels.

Training is DAgger (Ross et al. 2011): roll out a mixture of teacher and
student (the student's share grows each iteration), label every visited state
with the TEACHER's action, aggregate, retrain the student by regression.
Plain behaviour cloning trains only on the teacher's state distribution; the
student's own mistakes then lead to states it has never seen (compounding
error). DAgger fixes that by labelling the student's states too.

Images are rendered from the fast env's own geometry (same tilt, latency and
FOV model), at 96x72 (the 640x480 camera scaled by 0.15): marker quiet zone,
black square with an inner white block, rover body, ground, sensor noise.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from lander.envs import task as tk
from lander.envs.fast_env import HIST, FastVecEnv
from lander.perception import frames as fr

W, H = 96, 72
SCALE = W / 640.0


class Renderer:
    """Draws what the down camera would see, for every env, at low resolution."""

    def __init__(self, env: FastVecEnv, rng: np.random.Generator) -> None:
        self.core = env.core
        self.rng = rng
        k = self.core.cam.k.copy()
        k[:2] *= SCALE
        self.k = k
        half = self.core.half
        q = half + self.core.cfg["marker"]["border_m"]
        sq = lambda s, cx=0.0, cy=0.0: np.array([[cx + s, cy + s], [cx + s, cy - s], [cx - s, cy - s], [cx - s, cy + s]])  # noqa: E731
        # polygons in the deck frame (x along rover heading), drawn in order
        self.polys = [(sq(0.6, 0, 0) * np.array([1.0, 0.83]), 70),   # rover body 1.2 x 1.0 m
                      (sq(q), 235),                                   # white quiet zone
                      (sq(half), 20),                                 # black marker square
                      (sq(half * 0.35, half * 0.3, -half * 0.25), 235)]  # inner white block

    def render(self) -> np.ndarray:
        c = self.core
        n = c.n
        age = self.rng.uniform(*c.fe["detection"]["latency_s"], n)
        back = np.minimum(np.rint(age / c.sub_dt).astype(int), HIST - 1)
        slot = (c.ptr - back) % HIST
        rows = np.arange(n)
        rel = c.p_hist[rows, slot]
        roll, pitch = c.att_hist[rows, slot, 0], c.att_hist[rows, slot, 1]
        th = c.rover.heading
        cth, sth = np.cos(th), np.sin(th)
        imgs = np.full((n, H, W), 110, np.uint8)
        imgs += self.rng.integers(0, 12, (n, H, W), dtype=np.uint8)   # ground texture
        for poly, shade in self.polys:
            px = rel[:, None, 0] + poly[None, :, 0] * cth[:, None] - poly[None, :, 1] * sth[:, None]
            py = rel[:, None, 1] + poly[None, :, 0] * sth[:, None] + poly[None, :, 1] * cth[:, None]
            pz = np.repeat(rel[:, None, 2], 4, axis=1)
            body = fr.level_to_body(np.stack([px, py, pz], -1), roll[:, None], pitch[:, None])
            cam = (body - c.mount) @ fr.R_BODY_CAM
            z = np.maximum(cam[..., 2], 1e-3)
            u = self.k[0, 0] * cam[..., 0] / z + self.k[0, 2]
            v = self.k[1, 1] * cam[..., 1] / z + self.k[1, 2]
            pts = np.stack([u, v], -1)
            ok = (cam[..., 2] > 0.05).all(axis=1)
            for i in np.flatnonzero(ok):
                cv2.fillConvexPoly(imgs[i], np.round(pts[i] * 16).astype(np.int32), int(shade),
                                   lineType=cv2.LINE_AA, shift=4)
        noise = self.rng.normal(0, 3, imgs.shape)
        return np.clip(imgs + noise, 0, 255).astype(np.uint8)


class Student(nn.Module):
    """2 stacked frames + [drone vel (3), prev action (3)] -> action in [-1,1]^3."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        n_conv = 64 * 5 * 6
        self.head = nn.Sequential(nn.Linear(n_conv + 6, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Tanh())

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        x = self.conv(img.float() / 255.0 - 0.5)
        return self.head(torch.cat([x, vec], dim=1))


def _vec(obs: np.ndarray) -> np.ndarray:
    """Non-visual part of the observation the student is allowed to see."""
    return np.concatenate([obs[:, 5:8], obs[:, 10:13]], axis=1).astype(np.float32)


def dagger(cfg: dict, teacher, out_dir: Path, iters: int = 5, steps_per_iter: int = 600,
           n_env: int = 64, epochs: int = 4, seed: int = 0, device: str = "cuda", log=print) -> Student:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = FastVecEnv(n_env, cfg, seed=seed)       # DR on, final curriculum stage (full mix)
    rend = Renderer(env, rng)
    student = Student().to(device)
    opt = torch.optim.Adam(student.parameters(), lr=3e-4)
    imgs: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for it in range(iters):
        beta = 1.0 if it == 0 else 0.5 ** it       # teacher's share of the control
        obs = env.reset()
        prev = rend.render()
        t0 = time.time()
        for _ in range(steps_per_iter):
            cur = rend.render()
            stack = np.stack([prev, cur], axis=1)
            a_teacher = teacher(obs)
            imgs.append(stack)
            vecs.append(_vec(obs))
            labels.append(a_teacher.astype(np.float32))
            if beta < 1.0:
                with torch.no_grad():
                    a_student = student(torch.from_numpy(stack).to(device),
                                        torch.from_numpy(_vec(obs)).to(device)).cpu().numpy()
                use_t = rng.uniform(size=n_env) < beta
                act = np.where(use_t[:, None], a_teacher, a_student)
            else:
                act = a_teacher
            obs, _, done, _ = env.core.step(act)
            prev = np.where(done[:, None, None], rend.render(), cur)  # new episode: fresh "previous"
        X = torch.from_numpy(np.concatenate(imgs))
        V = torch.from_numpy(np.concatenate(vecs))
        Y = torch.from_numpy(np.concatenate(labels))
        n = len(Y)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for b in range(0, n, 512):
                i = perm[b:b + 512]
                loss = nn.functional.mse_loss(student(X[i].to(device), V[i].to(device)), Y[i].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        log(f"DAgger iter {it}: beta={beta:.2f} dataset {n} samples, last loss {loss.item():.4f}, "
            f"rollout {time.time() - t0:.0f} s")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out_dir / "student.pt")
    return student


def evaluate_student(cfg: dict, student: Student, grid: list[tuple[str, float]], episodes: int,
                     seed: int, device: str = "cuda") -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i, (kind, speed) in enumerate(grid):
        env = FastVecEnv(min(128, episodes), cfg, seed=seed + 31 * i, dr=True)
        env.core.fixed = (kind, speed)
        rend = Renderer(env, rng)
        obs = env.reset()
        prev = rend.render()
        cell: list[dict] = []
        while len(cell) < episodes:
            cur = rend.render()
            with torch.no_grad():
                a = student(torch.from_numpy(np.stack([prev, cur], 1)).to(device),
                            torch.from_numpy(_vec(obs)).to(device)).cpu().numpy()
            obs, _, done, infos = env.core.step(a)
            prev = np.where(done[:, None, None], rend.render(), cur)
            cell += [{**{k: v for k, v in inf.items() if k != "terminal_observation"},
                      "path": kind, "nominal_speed": speed} for inf in infos if "outcome" in inf]
        records += cell[:episodes]
    return records


def sample_frames(cfg: dict, out: Path, n: int = 6, seed: int = 3) -> Path:
    """Save a strip of rendered student inputs (for the README)."""
    env = FastVecEnv(n, cfg, seed=seed, dr=False)
    env.reset()
    rend = Renderer(env, np.random.default_rng(seed))
    for _ in range(5):
        env.core.step(np.zeros((n, 3)))
    imgs = rend.render()
    strip = np.concatenate([cv2.resize(im, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST) for im in imgs], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip)
    return out


__all__ = ["Renderer", "Student", "dagger", "evaluate_student", "sample_frames", "tk"]
