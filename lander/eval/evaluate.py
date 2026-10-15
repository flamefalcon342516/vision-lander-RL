"""Evaluation (Step 12): metrics per (path type, speed) for any policy, in either stage.

A *policy* is any callable obs (N, OBS_DIM) -> action (N, 3). PPO, the PID
baseline and the ONNX-exported policy all fit this interface, so every number
in the results tables comes from the same evaluation code.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from lander.envs import task as tk
from lander.eval import report

Policy = Callable[[np.ndarray], np.ndarray]


def ppo_policy(model_path: str | Path, device: str = "cpu") -> Policy:
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device=device)

    def act(obs: np.ndarray) -> np.ndarray:
        a, _ = model.predict(np.atleast_2d(obs), deterministic=True)
        return a

    return act


def load_policy(cfg: dict, run_or_zip: Path) -> Policy:
    """A training run dir (reads meta.json: residual or plain PPO) or a .zip."""
    import json

    run = Path(run_or_zip)
    if run.is_dir():
        meta = json.loads((run / "meta.json").read_text())
        if meta.get("residual"):
            from lander.training.residual import residual_policy

            return residual_policy(cfg, run / "final.zip", meta["residual_scale"])
        return ppo_policy(run / "final.zip")
    return ppo_policy(run)


def pid_policy(cfg: dict) -> Policy:
    from lander.baselines.pid import PIDLander

    return PIDLander(cfg["baseline"], cfg["task"])


def eval_fast(policy: Policy, cfg: dict, episodes: int, seed: int, dr: bool = True,
              grid: list[tuple[str, float]] | None = None, n_env: int = 256) -> list[dict]:
    from lander.envs.fast_env import FastVecEnv

    grid = grid or eval_grid(cfg)
    records: list[dict] = []
    for i, (kind, speed) in enumerate(grid):
        env = FastVecEnv(min(n_env, episodes), cfg, seed=seed + 7919 * i, dr=dr)
        env.core.fixed = (kind, speed)
        obs = env.reset()
        cell: list[dict] = []
        while len(cell) < episodes:
            obs, _, _, infos = env.core.step(policy(obs))
            cell += [{**{k: v for k, v in inf.items() if k != "terminal_observation"},
                      "path": kind, "nominal_speed": speed, "stage": "fast"}
                     for inf in infos if "outcome" in inf]
        records += cell[:episodes]
    return records


def eval_gazebo(policy: Policy, cfg: dict, episodes: int, seed: int, rtf: float = 5.0,
                grid: list[tuple[str, float]] | None = None, log=print, gui: bool = False) -> list[dict]:
    from lander.envs.gazebo_env import GazeboLandingEnv

    grid = grid or eval_grid(cfg)
    env = GazeboLandingEnv(cfg, gui=gui, rtf=rtf, seed=seed)
    records: list[dict] = []
    try:
        n_total, i = len(grid) * episodes, 0
        for kind, speed in grid:
            env.set_scenario(kind, speed)
            for ep in range(episodes):
                i += 1
                log(report.banner(i, n_total, kind, speed))
                obs, _ = env.reset()
                while True:
                    obs, _, term, trunc, info = env.step(policy(obs[None])[0])
                    if term or trunc:
                        break
                rec = dict(info, path=kind, nominal_speed=speed, stage="gazebo")
                records.append(rec)
                log(report.episode_line(kind, speed, tk.OUTCOME_NAMES[rec["outcome"]],
                                        rec["final_xy_err"], rec["time_s"]))
    finally:
        env.close()
    return records


def eval_grid(cfg: dict) -> list[tuple[str, float]]:
    e = cfg["scenarios"]["eval"]
    grid = []
    for kind in e["path_types"]:
        speeds = [0.0] if kind == "static" else e["speeds_mps"]
        grid += [(kind, float(s)) for s in speeds]
    return grid


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate (good for small n)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(r["path"], r["nominal_speed"]) for r in records},
                  key=lambda k: (["static", "straight", "circle", "figure8", "random"].index(k[0]), k[1]))
    for path, speed in keys + [("ALL", -1.0)]:
        rs = records if path == "ALL" else [r for r in records if r["path"] == path and r["nominal_speed"] == speed]
        n = len(rs)
        oc = np.array([r["outcome"] for r in rs])
        succ = [r for r in rs if r["outcome"] == tk.SUCCESS]
        lost = sum(r.get("lost_events", 0) for r in rs)
        rec = sum(r.get("recoveries", 0) for r in rs)
        lo, hi = wilson(len(succ), n)
        rows.append({
            "path": path, "speed": speed, "n": n,
            "success": len(succ) / n if n else 0.0, "success_ci_lo": lo, "success_ci_hi": hi,
            "xy_err_m": float(np.mean([r["final_xy_err"] for r in succ])) if succ else float("nan"),
            "rel_vxy_mps": float(np.mean([r["final_rel_vxy"] for r in succ])) if succ else float("nan"),
            "vz_mps": float(np.mean([r["final_vz"] for r in succ])) if succ else float("nan"),
            "time_s": float(np.mean([r["time_s"] for r in succ])) if succ else float("nan"),
            "recovery_rate": rec / lost if lost else float("nan"),
            **{f"frac_{name}": float(np.mean(oc == code)) if n else 0.0
               for code, name in tk.OUTCOME_NAMES.items() if code != tk.SUCCESS},
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict], title: str) -> str:
    out = [f"**{title}**", "",
           "| path | speed (m/s) | n | success (95% CI) | xy err (m) | rel v (m/s) | time (s) | recovery |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = "all" if r["speed"] < 0 else ("-" if r["path"] == "static" else f"{r['speed']:.1f}")
        rec = "-" if np.isnan(r["recovery_rate"]) else f"{r['recovery_rate']:.0%}"
        f = lambda x, d=2: "-" if np.isnan(x) else f"{x:.{d}f}"  # noqa: E731
        out.append(f"| {r['path']} | {sp} | {r['n']} | {r['success']:.0%} "
                   f"({max(r['success_ci_lo'], 0.0):.0%}-{r['success_ci_hi']:.0%}) | {f(r['xy_err_m'], 3)} | "
                   f"{f(r['rel_vxy_mps'])} | {f(r['time_s'], 1)} | {rec} |")
    return "\n".join(out)
