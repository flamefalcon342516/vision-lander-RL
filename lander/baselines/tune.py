"""Random search over PID baseline gains in Stage A (fair-comparison tuning).

Objective: success rate on the full curriculum stage (all path types, speeds
0-3 m/s) with domain randomisation ON, i.e. the same conditions the final PPO
policy is evaluated on.
"""
from __future__ import annotations

import numpy as np

from lander.baselines.pid import PIDLander
from lander.envs.fast_env import FastVecEnv
from lander.envs.task import SUCCESS

SPACE = {  # (low, high) per gain, sampled uniformly
    "kp_xy": (0.3, 2.5), "kd_xy": (0.0, 1.0), "feedforward": (0.6, 1.2),
    "align_radius_m": (0.1, 0.6), "align_per_m": (0.0, 0.3),
    "descent_gain": (0.2, 1.0), "descent_min_mps": (0.15, 0.6), "descent_max_mps": (0.5, 1.5),
    "lost_hold_s": (0.2, 1.0), "lost_climb_s": (0.8, 3.0),
}


def evaluate(gains: dict, cfg: dict, episodes: int, seed: int, n_env: int = 512) -> float:
    pid = PIDLander(gains, cfg["task"])
    env = FastVecEnv(n_env, cfg, seed=seed, dr=True)
    obs = env.reset()
    outcomes: list[int] = []
    while len(outcomes) < episodes:
        obs, _, _, infos = env.core.step(pid(obs))
        outcomes += [i["outcome"] for i in infos if "outcome" in i]
    return float(np.mean(np.array(outcomes[:episodes]) == SUCCESS))


def tune(cfg: dict, trials: int, episodes: int, seed: int = 0) -> tuple[dict, float, list]:
    rng = np.random.default_rng(seed)
    base = dict(cfg["baseline"])
    best, best_score = base, evaluate(base, cfg, episodes, seed)
    history = [(dict(base), best_score)]
    for _ in range(trials):
        g = dict(base)
        for k, (lo, hi) in SPACE.items():
            g[k] = float(rng.uniform(lo, hi))
        s = evaluate(g, cfg, episodes, seed)
        history.append((g, s))
        if s > best_score:
            best, best_score = g, s
    # re-score the winner on fresh seeds: random search over-fits its own noise
    confirm = evaluate(best, cfg, episodes * 4, seed + 1000)
    return best, confirm, history
