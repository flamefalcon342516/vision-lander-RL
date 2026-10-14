"""Warm-start PPO from the classical controller (behaviour cloning + critic pretraining).

Why (measured, Step 9): PPO from scratch fell into one of two local optima
depending on the reward/exploration settings:
  * "dive"  - descend at full speed without tracking: works on static rovers,
               0-8% on rovers moving at only 0.1 m/s (runs reward_v4, relobs);
  * "hover" - track the marker precisely but never commit to a descent,
               timing out (sweep run D, low exploration noise).
Landing needs both at once, and random exploration rarely finds that
combination. Starting the actor from the PID baseline puts PPO in the right
basin; PPO then has to *improve* on it (curves, fast rovers, marker loss),
and the evaluation compares it against exactly that starting controller.

Coverage: the PID's actions are perturbed with noise during collection while
the LABEL stays the PID's clean action at the visited state, so the clone has
seen states slightly off the PID's own trajectory (DART-style noise injection).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lander.baselines.pid import PIDLander
from lander.envs.fast_env import FastVecEnv


def collect(cfg: dict, steps: int, n_env: int, noise_std: float, seed: int) -> dict:
    pid = PIDLander(cfg["baseline"], cfg["task"])
    env = FastVecEnv(n_env, cfg, seed=seed, dr=True)        # final stage mix, full DR
    rng = np.random.default_rng(seed)
    obs = env.reset()
    O, A, R, D = [], [], [], []
    for _ in range(steps):
        a = pid(obs)
        exe = np.clip(a + noise_std * rng.standard_normal(a.shape), -1, 1)
        nxt, r, done, _ = env.core.step(exe)
        O.append(obs)
        A.append(a)
        R.append(r)
        D.append(done)
        obs = nxt
    O, A, R, D = (np.stack(x) for x in (O, A, R, D))          # (T, N, ...)
    gamma = cfg["task"]["gamma"]
    G = np.zeros_like(R)
    run = np.zeros(n_env)
    for t in range(len(R) - 1, -1, -1):                      # discounted return-to-go
        run = R[t] + gamma * run * (1.0 - D[t])
        G[t] = run
    return {"obs": O.reshape(-1, O.shape[-1]), "act": A.reshape(-1, 3), "ret": G.reshape(-1)}


def pretrain(model, data: dict, epochs: int, lr: float, device: str) -> dict:
    """Fit the PPO actor mean to the PID actions and the critic to scaled returns."""
    pol = model.policy
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(data["act"], dtype=torch.float32, device=device)
    # VecNormalize divides rewards by the std of discounted returns; mirror that scale.
    ret_np = data["ret"]
    ret = torch.as_tensor(ret_np / (ret_np.std() + 1e-8), dtype=torch.float32, device=device)
    actor = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    critic = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
    opt = torch.optim.Adam([{"params": actor}, {"params": critic}], lr=lr)
    n = len(act)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for b in range(0, n, 4096):
            i = perm[b:b + 4096]
            lat_pi = pol.mlp_extractor.forward_actor(obs[i])
            lat_vf = pol.mlp_extractor.forward_critic(obs[i])
            mean = pol.action_net(lat_pi)
            v = pol.value_net(lat_vf).squeeze(-1)
            loss_bc = nn.functional.mse_loss(mean, act[i])
            loss_v = nn.functional.mse_loss(v, ret[i])
            opt.zero_grad()
            (loss_bc + 0.5 * loss_v).backward()
            opt.step()
    return {"bc_mse": float(loss_bc), "value_mse": float(loss_v), "samples": n}
