"""Residual policy learning: PPO learns a correction ON TOP OF the PID baseline.

    a = clip( a_PID(obs) + scale * a_residual(obs), -1, 1 )

The residual network's output layer is initialised to zero, so training starts
EXACTLY at the PID's performance (not at an imperfect behaviour-cloned copy of
it: our BC clone reached 0.47 where the PID gets 0.80 on a 0.4 m/s straight
line). PPO then only has to learn *where the PID is wrong* - typically turns,
fast rovers and marker loss - which is a much smaller, better-conditioned
problem than learning landing from scratch, which fell into "dive" or "hover"
local optima here (Step 9 log). References: Silver et al. 2018 "Residual
Policy Learning"; Johannink et al. 2019 "Residual Reinforcement Learning for
Robot Control".

The same composition is used for training (ResidualVecEnv), evaluation
(residual_policy) and deployment (runtime: numpy PID + ONNX residual).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from lander.baselines.pid import PIDLander


def compose(a_pid: np.ndarray, a_res: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(a_pid + scale * np.clip(a_res, -1, 1), -1, 1)


class ResidualVecEnv(VecEnvWrapper):
    """The agent's action is the residual; the wrapped env receives PID + residual."""

    def __init__(self, venv, cfg: dict, scale: float) -> None:
        super().__init__(venv)
        self.pid = PIDLander(cfg["baseline"], cfg["task"])
        self.scale = scale
        self._obs: np.ndarray | None = None

    def reset(self):
        self._obs = self.venv.reset()
        return self._obs

    def step_async(self, actions):
        self.venv.step_async(compose(self.pid(self._obs), actions, self.scale))

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        self._obs = obs
        return obs, rew, done, infos


def zero_init_residual(model) -> None:
    """Make the initial deterministic residual exactly 0 (policy == PID)."""
    import torch

    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()


def residual_policy(cfg: dict, model_path, scale: float, device: str = "cpu"):
    from stable_baselines3 import PPO

    pid = PIDLander(cfg["baseline"], cfg["task"])
    model = PPO.load(str(model_path), device=device)

    def act(obs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        a_res, _ = model.predict(obs, deterministic=True)
        return compose(pid(obs), a_res, scale)

    return act
