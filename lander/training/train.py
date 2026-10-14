"""PPO training in Stage A (Step 9) with curriculum, DR, TensorBoard, checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from lander.envs import task as tk
from lander.envs.fast_env import FastVecEnv
from lander.envs.scenarios import Curriculum


class LandingStats(BaseCallback):
    """Logs outcome rates, curriculum stage and episode metrics; checkpoints."""

    def __init__(self, curriculum: Curriculum, run_dir: Path, every: int, ckpt_every: int):
        super().__init__()
        self.cur, self.dir, self.every, self.ckpt_every = curriculum, run_dir, every, ckpt_every
        self.buf: list[dict] = []
        self.rollouts = 0
        self.next_ckpt = ckpt_every
        self.stage_log: list[tuple[int, int]] = [(0, curriculum.stage)]

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.buf.append(info)
        if self.num_timesteps >= self.next_ckpt:
            self.model.save(self.dir / f"ckpt_{self.num_timesteps}")
            self.model.get_env().save(str(self.dir / f"vecnorm_{self.num_timesteps}.pkl"))
            self.next_ckpt += self.ckpt_every
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        if self.stage_log[-1][1] != self.cur.stage:
            self.stage_log.append((self.num_timesteps, self.cur.stage))
        if self.rollouts % self.every or not self.buf:
            return
        oc = np.array([i["outcome"] for i in self.buf])
        for code, name in tk.OUTCOME_NAMES.items():
            self.logger.record(f"outcome/{name}", float(np.mean(oc == code)))
        succ = [i for i in self.buf if i["outcome"] == tk.SUCCESS]
        if succ:
            self.logger.record("landing/xy_err_m", float(np.mean([i["final_xy_err"] for i in succ])))
            self.logger.record("landing/rel_vxy_mps", float(np.mean([i["final_rel_vxy"] for i in succ])))
            self.logger.record("landing/time_s", float(np.mean([i["time_s"] for i in succ])))
        self.logger.record("perception/visible_frac", float(np.mean([i["visible_frac"] for i in self.buf])))
        self.logger.record("curriculum/stage", self.cur.stage)
        self.buf = []


def train(cfg: dict, run_dir: Path, seed: int, curriculum: bool, dr: bool,
          total_steps: int | None = None) -> Path:
    p = cfg["ppo"]
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    cur = Curriculum(cfg["scenarios"], enabled=curriculum)
    venv = FastVecEnv(p["n_envs"], cfg, seed=seed, curriculum=cur, dr=dr)
    res = p.get("residual", {})
    if res.get("enabled", False):
        from lander.training.residual import ResidualVecEnv

        venv = ResidualVecEnv(venv, cfg, res["scale"])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=p["norm_reward"], gamma=cfg["task"]["gamma"])
    lr0 = p["learning_rate"]
    model = PPO(
        "MlpPolicy", venv, n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        learning_rate=lambda progress: lr0 * progress, gamma=cfg["task"]["gamma"],
        gae_lambda=p["gae_lambda"], clip_range=p["clip_range"], ent_coef=p["ent_coef"],
        vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"], target_kl=p["target_kl"],
        policy_kwargs=dict(net_arch=dict(pi=p["net_arch"], vf=p["net_arch"]),
                           activation_fn=getattr(torch.nn, {"tanh": "Tanh", "relu": "ReLU"}[p["activation"]]),
                           log_std_init=p["log_std_init"]),
        tensorboard_log=str(run_dir / "tb"), seed=seed, device=p["device"], verbose=0)
    if res.get("enabled", False):
        from lander.training.residual import zero_init_residual

        zero_init_residual(model)
    if p.get("warmstart", {}).get("enabled", False):
        from lander.training.warmstart import collect, pretrain

        w = p["warmstart"]
        data = collect(cfg, w["collect_steps"], w["collect_envs"], w["noise_std"], seed)
        stats = pretrain(model, data, w["epochs"], w["lr"], p["device"])
        (run_dir / "warmstart.json").write_text(json.dumps(stats, indent=2))
    steps = total_steps or p["total_timesteps"]
    cb = LandingStats(cur, run_dir, p["eval_every_rollouts"], p["checkpoint_every"])
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb, tb_log_name="ppo")
    model.save(run_dir / "final")
    venv.save(str(run_dir / "vecnorm_final.pkl"))
    meta = {"seed": seed, "curriculum": curriculum, "dr": dr, "steps": steps,
            "residual": res.get("enabled", False), "residual_scale": res.get("scale"),
            "wall_s": time.time() - t0, "final_stage": cur.stage, "stage_log": cb.stage_log}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir / "final.zip"
