"""Step 7 toy: PPO on CartPole-v1 with SB3, to see the RL loop before our env.

Prints mean episode return after each chunk of training. CartPole's maximum
return is 500 (balanced for 500 steps).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main(chunks: int = 8, steps_per_chunk: int = 10_000, seed: int = 0) -> list[tuple[int, float]]:
    model = PPO("MlpPolicy", "CartPole-v1", seed=seed, device="cpu", verbose=0)
    curve = [(0, float(np.mean(evaluate_policy(model, model.get_env(), 10, return_episode_rewards=True)[0])))]
    for i in range(1, chunks + 1):
        model.learn(steps_per_chunk, reset_num_timesteps=False)
        r, _ = evaluate_policy(model, model.get_env(), 10)
        curve.append((i * steps_per_chunk, float(r)))
        print(f"{i * steps_per_chunk:6d} steps: mean return {r:6.1f}", flush=True)
    return curve


if __name__ == "__main__":
    main()
