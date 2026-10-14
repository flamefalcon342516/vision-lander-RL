"""Sample rover scenarios from a curriculum stage (shared by both stages)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    kind: np.ndarray       # (n,) path type ids
    speed: np.ndarray      # (n,) m/s
    radius: np.ndarray     # (n,) m
    omega_max: np.ndarray  # (n,) rad/s


def sample(stage: dict, cfg: dict, rng: np.random.Generator, n: int) -> Scenario:
    probs = np.asarray(stage["probs"], dtype=float)
    kind = rng.choice(len(probs), size=n, p=probs / probs.sum())
    lo, hi = stage["speed_mps"]
    speed = np.where(kind == 0, 0.0, rng.uniform(lo, hi, n))
    radius = rng.uniform(*cfg["radius_m"], n)
    omega_max = rng.uniform(*cfg["random_omega_max_radps"], n)
    return Scenario(kind, speed, radius, omega_max)


def fixed(kind_name: str, speed: float, cfg: dict, rng: np.random.Generator, n: int) -> Scenario:
    """Evaluation: one path type at one nominal speed (radius etc. still random)."""
    kind = np.full(n, cfg["path_types"].index(kind_name))
    sp = np.full(n, 0.0 if kind_name == "static" else speed)
    return Scenario(kind, sp, rng.uniform(*cfg["radius_m"], n),
                    rng.uniform(*cfg["random_omega_max_radps"], n))


class Curriculum:
    """Success-gated stage scheduler."""

    def __init__(self, cfg: dict, start_stage: int = 0, enabled: bool = True) -> None:
        self.stages = cfg["curriculum"]["stages"]
        self.window = cfg["curriculum"]["window"]
        self.promote_at = cfg["curriculum"]["promote_at"]
        # Disabled curriculum = train on the final stage from the start (ablation).
        self.stage = start_stage if enabled else len(self.stages) - 1
        self.enabled = enabled
        self._results: list[bool] = []

    @property
    def current(self) -> dict:
        return self.stages[self.stage]

    def record(self, successes: np.ndarray) -> bool:
        """Add finished-episode results; return True if the stage advanced."""
        if not self.enabled or self.stage == len(self.stages) - 1:
            return False
        self._results.extend(bool(s) for s in successes)
        self._results = self._results[-self.window:]
        if len(self._results) >= self.window and np.mean(self._results) >= self.promote_at:
            self.stage += 1
            self._results = []
            return True
        return False
