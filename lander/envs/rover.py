"""Rover motion: one vectorised unicycle model shared by Stage A and Stage B.

The rover is a unicycle: state (x, y, heading) in a world ENU-like plane, inputs
(v, omega) = forward speed and turn rate. Path *types* only differ in how
(v, omega) evolve over time. In Stage A we integrate this directly; in Stage B
the same (v, omega) are sent to Gazebo's VelocityControl as the rover's body
twist, so both stages see the same rover behaviour by construction.

Path types (ids are stable; they index arrays):
  0 static   : v = 0
  1 straight : constant v, omega = 0
  2 circle   : constant v, omega = v / R
  3 figure8  : constant v, omega = +-v/R, sign flips after each full circle
  4 random   : v and omega follow mean-reverting random walks (seeded)
"""
from __future__ import annotations

import numpy as np

PATH_TYPES = ("static", "straight", "circle", "figure8", "random")


class RoverFleet:
    """N independent rovers, advanced together with `step(dt)`."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        self.n = n
        self.rng = rng
        self.pos = np.zeros((n, 2))
        self.heading = np.zeros(n)
        self.v = np.zeros(n)
        self.omega = np.zeros(n)
        self.kind = np.zeros(n, dtype=np.int64)
        self.speed = np.zeros(n)       # nominal speed for this episode
        self.radius = np.ones(n)
        self.t = np.zeros(n)           # time since reset (for figure-8 phase)
        # random-walk parameters (only used by kind 4)
        self.omega_max = np.zeros(n)

    def reset(self, idx: np.ndarray, kind: np.ndarray, speed: np.ndarray,
              radius: np.ndarray, omega_max: np.ndarray) -> None:
        """Re-initialise the rovers in `idx` at the origin with a random heading."""
        k = len(idx)
        self.pos[idx] = 0.0
        self.heading[idx] = self.rng.uniform(-np.pi, np.pi, k)
        self.kind[idx] = kind
        self.speed[idx] = speed
        self.radius[idx] = radius
        self.omega_max[idx] = omega_max
        self.t[idx] = 0.0
        self.v[idx] = np.where(kind == 0, 0.0, speed)
        self.omega[idx] = 0.0
        self._update_inputs(idx, dt=0.0)

    def _update_inputs(self, idx: np.ndarray, dt: float) -> None:
        kind, v0, r = self.kind[idx], self.speed[idx], self.radius[idx]
        omega = np.zeros(len(idx))
        circ = kind == 2
        omega[circ] = v0[circ] / r[circ]
        f8 = kind == 3
        if f8.any():
            period = 2 * np.pi * r[f8] / np.maximum(v0[f8], 1e-6)
            sign = np.where((self.t[idx][f8] // period) % 2 == 0, 1.0, -1.0)
            omega[f8] = sign * v0[f8] / r[f8]
        rnd = kind == 4
        v = np.where(kind == 0, 0.0, v0)
        if rnd.any():
            # Ornstein-Uhlenbeck turn rate: mean-reverting to 0, time constant
            # 2 s, stationary std ~ omega_max / 2. Speed wanders +-30% likewise.
            tau = 2.0
            wmax = self.omega_max[idx][rnd]
            w = self.omega[idx][rnd]
            w = w - w / tau * dt + wmax * np.sqrt(dt / tau) * self.rng.standard_normal(rnd.sum())
            omega[rnd] = np.clip(w, -wmax, wmax)
            vr = self.v[idx][rnd]
            vr = vr - (vr - v0[rnd]) / tau * dt + 0.3 * v0[rnd] * np.sqrt(dt / tau) \
                * self.rng.standard_normal(rnd.sum())
            v[rnd] = np.clip(vr, 0.5 * v0[rnd], 1.3 * v0[rnd])
        self.v[idx] = v
        self.omega[idx] = omega

    def step(self, dt: float) -> None:
        """Exact unicycle integration over dt with inputs held constant."""
        idx = np.arange(self.n)
        self._update_inputs(idx, dt)
        v, w, th = self.v, self.omega, self.heading
        small = np.abs(w) < 1e-6
        # Straight-line limit vs. exact arc (avoids division by ~0).
        dx = np.where(small, v * dt * np.cos(th),
                      v / np.where(small, 1, w) * (np.sin(th + w * dt) - np.sin(th)))
        dy = np.where(small, v * dt * np.sin(th),
                      -v / np.where(small, 1, w) * (np.cos(th + w * dt) - np.cos(th)))
        self.pos[:, 0] += dx
        self.pos[:, 1] += dy
        self.heading = (th + w * dt + np.pi) % (2 * np.pi) - np.pi
        self.t += dt

    def velocity(self) -> np.ndarray:
        """World-frame velocity (N, 2)."""
        return np.stack([self.v * np.cos(self.heading), self.v * np.sin(self.heading)], axis=1)
