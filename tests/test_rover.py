"""Rover path generator: kinematics must be exact, paths must be what they claim."""
import numpy as np

from lander.envs.rover import RoverFleet


def _fleet(kind: int, speed: float = 1.0, radius: float = 4.0, n: int = 1) -> RoverFleet:
    f = RoverFleet(n, np.random.default_rng(0))
    idx = np.arange(n)
    f.reset(idx, np.full(n, kind), np.full(n, speed), np.full(n, radius), np.full(n, 0.5))
    return f


def test_static_rover_does_not_move() -> None:
    f = _fleet(0)
    for _ in range(100):
        f.step(0.1)
    assert np.allclose(f.pos, 0)


def test_straight_distance_is_speed_times_time() -> None:
    f = _fleet(1, speed=1.5)
    for _ in range(100):
        f.step(0.1)
    assert np.isclose(np.linalg.norm(f.pos[0]), 15.0)


def test_circle_returns_to_start_after_one_period() -> None:
    f = _fleet(2, speed=2.0, radius=5.0)
    period = 2 * np.pi * 5.0 / 2.0
    n = 1000
    for _ in range(n):
        f.step(period / n)
    assert np.linalg.norm(f.pos[0]) < 1e-9          # exact arc integration
    assert np.max(np.abs(f.velocity())) <= 2.0 + 1e-9


def test_figure8_returns_to_start_after_two_circles() -> None:
    f = _fleet(3, speed=1.0, radius=3.0)
    period = 2 * np.pi * 3.0
    n = 2000
    for _ in range(n):
        f.step(2 * period / n)
    assert np.linalg.norm(f.pos[0]) < 0.05


def test_random_paths_are_bounded_and_seeded() -> None:
    a, b = _fleet(4, n=8), _fleet(4, n=8)
    for _ in range(300):
        a.step(0.1)
        b.step(0.1)
    assert np.allclose(a.pos, b.pos)                # same seed -> same paths
    assert np.all(np.abs(a.omega) <= 0.5 + 1e-12)
    assert np.all((a.v >= 0.5) & (a.v <= 1.3))
