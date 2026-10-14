"""Reward and termination logic (envs/task.py) - the MDP definition."""
from pathlib import Path

import numpy as np
import yaml

from lander.envs import task as tk

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/task.yaml").read_text())


def truth(d, v_rel=(0, 0, 0), h=None, ground=None) -> tk.StepTruth:
    d = np.atleast_2d(np.asarray(d, float))
    h = d[:, 2] if h is None else np.atleast_1d(h)
    return tk.StepTruth(d, np.atleast_2d(np.asarray(v_rel, float)), h,
                        h + 0.3 if ground is None else np.atleast_1d(ground))


def out(t, elapsed=1.0):
    return tk.check_termination(t, np.array([elapsed]), CFG)[0]


def test_good_touchdown_is_success():
    assert out(truth([0.1, 0.0, 0.2], v_rel=[0.1, 0, -0.4])) == tk.SUCCESS


def test_touchdown_too_far_or_too_fast_is_miss():
    assert out(truth([0.4, 0.0, 0.2])) == tk.MISS                       # off centre
    assert out(truth([0.0, 0.0, 0.2], v_rel=[0.8, 0, 0])) == tk.MISS    # sliding
    assert out(truth([0.0, 0.0, 0.2], v_rel=[0, 0, -1.4])) == tk.MISS   # slamming down


def test_out_of_bounds_and_timeout():
    assert out(truth([20.0, 0.0, 5.0])) == tk.OUT_OF_BOUNDS
    assert out(truth([0.0, 0.0, 20.0])) == tk.OUT_OF_BOUNDS
    assert out(truth([1.0, 0.0, 3.0]), elapsed=CFG["max_episode_s"]) == tk.TIMEOUT
    assert out(truth([1.0, 0.0, 3.0])) == tk.RUNNING


def test_outcome_ordering_success_beats_miss_beats_never_landing():
    """Reward v1 lesson: if never landing is not worse than a miss, PPO hovers."""
    r = CFG["reward"]
    assert r["r_success"] > 0 > r["r_miss"] > r["r_timeout"]
    assert r["r_fail"] <= r["r_timeout"]


def _r(prev, cur, outcome, visible=True, a=(0, 0, 0), pa=(0, 0, 0), terms=None):
    return tk.reward(prev, cur, np.array([outcome]), np.array([visible]), np.atleast_2d(a),
                     np.atleast_2d(pa), CFG, terms)[0]


def test_shaping_rewards_progress_and_telescopes():
    far, near = truth([2.0, 0, 5.0]), truth([1.0, 0, 4.0])
    assert _r(far, near, tk.RUNNING) > _r(near, far, tk.RUNNING)
    # potential-based: sum over a path depends only on endpoints (gamma=1 check)
    cfg1 = dict(CFG, gamma=1.0)
    states = [truth([2.0 - 0.1 * i, 0, 5.0 - 0.3 * i]) for i in range(10)]
    total = sum(tk.reward(a, b, np.array([0]), np.array([True]), np.zeros((1, 3)), np.zeros((1, 3)),
                          cfg1, t := {}) [0] - t["lost"][0] - t["smooth"][0] - t["time"][0]
                for a, b in zip(states, states[1:]))
    direct = tk.potential(states[-1], CFG)[0] - tk.potential(states[0], CFG)[0]
    assert np.isclose(total, direct)


def test_penalties_for_lost_marker_and_jerky_actions():
    s = truth([1.0, 0, 3.0])
    assert _r(s, s, tk.RUNNING, visible=False) < _r(s, s, tk.RUNNING, visible=True)
    assert _r(s, s, tk.RUNNING, a=(1, -1, 1), pa=(-1, 1, -1)) < _r(s, s, tk.RUNNING)


def test_success_bonus_prefers_precise_landings():
    prev = truth([0.1, 0, 0.5])
    precise = _r(prev, truth([0.0, 0, 0.2]), tk.SUCCESS)
    sloppy = _r(prev, truth([0.2, 0, 0.2]), tk.SUCCESS)
    assert precise > sloppy > 0


def test_action_scaling_and_clipping():
    v = tk.action_to_velocity(np.array([[1.0, -2.0, 0.5]]), CFG)[0]
    assert np.allclose(v, [CFG["action"]["v_xy_max_mps"], -CFG["action"]["v_xy_max_mps"],
                           0.5 * CFG["action"]["v_z_max_mps"]])


def test_miss_penalty_is_graded_but_never_worse_than_timeout():
    """Reward v3 lesson: a flat miss penalty gave no gradient from far to near misses."""
    prev = truth([0.5, 0, 0.5])
    near = _r(prev, truth([0.3, 0, 0.2]), tk.MISS)
    far = _r(prev, truth([2.5, 0, 0.2]), tk.MISS)
    terms_far, terms_worst = {}, {}
    _r(prev, truth([2.5, 0, 0.2]), tk.MISS, terms=terms_far)
    _r(prev, truth([9.0, 0, 0.2], v_rel=[5, 0, 0]), tk.MISS, terms=terms_worst)
    assert terms_far["terminal"][0] < -5.0                     # graded below the base penalty
    assert terms_worst["terminal"][0] > CFG["reward"]["r_timeout"]   # trying beats never landing
    assert near > far or terms_far["terminal"][0] < 0


def test_descending_while_misaligned_loses_potential():
    """Reward v4 lesson: descent must not pay when far off-centre (alignment cone)."""
    far_hi, far_lo = truth([1.0, 0, 3.0]), truth([1.0, 0, 0.5])
    near_hi, near_lo = truth([0.1, 0, 3.0]), truth([0.1, 0, 0.5])
    assert tk.potential(far_lo, CFG)[0] < tk.potential(far_hi, CFG)[0]
    assert tk.potential(near_lo, CFG)[0] > tk.potential(near_hi, CFG)[0]
