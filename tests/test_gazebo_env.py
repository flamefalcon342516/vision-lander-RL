"""Stage B env integration test (slow: launches Gazebo + SITL)."""
import shutil

import numpy as np
import pytest

from lander.envs.task import OBS_DIM

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(shutil.which("gz") is None, reason="gz not installed")]


def test_gazebo_env_spaces_reset_and_step():
    from lander.config import load_all
    from lander.envs.gazebo_env import GazeboLandingEnv

    env = GazeboLandingEnv(load_all(), rtf=5.0, seed=0)
    env.set_scenario("static", 0.0)
    try:
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
        assert info["reset_wall_s"] < 30
        for _ in range(10):
            obs, r, term, trunc, info = env.step(np.zeros(3, np.float32))
            assert np.isfinite(obs).all() and np.isfinite(r)
        assert env.ep_info["visible_steps"] > 0     # the marker is seen from the start pose
    finally:
        env.close()
