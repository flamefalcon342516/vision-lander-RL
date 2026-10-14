"""Stage A env: Gymnasium API compliance, determinism, dynamics = sysid model."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from lander.config import load_all
from lander.envs.fast_env import FastLandingEnv, FastVecEnv
from lander.envs.task import OBS_DIM
from lander.sim.sysid import simulate_axis

CFG = load_all()


def test_gymnasium_check_env():
    check_env(FastLandingEnv(CFG, seed=0), skip_render_check=True)


def test_vec_env_shapes_and_autoreset():
    env = FastVecEnv(32, CFG, seed=0)
    obs = env.reset()
    assert obs.shape == (32, OBS_DIM) and obs.dtype == np.float32
    finished = 0
    for _ in range(500):
        env.step_async(np.zeros((32, 3)))
        obs, rew, done, infos = env.step_wait()
        assert np.isfinite(obs).all() and np.isfinite(rew).all()
        for d, info in zip(done, infos):
            if d:
                finished += 1
                assert "terminal_observation" in info and "episode" in info
    assert finished > 0   # hovering forever times out -> episodes do end


def test_same_seed_same_rollout():
    a, b = FastVecEnv(8, CFG, seed=3), FastVecEnv(8, CFG, seed=3)
    oa, ob = a.reset(), b.reset()
    act = np.random.default_rng(0).uniform(-1, 1, (50, 8, 3))
    for x in act:
        oa = a.core.step(x)[0]
        ob = b.core.step(x)[0]
    assert np.array_equal(oa, ob)


def test_fast_env_velocity_response_matches_sysid_model():
    """Stage A dynamics without DR/gusts must equal the identified model."""
    cfg = load_all()
    cfg["fast_env"]["disturbance"]["gust_std_mps"] = 0.0
    env = FastVecEnv(1, cfg, seed=0, dr=False)
    env.reset()
    d = cfg["fast_env"]["dynamics"]
    vmax = cfg["task"]["action"]["v_xy_max_mps"]
    steps = 30
    vx = []
    for _ in range(steps):
        for _ in range(env.core.n_sub):
            env.core._substep(np.array([[2.0, 0.0, 0.0]]))
            vx.append(env.core.v[0, 0])
    # simulate_axis pads its delay line with u[0]; start the input at 0 so the
    # reference sees the same step-from-rest as the env's zero-initialised history.
    ref = simulate_axis(np.r_[0.0, np.full(len(vx), 2.0)], cfg["fast_env"]["substep_s"],
                        d["tau_xy_s"], d["delay_xy_s"], d["a_max_xy_mps2"], d["jerk_xy_mps3"])[1:]
    assert np.allclose(vx, ref, atol=0.03), (vx[-5:], ref[-5:])
    assert vmax > 2.0
