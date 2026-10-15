"""Stage A: a fast, vectorised NumPy landing simulator.

Thousands of envs step together. Everything that matters for transfer is
either *measured* (sysid dynamics, detection range/noise/latency in Gazebo) or
*randomised* around the measurement (domain randomisation), and everything
the agent touches is the SAME code as Stage B:
  observation  -> envs/task.build_obs     action scaling -> envs/task
  reward/termination -> envs/task         Kalman filter  -> perception/kalman
  rover motion -> envs/rover              frames         -> perception/frames

World: yaw fixed at 0, so the level frame equals NED (x north, y east, z down).
The deck is at z = 0; the drone's height above it is h = -z.

Per substep (0.02 s):
  u_d   = command delayed by `delay`                 (transport delay)
  a_des = clip((u_d - v) / tau, +-a_max)             (sysid model, |a_xy| clipped as a vector)
  a    += clip(a_des - a, +-jerk dt)                  (jerk limit: tilt ramps, never steps)
  v    += a dt;  v_true = v + gust                    (OU wind gust)
  tilt  = gain * atan(a / g)                          (camera tilts with the body)
Per control step (0.1 s): a camera measurement of the marker as it was `age`
seconds ago (latency), seen through the tilted camera, FOV + pixel-size-based
detection probability, noise, depth bias; de-rotated with the (noisy) attitude
AT THE IMAGE TIME, exactly as gazebo_env does; then KF predict/update.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from lander.envs import scenarios as sc
from lander.envs import task as tk
from lander.envs.rover import RoverFleet
from lander.perception import frames as fr
from lander.perception.detector import CameraModel
from lander.perception.kalman import MarkerKF

G = 9.81
HIST = 16          # substeps of latency history (0.32 s)
CMD_HIST = 32      # substeps of command history (0.64 s)


class FastLandingCore:
    """Vectorised simulator; `step` auto-resets finished envs (VecEnv semantics)."""

    def __init__(self, n: int, cfg: dict, seed: int = 0, curriculum: sc.Curriculum | None = None,
                 dr: bool | None = None) -> None:
        self.n, self.cfg = n, cfg
        self.task, self.fe = cfg["task"], cfg["fast_env"]
        self.rng = np.random.default_rng(seed)
        self.curriculum = curriculum
        self.dr = self.fe["dr"]["enabled"] if dr is None else dr
        self.fixed: tuple[str, float] | None = None
        self.cam = CameraModel.from_config(cfg["camera"])
        self.mount = np.array(cfg["camera"]["mount_offset_body_frd_m"])
        self.half = cfg["marker"]["marker_size_m"] / 2
        nest = cfg["marker"].get("nested", {})
        self.nested = bool(nest.get("enabled"))
        if self.nested:   # inner marker: half size and centre offset on the deck (see marker.yaml)
            cell = cfg["marker"]["marker_size_m"] / 6
            r, c = nest["cell"]
            self.inner_half = cell * nest["inner_fraction"] / 2
            self.inner_off = np.array([(c + 0.5 - 3) * cell, (3 - (r + 0.5)) * cell])
            self.size_ratio = self.half / self.inner_half
        self.sub_dt = self.fe["substep_s"]
        self.n_sub = int(round(self.task["control_dt_s"] / self.sub_dt))
        self.rover = RoverFleet(n, self.rng)
        self.kf = MarkerKF(n, cfg["perception"]["kf"])
        z3 = lambda: np.zeros((n, 3))  # noqa: E731
        self.pos, self.v, self.gust, self.acc = z3(), z3(), z3(), z3()
        self.cmd_hist = np.zeros((n, CMD_HIST, 3))
        self.p_hist = np.zeros((n, HIST, 3))
        self.att_hist = np.zeros((n, HIST, 2))
        self.ptr = 0
        self.prev_action = np.zeros((n, 3))
        self.elapsed = np.zeros(n)
        self.ep_ret = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=int)
        self.vis_steps = np.zeros(n, dtype=int)
        self.lost_events = np.zeros(n, dtype=int)
        self.recoveries = np.zeros(n, dtype=int)
        self._lost_flag = np.zeros(n, dtype=bool)
        self.kind = np.zeros(n, dtype=int)
        self._params = {}
        self.reset_all()

    # ------------------------------------------------------------ episode setup
    def _dr_level(self) -> float:
        """0 = nominal (measured) values, 1 = full DR ranges. The curriculum's
        current stage sets it during training; evaluation uses 1 (dr=True) or 0."""
        if not self.dr:
            return 0.0
        if self.curriculum is not None and self.fixed is None:
            return float(self.curriculum.current.get("dr_level", 1.0))
        return 1.0

    def _sample_params(self, idx: np.ndarray) -> None:
        d, dr, det = self.fe["dynamics"], self.fe["dr"], self.fe["detection"]
        k = len(idx)
        lvl = self._dr_level()

        def draw(lo_hi, nominal):
            # shrink [lo, hi] towards the nominal value by the DR level
            lo, hi = lo_hi
            return self.rng.uniform(nominal + lvl * (lo - nominal), nominal + lvl * (hi - nominal), k)

        dyn = draw(dr["dyn_scale"], 1.0)
        dl = draw(dr["delay_scale"], 1.0)
        p = self._params
        for key, val in {
            "tau_xy": d["tau_xy_s"] * dyn, "a_xy": d["a_max_xy_mps2"] / dyn,
            "tau_z": d["tau_z_s"] * dyn, "a_z": d["a_max_z_mps2"] / dyn,
            "j_xy": d["jerk_xy_mps3"] / dyn, "j_z": d["jerk_z_mps3"] / dyn,
            "k_xy": np.rint(d["delay_xy_s"] * dl / self.sub_dt).astype(int),
            "k_z": np.rint(d["delay_z_s"] * dl / self.sub_dt).astype(int),
            "side50": det["side50_px"] + draw(dr["side50_shift_px"], 0.0),
            "noise": draw(dr["noise_scale"], 1.0),
            "dropout": draw(dr["dropout"], det["dropout"]),
            "bias": draw(dr["depth_bias_frac"], det["depth_bias_frac"]),
            "gust": draw(dr["gust_std_mps"], self.fe["disturbance"]["gust_std_mps"]),
            "lat": draw(dr["latency_scale"], 1.0),
        }.items():
            val = np.broadcast_to(np.asarray(val), (k,))
            if key not in p:
                p[key] = np.zeros(self.n, dtype=val.dtype)
            p[key][idx] = val

    def reset_idx(self, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        k = len(idx)
        scfg = self.cfg["scenarios"]
        if self.fixed is not None:
            scen = sc.fixed(*self.fixed, scfg, self.rng, k)
        else:
            stage = (self.curriculum.current if self.curriculum
                     else scfg["curriculum"]["stages"][-1])
            scen = sc.sample(stage, scfg, self.rng, k)
        self.kind[idx] = scen.kind
        self.rover.reset(idx, scen.kind, scen.speed, scen.radius, scen.omega_max)
        self._sample_params(idx)
        s = self.task["start"]
        h0 = self.rng.uniform(*s["height_m"], k)
        r = s["offset_frac_of_height"] * h0 * np.sqrt(self.rng.uniform(size=k))
        ang = self.rng.uniform(-np.pi, np.pi, k)
        self.pos[idx] = np.stack([r * np.cos(ang), r * np.sin(ang), -h0], axis=1)
        self.v[idx] = 0.0
        self.gust[idx] = 0.0
        self.acc[idx] = 0.0
        self.cmd_hist[idx] = 0.0
        rel = self._rel_pos()[idx]
        self.p_hist[idx] = rel[:, None, :]
        self.att_hist[idx] = 0.0
        self.kf.reset(idx)
        self.prev_action[idx] = 0.0
        self.elapsed[idx] = 0.0
        self.ep_ret[idx] = 0.0
        self.ep_len[idx] = 0
        self.vis_steps[idx] = 0
        self.lost_events[idx] = 0
        self.recoveries[idx] = 0
        self._lost_flag[idx] = False

    def reset_all(self) -> np.ndarray:
        idx = np.arange(self.n)
        self.reset_idx(idx)
        vis = self._perceive(idx)
        self.vis_steps += vis
        return self._obs(vis)

    # ------------------------------------------------------------ physics
    def _rel_pos(self) -> np.ndarray:
        deck = np.concatenate([self.rover.pos, np.zeros((self.n, 1))], axis=1)
        return deck - self.pos

    def _substep(self, cmd: np.ndarray) -> None:
        p, dt = self._params, self.sub_dt
        self.ptr = (self.ptr + 1) % CMD_HIST
        self.cmd_hist[:, self.ptr] = cmd
        rows = np.arange(self.n)
        u = np.empty((self.n, 3))
        u[:, :2] = self.cmd_hist[rows, (self.ptr - p["k_xy"]) % CMD_HIST, :2]
        u[:, 2] = self.cmd_hist[rows, (self.ptr - p["k_z"]) % CMD_HIST, 2]
        a_des = np.empty((self.n, 3))
        a_des[:, :2] = (u[:, :2] - self.v[:, :2]) / p["tau_xy"][:, None]
        norm = np.linalg.norm(a_des[:, :2], axis=1)
        a_des[:, :2] *= np.minimum(1.0, p["a_xy"] / np.maximum(norm, 1e-9))[:, None]
        a_des[:, 2] = np.clip((u[:, 2] - self.v[:, 2]) / p["tau_z"], -p["a_z"], p["a_z"])
        da = a_des - self.acc
        dn = np.linalg.norm(da[:, :2], axis=1)
        da[:, :2] *= np.minimum(1.0, p["j_xy"] * dt / np.maximum(dn, 1e-9))[:, None]
        da[:, 2] = np.clip(da[:, 2], -p["j_z"] * dt, p["j_z"] * dt)
        self.acc = self.acc + da
        a = self.acc
        self.v += a * dt
        # OU gust in the horizontal plane (and 30% of it vertically)
        tau_g = self.fe["disturbance"]["gust_tau_s"]
        scale = (p["gust"] * np.sqrt(2 * dt / tau_g))[:, None] * np.array([1, 1, 0.3])
        self.gust += -self.gust / tau_g * dt + scale * self.rng.standard_normal((self.n, 3))
        self.pos += (self.v + self.gust) * dt
        self.rover.step(dt)
        gain = self.fe["dynamics"]["tilt_gain"]
        pitch = -gain * np.arctan(a[:, 0] / G)       # accelerate forward = nose down
        roll = gain * np.arctan(a[:, 1] / G)         # accelerate right = right wing down
        h = (self.ptr % HIST)
        self.p_hist[:, h] = self._rel_pos()
        self.att_hist[:, h, 0] = roll
        self.att_hist[:, h, 1] = pitch

    # ------------------------------------------------------------ perception
    def _perceive(self, idx: np.ndarray) -> np.ndarray:
        """Simulated detection + KF update for envs in idx. Returns visible (n,)."""
        det, p = self.fe["detection"], self._params
        rows = idx
        k = len(rows)
        age = self.rng.uniform(*det["latency_s"], k) * p["lat"][rows]
        back = np.minimum(np.rint(age / self.sub_dt).astype(int), HIST - 1)
        slot = (self.ptr - back) % HIST
        rel = self.p_hist[rows, slot]                      # marker rel. pos when the image was taken
        roll, pitch = self.att_hist[rows, slot, 0], self.att_hist[rows, slot, 1]
        # marker corners (orientation = rover heading) in the level frame
        th = self.rover.heading[rows]
        inside, side = self._project_square(rel, th, roll, pitch, self.half, np.zeros(2))
        p_det = 1 / (1 + np.exp(-(side - p["side50"][rows]) / det["side_width_px"]))
        p_det *= 1 - p["dropout"][rows]
        u_rand = self.rng.uniform(size=k)
        depth_scale = np.ones(k)
        if self.nested:
            nd = det["nested"]
            in_inside, in_side = self._project_square(rel, th, roll, pitch, self.inner_half, self.inner_off)
            # once the inner marker resolves, it corrupts one cell of the outer (measured handover)
            p_det *= 1 - 1 / (1 + np.exp(-(in_side - nd["outer_lost_inner_side_px"]) / nd["side_width_px"]))
            p_in = 1 / (1 + np.exp(-(in_side - nd["inner_side50_px"] * p["side50"][rows] / det["side50_px"])
                                   / nd["side_width_px"])) * (1 - p["dropout"][rows])
            outer = inside & (u_rand < p_det)
            inner = ~outer & in_inside & (self.rng.uniform(size=k) < p_in)
            visible = outer | inner
            depth_scale[inner] = self.size_ratio    # PnP depth error ~ h^2 / marker size
        else:
            visible = inside & (u_rand < p_det)
        # measurement: centre in camera frame, noise, depth bias (scales the ray)
        centre = (fr.level_to_body(rel, roll, pitch) - self.mount) @ fr.R_BODY_CAM
        hgt = np.maximum(rel[:, 2], 0.1)
        ns = p["noise"][rows]
        centre[:, :2] += self.rng.standard_normal((k, 2)) * (det["lat_std_m_per_m"] * hgt * ns)[:, None]
        centre[:, 2] += self.rng.standard_normal(k) * det["depth_std_m_per_m2"] * hgt ** 2 * ns * depth_scale
        centre *= (1 + p["bias"][rows])[:, None]
        # de-rotate with the attitude at the IMAGE time (+ EKF noise), as gazebo_env does
        an = det["attitude_noise_rad"]
        roll_m = roll + an * self.rng.standard_normal(k)
        pitch_m = pitch + an * self.rng.standard_normal(k)
        meas = fr.body_to_level(centre @ fr.R_BODY_CAM.T + self.mount, roll_m, pitch_m)
        upd = rows[visible]
        accepted = self.kf.update(upd, meas[visible])     # all True unless gating is on
        out = np.zeros(self.n, dtype=bool)
        out[upd[accepted]] = True
        return out

    def _project_square(self, rel, th, roll, pitch, half, off):
        """Project a deck square (half-size `half`, centre offset `off` in the deck frame) into
        the camera. Returns (all corners inside the image, mean side length in px)."""
        c, s = np.cos(th), np.sin(th)
        offs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * half + off
        cx = rel[:, None, 0] + offs[None, :, 0] * c[:, None] - offs[None, :, 1] * s[:, None]
        cy = rel[:, None, 1] + offs[None, :, 0] * s[:, None] + offs[None, :, 1] * c[:, None]
        cz = np.repeat(rel[:, None, 2], 4, axis=1)
        corners = np.stack([cx, cy, cz], axis=-1)          # (k,4,3)
        body = fr.level_to_body(corners, roll[:, None], pitch[:, None])
        cam = (body - self.mount) @ fr.R_BODY_CAM          # R^T applied row-wise
        z = cam[..., 2]
        kk = self.cam.k
        u = kk[0, 0] * cam[..., 0] / np.maximum(z, 1e-6) + kk[0, 2]
        v = kk[1, 1] * cam[..., 1] / np.maximum(z, 1e-6) + kk[1, 2]
        m = self.fe["camera"]["edge_margin_px"]
        inside = ((u > m) & (u < self.cam.width - m) & (v > m) & (v < self.cam.height - m)
                  & (z > 0.1)).all(axis=1)
        side = np.mean(np.hypot(u - np.roll(u, 1, axis=1), v - np.roll(v, 1, axis=1)), axis=1)
        return inside, side

    def _drone_vel_meas(self) -> np.ndarray:
        return self.v + self.gust + 0.03 * self.rng.standard_normal((self.n, 3))

    def _obs(self, visible: np.ndarray) -> np.ndarray:
        return tk.build_obs(self.kf.rel_pos, self.kf.rover_vel, self.kf.initialised,
                            self._drone_vel_meas(), visible, self.kf.since_meas,
                            self.prev_action, self.task)

    def _truth(self) -> tk.StepTruth:
        rel = self._rel_pos()
        rv = self.rover.velocity()
        v_rel = np.concatenate([rv, np.zeros((self.n, 1))], axis=1) - (self.v + self.gust)
        h = -self.pos[:, 2]
        return tk.StepTruth(rel, v_rel, h, h + self.cfg["gazebo_env"]["deck_above_rover_origin_m"])

    # ------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        a = np.clip(action, -1, 1)
        cmd = tk.action_to_velocity(a, self.task, tk.feedforward(self.kf.rover_vel, self.kf.initialised))
        prev = self._truth()
        for _ in range(self.n_sub):
            self._substep(cmd)
        dt = self.task["control_dt_s"]
        self.kf.predict(dt, self._drone_vel_meas())
        visible = self._perceive(np.arange(self.n))
        self.elapsed += dt
        cur = self._truth()
        outcome = tk.check_termination(cur, self.elapsed, self.task)
        rew = tk.reward(prev, cur, outcome, visible, a, self.prev_action, self.task)
        self.prev_action = a.copy()
        self.ep_ret += rew
        self.ep_len += 1
        self.vis_steps += visible
        # marker-loss recovery bookkeeping: a loss = >= 1 s without detection
        lost_now = self.kf.since_meas >= 1.0
        new_loss = lost_now & ~self._lost_flag
        self.lost_events += new_loss
        self.recoveries += self._lost_flag & visible
        self._lost_flag = np.where(visible, False, self._lost_flag | lost_now)
        obs = self._obs(visible)
        done = outcome != tk.RUNNING
        infos: list[dict] = [{} for _ in range(self.n)]
        idx = np.flatnonzero(done)
        if len(idx):
            for i in idx:
                infos[i] = {
                    "terminal_observation": obs[i].copy(),
                    "TimeLimit.truncated": bool(outcome[i] == tk.TIMEOUT),
                    "outcome": int(outcome[i]), "kind": int(self.kind[i]),
                    "speed": float(self.rover.speed[i]),
                    "final_xy_err": float(np.linalg.norm(cur.d[i, :2])),
                    "final_rel_vxy": float(np.linalg.norm(cur.v_rel[i, :2])),
                    "final_vz": float(-cur.v_rel[i, 2]),
                    "time_s": float(self.elapsed[i]),
                    "visible_frac": float(self.vis_steps[i] / max(self.ep_len[i], 1)),
                    "lost_events": int(self.lost_events[i]), "recoveries": int(self.recoveries[i]),
                    "episode": {"r": float(self.ep_ret[i]), "l": int(self.ep_len[i]),
                                "t": float(self.elapsed[i])},
                }
            if self.curriculum is not None:
                self.curriculum.record(outcome[idx] == tk.SUCCESS)
            self.reset_idx(idx)
            vis2 = self._perceive(idx)
            self.vis_steps[idx] += vis2[idx]
            new_obs = self._obs(vis2 | visible)
            obs[idx] = new_obs[idx]
            # the first observation of the new episode uses only the new detection
            obs[idx, 8] = vis2[idx]
        return obs, rew.astype(np.float32), done, infos


class FastVecEnv(VecEnv):
    """Stable-Baselines3 VecEnv over FastLandingCore (no Python loop over envs)."""

    def __init__(self, n: int, cfg: dict, seed: int = 0, curriculum: sc.Curriculum | None = None,
                 dr: bool | None = None) -> None:
        self.core = FastLandingCore(n, cfg, seed, curriculum, dr)
        obs_space = gym.spaces.Box(-np.inf, np.inf, (tk.OBS_DIM,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (tk.ACT_DIM,), np.float32)
        super().__init__(n, obs_space, act_space)
        self._actions = np.zeros((n, tk.ACT_DIM))

    def reset(self):
        return self.core.reset_all()

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        return self.core.step(self._actions)

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self._n(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self._n(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._n(indices)

    def seed(self, seed=None):
        self.core.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def _n(self, indices) -> int:
        if indices is None:
            return self.num_envs
        return 1 if isinstance(indices, int) else len(list(indices))


class FastLandingEnv(gym.Env):
    """Single-env Gymnasium view of the core (for check_env and simple scripts)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int = 0, dr: bool | None = None) -> None:
        self.cfg, self._seed, self._dr = cfg, seed, dr
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (tk.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (tk.ACT_DIM,), np.float32)
        self.core = FastLandingCore(1, cfg, seed, None, dr)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.core = FastLandingCore(1, self.cfg, seed, None, self._dr)
            return self.core.reset_all()[0], {}
        return self.core.reset_all()[0], {}

    def step(self, action):
        obs, rew, done, infos = self.core.step(np.asarray(action, dtype=float)[None])
        info = infos[0]
        if done[0]:
            obs_out = info["terminal_observation"]
            trunc = info["TimeLimit.truncated"]
            return obs_out, float(rew[0]), not trunc, trunc, info
        return obs[0], float(rew[0]), False, False, info
