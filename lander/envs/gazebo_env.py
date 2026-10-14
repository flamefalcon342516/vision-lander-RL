"""Stage B: the landing task in Gazebo + ArduPilot SITL with the real camera pipeline.

Same observation/action spaces and reward as fast_env (both use envs/task.py).

Reset strategy (chosen after measuring the alternatives, see docs):
  * Restart SITL every episode: >= 4 s wall just for EKF convergence at max
    speed (Step 2), plus re-arm and takeoff. Too slow.
  * Reset world + SITL: Gazebo freezes while SITL is down (lockstep, Step 2).
  * Teleport the drone: the EKF sees a position jump, rejects GPS/innovations,
    and needs time to recover; teleports also don't zero velocity (Step 1).
  * **Chosen: re-position in flight.** SITL keeps running, the drone stays
    armed. Between episodes we stop the rover (optionally teleport it, which
    is safe because nothing estimates its state), fly the drone with a GUIDED
    position setpoint to a random start relative to the rover, and begin.
    "Touchdown" is virtual (task.yaml): the episode ends a few cm above the
    deck, so no physical landing/disarm happens. Reset cost is measured and
    reported in info["reset_wall_s"].
  * If the drone ever crashes physically (flipped / disarmed), a hard reset
    respawns it and restarts SITL.
"""
from __future__ import annotations

import math
import threading
import time

import gymnasium as gym
import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.image_pb2 import Image
from gz.msgs10.light_pb2 import Light
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node

from lander.envs import scenarios as sc
from lander.envs import task as tk
from lander.envs.rover import RoverFleet
from lander.mav import MavLink
from lander.perception import frames as fr
from lander.perception.detector import CameraModel, MarkerDetector
from lander.perception.kalman import MarkerKF
from lander.sim.demo import _stop
from lander.sim.gz_client import GzWorld, Vec3, launch_gz, project_root
from lander.sim.sitl import launch_sitl, stop_process


class _Truth:
    """Latest stamped ground-truth poses from the PosePublisher topics."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}

    def cb(self, name: str):
        def f(m: Pose) -> None:
            t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
            p = np.array([m.position.x, m.position.y, m.position.z])
            q = np.array([m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z])
            with self.lock:
                self.data[name] = (t, p, q)
        return f

    def get(self, name: str):
        with self.lock:
            return self.data.get(name)


class GazeboLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, gui: bool = False, rtf: float = 5.0, seed: int = 0,
                 stage: dict | None = None) -> None:
        """cfg: dict of the loaded YAMLs: sim, sitl, paths, task, camera, marker,
        perception, scenarios, gazebo_env."""
        self.cfg = cfg
        self.task = cfg["task"]
        self.genv = cfg["gazebo_env"]
        self.gui, self.rtf = gui, rtf
        self.rng = np.random.default_rng(seed)
        self.stage = stage or cfg["scenarios"]["curriculum"]["stages"][-1]
        self.fixed_scenario: tuple[str, float] | None = None
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (tk.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (tk.ACT_DIM,), np.float32)
        self.cam = CameraModel.from_config(cfg["camera"])
        self.detector = MarkerDetector(self.cam, cfg["marker"])
        self.mount = np.array(cfg["camera"]["mount_offset_body_frd_m"])
        self.kf = MarkerKF(1, cfg["perception"]["kf"])
        self.rover = RoverFleet(1, self.rng)
        self._started = False
        self._gz = self._sitl = self.world = self.mav = None
        self.truth = _Truth()
        self._img_lock = threading.Lock()
        self._img: Image | None = None
        self.detection_log: list[dict] = []   # for the Stage A noise model

    # ------------------------------------------------------------ lifecycle
    def _start(self) -> None:
        c = self.cfg
        self._gz = launch_gz(str(project_root() / c["sim"]["world_file"]), headless=not self.gui,
                             verbosity=c["sim"]["verbosity"])
        self.world = GzWorld(c["sim"]["world_name"], c["sim"]["service_timeout_ms"],
                             c["sim"]["physics_step_s"])
        self.world.wait_until_up(c["sim"]["startup_timeout_s"])
        self.node = Node()
        self.node.subscribe(Image, c["camera"]["image_topic"], self._on_image)
        self.node.subscribe(Pose, "/model/rover/pose", self.truth.cb("rover"))
        d = c["sitl"]["drone"]
        self.node.subscribe(Pose, f"/model/{d['name']}/pose", self.truth.cb("drone"))
        self.rover_pub = self.node.advertise("/model/rover/cmd_vel", Twist)
        w = c["sim"]["world_name"]
        self.wrench_pub = self.node.advertise(f"/world/{w}/wrench/persistent", EntityWrench)
        self.wrench_clear_pub = self.node.advertise(f"/world/{w}/wrench/clear", Entity)
        self.world.spawn("model://lander_rover", "rover", Vec3(0, 0, self.genv["rover_spawn_z_m"]))
        self._spawn_drone_and_sitl()
        self._started = True

    def _spawn_drone_and_sitl(self) -> None:
        c, d = self.cfg, self.cfg["sitl"]["drone"]
        sx, sy = self.genv["drone_spawn_xy_m"]
        self.spawn_enu = np.array([sx, sy, d["spawn_z_m"]])
        self.world.spawn(d["model_uri"], d["name"], Vec3(*self.spawn_enu), math.radians(d["spawn_yaw_deg"]))
        self.world.set_real_time_factor(self.rtf)
        self.world.pause(False)
        self._sitl = launch_sitl({**c["sitl"], "speedup": self.rtf}, c["paths"], project_root())
        self.mav = MavLink(c["sitl"]["mavlink_url"],
                           rates_hz={"LOCAL_POSITION_NED": 50, "ATTITUDE": 50},
                           clock=lambda: self.world.sim_time)
        self.mav.arm_guided(c["sitl"]["fly_check"]["ekf_ready_timeout_s"])
        self.mav.takeoff(self.genv["takeoff_alt_m"])
        self._wait_sim(self.genv["takeoff_wait_s"])

    def _hard_reset(self) -> None:
        """Drone flipped/disarmed: remove it (while SITL still runs, else Gazebo
        is frozen by lockstep), stop SITL, respawn, restart SITL."""
        self.mav.close()
        self.world.remove(self.cfg["sitl"]["drone"]["name"])
        stop_process(self._sitl)
        self._spawn_drone_and_sitl()

    def close(self) -> None:
        if self.mav:
            self.mav.close()
        if self._sitl:
            stop_process(self._sitl)
        if self.world:
            self.world.close()
        if self._gz:
            _stop(self._gz)
        self._started = False

    # ------------------------------------------------------------ helpers
    def _on_image(self, m: Image) -> None:
        with self._img_lock:
            self._img = m

    def _wait_sim(self, dt: float) -> None:
        t_end = self.world.sim_time + dt
        deadline = time.monotonic() + max(30.0, 5 * dt)
        while self.world.sim_time < t_end:
            if time.monotonic() > deadline:
                raise TimeoutError("simulation stalled")
            time.sleep(0.0005)

    def _deck_enu(self) -> np.ndarray:
        _, p, _ = self.truth.get("rover")
        return p + np.array([0, 0, self.genv["deck_above_rover_origin_m"]])

    def _true_state(self) -> tuple[tk.StepTruth, np.ndarray]:
        _, dp, dq = self.truth.get("drone")
        _, _, yaw = self.mav.attitude()
        d_ned = fr.enu_to_ned(self._deck_enu() - dp)
        d_level = fr.ned_to_level(d_ned, yaw)
        rover_v = self.rover.velocity()[0]            # NE plane (x=N, y=E), see rover.py
        v_rel_ned = np.array([rover_v[0], rover_v[1], 0.0]) - self.mav.vel_ned()
        v_rel = fr.ned_to_level(v_rel_ned, yaw)
        h = dp[2] - self._deck_enu()[2]
        t = tk.StepTruth(d_level[None], v_rel[None], np.array([h]), np.array([dp[2]]))
        return t, dq

    def _perceive(self) -> tuple[bool, float]:
        """Latest camera frame -> detection -> level frame -> KF update."""
        with self._img_lock:
            m = self._img
        if m is None:
            return False, float("nan")
        img_t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        det = self.detector.detect(img)
        if det is None:
            return False, img_t
        # attitude at the moment the image was captured, not now (see MavLink.attitude_at)
        roll, pitch, _ = self.mav.attitude_at(img_t)
        p_level = fr.body_to_level(fr.cam_to_body(det.p_cam, self.mount), roll, pitch)
        self.kf.update(np.array([0]), p_level[None])
        truth, _ = self._true_state()
        self.detection_log.append({"meas": p_level, "truth": truth.d[0], "h": float(truth.h[0]),
                                   "age_s": self.world.sim_time - img_t,
                                   "reproj_px": det.reproj_err_px, "side_px": det.side_px})
        return True, img_t

    def _obs(self, visible: bool) -> np.ndarray:
        _, _, yaw = self.mav.attitude()
        dv = fr.ned_to_level(self.mav.vel_ned(), yaw)[None]
        return tk.build_obs(self.kf.rel_pos, self.kf.rover_vel, self.kf.initialised, dv,
                            np.array([visible]), self.kf.since_meas, self.prev_action[None],
                            self.task)[0]

    def _rover_cmd(self) -> None:
        # rover.py heading is NED (from north, clockwise) -> ENU yaw-rate sign flips.
        self.rover_pub.publish(Twist(linear={"x": float(self.rover.v[0])},
                                     angular={"z": float(-self.rover.omega[0])}))

    # ------------------------------------------------------------ Step 11: DR
    def _dr_reset(self) -> None:
        dr = self.genv["dr"]
        self.wind = np.zeros(2)
        self.gust = np.zeros(2)
        if not dr["enabled"]:
            return
        ang = self.rng.uniform(-np.pi, np.pi)
        self.wind = self.rng.uniform(*dr["wind_mean_mps"]) * np.array([np.cos(ang), np.sin(ang)])
        self.gust_std = self.rng.uniform(*dr["gust_std_mps"])
        # lighting: sun elevation/azimuth and intensity
        el = np.radians(self.rng.uniform(*dr["sun_elevation_deg"]))
        az = self.rng.uniform(-np.pi, np.pi)
        diff = self.rng.uniform(*dr["sun_diffuse"])
        light = Light(name="sun", type=Light.DIRECTIONAL, cast_shadows=True)
        light.direction.x, light.direction.y, light.direction.z = (
            np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el))
        light.diffuse.r = light.diffuse.g = light.diffuse.b = diff
        light.diffuse.a = 1.0
        self.world._services.request(f"/world/{self.world.world}/light_config", light, Boolean,
                                     self.cfg["sim"]["service_timeout_ms"])
        self.ep_light = float(diff)

    def _apply_wind(self, dt: float) -> None:
        dr = self.genv["dr"]
        if not dr["enabled"]:
            return
        tau = dr["gust_tau_s"]
        self.gust += -self.gust / tau * dt + self.gust_std * np.sqrt(2 * dt / tau) * self.rng.standard_normal(2)
        v_ned = self.mav.vel_ned()
        v_enu = np.array([v_ned[1], v_ned[0]])
        w_enu = self.wind + self.gust
        v_air = w_enu - v_enu
        f = dr["drag_n_per_m2s2"] * np.linalg.norm(v_air) * v_air
        msg = EntityWrench()
        msg.entity.name = dr["drone_link"]
        msg.entity.type = Entity.LINK
        msg.wrench.force.x, msg.wrench.force.y = float(f[0]), float(f[1])
        # persistent wrenches stay applied every physics step; replace, don't stack
        self.wrench_clear_pub.publish(msg.entity)
        self.wrench_pub.publish(msg)

    # ------------------------------------------------------------ gym API
    def set_scenario(self, kind: str | None, speed: float = 0.0) -> None:
        """Evaluation hook: fix the next episodes' path type and speed."""
        self.fixed_scenario = None if kind is None else (kind, speed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.rover.rng = self.rng
        if not self._started:
            self._start()
        wall0 = time.monotonic()
        roll, pitch, _ = self.mav.attitude()
        if (not self.mav.armed or abs(roll) > 1.0 or abs(pitch) > 1.0
                or self.mav.mode != "GUIDED"):
            self._hard_reset()
        g = self.genv
        # 1. stop the rover; bring it home if it wandered off
        self.rover.v[:] = 0
        self.rover.omega[:] = 0
        self._rover_cmd()
        scfg = self.cfg["scenarios"]
        scen = (sc.fixed(*self.fixed_scenario, scfg, self.rng, 1) if self.fixed_scenario
                else sc.sample(self.stage, scfg, self.rng, 1))
        self.rover.reset(np.array([0]), scen.kind, scen.speed, scen.radius, scen.omega_max)
        rp = self.truth.get("rover")[1]
        if np.linalg.norm(rp[:2]) > g["rover_recenter_dist_m"]:
            rp = np.array([0.0, 0.0, rp[2]])
        # heading (NED, from north clockwise) -> Gazebo ENU yaw
        self.world.set_pose("rover", Vec3(rp[0], rp[1], g["rover_spawn_z_m"]),
                            math.pi / 2 - float(self.rover.heading[0]))
        self._wait_sim(0.2)
        # 2. fly to a random start relative to the deck
        s = self.task["start"]
        h0 = self.rng.uniform(*s["height_m"])
        r = s["offset_frac_of_height"] * h0 * np.sqrt(self.rng.uniform())
        ang = self.rng.uniform(-np.pi, np.pi)
        deck = self._deck_enu()
        target_enu = deck + np.array([r * np.cos(ang), r * np.sin(ang), h0])
        target_ned = fr.enu_to_ned(target_enu - self.spawn_enu)
        t0 = self.world.sim_time
        while True:
            self.mav.send_position_ned(target_ned)
            self._wait_sim(0.1)
            err = np.linalg.norm(self.mav.pos_ned() - target_ned)
            if err < g["reposition_tol_m"] and np.linalg.norm(self.mav.vel_ned()) < g["reposition_vel_tol_mps"]:
                break
            if self.world.sim_time - t0 > g["reposition_timeout_s"]:
                break   # start from wherever we are; the episode still runs
        # 3. start the episode
        self._dr_reset()
        self.kf.reset(np.array([0]))
        self.prev_action = np.zeros(3)
        self.elapsed = 0.0
        self.mav.send_velocity_level(np.zeros(3))
        self._rover_cmd()
        self._wait_sim(self.task["control_dt_s"])
        yaw = self.mav.attitude()[2]
        dv = fr.ned_to_level(self.mav.vel_ned(), yaw)[None]
        self.kf.predict(self.task["control_dt_s"], dv)
        visible, _ = self._perceive()
        self.prev_truth, _ = self._true_state()
        self.ep_info = {"kind": int(scen.kind[0]), "speed": float(scen.speed[0]),
                        "start_h": float(h0), "reset_wall_s": time.monotonic() - wall0,
                        "visible_steps": int(visible), "steps": 0,
                        "lost_events": 0, "recoveries": 0}
        self._lost_flag = False
        return self._obs(visible), dict(self.ep_info)

    def step(self, action: np.ndarray):
        dt = self.task["control_dt_s"]
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        ff = tk.feedforward(self.kf.rover_vel, self.kf.initialised)
        self.mav.send_velocity_level(tk.action_to_velocity(a[None], self.task, ff)[0])
        self.rover.step(dt)
        self._rover_cmd()
        self._apply_wind(dt)
        wall0 = time.monotonic()
        self._wait_sim(dt)
        step_wall = time.monotonic() - wall0
        _, _, yaw = self.mav.attitude()
        self.kf.predict(dt, fr.ned_to_level(self.mav.vel_ned(), yaw)[None])
        visible, _ = self._perceive()
        self.elapsed += dt
        cur, _ = self._true_state()
        outcome = tk.check_termination(cur, np.array([self.elapsed]), self.task)
        rew = tk.reward(self.prev_truth, cur, outcome, np.array([visible]), a[None],
                        self.prev_action[None], self.task)[0]
        self.prev_truth = cur
        self.prev_action = a
        self.ep_info["steps"] += 1
        self.ep_info["visible_steps"] += int(visible)
        # marker-loss recovery bookkeeping (same definition as fast_env): a
        # loss = >= 1 s without detection; recovery = detected again afterwards
        lost_now = bool(self.kf.since_meas[0] >= 1.0)
        if lost_now and not self._lost_flag:
            self.ep_info["lost_events"] += 1
        if self._lost_flag and visible:
            self.ep_info["recoveries"] += 1
        self._lost_flag = False if visible else (self._lost_flag or lost_now)
        o = int(outcome[0])
        terminated = o not in (tk.RUNNING, tk.TIMEOUT)
        truncated = o == tk.TIMEOUT
        info = {"outcome": o, "step_wall_s": step_wall}
        if terminated or truncated:
            info.update(self.ep_info, final_xy_err=float(np.linalg.norm(cur.d[0, :2])),
                        final_rel_vxy=float(np.linalg.norm(cur.v_rel[0, :2])),
                        final_vz=float(-cur.v_rel[0, 2]), time_s=self.elapsed,
                        visible_frac=self.ep_info["visible_steps"] / max(self.ep_info["steps"] + 1, 1),
                        speed=self.ep_info["speed"], kind=self.ep_info["kind"])
            # climb away from the deck right away so a virtual touchdown never
            # turns into a physical contact while the next reset is prepared
            self.mav.send_velocity_level(np.array([0.0, 0.0, -1.0]))
        return self._obs(visible), float(rew), terminated, truncated, info
