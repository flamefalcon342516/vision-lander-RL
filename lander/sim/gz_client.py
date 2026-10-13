"""Thin gz-transport client for one Gazebo world: launch, spawn, move, step.

Everything here is a *service call* (request/response, like a function call
across processes) or a *topic subscription* (stream of messages). These are the
same primitives the `gz service` / `gz topic` CLI tools use; this module just
makes them callable from Python so the RL env (Step 10) can reset episodes.

Gazebo world frame is ENU, z up. Poses here are (x, y, z) in metres plus an
optional quaternion; we don't need orientation until Step 2.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.physics_pb2 import Physics
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float


def _service_worker(conn) -> None:
    """Runs in a separate process: performs blocking gz service requests.

    Why a process: gz-transport's Python `request()` blocks while holding the
    GIL. All Nodes in a process share one receive thread, and if any Python
    subscription exists, that thread blocks waiting for the GIL to run the
    callback, so the service *response* is never processed and every request
    times out. (Measured in Step 1: 100% failure with any active subscription,
    0.2 s success without.) A process has its own GIL and its own transport.
    """
    node = Node()
    while (job := conn.recv()) is not None:
        service, payload, req_type, rep_type, timeout_ms = job
        ok, raw = node.request_raw(service, payload, req_type, rep_type, timeout_ms)
        conn.send((ok, raw))


class _ServiceProxy:
    """Forwards requests to _service_worker over a pipe (~1 ms per call)."""

    def __init__(self) -> None:
        # "spawn", not "fork": forking a process that already runs
        # gz-transport threads copies their locks in an undefined state.
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_service_worker, args=(child,), daemon=True)
        self._proc.start()

    def request(self, service: str, req, rep_type, timeout_ms: int):
        self._conn.send((service, req.SerializeToString(), req.DESCRIPTOR.full_name,
                         rep_type.DESCRIPTOR.full_name, timeout_ms))
        ok, raw = self._conn.recv()
        return ok, rep_type.FromString(raw)

    def close(self) -> None:
        if self._proc.is_alive():
            self._conn.send(None)
            self._proc.join(timeout=5)


def launch_gz(world_file: str, headless: bool, verbosity: int) -> subprocess.Popen:
    """Start `gz sim` as a child process.

    -s     : server only (no GUI). Physics and sensors still run; this is what
             training uses. Headless *rendering* (cameras) still needs a GPU/EGL.
    -r     : start running instead of paused.
    We start paused (no -r) so the caller decides when time begins: that keeps
    runs reproducible.
    """
    cmd = ["gz", "sim", "-v", str(verbosity), world_file]
    if headless:
        cmd.insert(2, "-s")
    # New session: lets us kill gz's whole process group (server + GUI) at once.
    return subprocess.Popen(cmd, env=os.environ.copy(), start_new_session=True)


class GzWorld:
    """Client for the services and topics of one running world."""

    def __init__(self, world: str, service_timeout_ms: int, physics_step_s: float) -> None:
        self.world = world
        self._step_s = physics_step_s
        self.timeout_ms = service_timeout_ms
        self._node = Node()  # subscriptions only - see _service_worker
        self._services = _ServiceProxy()
        self._lock = threading.Lock()
        self._sim_time = -1.0
        self._poses: dict[str, tuple[float, Vec3]] = {}
        # Callbacks run on gz-transport's own thread, hence the lock.
        self._node.subscribe(Clock, f"/world/{world}/clock", self._on_clock)
        self._node.subscribe(Pose_V, f"/world/{world}/pose/info", self._on_poses)

    def close(self) -> None:
        self._services.close()

    def __enter__(self) -> "GzWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- topic callbacks ----------
    def _on_clock(self, msg: Clock) -> None:
        with self._lock:
            self._sim_time = msg.sim.sec + msg.sim.nsec * 1e-9

    def _on_poses(self, msg: Pose_V) -> None:
        # pose/info contains every entity with a pose: models (relative to the
        # world) AND their links/visuals (relative to their parent), by bare
        # name. Keying by name is only safe for models whose name doesn't
        # clash with any link name - keep model names distinctive.
        # header.stamp is the *sim* time of the step this pose belongs to.
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        with self._lock:
            for p in msg.pose:
                self._poses[p.name] = (t, Vec3(p.position.x, p.position.y, p.position.z))

    # ---------- state queries ----------
    @property
    def sim_time(self) -> float:
        """Latest simulated time in seconds (-1 before the first clock message)."""
        with self._lock:
            return self._sim_time

    def pose(self, name: str) -> Vec3 | None:
        with self._lock:
            entry = self._poses.get(name)
        return entry[1] if entry else None

    def wait_for_pose(self, name: str, t: float, timeout_s: float) -> Vec3:
        """Block until we hold `name`'s pose from sim time >= t; return it.

        The clock and pose topics arrive independently, so seeing sim_time = t
        does NOT mean the pose for t has arrived yet.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                entry = self._poses.get(name)
            if entry and entry[0] >= t - 5e-4:
                return entry[1]
            if time.monotonic() > deadline:
                raise TimeoutError(f"no pose of '{name}' at sim time {t:.3f}")
            time.sleep(0.0005)

    def wait_until_up(self, timeout_s: float) -> None:
        """Block until the world publishes a clock (i.e. the server is loaded)."""
        deadline = time.monotonic() + timeout_s
        while self.sim_time < 0:
            if time.monotonic() > deadline:
                raise TimeoutError(f"world '{self.world}' not up after {timeout_s} s")
            time.sleep(0.1)

    # ---------- services ----------
    def _call(self, service: str, req) -> None:
        ok, rep = self._services.request(
            f"/world/{self.world}/{service}", req, Boolean, self.timeout_ms
        )
        # Two different failures: `ok` False = no response (service missing,
        # timeout); rep.data False = Gazebo answered "no" (e.g. name clash).
        if not (ok and rep.data):
            raise RuntimeError(f"gz service '{service}' failed (responded={ok})")

    def spawn(self, sdf_uri: str, name: str, at: Vec3, yaw_rad: float = 0.0) -> None:
        """Insert a model. sdf_filename accepts a model:// URI or a file path.

        yaw_rad is about world +z (ENU, counter-clockwise from East).
        """
        req = EntityFactory(sdf_filename=sdf_uri, name=name, allow_renaming=False)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = at.x, at.y, at.z
        # Quaternion of a pure yaw: (w, z) = (cos(yaw/2), sin(yaw/2)).
        # Identity is w=1; an all-zero quaternion is invalid.
        req.pose.orientation.w = math.cos(yaw_rad / 2)
        req.pose.orientation.z = math.sin(yaw_rad / 2)
        self._call("create", req)

    def remove(self, name: str) -> None:
        self._call("remove", Entity(name=name, type=Entity.MODEL))
        with self._lock:
            self._poses.pop(name, None)

    def set_pose(self, name: str, at: Vec3) -> None:
        """Teleport a model. Velocity is NOT reset by this - remember for Step 10."""
        req = Pose(name=name)
        req.position.x, req.position.y, req.position.z = at.x, at.y, at.z
        req.orientation.w = 1.0
        self._call("set_pose", req)

    def pause(self, paused: bool = True) -> None:
        self._call("control", WorldControl(pause=paused))

    def set_real_time_factor(self, rtf: float) -> None:
        """Target sim/wall speed ratio (1.0 = real time).

        Also paces `step()`: with rtf=1, 100 steps of 1 ms take 0.1 s of wall
        time even though the work takes far less.

        Measured (gz-sim 8.15): <real_time_factor>0</real_time_factor> in the
        SDF means "unlimited", but rtf=0 sent through this service is silently
        ignored (the call still returns success). For "as fast as possible" at
        runtime, send a large value instead.
        """
        if rtf <= 0:
            raise ValueError("rtf <= 0 is ignored by /set_physics; pass a large value instead")
        self._call("set_physics", Physics(real_time_factor=rtf, max_step_size=self._step_s))

    def step(self, n: int) -> None:
        """Advance exactly n physics steps while paused (asynchronous!).

        The service returns once the request is *accepted*, not when the steps
        are done. Use wait_for_sim_time() to synchronise.
        """
        self._call("control", WorldControl(pause=True, multi_step=n))

    def wait_for_sim_time(self, t: float, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        # Half a step of slack absorbs float rounding of sec/nsec.
        while self.sim_time < t - 5e-4:
            if time.monotonic() > deadline:
                raise TimeoutError(f"sim time stuck at {self.sim_time:.3f}, wanted {t:.3f}")
            time.sleep(0.001)

    def reset(self) -> None:
        """Reset all models to their initial poses and time to 0.

        Spawned models stay. Note: SITL is not reset with Gazebo; this is
        exactly the reset problem we study in Step 10.
        """
        req = WorldControl(pause=True)
        req.reset.all = True
        self._call("control", req)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
