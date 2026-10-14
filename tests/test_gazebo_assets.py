"""Gazebo assets: SDF validity (fast) and a free-fall physics check (slow)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SDF_FILES = sorted((ROOT / "gazebo").rglob("*.sdf"))
needs_gz = pytest.mark.skipif(shutil.which("gz") is None, reason="gz not installed")


@needs_gz
@pytest.mark.parametrize("sdf", SDF_FILES, ids=lambda p: p.name)
def test_sdf_is_valid(sdf: Path) -> None:
    # `gz sdf -k` checks the file against the SDF schema: catches typos in
    # element names that Gazebo would otherwise silently ignore at load time.
    # gz sdf resolves model:// via SDF_PATH; build it here so the test doesn't
    # depend on the caller having sourced env.sh.
    models = [ROOT / "gazebo" / "models",
              Path.home() / "gz_ws/src/ardupilot_gazebo/models"]
    env = {**os.environ, "SDF_PATH": ":".join(map(str, models))}
    out = subprocess.run(["gz", "sdf", "-k", str(sdf)], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "Valid" in out.stdout, out.stdout + out.stderr


def test_every_model_has_config() -> None:
    # model:// URIs only resolve if model.config sits next to model.sdf.
    for model_sdf in (ROOT / "gazebo" / "models").rglob("model.sdf"):
        assert (model_sdf.parent / "model.config").exists(), model_sdf


def test_world_name_matches_config() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    assert f'<world name="{cfg["world_name"]}">' in (ROOT / cfg["world_file"]).read_text()


@needs_gz
@pytest.mark.slow
def test_crate_free_fall_matches_theory() -> None:
    from lander.sim.demo import drop_test

    cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    assert drop_test(cfg, gui=False) == 0


@needs_gz
@pytest.mark.slow
def test_sitl_takeoff_hover_land_and_frames() -> None:
    """Full Gazebo + SITL loop at 5x: arm, take off, hover, land; ENU->NED agrees."""
    from lander.sim.fly_check import fly_check

    load = lambda name: yaml.safe_load((ROOT / "configs" / name).read_text())  # noqa: E731
    sitl_cfg = load("sitl.yaml")
    sitl_cfg["speedup"] = 5
    assert fly_check(load("sim.yaml"), sitl_cfg, load("paths.yaml"), gui=False, rtf=5) == 0
