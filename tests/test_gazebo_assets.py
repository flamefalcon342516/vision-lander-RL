"""Gazebo assets: SDF validity (fast) and a free-fall physics check (slow)."""
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
    out = subprocess.run(["gz", "sdf", "-k", str(sdf)], capture_output=True, text=True)
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
