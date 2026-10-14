"""Load every project YAML into one dict keyed by file stem."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FILES = ("sim", "sitl", "paths", "task", "camera", "marker", "perception", "scenarios",
         "gazebo_env", "fast_env", "baseline", "ppo")


def load_all(config_dir: Path | None = None) -> dict:
    d = config_dir or ROOT / "configs"
    return {name: yaml.safe_load((d / f"{name}.yaml").read_text())
            for name in FILES if (d / f"{name}.yaml").exists()}
