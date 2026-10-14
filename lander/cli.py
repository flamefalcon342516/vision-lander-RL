"""Command-line entry point: `python -m lander <command>`.

Subcommands are added as their step is built; the placeholders state which
step will implement them so the CLI doubles as a roadmap.
"""
from __future__ import annotations

import argparse

# command -> step that implements it
_PLANNED = {
    "train": "Step 9 (PPO in the fast env) / Step 11 (Gazebo fine-tune)",
    "eval": "Step 12 (metrics, baselines, ablations)",
    "deploy": "Step 13 (companion-computer runtime)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lander", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="verify the toolchain on this machine")
    doctor.add_argument("--paths", default="configs/paths.yaml", help="machine paths YAML")
    sim = sub.add_parser("sim", help="Gazebo tools")
    sim.add_argument("action", choices=["up", "drop-test", "fly-check", "sysid", "make-marker"])
    sim.add_argument("--gui", action="store_true", help="also open the Gazebo GUI")
    sim.add_argument("--drone", action="store_true", help="up: also spawn the iris")
    sim.add_argument("--rtf", type=float, default=1.0, help="fly-check: Gazebo real-time factor")
    sim.add_argument("--speedup", type=float, help="fly-check: override SITL --speedup")
    sim.add_argument("--config", default="configs/sim.yaml")
    sim.add_argument("--sitl-config", default="configs/sitl.yaml")
    sim.add_argument("--paths", default="configs/paths.yaml")
    for name, step in _PLANNED.items():
        sub.add_parser(name, help=f"not yet implemented: {step}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        from lander.doctor import run_doctor

        return run_doctor(args.paths)
    if args.command == "sim":
        import yaml

        from lander.sim import demo

        cfg = yaml.safe_load(open(args.config))
        sitl_cfg = yaml.safe_load(open(args.sitl_config))
        if args.action == "up":
            return demo.world_up(cfg, args.gui, sitl_cfg["drone"] if args.drone else None)
        if args.action == "drop-test":
            return demo.drop_test(cfg, args.gui)
        paths = yaml.safe_load(open(args.paths))
        if args.action == "sysid":
            from pathlib import Path

            from lander.sim.sysid import run_sysid

            res = run_sysid(cfg, sitl_cfg, paths, Path("results/sysid"), args.rtf)
            print(yaml.safe_dump(res, sort_keys=False))
            return 0
        if args.action == "make-marker":
            from pathlib import Path

            from lander.sim.assets import make_marker_texture

            out = Path("gazebo/models/lander_rover/materials/textures/aruco_marker.png")
            print(make_marker_texture(yaml.safe_load(open("configs/marker.yaml")), out))
            return 0
        from lander.sim.fly_check import fly_check

        if args.speedup:
            sitl_cfg["speedup"] = args.speedup
        return fly_check(cfg, sitl_cfg, paths, args.gui, args.rtf)
    print(f"'{args.command}' is not implemented yet: {_PLANNED[args.command]}")
    return 2
