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
    sim.add_argument("action", choices=["up", "drop-test"])
    sim.add_argument("--gui", action="store_true", help="also open the Gazebo GUI")
    sim.add_argument("--config", default="configs/sim.yaml")
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
        return {"up": demo.world_up, "drop-test": demo.drop_test}[args.action](cfg, args.gui)
    print(f"'{args.command}' is not implemented yet: {_PLANNED[args.command]}")
    return 2
