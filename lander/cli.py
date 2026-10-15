"""Command-line entry point: `python -m lander <command>`.

Subcommands are added as their step is built; the placeholders state which
step will implement them so the CLI doubles as a roadmap.
"""
from __future__ import annotations

import argparse



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lander", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="verify the toolchain on this machine")
    doctor.add_argument("--paths", default="configs/paths.yaml", help="machine paths YAML")
    tr = sub.add_parser("train", help="PPO in the fast env (Stage A)")
    tr.add_argument("--run", required=True, help="run name under results/runs/")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--steps", type=int, default=None)
    tr.add_argument("--no-curriculum", action="store_true")
    tr.add_argument("--no-dr", action="store_true")
    ev = sub.add_parser("eval", help="evaluate a policy over the scenario grid")
    ev.add_argument("--policy", required=True, help="'pid', or a PPO .zip / run dir")
    ev.add_argument("--stage", choices=["fast", "gazebo"], default="fast")
    ev.add_argument("--episodes", type=int, default=500, help="per grid cell")
    ev.add_argument("--no-dr", action="store_true", help="fast: nominal parameters")
    ev.add_argument("--seed", type=int, default=12345)
    ev.add_argument("--rtf", type=float, default=5.0)
    ev.add_argument("--out", required=True, help="output name under results/eval/")
    ev.add_argument("--gui", action="store_true", help="gazebo: open the Gazebo GUI to watch")
    ev.add_argument("--kind", default=None, help="gazebo: only this path type (static/straight/circle/figure8/random)")
    ev.add_argument("--speed", type=float, default=None, help="gazebo: with --kind, rover speed in m/s")
    ev.add_argument("--cells", default=None,
                    help='gazebo: scenario sequence "kind:speed,...", e.g. static:0,straight:0.5,circle:1.5')
    sim = sub.add_parser("sim", help="Gazebo tools")
    sim.add_argument("action", choices=["up", "drop-test", "fly-check", "sysid", "make-marker"])
    sim.add_argument("--gui", action="store_true", help="also open the Gazebo GUI")
    sim.add_argument("--drone", action="store_true", help="up: also spawn the iris")
    sim.add_argument("--rtf", type=float, default=1.0, help="fly-check: Gazebo real-time factor")
    sim.add_argument("--speedup", type=float, help="fly-check: override SITL --speedup")
    sim.add_argument("--config", default="configs/sim.yaml")
    sim.add_argument("--sitl-config", default="configs/sitl.yaml")
    sim.add_argument("--paths", default="configs/paths.yaml")
    dp = sub.add_parser("deploy", help="ONNX export, timing, SITL rehearsal, hardware run")
    dp.add_argument("action", choices=["export", "timing", "rehearse", "run"])
    dp.add_argument("--run", default="main_s0", help="training run under results/runs/")
    dp.add_argument("--onnx", default=None, help="default: results/runs/<run>/policy.onnx")
    dp.add_argument("--kind", default="straight")
    dp.add_argument("--speed", type=float, default=1.0)
    dp.add_argument("--episodes", type=int, default=3)
    dp.add_argument("--rtf", type=float, default=5.0)
    dp.add_argument("--gui", action="store_true", help="rehearse: open the Gazebo GUI to watch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        from lander.doctor import run_doctor

        return run_doctor(args.paths)
    if args.command == "train":
        from pathlib import Path

        from lander.config import load_all
        from lander.training.train import train

        out = train(load_all(), Path("results/runs") / args.run, args.seed,
                    curriculum=not args.no_curriculum, dr=not args.no_dr, total_steps=args.steps)
        print(f"saved {out}")
        return 0
    if args.command == "eval":
        import json
        from pathlib import Path

        from lander.config import load_all
        from lander.eval import evaluate as ev_mod

        cfg = load_all()
        if args.policy == "pid":
            pol = ev_mod.pid_policy(cfg)
        else:
            p = Path(args.policy)
            pol = ev_mod.load_policy(cfg, p)
        if args.stage == "fast":
            recs = ev_mod.eval_fast(pol, cfg, args.episodes, args.seed, dr=not args.no_dr)
        else:
            grid = [(args.kind, args.speed or 0.0)] if args.kind else None
            if args.cells:
                grid = [(c.split(":")[0].strip(), float(c.split(":")[1]) if ":" in c else 0.0)
                        for c in args.cells.split(",") if c.strip()]
            recs = ev_mod.eval_gazebo(pol, cfg, args.episodes, args.seed, rtf=args.rtf, grid=grid, gui=args.gui)
        out = Path("results/eval") / args.out
        out.mkdir(parents=True, exist_ok=True)
        rows = ev_mod.summarize(recs)
        ev_mod.write_csv(rows, out / "summary.csv")
        (out / "episodes.json").write_text(json.dumps(recs, default=float))
        md = ev_mod.markdown_table(rows, f"{args.policy} - {args.stage}")
        (out / "summary.md").write_text(md + "\n")
        from lander.eval import report

        print(report.summary(rows, f"{Path(args.policy).name} · {args.stage}"))
        print(report.dim(f"  saved {out}/summary.md, summary.csv, episodes.json"))
        return 0
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
    if args.command == "deploy":
        return _deploy(args)
    return 2


def _deploy(args) -> int:
    import json
    from pathlib import Path

    from lander.config import load_all

    cfg = load_all()
    run = Path("results/runs") / args.run
    onnx = Path(args.onnx) if args.onnx else run / "policy.onnx"
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    res_scale = meta.get("residual_scale") if meta.get("residual") else None
    if args.action == "export":
        from lander.deploy.export import export_onnx

        print(f"exported {export_onnx(run / 'final.zip', onnx)} (ONNX == SB3 to 1e-4)")
        return 0
    if args.action == "timing":
        from lander.deploy.export import time_inference

        for th in (1, 4):
            print(json.dumps(time_inference(onnx, th)))
        return 0
    if args.action == "rehearse":
        from lander.deploy.rehearsal import rehearse

        out = Path("results/rehearsal")
        res = rehearse(cfg, onnx, args.kind, args.speed, args.episodes, out, rtf=args.rtf, gui=args.gui,
                       residual_scale=res_scale)
        (out / f"summary_{args.kind}_{args.speed}.json").write_text(json.dumps(res, indent=1))
        return 0
    # hardware: real camera + serial link. Never exercised in this repo's tests.
    from lander.deploy.runtime import CvFrameSource, Runtime, RuntimeConfig
    from lander.deploy.safety import SafetyConfig
    from lander.mav import MavLink

    d = cfg["deploy"]
    mav = MavLink(d["hardware_url"], rates_hz={"LOCAL_POSITION_NED": 50, "ATTITUDE": 50, "HEARTBEAT": 2})
    frames = CvFrameSource(d["camera_device"], cfg["camera"]["width"], cfg["camera"]["height"])
    rt = Runtime(cfg, onnx, frames, mav, SafetyConfig(**d["safety"]), RuntimeConfig(**d["runtime"]),
                 Path("results/flight_logs") / f"flight_{int(__import__('time').time())}.csv",
                 residual_scale=res_scale)
    print(json.dumps(rt.run(), indent=1))
    return 0
