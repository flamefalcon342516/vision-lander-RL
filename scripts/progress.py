#!/usr/bin/env python3
"""Progress bars for everything running: PPO training (TensorBoard logs in results/runs/)
and Gazebo evaluations (episode lines in the eval's log file).

    .venv/bin/python scripts/progress.py          # once
    .venv/bin/python scripts/progress.py --watch  # refresh every 5 s (Ctrl-C to quit)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "runs"
DEFAULT_TOTAL = 20_000_000
GRID_CELLS = 13          # static + 4 paths x 3 speeds (configs/scenarios.yaml)
EPISODE = re.compile(r"(LANDED|✘|gazebo \w+@)")


def bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def training() -> list[str]:
    from lander.eval.tb import scalars

    out = []
    for run in sorted(RUNS.glob("*")):
        if not (run / "tb").exists() or (run / "final.zip").exists():
            continue       # finished, or not a training run
        total = json.loads((run / "meta.json").read_text())["steps"] if (run / "meta.json").exists() else DEFAULT_TOTAL
        s, _ = scalars(run, "rollout/ep_rew_mean")
        _, succ = scalars(run, "outcome/success")
        steps = s[-1] if s else 0
        frac = min(steps / total, 1.0)
        age = time.time() - max(f.stat().st_mtime for f in (run / "tb").rglob("*"))
        if age > 6 * 3600:
            continue       # abandoned experiment, not a live run
        status = "training" if age < 120 else f"stalled? last write {fmt_time(age)} ago"
        out.append(f"  {run.name:18} {bar(frac)} {frac:4.0%}  {steps / 1e6:5.1f}M/{total / 1e6:.0f}M  "
                   f"success {succ[-1] if succ else 0:.2f}  {status}")
    return out


def _arg(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def evaluations() -> list[str]:
    out = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue
        if "lander" not in argv or "eval" not in argv or _arg(argv, "--stage") != "gazebo":
            continue
        cells = _arg(argv, "--cells")
        n_cells = len([c for c in cells.split(",") if c.strip()]) if cells else (1 if "--kind" in argv else GRID_CELLS)
        total = n_cells * int(_arg(argv, "--episodes", 500))
        name = f"{Path(_arg(argv, '--policy', '?')).name} -> {_arg(argv, '--out', '?')}"
        try:
            log = Path(os.readlink(f"/proc/{pid}/fd/1"))
            lines = log.read_text(errors="ignore").splitlines() if log.is_file() else []
        except OSError:
            lines = []
        if not lines:
            out.append(f"  {name:32} running (output goes to a terminal, so no count)")
            continue
        eps = [ln for ln in lines if EPISODE.search(ln) and "▶" not in ln]
        done = len(eps)
        landed = sum(("LANDED" in ln) or (": success" in ln) for ln in eps)
        started = Path(f"/proc/{pid}").stat().st_ctime
        el = time.time() - started
        eta = f"ETA {fmt_time(el / done * (total - done))}" if done else "starting"
        out.append(f"  {name:32} {bar(done / total)} {done:3d}/{total}  landed {landed}/{done}  "
                   f"{fmt_time(el)} elapsed, {eta}")
    return out


def show() -> None:
    t, e = training(), evaluations()
    print(time.strftime("%H:%M:%S"))
    print("Training (PPO)")
    print("\n".join(t) if t else "  nothing training")
    print("Gazebo evaluations")
    print("\n".join(e) if e else "  nothing running")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        try:
            while True:
                print("\033[2J\033[H", end="")
                show()
                time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        show()
