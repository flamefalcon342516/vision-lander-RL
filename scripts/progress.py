#!/usr/bin/env python3
"""Training progress bars for all runs in results/runs/ (reads TensorBoard logs).

    .venv/bin/python scripts/progress.py          # once
    .venv/bin/python scripts/progress.py --watch  # refresh every 10 s (Ctrl-C to quit)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lander.eval.tb import scalars  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "results" / "runs"
DEFAULT_TOTAL = 20_000_000


def bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def show() -> None:
    print(f"{'run':16} {'progress':32} {'steps':>9}  {'success':>7}  status")
    for run in sorted(RUNS.glob("*_s*")):
        if run.name.startswith("reward_"):
            continue
        done = (run / "final.zip").exists()
        total = json.loads((run / "meta.json").read_text())["steps"] if (run / "meta.json").exists() else DEFAULT_TOTAL
        s, _ = scalars(run, "rollout/ep_rew_mean")
        _, succ = scalars(run, "outcome/success")
        steps = total if done else (s[-1] if s else 0)
        frac = min(steps / total, 1.0)
        print(f"{run.name:16} {bar(frac)} {frac:4.0%} {steps / 1e6:6.1f}M  "
              f"{succ[-1] if succ else 0:7.2f}  {'done' if done else 'training'}")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        try:
            while True:
                print("\033[2J\033[H", end="")
                show()
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        show()
