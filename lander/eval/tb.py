"""Read scalars from TensorBoard event files (for plots and progress checks)."""
from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    files = sorted((run_dir / "tb").rglob("events.out.tfevents.*"))
    steps, vals = [], []
    for f in files:
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                steps.append(e.step)
                vals.append(e.value)
    return steps, vals
