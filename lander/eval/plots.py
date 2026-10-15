"""Step 12 figures for the README (static PNGs, light surface).

Palette: the reference categorical slots 1-3 (blue, orange, aqua), which are
documented to pass all-pairs CVD/normal-vision checks on the light surface;
never more than three series per panel. Grey is reserved for references
(diagonals, theory). One y-axis per panel; multi-measure figures are small
multiples.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PATHS = ["static", "straight", "circle", "figure8", "random"]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
    })


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def training_curves(runs: dict[str, list[Path]], out: Path) -> Path:
    """runs: label -> run dirs (seeds). Panels: curriculum stage, success (per current stage)."""
    from lander.eval.tb import scalars

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for (label, dirs), color in zip(runs.items(), (BLUE, ORANGE, AQUA)):
        for k, (tag, ax) in enumerate((("curriculum/stage", axes[0]), ("outcome/success", axes[1]))):
            curves = [scalars(d, tag) for d in dirs if d.exists()]
            curves = [c for c in curves if c[0]]
            if not curves:
                continue
            grid = np.linspace(0, min(c[0][-1] for c in curves), 200)
            ys = np.array([np.interp(grid, *c) for c in curves])
            ax.plot(grid / 1e6, ys.mean(0), color=color, label=f"{label} (n={len(curves)})")
            if len(curves) > 1:
                ax.fill_between(grid / 1e6, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)
    axes[0].set(title="Curriculum stage reached", xlabel="environment steps (M)", ylabel="stage (0 = static ... 4 = full + DR)")
    axes[1].set(title="Training success rate (on the current stage)", xlabel="environment steps (M)", ylabel="success", ylim=(0, 1))
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    return _save(fig, out)


def success_by_speed(summaries: dict[str, list[dict]], out: Path, title: str) -> Path:
    """Small multiples per path type: success vs rover speed with 95% Wilson CIs."""
    _style()
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for (label, rows), color, dx in zip(summaries.items(), (BLUE, ORANGE, AQUA), (-0.06, 0.06, 0.0)):
        for ax, path in zip(axes, PATHS):
            rs = [r for r in rows if r["path"] == path]
            if not rs:
                continue
            x = np.array([float(r["speed"]) for r in rs]) + dx
            y = np.array([float(r["success"]) for r in rs])
            lo = y - np.array([float(r["success_ci_lo"]) for r in rs])
            hi = np.array([float(r["success_ci_hi"]) for r in rs]) - y
            ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", capsize=3, label=label)
            ax.set_title(path)
            ax.set_xlabel("rover speed (m/s)")
            if path == "static":
                ax.set(xlim=(-0.5, 0.5), xticks=[0])
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(loc="upper right")
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    return _save(fig, out)


def sim_to_sim(pairs: dict[str, tuple[list[dict], list[dict]]], out: Path) -> Path:
    """x = Stage A success, y = Gazebo success, per grid cell; diagonal = no gap."""
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="no gap")
    for (label, (fast, gz)), color in zip(pairs.items(), (BLUE, ORANGE)):
        key = lambda r: (r["path"], round(float(r["speed"]), 1))  # noqa: E731
        f = {key(r): float(r["success"]) for r in fast if r["path"] != "ALL"}
        g = {key(r): float(r["success"]) for r in gz if r["path"] != "ALL"}
        common = sorted(set(f) & set(g))
        ax.scatter([f[k] for k in common], [g[k] for k in common], color=color, s=50,
                   edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="success in Stage A (fast env)",
           ylabel="success in Stage B (Gazebo + SITL)", title="Sim-to-sim transfer per scenario")
    ax.legend(loc="upper left")
    return _save(fig, out)


def sysid_fit(log_path: Path, fit: dict, out: Path, windows: dict[str, tuple[float, float]]) -> Path:
    from lander.sim.sysid import simulate_axis

    _style()
    a = np.load(log_path)
    t = a[:, 0]
    dt = 0.02
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, i, axis in ((axes[0], 0, "x"), (axes[1], 2, "z")):
        lo, hi = windows[axis]
        tg = np.arange(max(lo, t[0]), min(hi, t[-1]), dt)
        u = np.interp(tg, t, a[:, 1 + i])
        v = np.interp(tg, t, a[:, 4 + i])
        p = fit[axis]
        m = simulate_axis(u, dt, p["tau_s"], p["delay_s"], p["a_max_mps2"], p["jerk_mps3"], v[0])
        ax.plot(tg - tg[0], u, color=INK2, lw=1, ls="--", label="command")
        ax.plot(tg - tg[0], v, color=BLUE, label="SITL (measured)")
        ax.plot(tg - tg[0], m, color=ORANGE, label="identified model")
        ax.set(title=f"{axis}-velocity step response (model RMSE {p['rmse_mps']:.3f} m/s)",
               xlabel="time (s)", ylabel="m/s")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    return _save(fig, out)


def detection_range(path: Path, side50: float, width: float, out: Path) -> Path:
    _style()
    d = json.loads(path.read_text())
    h = np.array(d["height_m"], float)
    rate = np.array(d["detect"]) / np.array(d["tries"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    hh = np.linspace(1.5, 12, 200)
    # marker side in px vs height: f*s/h = 320*0.5/h; the measured side_px*h is 155-161
    side = 160.0 / hh
    ax.plot(hh, 1 / (1 + np.exp(-(side - side50) / width)), color=ORANGE, label="Stage A model")
    ax.plot(h, rate, "o", color=BLUE, label="Gazebo (measured)")
    ax.set(xlabel="height above marker (m)", ylabel="detection rate", ylim=(-0.03, 1.05),
           title="ArUco detection range (0.5 m marker, 640 px, 90° FOV)")
    ax.legend(loc="lower left")
    return _save(fig, out)


def landing_error(records: dict[str, list[dict]], out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(0, 0.25, 26)
    for (label, recs), color in zip(records.items(), (BLUE, ORANGE)):
        e = [r["final_xy_err"] for r in recs if r["outcome"] == 1]
        if e:
            ax.hist(e, bins=bins, histtype="step", lw=2, color=color, density=True, cumulative=True,
                    label=f"{label} (median {np.median(e) * 100:.1f} cm)")
    ax.set(xlabel="horizontal error at touchdown (m), successful landings",
           ylabel="fraction of landings ≤ error", title="Landing precision (cumulative)", ylim=(0, 1.02))
    ax.legend(loc="upper left")
    return _save(fig, out)
