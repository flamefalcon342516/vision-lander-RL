"""Terminal output for evaluations: scenario banners, one line per landing, and a
boxed summary. Plain ANSI (no extra dependency); colour only on a real terminal
and never when NO_COLOR is set. The files in results/eval/ stay plain Markdown/CSV.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def bold(s: str) -> str: return _c("1", s)
def dim(s: str) -> str: return _c("2", s)
def green(s: str) -> str: return _c("32", s)
def red(s: str) -> str: return _c("31", s)
def yellow(s: str) -> str: return _c("33", s)
def cyan(s: str) -> str: return _c("36", s)


OUTCOME_TEXT = {
    "success": "landed on the deck",
    "miss": "touched down off the deck",
    "crash": "crashed",
    "out_of_bounds": "flew out of the arena",
    "timeout": "ran out of time",
}


def _speed(kind: str, speed: float) -> str:
    return "parked" if kind == "static" else f"{speed:.1f} m/s"


def banner(i: int, n: int, kind: str, speed: float) -> str:
    title = f" {i}/{n}  {kind} rover · {_speed(kind, speed)} "
    return "\n" + cyan(bold("▶" + title)) + dim("  take off → follow the rover → land")


def episode_line(kind: str, speed: float, outcome: str, xy_err: float, t: float) -> str:
    if outcome == "success":
        return "   " + green(bold("✔ LANDED")) + f"   {xy_err * 100:4.0f} cm off centre   {t:4.1f} s"
    return "   " + red(bold("✘ " + outcome.upper().replace("_", " "))) + "   " + dim(OUTCOME_TEXT.get(outcome, outcome))


def _bar(p: float, width: int = 10) -> str:
    full = int(round(p * width))
    s = "█" * full + "░" * (width - full)
    return green(s) if p >= 0.7 else (yellow(s) if p >= 0.35 else red(s))


def summary(rows: list[dict], title: str) -> str:
    """rows: evaluate.summarize() output (the last row is ALL)."""
    head = f" {'rover path':<11}{'speed':>10}  {'landed':<17}{'off centre':>11}{'time':>8} "
    width = len(head)
    line = "─" * width
    out = ["", bold(f"┌{line}┐"), bold("│") + bold(f" {title}".ljust(width)) + bold("│"),
           bold(f"├{line}┤"), bold("│") + dim(head) + bold("│"), bold(f"├{line}┤")]
    for r in rows:
        is_all = r["speed"] < 0
        if is_all:
            out.append(bold(f"├{line}┤"))
        k = int(round(r["success"] * r["n"]))
        landed = f"{k}/{r['n']}".rjust(5)
        cm = "-" if np.isnan(r["xy_err_m"]) else f"{r['xy_err_m'] * 100:.0f} cm"
        tt = "-" if np.isnan(r["time_s"]) else f"{r['time_s']:.1f} s"
        name = "ALL" if is_all else r["path"]
        sp = "" if is_all else _speed(r["path"], float(r["speed"]))
        cells = f" {name:<11}{sp:>10}  {landed} {_bar(r['success'])} {cm:>11}{tt:>8} "
        visible = len(f" {name:<11}{sp:>10}  {landed} {'x' * 10} {cm:>11}{tt:>8} ")
        cells += " " * (width - visible)
        out.append(bold("│") + (bold(cells) if is_all else cells) + bold("│"))
    out.append(bold(f"└{line}┘"))
    a = rows[-1]
    pct = f"{a['success']:.0%}"
    out.append(f"  {bold(pct)} landed overall  "
               + dim(f"(95% CI {max(a['success_ci_lo'], 0):.0%}–{a['success_ci_hi']:.0%}, n={a['n']})"))
    fails = {k: a[f"frac_{k}"] for k in ("miss", "crash", "out_of_bounds", "timeout") if a.get(f"frac_{k}", 0) > 0}
    if fails:
        out.append(dim("  failures: " + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in fails.items())))
    return "\n".join(out)
