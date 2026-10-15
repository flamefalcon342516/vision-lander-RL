"""Step 12: regenerate the README figures from results/eval and results/runs."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lander.eval import plots  # noqa: E402

EVAL, RUNS, OUT = Path("results/eval"), Path("results/runs"), Path("docs/figures")


def rows(name):
    p = EVAL / name / "summary.csv"
    return list(csv.DictReader(open(p))) if p.exists() else None


def episodes(name):
    p = EVAL / name / "episodes.json"
    return json.loads(p.read_text()) if p.exists() else None


made = []
made.append(plots.training_curves(
    {"residual PPO, full DR": [RUNS / f"main_s{i}" for i in range(3)],
     "residual, curriculum": [RUNS / "curriculum_s0"],
     "residual, no DR": [RUNS / "no_dr_s0"]}, OUT / "training_curves.png"))
made.append(plots.success_by_speed(
    {"PID (tuned)": rows("pid_fast_dr"), "residual PPO (seed 0)": rows("res_s0_fast_dr")},
    OUT / "success_fast_dr.png", "Stage A, full domain randomisation (95% CIs)"))
made.append(plots.success_by_speed(
    {"full DR": rows("res_s0_fast_dr"), "curriculum": rows("curriculum_s0_fast_dr"),
     "no DR": rows("no_dr_s0_fast_dr")},
    OUT / "ablations_fast_dr.png", "Ablations, evaluated under full domain randomisation"))
made.append(plots.landing_error(
    {"PID": episodes("pid_fast_dr"), "residual PPO": episodes("res_s0_fast_dr")},
    OUT / "landing_error.png"))
if rows("res_s0_gazebo"):
    made.append(plots.sim_to_sim(
        {"PID": (rows("pid_fast_nominal"), rows("pid_gazebo")),
         "residual PPO": (rows("res_s0_fast_nominal"), rows("res_s0_gazebo"))},
        OUT / "sim_to_sim.png"))
    made.append(plots.success_by_speed(
        {"PID (tuned)": rows("pid_gazebo"), "residual PPO": rows("res_s0_gazebo")},
        OUT / "success_gazebo.png", "Stage B, Gazebo + ArduCopter SITL (3 episodes per cell)"))
print("\n".join(map(str, made)))
