# Vision-Based Autonomous Landing on a Moving Platform with PPO

A quadcopter (Gazebo Harmonic + ArduPilot SITL) learns with PPO to track and land
on a moving rover carrying an ArUco marker, using its downward camera, with an
ONNX runtime and safety layer for a real companion computer.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and
[LEARNING.md](LEARNING.md) for the concepts and measurements behind each step.

## Results (measured 2026-10-15)
The final controller is **residual PPO**: a tuned PID visual-servoing controller plus a
learned correction, `a = clip(a_PID + 0.5·a_res)`. PPO from scratch failed (dive / hover
local optima); that negative result is documented in LEARNING.md Step 9.

| Controller | Stage A, full DR | Stage A, nominal | Gazebo + SITL |
|---|---|---|---|
| PID (tuned) | 31% (n=13000) | 48% | 54% (39–68%, n=39), 7.3 s, 4.8 cm |
| **Residual PPO**, 3 seeds | **37–38%** (n=6500 each) | **51%** | 51% (36–66%, n=39), **4.7 s**, 11.8 cm |
| ablation: curriculum | 36% | 48% | – |
| ablation: no DR in training | 38% | 50% | – |
| pixel student (DAgger, Step 15) | 34.9% (n=5200) | – | – |

Grid: static + {straight, circle, figure-8, random} × {0.5, 1.5, 2.5} m/s; success =
touchdown within the marker with low relative velocity.

- **Stage A:** the residual beats the PID by ~7 points on every seed and lands ~35% faster;
  the gains are on curved and fast paths (circle 1.5 m/s: 28% vs 18%).
- **Gazebo:** no measurable gain (51% vs 54%, CIs overlap almost fully at 3 episodes per cell);
  the residual lands faster but less centred — a sim-to-sim gap.
- **Ablations:** neither DR nor the curriculum helped in this setup; the curriculum slightly hurt
  on curves (it re-learns the static "dive" habit, as with PPO from scratch).
- **Pixel student:** a CNN on two 96×72 frames keeps ~90% of the teacher's success. Its frames
  come from a synthetic renderer, not the Gazebo camera, so it is a feasibility result.
- **Runtime:** ONNX == SB3 to 1e-4; 0.021 ms mean / 0.044 ms max per inference (laptop CPU).

![Stage A success by path and speed](docs/figures/success_fast_dr.png)
![Gazebo success by path and speed](docs/figures/success_gazebo.png)
![Sim-to-sim transfer](docs/figures/sim_to_sim.png)
![Ablations](docs/figures/ablations_fast_dr.png)
![Training curves](docs/figures/training_curves.png)
![Landing precision](docs/figures/landing_error.png)

Reproduce: `bash scripts/train_all.sh`, `python -m lander eval --policy results/runs/main_s0 --stage fast|gazebo --out NAME`,
`python scripts/run_dagger.py`, `python scripts/make_figures.py`.

## Quick start
```bash
make venv            # create .venv (torch cu126, SB3, Gymnasium, OpenCV, pymavlink)
source env.sh        # Gazebo/ArduPilot paths, gz loopback, protobuf backend
python -m lander doctor
make test
```

## Stack (verified 2026-10-13)
Ubuntu 22.04 · ROS 2 Humble (optional) · Gazebo Harmonic 8.15 · ArduCopter SITL
(master @ 6870d06) · ardupilot_gazebo (@ 50e62a5) · Python 3.10 · torch 2.14.1+cu126 ·
Stable-Baselines3 2.9.0 · Gymnasium 1.4.0 · OpenCV 4.10 · pymavlink 2.4.50
