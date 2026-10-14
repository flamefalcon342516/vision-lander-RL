#!/bin/bash
# Main runs (3 seeds) + ablations. Sequential: one GPU.
set -e
cd "$(dirname "$0")/.."
unset PYTHONPATH
for s in 0 1 2; do .venv/bin/python -m lander train --run main_s$s --seed $s; done
.venv/bin/python -m lander train --run no_curriculum_s0 --seed 0 --no-curriculum
.venv/bin/python -m lander train --run no_dr_s0 --seed 0 --no-dr
