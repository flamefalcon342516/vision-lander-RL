#!/bin/bash
# Final runs: residual PPO (configs/ppo.yaml) - 3 seeds of the main setting
# plus two ablations. 3 processes in parallel on CPU (GPU unavailable after
# suspend; ~16k steps/s each). From-scratch PPO attempts are kept as
# reward_v*/relobs/sweep runs and documented in LEARNING.md.
cd "$(dirname "$0")/.."
unset PYTHONPATH
STEPS=${STEPS:-20000000}
run() { .venv/bin/python -m lander train --steps $STEPS "$@" > results/train_$2.log 2>&1; }
run --run main_s0 --seed 0 --no-curriculum &
run --run main_s1 --seed 1 --no-curriculum &
run --run main_s2 --seed 2 --no-curriculum &
wait
run --run curriculum_s0 --seed 0 &
run --run no_dr_s0 --seed 0 --no-curriculum --no-dr &
wait
