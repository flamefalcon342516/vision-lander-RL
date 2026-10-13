# Vision-Based Autonomous Landing on a Moving Platform with PPO

A quadcopter (Gazebo Harmonic + ArduPilot SITL) learns with PPO to track and land
on a moving rover carrying an ArUco marker, using only its downward camera, then
deploys to a real drone via a companion computer.

> Work in progress — built step by step. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
> for the design and [LEARNING.md](LEARNING.md) for the concepts behind each step.

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
