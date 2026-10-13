PY ?= .venv/bin/python

# ROS 2 setup.bash leaks ROS pytest plugins via PYTHONPATH; we don't use ROS here.
unexport PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = python
export GZ_IP = 127.0.0.1

.PHONY: help venv doctor test test-fast sim-up sim-gui drop-test

help:
	@echo "make venv | doctor | test | test-fast | sim-up | sim-gui | drop-test"

venv:
	uv venv --system-site-packages --python /usr/bin/python3.10 .venv
	uv pip install --python $(PY) -r requirements.txt \
	  --index-url https://download.pytorch.org/whl/cu126 \
	  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

doctor:
	$(PY) -m lander doctor

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

# Gazebo needs the resource/plugin paths from env.sh: run these after `source env.sh`.
sim-up:
	$(PY) -m lander sim up

sim-gui:
	$(PY) -m lander sim up --gui

drop-test:
	$(PY) -m lander sim drop-test
