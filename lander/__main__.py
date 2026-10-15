"""Allows `python -m lander ...`."""
import signal
import sys

from lander.cli import main

# `kill <pid>` sends SIGTERM, which by default skips `finally` blocks and atexit hooks,
# leaving Gazebo (its own session) and SITL running. Turn it into a normal exit.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

raise SystemExit(main())
